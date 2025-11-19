"""GPU scheduling service for managing concurrent training jobs."""

import logging
import os
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from redis import Redis

from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GPUAllocation:
    """Represents a GPU allocation for a job."""
    job_id: str
    device_id: int  # Which GPU (0, 1, 2, etc.)
    estimated_memory_mb: int
    allocated_at: str  # ISO format string for Redis serialization
    cuda_visible_devices: str  # e.g., "0" or "1"
    worker_id: str  # Which worker process allocated this


class GPUScheduler:
    """Manages GPU allocations for concurrent training jobs using Redis for coordination."""
    
    _REDIS_KEY_PREFIX = "gpu:allocations"
    _REDIS_LOCK_KEY = "gpu:lock"
    _LOCK_TIMEOUT = 5.0  # seconds
    
    def __init__(self, total_gpu_memory_mb: int = 40960, reserved_memory_mb: int = 2048, worker_id: Optional[str] = None):
        """
        Initialize GPU scheduler.
        
        Args:
            total_gpu_memory_mb: Total GPU memory in MB (A100 40GB = 40960 MB)
            reserved_memory_mb: Reserved memory for system/CUDA overhead
            worker_id: Unique identifier for this worker process
        """
        self.total_gpu_memory_mb = total_gpu_memory_mb
        self.available_memory_mb = total_gpu_memory_mb - reserved_memory_mb
        self.max_concurrent_jobs = 3  # Will be overridden by settings
        self.worker_id = worker_id or f"worker-{os.getpid()}-{int(time.time())}"
        
        # Initialize Redis connection
        settings = get_settings()
        if not settings.QUEUE_URL:
            raise RuntimeError("QUEUE_URL must be configured for GPU scheduler")
        self._redis = Redis.from_url(
            settings.QUEUE_URL,
            decode_responses=True,
        )
        
        logger.info(
            f"GPU Scheduler initialized (worker_id={self.worker_id}): "
            f"total={total_gpu_memory_mb}MB, reserved={reserved_memory_mb}MB"
        )
    
    def _get_allocations_key(self) -> str:
        """Get Redis key for storing allocations."""
        return f"{self._REDIS_KEY_PREFIX}:active"
    
    def _get_allocation_key(self, job_id: str) -> str:
        """Get Redis key for a specific job allocation."""
        return f"{self._REDIS_KEY_PREFIX}:job:{job_id}"
    
    def _load_allocations(self) -> Dict[str, GPUAllocation]:
        """Load all active allocations from Redis."""
        allocations = {}
        try:
            # Get all allocation keys
            pattern = f"{self._REDIS_KEY_PREFIX}:job:*"
            keys = self._redis.keys(pattern)
            
            for key in keys:
                data = self._redis.get(key)
                if data:
                    try:
                        alloc_dict = json.loads(data)
                        allocations[alloc_dict["job_id"]] = GPUAllocation(**alloc_dict)
                    except Exception as e:
                        logger.warning(f"Failed to parse allocation from {key}: {e}")
        except Exception as e:
            logger.error(f"Error loading allocations from Redis: {e}")
        return allocations
    
    def _save_allocation(self, allocation: GPUAllocation):
        """Save an allocation to Redis."""
        try:
            key = self._get_allocation_key(allocation.job_id)
            data = json.dumps(asdict(allocation))
            # Set with expiration (2 hours) to prevent stale allocations
            self._redis.setex(key, 7200, data)
        except Exception as e:
            logger.error(f"Error saving allocation to Redis: {e}")
    
    def _delete_allocation(self, job_id: str):
        """Delete an allocation from Redis."""
        try:
            key = self._get_allocation_key(job_id)
            self._redis.delete(key)
        except Exception as e:
            logger.error(f"Error deleting allocation from Redis: {e}")
    
    def estimate_job_memory(self, config: dict) -> int:
        """
        Estimate GPU memory requirements for a job based on config.
        
        Args:
            config: Job configuration dictionary
            
        Returns:
            Estimated memory in MB
        """
        # Extract model information
        model_cfg = config.get("model", {})
        model_params = model_cfg.get("params", {})
        
        # Get model name/type
        model_name = model_cfg.get("name", "").lower()
        
        # Get batch size
        dataloader_cfg = config.get("dataloader", {})
        batch_size = dataloader_cfg.get("params", {}).get("batch_size", 32)
        
        # Rough memory estimation based on model type
        # These are conservative estimates for common model architectures
        base_memory_mb = {
            "resnet": 2000,      # ~2GB for ResNet variants
            "vit": 4000,         # ~4GB for Vision Transformers
            "convnext": 3000,    # ~3GB for ConvNeXt
            "mae": 5000,         # ~5GB for MAE
            "dino": 4500,        # ~4.5GB for DINO
            "autoencoder": 1500, # ~1.5GB for autoencoders
            "simclr": 3000,      # ~3GB for SimCLR
            "srgan": 4000,       # ~4GB for SRGAN
        }
        
        # Get base memory for model type
        estimated_mb = 2000  # Default fallback for unknown models
        for model_key, memory in base_memory_mb.items():
            if model_key in model_name:
                estimated_mb = memory
                break
        
        # Scale by batch size (rough approximation)
        # Larger batch sizes require more memory for activations
        batch_multiplier = max(1.0, batch_size / 32)
        estimated_mb = int(estimated_mb * batch_multiplier)
        
        # Add overhead for gradients, optimizer states, activations
        # Total = model * 4 (weights + gradients + optimizer + activations)
        # This is a conservative estimate
        estimated_mb = estimated_mb * 4
        
        # Add safety margin (20%) to prevent OOM
        estimated_mb = int(estimated_mb * 1.2)
        
        logger.info(
            f"Estimated GPU memory for job: {estimated_mb}MB "
            f"(model: {model_name}, batch_size: {batch_size})"
        )
        
        return estimated_mb
    
    def can_allocate(self, estimated_memory_mb: int) -> Tuple[bool, Optional[str]]:
        """
        Check if a job can be allocated GPU resources.
        
        Args:
            estimated_memory_mb: Estimated memory requirement in MB
            
        Returns:
            Tuple of (can_allocate, reason_if_not)
        """
        # Load current allocations from Redis (shared state)
        allocations = self._load_allocations()
        
        # Check concurrent job limit
        if len(allocations) >= self.max_concurrent_jobs:
            return False, f"Maximum concurrent jobs ({self.max_concurrent_jobs}) reached"
        
        # Check available memory
        used_memory = sum(alloc.estimated_memory_mb for alloc in allocations.values())
        available = self.available_memory_mb - used_memory
        
        if estimated_memory_mb > available:
            return False, f"Insufficient GPU memory (need {estimated_memory_mb}MB, have {available}MB)"
        
        return True, None
    
    def allocate_gpu(self, job_id: str, config: dict) -> Optional[GPUAllocation]:
        """
        Allocate GPU resources for a job.
        
        This method uses Redis for coordination across multiple workers.
        
        Args:
            job_id: Job identifier
            config: Job configuration
            
        Returns:
            GPUAllocation if successful, None otherwise
        """
        estimated_memory = self.estimate_job_memory(config)
        
        # Check if allocation is possible (loads current state from Redis)
        can_allocate, reason = self.can_allocate(estimated_memory)
        if not can_allocate:
            logger.warning(f"Cannot allocate GPU for job {job_id}: {reason}")
            return None
        
        # Create allocation
        allocation = GPUAllocation(
            job_id=job_id,
            device_id=0,  # For now, use GPU 0 (single GPU setup)
            estimated_memory_mb=estimated_memory,
            allocated_at=datetime.utcnow().isoformat(),
            cuda_visible_devices="0",
            worker_id=self.worker_id,
        )
        
        # Save to Redis (this is the source of truth)
        self._save_allocation(allocation)
        
        # Double-check we didn't exceed limits (race condition protection)
        allocations = self._load_allocations()
        if len(allocations) > self.max_concurrent_jobs:
            # Another worker allocated at the same time, release this one
            self._delete_allocation(job_id)
            logger.warning(
                f"Race condition detected: too many allocations after allocation. "
                f"Released allocation for job {job_id}"
            )
            return None
        
        used_memory = sum(alloc.estimated_memory_mb for alloc in allocations.values())
        if used_memory > self.available_memory_mb:
            # Memory exceeded, release this allocation
            self._delete_allocation(job_id)
            logger.warning(
                f"Memory limit exceeded after allocation. Released allocation for job {job_id}"
            )
            return None
        
        logger.info(
            f"Allocated GPU for job {job_id} (worker={self.worker_id}): "
            f"{estimated_memory}MB on device {allocation.device_id} "
            f"(total active jobs: {len(allocations)})"
        )
        
        return allocation
    
    def release_gpu(self, job_id: str) -> Optional[GPUAllocation]:
        """
        Release GPU allocation for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            GPUAllocation that was released, or None if not found
        """
        allocations = self._load_allocations()
        if job_id in allocations:
            allocation = allocations[job_id]
            self._delete_allocation(job_id)
            logger.info(
                f"Released GPU allocation for job {job_id} (worker={self.worker_id}) "
                f"(remaining active jobs: {len(allocations) - 1})"
            )
            return allocation
        logger.warning(f"Attempted to release GPU for job {job_id} but no allocation found")
        return None
    
    def get_current_usage(self) -> Dict:
        """Get current GPU usage statistics from Redis."""
        allocations = self._load_allocations()
        used_memory = sum(alloc.estimated_memory_mb for alloc in allocations.values())
        available = self.available_memory_mb - used_memory
        utilization = (used_memory / self.available_memory_mb) * 100 if self.available_memory_mb > 0 else 0
        
        return {
            "active_jobs": len(allocations),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "used_memory_mb": used_memory,
            "available_memory_mb": available,
            "total_memory_mb": self.total_gpu_memory_mb,
            "reserved_memory_mb": self.total_gpu_memory_mb - self.available_memory_mb,
            "utilization_percent": round(utilization, 2),
            "allocations": [
                {
                    "job_id": alloc.job_id,
                    "device_id": alloc.device_id,
                    "memory_mb": alloc.estimated_memory_mb,
                    "allocated_at": alloc.allocated_at,
                    "cuda_visible_devices": alloc.cuda_visible_devices,
                    "worker_id": alloc.worker_id,
                }
                for alloc in allocations.values()
            ]
        }


# Global scheduler instance
_gpu_scheduler: Optional[GPUScheduler] = None


def get_gpu_scheduler() -> GPUScheduler:
    """Get or create GPU scheduler instance."""
    global _gpu_scheduler
    if _gpu_scheduler is None:
        from config import get_settings
        settings = get_settings()
        _gpu_scheduler = GPUScheduler(
            total_gpu_memory_mb=settings.GPU_TOTAL_MEMORY_MB,
            reserved_memory_mb=settings.GPU_RESERVED_MEMORY_MB,
        )
        _gpu_scheduler.max_concurrent_jobs = settings.GPU_MAX_CONCURRENT_JOBS
        logger.info(
            f"GPU Scheduler initialized: "
            f"total={settings.GPU_TOTAL_MEMORY_MB}MB, "
            f"reserved={settings.GPU_RESERVED_MEMORY_MB}MB, "
            f"max_concurrent={settings.GPU_MAX_CONCURRENT_JOBS}"
        )
    return _gpu_scheduler
