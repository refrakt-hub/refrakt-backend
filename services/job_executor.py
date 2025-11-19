"""SAQ worker entrypoints for Refrakt jobs."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from config import get_settings
from services.job_service import JobService
from services.r2_service import R2Service
from services.websocket_service import WebSocketService
from services.gpu_scheduler import get_gpu_scheduler

logger = logging.getLogger(__name__)

_job_service = JobService(R2Service(), WebSocketService())


async def execute_training_job(ctx: Dict[str, Any], *, job_id: str, source: str = "api") -> Dict[str, Any]:
    """
    SAQ task that runs a training job by ID with GPU scheduling.
    
    This function:
    1. Allocates GPU resources for the job
    2. Sets CUDA_VISIBLE_DEVICES environment variable
    3. Executes the training job
    4. Releases GPU resources when done
    """
    settings = get_settings()
    allocation = None
    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    # Get job configuration for GPU allocation
    job_record = _job_service.repository.get_job(job_id, include_logs=False)
    if not job_record:
        raise ValueError(f"Job {job_id} not found")
    
    config = job_record.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Job {job_id} missing configuration")
    
    # Allocate GPU resources if scheduler is enabled
    if settings.GPU_SCHEDULER_ENABLED:
        scheduler = get_gpu_scheduler()
        allocation = scheduler.allocate_gpu(job_id, config)
        
        if not allocation:
            error_msg = f"Could not allocate GPU resources for job {job_id}"
            logger.error(error_msg)
            _job_service.update_job_status(job_id, "error", error=error_msg)
            raise RuntimeError(error_msg)
        
        # Set CUDA_VISIBLE_DEVICES to restrict job to allocated GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = allocation.cuda_visible_devices
        logger.info(
            f"Job {job_id} allocated GPU device {allocation.device_id} "
            f"(estimated memory: {allocation.estimated_memory_mb}MB)"
        )
    
    try:
        # Execute the training job
    await _job_service.run_job(job_id)
    except Exception as e:
        logger.error(f"Job {job_id} failed with exception: {e}", exc_info=True)
        raise
    finally:
        # Always release GPU allocation and restore environment
        if settings.GPU_SCHEDULER_ENABLED and allocation:
            scheduler = get_gpu_scheduler()
            scheduler.release_gpu(job_id)
            logger.info(f"Released GPU allocation for job {job_id}")
        
        # Restore original CUDA_VISIBLE_DEVICES
        if original_cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    
    return {"job_id": job_id, "source": source}


