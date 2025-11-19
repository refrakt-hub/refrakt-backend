"""SAQ worker configuration for processing Refrakt training jobs."""

from __future__ import annotations

from typing import Dict

from config import get_settings
from services.job_executor import execute_training_job
from services.queue import get_queue


def build_settings() -> Dict[str, object]:
    settings = get_settings()

    # Determine worker concurrency based on GPU scheduler if enabled
    # This allows multiple jobs to run concurrently on the GPU
    if settings.GPU_SCHEDULER_ENABLED:
        # Use GPU scheduler's max concurrent jobs, but don't exceed worker concurrency setting
        from services.gpu_scheduler import get_gpu_scheduler
        scheduler = get_gpu_scheduler()
        concurrency = min(
            scheduler.max_concurrent_jobs,
            settings.QUEUE_WORKER_CONCURRENCY if settings.QUEUE_WORKER_CONCURRENCY > 0 else scheduler.max_concurrent_jobs
        )
    else:
        # Fall back to configured worker concurrency (typically 1 for sequential execution)
        concurrency = settings.QUEUE_WORKER_CONCURRENCY

    return {
        "queue": get_queue(),
        "functions": [("train_job", execute_training_job)],
        "concurrency": concurrency,
    }


settings: Dict[str, object] = build_settings()


