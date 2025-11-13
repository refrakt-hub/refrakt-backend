"""Shared queue helpers for background job orchestration."""

from __future__ import annotations

from functools import lru_cache

from saq import Queue

from config import get_settings
from services.metrics import record_job_enqueued


@lru_cache(maxsize=1)
def get_queue() -> Queue:
    """Return a singleton SAQ queue instance."""
    settings = get_settings()
    if not settings.QUEUE_URL:
        raise RuntimeError("QUEUE_URL must be configured before accessing the task queue")
    return Queue.from_url(
        settings.QUEUE_URL,
        name=settings.QUEUE_NAME,
    )


async def enqueue_training_job(job_id: str, *, source: str = "api") -> str:
    """Publish a training job to the queue and return the job identifier."""
    settings = get_settings()
    queue = get_queue()
    job = await queue.enqueue(
        "train_job",
        job_id=job_id,
        source=source,
        timeout=settings.QUEUE_DEFAULT_TIMEOUT,
        retries=settings.QUEUE_RETRY_LIMIT,
        retry_delay=settings.QUEUE_RETRY_DELAY,
        retry_backoff=settings.QUEUE_RETRY_BACKOFF,
    )
    record_job_enqueued(source)
    return job.key


