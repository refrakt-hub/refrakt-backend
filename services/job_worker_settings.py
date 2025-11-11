"""SAQ worker configuration for processing Refrakt training jobs."""

from __future__ import annotations

from typing import Dict

from config import get_settings
from services.job_executor import execute_training_job
from services.queue import get_queue


def build_settings() -> Dict[str, object]:
    settings = get_settings()

    return {
        "queue": get_queue(),
        "functions": [("train_job", execute_training_job)],
        "concurrency": settings.QUEUE_WORKER_CONCURRENCY,
    }


settings: Dict[str, object] = build_settings()


