"""SAQ worker entrypoints for Refrakt jobs."""

from __future__ import annotations

from typing import Any, Dict

from services.job_service import JobService
from services.r2_service import R2Service
from services.websocket_service import WebSocketService


_job_service = JobService(R2Service(), WebSocketService())


async def execute_training_job(ctx: Dict[str, Any], *, job_id: str, source: str = "api") -> Dict[str, Any]:
    """SAQ task that runs a training job by ID."""
    await _job_service.run_job(job_id)
    return {"job_id": job_id, "source": source}


