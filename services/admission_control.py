"""Admission control helpers for queue back-pressure."""

import asyncio
from datetime import datetime, timezone
from typing import Optional, Tuple

from fastapi import HTTPException, Request, status

from config import Settings, get_settings
from services.job_repository import get_job_repository

PENDING_STATUSES = {"pending", "generating", "queued"}


def _compute_queue_health(settings: Settings) -> Tuple[int, Optional[float]]:
    repository = get_job_repository()
    jobs = repository.list_jobs(limit=500)
    now = datetime.now(timezone.utc)
    pending = 0
    oldest_age: Optional[float] = None

    for record in jobs:
        status = record.get("status")
        if status not in PENDING_STATUSES:
            continue
        pending += 1
        created_at = record.get("queued_at") or record.get("created_at")
        if not created_at:
            continue
        try:
            ts = datetime.fromisoformat(created_at)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = (now - ts).total_seconds()
        except Exception:
            continue
        if oldest_age is None or age > oldest_age:
            oldest_age = age
    return pending, oldest_age


async def ensure_queue_capacity(request: Request) -> None:
    """Reject requests when queue length or wait time exceeds thresholds."""
    settings = get_settings()
    if settings.QUEUE_MAX_PENDING <= 0 and settings.QUEUE_MAX_AGE_SECONDS <= 0:
        return

    pending, oldest_age = await asyncio.to_thread(_compute_queue_health, settings)

    if settings.QUEUE_MAX_PENDING > 0 and pending >= settings.QUEUE_MAX_PENDING:
        retry_after = max(int(settings.QUEUE_RETRY_DELAY), 30)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue is saturated. Please retry later.",
            headers={"Retry-After": str(retry_after)},
        )

    if settings.QUEUE_MAX_AGE_SECONDS > 0 and oldest_age is not None and oldest_age >= settings.QUEUE_MAX_AGE_SECONDS:
        retry_after = max(int(settings.QUEUE_RETRY_DELAY), 30)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue is experiencing high latency. Please retry later.",
            headers={"Retry-After": str(retry_after)},
        )

