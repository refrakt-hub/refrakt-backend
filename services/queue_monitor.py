"""Background task that samples job repository for queue metrics."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from config import Settings, get_settings
from services.job_repository import get_job_repository, JobRepository
from services.metrics import update_queue_snapshot

PENDING_STATUSES = {"pending", "generating", "queued"}
RUNNING_STATUSES = {"running", "uploading"}


class QueueMonitor:
    """Periodically aggregates queue metrics for Prometheus gauges."""

    def __init__(
        self,
        repository: Optional[JobRepository] = None,
        settings: Optional[Settings] = None,
    ):
        self._settings = settings or get_settings()
        self._repository = repository or get_job_repository()
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if not self._settings.PROMETHEUS_ENABLED or self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        interval = max(self._settings.PROMETHEUS_QUEUE_POLL_INTERVAL, 1.0)
        while not self._stop_event.is_set():
            await self._sample_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    async def _sample_once(self) -> None:
        try:
            jobs = await asyncio.to_thread(self._repository.list_jobs, 500)
        except Exception:
            return

        total_jobs = len(jobs)
        pending_jobs = 0
        running_jobs = 0
        oldest_pending_seconds: Optional[float] = None
        now = datetime.now(timezone.utc)

        for job in jobs:
            status = job.get("status")
            if status in PENDING_STATUSES:
                pending_jobs += 1
                queued_iso = job.get("queued_at") or job.get("created_at")
                if queued_iso:
                    age = _age_seconds(queued_iso, now)
                    if age is not None:
                        if oldest_pending_seconds is None or age > oldest_pending_seconds:
                            oldest_pending_seconds = age
            elif status in RUNNING_STATUSES:
                running_jobs += 1

        update_queue_snapshot(
            total_jobs=total_jobs,
            pending_jobs=pending_jobs,
            running_jobs=running_jobs,
            oldest_pending_seconds=oldest_pending_seconds,
        )


def _age_seconds(iso_str: str, now: datetime) -> Optional[float]:
    try:
        timestamp = datetime.fromisoformat(iso_str)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return (now - timestamp).total_seconds()

