"""Prometheus metrics definitions and helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram

JOB_ENQUEUED_TOTAL = Counter(
    "refrakt_job_enqueued_total",
    "Number of training jobs enqueued",
    ("source",),
)

JOB_STATUS_TRANSITIONS_TOTAL = Counter(
    "refrakt_job_status_transitions_total",
    "Number of job status transitions",
    ("from_status", "to_status"),
)

JOB_QUEUE_WAIT_SECONDS = Histogram(
    "refrakt_job_queue_wait_seconds",
    "Time a job spends waiting in the queue before running",
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1200, float("inf")),
)

JOB_RUN_DURATION_SECONDS = Histogram(
    "refrakt_job_run_duration_seconds",
    "Time a job spends executing (running -> terminal state)",
    buckets=(30, 60, 120, 300, 600, 1200, 1800, 3600, float("inf")),
)

JOB_TOTAL_GAUGE = Gauge(
    "refrakt_jobs_total",
    "Total tracked jobs in repository",
)

QUEUE_PENDING_GAUGE = Gauge(
    "refrakt_queue_pending_jobs",
    "Jobs waiting to start (pending/generating/queued)",
)

QUEUE_RUNNING_GAUGE = Gauge(
    "refrakt_queue_running_jobs",
    "Jobs currently executing (running/uploading)",
)

QUEUE_OLDEST_WAIT_SECONDS = Gauge(
    "refrakt_queue_oldest_pending_seconds",
    "Age in seconds of the oldest pending job",
)


def record_job_enqueued(source: str) -> None:
    """Increment counter when a job is enqueued."""
    try:
        JOB_ENQUEUED_TOTAL.labels(source=source).inc()
    except Exception:
        pass


def record_job_status_transition(
    previous_job: Optional[Dict[str, Any]],
    new_status: str,
    timestamp: datetime,
) -> None:
    """Capture status transition metrics and durations."""
    previous_status = (previous_job or {}).get("status") or "unknown"

    try:
        JOB_STATUS_TRANSITIONS_TOTAL.labels(previous_status, new_status).inc()
    except Exception:
        pass

    if previous_job is None:
        return

    try:
        if new_status == "running":
            queued_at = _first_present(
                previous_job.get("queued_at"),
                previous_job.get("created_at"),
            )
            if queued_at:
                wait_seconds = _duration_seconds(queued_at, timestamp)
                if wait_seconds is not None and wait_seconds >= 0:
                    JOB_QUEUE_WAIT_SECONDS.observe(wait_seconds)

        if new_status in {"completed", "error"}:
            started_at = _first_present(
                previous_job.get("started_at"),
                previous_job.get("running_at"),
            )
            if started_at:
                run_seconds = _duration_seconds(started_at, timestamp)
                if run_seconds is not None and run_seconds >= 0:
                    JOB_RUN_DURATION_SECONDS.observe(run_seconds)
    except Exception:
        pass


def update_queue_snapshot(
    *,
    total_jobs: int,
    pending_jobs: int,
    running_jobs: int,
    oldest_pending_seconds: Optional[float],
) -> None:
    """Update gauges describing current queue health."""
    try:
        JOB_TOTAL_GAUGE.set(total_jobs)
        QUEUE_PENDING_GAUGE.set(pending_jobs)
        QUEUE_RUNNING_GAUGE.set(running_jobs)
        if oldest_pending_seconds is None:
            QUEUE_OLDEST_WAIT_SECONDS.set(0)
        else:
            QUEUE_OLDEST_WAIT_SECONDS.set(max(oldest_pending_seconds, 0))
    except Exception:
        pass


def _first_present(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _duration_seconds(start_iso: str, end_time: datetime) -> Optional[float]:
    try:
        start_dt = datetime.fromisoformat(start_iso)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    delta = end_time - start_dt
    return delta.total_seconds()

