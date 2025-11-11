"""Persistence layer for training job metadata."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from redis import Redis

from config import get_settings


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _utc_now_epoch() -> float:
    return time.time()


class JobRepository:
    """Read/write job state using Redis."""

    _JOB_KEY_TEMPLATE = "job:{job_id}"
    _JOB_LOGS_KEY_TEMPLATE = "job:{job_id}:logs"
    _JOBS_INDEX_KEY = "jobs:index"

    def __init__(self, redis_client: Optional[Redis] = None):
        if redis_client is None:
            settings = get_settings()
            if not settings.QUEUE_URL:
                raise RuntimeError("QUEUE_URL must be configured before using JobRepository")
            redis_client = Redis.from_url(
                settings.QUEUE_URL,
                decode_responses=True,
            )
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_job(self, job_id: str, prompt: str, user_id: str) -> Dict[str, Any]:
        now_iso = _utc_now_iso()
        job_record = {
            "job_id": job_id,
            "prompt": prompt,
            "user_id": user_id,
            "status": "pending",
            "created_at": now_iso,
            "updated_at": now_iso,
            "r2_uploaded": False,
            "local_cleaned": False,
            "queue_job_id": None,
        }

        job_key = self._job_key(job_id)
        pipeline = self._redis.pipeline()
        pipeline.set(job_key, json.dumps(job_record))
        pipeline.zadd(self._JOBS_INDEX_KEY, {job_id: _utc_now_epoch()})
        pipeline.delete(self._job_logs_key(job_id))
        pipeline.execute()
        return job_record

    def update_job(self, job_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
        job = self.get_job(job_id, include_logs=False)
        if not job:
            return None

        if fields:
            job.update(fields)
        job["updated_at"] = _utc_now_iso()

        self._redis.set(self._job_key(job_id), json.dumps(job))
        return job

    def get_job(self, job_id: str, include_logs: bool = True, logs_limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        raw = self._redis.get(self._job_key(job_id))
        if not raw:
            return None
        job: Dict[str, Any] = json.loads(raw)
        if include_logs:
            job["logs"] = self.get_logs(job_id, limit=logs_limit)
        return job

    def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        job_ids = self._redis.zrevrange(self._JOBS_INDEX_KEY, 0, limit - 1)
        jobs: List[Dict[str, Any]] = []
        for job_id in job_ids:
            job = self.get_job(job_id, include_logs=False)
            if job:
                jobs.append(job)
        return jobs

    def append_log(self, job_id: str, log_line: str) -> None:
        self._redis.rpush(self._job_logs_key(job_id), log_line)
        # keep list bounded (last 500 lines)
        self._redis.ltrim(self._job_logs_key(job_id), -500, -1)

    def get_logs(self, job_id: str, limit: Optional[int] = None) -> List[str]:
        if limit is None:
            logs = self._redis.lrange(self._job_logs_key(job_id), 0, -1)
        else:
            logs = self._redis.lrange(self._job_logs_key(job_id), -limit, -1)
        return logs or []

    def delete_job(self, job_id: str) -> None:
        pipeline = self._redis.pipeline()
        pipeline.delete(self._job_key(job_id))
        pipeline.delete(self._job_logs_key(job_id))
        pipeline.zrem(self._JOBS_INDEX_KEY, job_id)
        pipeline.execute()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _job_key(self, job_id: str) -> str:
        return self._JOB_KEY_TEMPLATE.format(job_id=job_id)

    def _job_logs_key(self, job_id: str) -> str:
        return self._JOB_LOGS_KEY_TEMPLATE.format(job_id=job_id)


_repository: Optional[JobRepository] = None


def get_job_repository() -> JobRepository:
    global _repository
    if _repository is None:
        _repository = JobRepository()
    return _repository


