"""Build contextual summaries for training jobs."""

from __future__ import annotations

from typing import Dict, List, Optional

from services.job_service import JobService


class JobContextService:
    """Generate textual context snippets for job-related questions."""

    def __init__(self, job_service: JobService, max_logs: int = 5):
        self._job_service = job_service
        self._max_logs = max_logs

    def build_job_context(self, job_ids: List[str]) -> List[str]:
        contexts: List[str] = []
        seen = set()
        for job_id in job_ids:
            if job_id in seen:
                continue
            seen.add(job_id)
            job = self._job_service.get_job(job_id)
            if not job:
                contexts.append(f"[job:{job_id}] No job found for this identifier.")
                continue
            contexts.append(self._format_job_summary(job))
        return contexts

    def _format_job_summary(self, job: Dict[str, object]) -> str:
        job_id = job.get("job_id", "unknown")
        lines = [
            f"[job:{job_id}]",
            f"Status: {job.get('status')}",
            f"Created at: {job.get('created_at')}",
            f"Updated at: {job.get('updated_at')}",
        ]

        prompt = job.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            lines.append(f"Original prompt: {prompt.strip()[:500]}")

        dataset_meta = job.get("dataset") or {}
        if isinstance(dataset_meta, dict) and dataset_meta:
            dataset_parts = []
            name = dataset_meta.get("original_name") or dataset_meta.get("name")
            if name:
                dataset_parts.append(f"name={name}")
            num_classes = dataset_meta.get("num_classes")
            if num_classes:
                dataset_parts.append(f"classes={num_classes}")
            task_type = dataset_meta.get("task_type")
            if task_type:
                dataset_parts.append(f"task={task_type}")
            if dataset_parts:
                lines.append("Dataset: " + ", ".join(str(part) for part in dataset_parts))

        error = job.get("error")
        if isinstance(error, str) and error.strip():
            lines.append(f"Error: {error.strip()[:400]}")

        artifacts = job.get("artifact_metadata") or []
        if isinstance(artifacts, list) and artifacts:
            artifact_lines = [
                f"{item.get('name')} ({item.get('size')} bytes)"
                for item in artifacts[:5]
                if isinstance(item, dict)
            ]
            if artifact_lines:
                lines.append("Artifacts: " + "; ".join(artifact_lines))

        logs = job.get("logs") or []
        if isinstance(logs, list) and logs:
            recent_logs = [str(entry) for entry in logs[-self._max_logs :]]
            if recent_logs:
                lines.append("Recent logs:")
                for log_line in recent_logs:
                    lines.append(f"- {log_line}")

        return "\n".join(lines)


