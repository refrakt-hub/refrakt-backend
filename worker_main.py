"""Executable entrypoint for SAQ workers."""

from __future__ import annotations

import asyncio
import logging
import os

from saq.worker import Worker

from services.job_worker_settings import build_settings
from utils.logging_config import setup_logging, get_logger

# Setup logging before any other imports that might log
environment = os.getenv("ENVIRONMENT", "development")
log_level = os.getenv("LOG_LEVEL", "INFO")
setup_logging(environment, log_level)
logger = get_logger(__name__)


async def _run_worker():
    settings = build_settings()
    queue = settings.pop("queue")
    functions = settings.pop("functions")
    worker = Worker(queue, functions, **settings)
    try:
        await queue.connect()
        logger.info(
            "Queue worker started",
            extra={"worker_id": worker.id, "concurrency": worker.concurrency}
        )
        await worker.start()
        await worker.wait()
    finally:
        await queue.disconnect()


def main():
    try:
        asyncio.run(_run_worker())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


