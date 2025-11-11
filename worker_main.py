"""Executable entrypoint for SAQ workers."""

from __future__ import annotations

import asyncio
import logging

from saq.worker import Worker

from services.job_worker_settings import build_settings


async def _run_worker():
    settings = build_settings()
    queue = settings.pop("queue")
    functions = settings.pop("functions")
    worker = Worker(queue, functions, **settings)
    try:
        await queue.connect()
        logging.getLogger("refrakt.worker").info(
            "Queue worker started (id=%s, concurrency=%s)", worker.id, worker.concurrency
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


