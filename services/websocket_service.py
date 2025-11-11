"""WebSocket service for real-time log streaming backed by Redis pub/sub."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Dict, List, Optional

from fastapi import WebSocket
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.client import PubSub

from config import get_settings
from services.job_repository import get_job_repository


class WebSocketService:
    """Service for managing WebSocket connections across processes."""

    def __init__(self):
        settings = get_settings()
        if not settings.QUEUE_URL:
            raise RuntimeError("QUEUE_URL must be configured before using WebSocketService")

        self._redis: AsyncRedis = AsyncRedis.from_url(
            settings.QUEUE_URL,
            decode_responses=True,
        )
        self._repository = get_job_repository()

        self.active_connections: Dict[str, List[WebSocket]] = {}
        self._listener_tasks: Dict[str, asyncio.Task[None]] = {}

    async def add_connection(self, job_id: str, websocket: WebSocket):
        """Register a WebSocket connection and replay recent logs."""
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

        # Prime client with existing logs
        await self._send_existing_logs(job_id, websocket)

        # Ensure a single listener task per job
        if job_id not in self._listener_tasks:
            self._listener_tasks[job_id] = asyncio.create_task(self._stream_job_logs(job_id))

    async def remove_connection(self, job_id: str, websocket: WebSocket):
        """Remove a WebSocket connection, cleaning up listeners when idle."""
        connections = self.active_connections.get(job_id)
        if not connections:
            return
        if websocket in connections:
            connections.remove(websocket)
        if not connections:
            self.active_connections.pop(job_id, None)
            task = self._listener_tasks.pop(job_id, None)
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def broadcast_log(self, job_id: str, message: str):
        """Publish a log message to all listeners via Redis."""
        await self._redis.publish(self._channel(job_id), message)

    def get_connections_count(self, job_id: str) -> int:
        """Return the number of active WebSocket connections for a job."""
        return len(self.active_connections.get(job_id, []))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _stream_job_logs(self, job_id: str):
        pubsub: Optional[PubSub] = None
        try:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(self._channel(job_id))
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                payload = message["data"]
                await self._dispatch(job_id, payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            # On failure, attempt to restart listener
            await asyncio.sleep(1.0)
            if job_id in self.active_connections:
                self._listener_tasks[job_id] = asyncio.create_task(self._stream_job_logs(job_id))
        finally:
            if pubsub:
                with contextlib.suppress(Exception):
                    await pubsub.unsubscribe(self._channel(job_id))
                with contextlib.suppress(Exception):
                    await pubsub.close()

    async def _dispatch(self, job_id: str, payload: str):
        connections = self.active_connections.get(job_id)
        if not connections:
            return
        stale: List[WebSocket] = []
        for ws in connections:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            if ws in connections:
                connections.remove(ws)

    async def _send_existing_logs(self, job_id: str, websocket: WebSocket):
        logs = await asyncio.to_thread(self._repository.get_logs, job_id, 200)
        for log_line in logs:
            try:
                await websocket.send_text(log_line)
            except Exception:
                break

    @staticmethod
    def _channel(job_id: str) -> str:
        return f"job:{job_id}:logs"

