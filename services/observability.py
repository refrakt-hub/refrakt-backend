"""Observability helpers for exposing Prometheus telemetry."""

from __future__ import annotations

from typing import List

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from config import Settings


def init_observability(app: FastAPI, settings: Settings) -> None:
    """Wire Prometheus metrics endpoint if enabled."""
    if not settings.PROMETHEUS_ENABLED:
        return

    route = settings.PROMETHEUS_METRICS_ROUTE.rstrip("/") or "/metrics"
    if not route.startswith("/"):
        route = f"/{route}"

    dependencies: List[Depends] = []
    if settings.PROMETHEUS_BEARER_TOKEN:
        async def _authorize(request: Request) -> None:
            auth_header = request.headers.get("Authorization") or ""
            expected = f"Bearer {settings.PROMETHEUS_BEARER_TOKEN}"
            if auth_header != expected:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

        dependencies = [Depends(_authorize)]

    async def metrics_endpoint() -> Response:
        payload = generate_latest(REGISTRY)
        return Response(payload, media_type=CONTENT_TYPE_LATEST)

    app.add_api_route(
        route,
        metrics_endpoint,
        methods=["GET"],
        include_in_schema=False,
        dependencies=dependencies,
        name="prometheus_metrics",
    )

