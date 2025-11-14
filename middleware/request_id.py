"""Request ID tracking middleware for FastAPI"""

import logging
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config import get_settings

logger = logging.getLogger(__name__)

# Context variable for storing request ID (async-safe)
_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Get the current request ID from context"""
    return _request_id_ctx.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to track request IDs for log correlation"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
        self.header_name = self.settings.REQUEST_ID_HEADER

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Generate or extract request ID and add to response"""
        # Check if request ID is already in headers
        request_id = request.headers.get(self.header_name)

        # Generate new request ID if not present
        if not request_id:
            request_id = str(uuid.uuid4())

        # Store in context variable for logging
        _request_id_ctx.set(request_id)

        # Store in request state for access in route handlers
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers[self.header_name] = request_id

        # Clear context after request
        _request_id_ctx.set(None)

        return response

