"""Request tracking middleware for graceful shutdown"""

import asyncio
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RequestTrackerMiddleware(BaseHTTPMiddleware):
    """Middleware to track in-flight requests for graceful shutdown"""
    
    # Class-level reference to the active instance (set by FastAPI on instantiation)
    _instance: "RequestTrackerMiddleware | None" = None

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._in_flight_requests = 0
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()
        # Store this instance as the active one
        RequestTrackerMiddleware._instance = self
    
    @classmethod
    def get_instance(cls) -> "RequestTrackerMiddleware | None":
        """Get the active middleware instance"""
        return cls._instance

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track request start and completion"""
        # Check if shutdown is in progress
        if self._shutdown_event.is_set():
            from fastapi import status
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Service is shutting down"},
            )

        # Increment in-flight counter
        async with self._lock:
            self._in_flight_requests += 1

        try:
            # Process request
            response = await call_next(request)
            return response
        finally:
            # Decrement in-flight counter
            async with self._lock:
                self._in_flight_requests -= 1

    def start_shutdown(self) -> None:
        """Signal that shutdown has started"""
        self._shutdown_event.set()

    async def wait_for_requests(self, timeout: float) -> bool:
        """
        Wait for all in-flight requests to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all requests completed, False if timeout
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            async with self._lock:
                if self._in_flight_requests == 0:
                    return True
            
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                async with self._lock:
                    remaining = self._in_flight_requests
                logger.warning(
                    f"Shutdown timeout reached. {remaining} requests still in flight"
                )
                return False
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)

    def get_in_flight_count(self) -> int:
        """Get current number of in-flight requests (thread-safe)"""
        # Note: This is a best-effort read without lock for performance
        # For exact count during shutdown, use wait_for_requests which uses the lock
        return self._in_flight_requests

