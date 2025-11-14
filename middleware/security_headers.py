"""Security headers middleware for FastAPI"""

import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config import get_settings

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
        self._headers = self._build_headers()

    def _build_headers(self) -> dict[str, str]:
        """Build security headers based on configuration"""
        headers: dict[str, str] = {}

        if not self.settings.SECURITY_HEADERS_ENABLED:
            return headers

        # Essential security headers (always include if enabled)
        headers["X-Content-Type-Options"] = "nosniff"
        headers["X-Frame-Options"] = "DENY"
        headers["X-XSS-Protection"] = "1; mode=block"
        headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # HSTS (HTTP Strict Transport Security) - only in production
        if self.settings.is_production and self.settings.HSTS_MAX_AGE > 0:
            hsts_value = f"max-age={self.settings.HSTS_MAX_AGE}"
            if self.settings.HSTS_INCLUDE_SUBDOMAINS:
                hsts_value += "; includeSubDomains"
            if self.settings.HSTS_PRELOAD:
                hsts_value += "; preload"
            headers["Strict-Transport-Security"] = hsts_value

        # Content Security Policy (optional)
        if self.settings.CSP_POLICY:
            headers["Content-Security-Policy"] = self.settings.CSP_POLICY

        return headers

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response"""
        response = await call_next(request)

        # Add security headers to response
        for header_name, header_value in self._headers.items():
            response.headers[header_name] = header_value

        return response

