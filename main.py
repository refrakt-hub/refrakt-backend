#!/usr/bin/env python3
"""
Refrakt Backend API Server - Enhanced with Cloudflare R2 Storage

Features:
1. Upload job artifacts to Cloudflare R2
2. Generate presigned URLs for secure downloads
3. Real-time log streaming via WebSocket
4. Job artifacts management
"""

import logging
import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import router
from config import get_settings
from middleware.request_id import RequestIDMiddleware
from middleware.request_tracker import RequestTrackerMiddleware
from middleware.security_headers import SecurityHeadersMiddleware
from services.observability import init_observability
from services.queue_monitor import QueueMonitor
from services.rate_limiter import init_rate_limiter, shutdown_rate_limiter
from utils.logging_config import setup_logging

# Setup basic logging first (before Settings initialization)
# This allows Settings to use logging during initialization
environment = os.getenv("ENVIRONMENT", "development")
log_level = os.getenv("LOG_LEVEL", "INFO")
setup_logging(environment, log_level)
logger = logging.getLogger(__name__)

# Now initialize settings (which may log during initialization)
settings = get_settings()

# Reconfigure logging with settings (in case they differ from env vars)
if settings.ENVIRONMENT != environment or settings.LOG_LEVEL != log_level:
    setup_logging(settings.ENVIRONMENT, settings.LOG_LEVEL)

queue_monitor = QueueMonitor(settings=settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with graceful shutdown support"""
    # Initialize services
    init_observability(app, settings)
    await init_rate_limiter(settings)
    await queue_monitor.start()
    
    logger.info("Application started successfully")
    
    try:
        yield
    finally:
        # Start graceful shutdown
        logger.info("Starting graceful shutdown...")
        
        # Get the request tracker instance (set by FastAPI when middleware is instantiated)
        request_tracker = RequestTrackerMiddleware.get_instance()
        
        if request_tracker:
            # Signal shutdown to stop accepting new requests
            request_tracker.start_shutdown()
            
            # Wait for in-flight requests to complete
            shutdown_timeout = settings.SHUTDOWN_TIMEOUT
            in_flight = request_tracker.get_in_flight_count()
            
            if in_flight > 0:
                logger.info(f"Waiting for {in_flight} in-flight requests to complete (timeout: {shutdown_timeout}s)")
                all_completed = await request_tracker.wait_for_requests(shutdown_timeout)
                
                if all_completed:
                    logger.info("All in-flight requests completed")
                else:
                    remaining = request_tracker.get_in_flight_count()
                    logger.warning(f"Shutdown timeout reached. {remaining} requests may have been interrupted")
            else:
                logger.info("No in-flight requests")
        
        # Stop services
        logger.info("Stopping services...")
        await queue_monitor.stop()
        await shutdown_rate_limiter()
        logger.info("Graceful shutdown completed")


# Initialize FastAPI app
app = FastAPI(
    title="Refrakt Backend API",
    description="Backend API for Refrakt ML Framework with R2 Storage",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware with environment-based configuration
cors_origins, cors_origin_regex = settings.get_cors_origins()
logger.info(
    f"CORS configured",
    extra={
        "allowed_origins": cors_origins if cors_origins != ["*"] else ["*"],
        "origin_count": len(cors_origins) if cors_origins != ["*"] else "unlimited",
        "origin_regex_count": len(cors_origin_regex),
    },
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # Empty list is valid (no CORS), don't pass None
    allow_origin_regex="|".join(cors_origin_regex) if cors_origin_regex else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Request ID middleware (early in stack to capture all requests)
app.add_middleware(RequestIDMiddleware)

# Add Request Tracker middleware (for graceful shutdown)
# FastAPI will instantiate this, and the instance will be accessible via class method
app.add_middleware(RequestTrackerMiddleware)

# Add Security Headers middleware
if settings.SECURITY_HEADERS_ENABLED:
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("Security headers middleware enabled")

# Include API routes
app.include_router(router)


if __name__ == "__main__":
    # Port comes from PORT environment variable (set in env files or docker-compose)
    port = int(os.getenv("PORT", "8000"))
    
    # Determine reload based on environment
    # Command-line args can override reload setting for backward compatibility
    if len(sys.argv) > 1:
        env_arg = sys.argv[1]
        if env_arg == "dev":
            reload = True
        elif env_arg == "prod":
            reload = False
        else:
            reload = False
    else:
        # Use environment from settings (prioritize ENVIRONMENT env var)
        reload = settings.is_development
    
    # Log startup information
    if settings.is_development:
        logger.info(f"Starting Refrakt Backend in DEVELOPMENT mode on port {port}")
        logger.info(f"Access via: http://localhost:{port}")
        if reload:
            logger.info("Hot reload enabled for development")
    elif settings.is_production:
        logger.info(f"Starting Refrakt Backend in PRODUCTION mode on port {port}")
    else:
        logger.info(f"Starting Refrakt Backend in {settings.ENVIRONMENT.upper()} mode on port {port}")
    
    # Start uvicorn server
    if reload:
        # Development mode with hot reload
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            log_level=settings.UVICORN_LOG_LEVEL.lower(),
            access_log=settings.UVICORN_ACCESS_LOG,
        )
    else:
        # Production mode - use uvicorn directly (workers handled by orchestrator or uvicorn CLI)
        # For production with multiple workers, use: uvicorn main:app --workers N --host 0.0.0.0 --port PORT
        # For single worker (current), use uvicorn.run with production settings
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level=settings.UVICORN_LOG_LEVEL.lower(),
            access_log=settings.UVICORN_ACCESS_LOG,
            timeout_keep_alive=settings.UVICORN_TIMEOUT_KEEP_ALIVE,
            timeout_graceful_shutdown=settings.UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN,
        )
