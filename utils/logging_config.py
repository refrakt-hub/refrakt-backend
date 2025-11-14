"""Centralized logging configuration for Refrakt Backend"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

# Import request ID getter (will be None if not in request context)
try:
    from middleware.request_id import get_request_id
except ImportError:
    # Fallback if middleware not imported yet
    def get_request_id() -> Optional[str]:
        return None


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add request ID if available (from request context)
        request_id = get_request_id()
        if request_id:
            log_data["request_id"] = request_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add any extra fields passed via the 'extra' parameter
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_data.update(record.extra)
        else:
            # Extract extra fields from record attributes (excluding standard ones)
            standard_attrs = {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs", "message",
                "pathname", "process", "processName", "relativeCreated", "thread",
                "threadName", "exc_info", "exc_text", "stack_info", "getMessage"
            }
            for key, value in record.__dict__.items():
                if key not in standard_attrs and not key.startswith("_"):
                    log_data[key] = value
        
        return json.dumps(log_data, default=str)


def setup_logging(environment: str = "development", log_level: str = "INFO") -> None:
    """
    Configure logging for the application.
    
    Args:
        environment: Environment name (development, production, staging)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    
    # Choose formatter based on environment
    if environment.lower() == "production":
        # JSON formatter for production
        formatter = JSONFormatter()
    else:
        # Human-readable formatter for development
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)
    
    # Set levels for third-party loggers to reduce noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "environment": environment,
            "log_level": log_level,
            "format": "json" if environment.lower() == "production" else "text",
        },
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

