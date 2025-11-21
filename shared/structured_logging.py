"""
Structured logging utilities for ADRIAN services.

Provides structured logging with correlation IDs, trace context, and
JSON-formatted logs for better observability and debugging.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any
from functools import wraps

# Context variables for request tracing
_correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
_trace_context: ContextVar[Dict[str, Any]] = ContextVar('trace_context', default_factory=dict)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context."""
    _correlation_id.set(correlation_id)


def get_trace_context() -> Dict[str, Any]:
    """Get current trace context."""
    return _trace_context.get().copy()


def set_trace_context(**kwargs) -> None:
    """Update trace context with key-value pairs."""
    ctx = _trace_context.get()
    ctx.update(kwargs)
    _trace_context.set(ctx)


def clear_trace_context() -> None:
    """Clear trace context."""
    _trace_context.set({})


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation ID if available
        correlation_id = getattr(record, 'correlation_id', None) or get_correlation_id()
        if correlation_id:
            log_data["correlation_id"] = correlation_id
        
        # Add trace context
        trace_ctx = get_trace_context()
        if trace_ctx:
            log_data["trace"] = trace_ctx
        
        # Add service name if available
        if hasattr(record, 'service_name'):
            log_data["service"] = record.service_name
        
        # Add extra fields from record
        if hasattr(record, 'extra_data'):
            log_data["extra"] = record.extra_data
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class TraceContextFilter(logging.Filter):
    """Filter that adds trace context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID and trace context to record."""
        correlation_id = get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        
        trace_ctx = get_trace_context()
        if trace_ctx:
            record.trace_context = trace_ctx
        
        return True


def trace_function(logger: logging.Logger, operation: str):
    """
    Decorator to trace function execution with structured logging.
    
    Usage:
        @trace_function(logger, "process_utterance")
        async def process_utterance(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            correlation_id = get_correlation_id() or str(uuid.uuid4())
            set_correlation_id(correlation_id)
            
            start_time = time.perf_counter()
            logger.info(
                f"Starting {operation}",
                extra={
                    "extra_data": {
                        "operation": operation,
                        "correlation_id": correlation_id,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                }
            )
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    f"Completed {operation}",
                    extra={
                        "extra_data": {
                            "operation": operation,
                            "correlation_id": correlation_id,
                            "duration_ms": round(duration_ms, 2),
                            "success": True
                        }
                    }
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Failed {operation}: {str(e)}",
                    exc_info=True,
                    extra={
                        "extra_data": {
                            "operation": operation,
                            "correlation_id": correlation_id,
                            "duration_ms": round(duration_ms, 2),
                            "success": False,
                            "error_type": type(e).__name__
                        }
                    }
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            correlation_id = get_correlation_id() or str(uuid.uuid4())
            set_correlation_id(correlation_id)
            
            start_time = time.perf_counter()
            logger.info(
                f"Starting {operation}",
                extra={
                    "extra_data": {
                        "operation": operation,
                        "correlation_id": correlation_id,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                }
            )
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    f"Completed {operation}",
                    extra={
                        "extra_data": {
                            "operation": operation,
                            "correlation_id": correlation_id,
                            "duration_ms": round(duration_ms, 2),
                            "success": True
                        }
                    }
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Failed {operation}: {str(e)}",
                    exc_info=True,
                    extra={
                        "extra_data": {
                            "operation": operation,
                            "correlation_id": correlation_id,
                            "duration_ms": round(duration_ms, 2),
                            "success": False,
                            "error_type": type(e).__name__
                        }
                    }
                )
                raise
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    correlation_id: Optional[str] = None,
    **context
):
    """
    Log a message with structured context.
    
    Args:
        logger: Logger instance
        level: Log level (logging.INFO, logging.ERROR, etc.)
        message: Log message
        correlation_id: Optional correlation ID
        **context: Additional context fields
    """
    if correlation_id:
        set_correlation_id(correlation_id)
    
    if context:
        set_trace_context(**context)
    
    extra_data = {
        "correlation_id": correlation_id or get_correlation_id(),
        **context
    }
    
    logger.log(level, message, extra={"extra_data": extra_data})


def setup_structured_logging(
    service_name: str,
    enable_json: bool = False,
    correlation_id: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured logging for a service.
    
    Args:
        service_name: Name of the service
        enable_json: Whether to use JSON formatting
        correlation_id: Optional initial correlation ID
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(service_name)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Add trace context filter
    trace_filter = TraceContextFilter()
    logger.addFilter(trace_filter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    if enable_json:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            f'%(asctime)s - {service_name} - [%(correlation_id)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Set correlation ID if provided
    if correlation_id:
        set_correlation_id(correlation_id)
    
    return logger

