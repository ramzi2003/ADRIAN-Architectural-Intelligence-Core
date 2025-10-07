"""
Centralized logging configuration for ADRIAN services.
Logs to both console and SQLite database for queryability.
"""
import logging
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from shared.config import get_settings


class SQLiteHandler(logging.Handler):
    """Custom logging handler that writes to SQLite database."""
    
    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Create logs table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                service_name TEXT,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                correlation_id TEXT,
                extra_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def emit(self, record: logging.LogRecord):
        """Write log record to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract extra fields
            correlation_id = getattr(record, 'correlation_id', None)
            service_name = getattr(record, 'service_name', record.name)
            extra_data = getattr(record, 'extra_data', None)
            
            cursor.execute("""
                INSERT INTO logs (timestamp, service_name, level, message, correlation_id, extra_data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.fromtimestamp(record.created).isoformat(),
                service_name,
                record.levelname,
                record.getMessage(),
                correlation_id,
                json.dumps(extra_data) if extra_data else None
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            # Avoid infinite loop if logging fails
            print(f"Failed to write log to database: {e}")


def setup_logging(service_name: str, correlation_id: Optional[str] = None):
    """
    Setup logging for a service.
    
    Args:
        service_name: Name of the service (e.g., 'io-service')
        correlation_id: Optional correlation ID for request tracing
    """
    settings = get_settings()
    
    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with formatting
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        f'%(asctime)s - {service_name} - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # SQLite handler for persistent, queryable logs
    sqlite_handler = SQLiteHandler(settings.log_db_path)
    logger.addHandler(sqlite_handler)
    
    # Add service name to all log records
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.service_name = service_name
        if correlation_id:
            record.correlation_id = correlation_id
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)

