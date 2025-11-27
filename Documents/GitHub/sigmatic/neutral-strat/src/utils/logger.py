"""
Logging Configuration
Sets up logging for the trading system
"""

import logging
import logging.handlers
import os
from typing import Dict, Any
from pathlib import Path


def setup_logging(config: Dict[str, Any] = None):
    """Setup logging configuration"""
    if config is None:
        config = {}

    # Default configuration
    log_level = config.get('level', 'INFO')
    log_file = config.get('file', 'logs/trading.log')
    console_logging = config.get('console', True)
    max_size = config.get('max_size', '10MB')
    backup_count = config.get('backup_count', 5)

    # Create logs directory
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler with rotation
    if log_file:
        max_bytes = _parse_size(max_size)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Console handler
    if console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Set specific loggers to INFO to avoid spam
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('ccxt').setLevel(logging.WARNING)

    logging.info("Logging configured successfully")


def _parse_size(size_str: str) -> int:
    """Parse size string to bytes"""
    size_str = size_str.upper()

    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(name)