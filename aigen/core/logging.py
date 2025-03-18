"""
Logging Module - Provides logging facilities for the AI Agent Framework.

This module offers consistent logging capabilities across the framework, supporting
different log levels, formatting options, and output destinations. It allows for
contextual logging with metadata and integration with various logging backends.
"""

import logging
import sys
from typing import Dict, Any, Optional
from enum import Enum
import os
from datetime import datetime
import json


class LogLevel(Enum):
    """Log level enum for the logging system."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Logger:
    """
    Enhanced logging system with structured logging support.
    Provides console and file logging with various formatting options.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[94m",  # Blue
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[41m\033[97m",  # White on red background
        "RESET": "\033[0m",  # Reset
    }

    EMOJI = {
        "DEBUG": "🔍",
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🔥",
    }

    def __init__(self, name: str = "aigen", level: str = "INFO"):
        """
        Initialize the logger.

            name: Logger name
            level: Initial log level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_log_level(level))

        if not self.logger.handlers:
            self._setup_console_handler()

    def _get_log_level(self, level: str) -> int:
        """Convert string level to logging level."""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return level_map.get(level.upper(), logging.INFO)

    def _setup_console_handler(self) -> None:
        """Set up console logging handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    def add_file_handler(self, file_path: str, level: str = "INFO") -> None:
        """
        Add a file logging handler.

            file_path: Path to log file
            level: Log level for file handler
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        handler = logging.FileHandler(file_path)
        handler.setLevel(self._get_log_level(level))

        class JsonFormatter(logging.Formatter):
            """
            Custom formatter that outputs log records in JSON format.
            Structures log data for machine readability and easy integration
            with log processing tools.
            """
            def format(self, record):
                log_data = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "logger": record.name,
                }

                for key, value in record.__dict__.items():
                    if key not in log_data and not key.startswith("_") and key != "msg":
                        log_data[key] = str(value)

                return json.dumps(log_data)

        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)

    def set_level(self, level: str) -> None:
        """
        Set the logger level.

            level: The log level to set
        """
        self.logger.setLevel(self._get_log_level(level))

    def _format_message(
        self, level: str, message: str, extras: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format log message with timestamp, level, and color."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        color = self.COLORS.get(level, "")
        reset = self.COLORS["RESET"]
        emoji = self.EMOJI.get(level, "")

        formatted = f"[{timestamp}] {emoji} {color}{level}{reset}: {message}"

        if extras:
            extra_str = " ".join(f"{k}={v}" for k, v in extras.items())
            formatted += f" ({extra_str})"

        return formatted

    def debug(self, message: str, **kwargs) -> None:
        """
        Log a debug message.

            message: The message to log
            **kwargs: Additional context to include in the log
        """
        formatted = self._format_message("DEBUG", message, kwargs)
        self.logger.debug(formatted, extra=kwargs)

    def info(self, message: str, **kwargs) -> None:
        """
        Log an info message.

            message: The message to log
            **kwargs: Additional context to include in the log
        """
        formatted = self._format_message("INFO", message, kwargs)
        self.logger.info(formatted, extra=kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """
        Log a warning message.

            message: The message to log
            **kwargs: Additional context to include in the log
        """
        formatted = self._format_message("WARNING", message, kwargs)
        self.logger.warning(formatted, extra=kwargs)

    def error(self, message: str, **kwargs) -> None:
        """
        Log an error message.

            message: The message to log
            **kwargs: Additional context to include in the log
        """
        formatted = self._format_message("ERROR", message, kwargs)
        self.logger.error(formatted, extra=kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """
        Log a critical message.

            message: The message to log
            **kwargs: Additional context to include in the log
        """
        formatted = self._format_message("CRITICAL", message, kwargs)
        self.logger.critical(formatted, extra=kwargs)

    def success(self, message: str, **kwargs) -> None:
        """
        Log a success message (INFO level with success formatting).

            message: The message to log
            **kwargs: Additional context to include in the log
        """
        formatted = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ \033[92mSUCCESS\033[0m: {message}"

        if kwargs:
            extra_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
            formatted += f" ({extra_str})"

        self.logger.info(formatted, extra=kwargs)


logger = Logger()


def get_logger(name: str = None) -> Logger:
    """
    Get a logger instance.

        name: Optional logger name

    """
    if name:
        return Logger(name)
    return logger


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure global logging settings.

        level: Log level to set
        log_file: Optional file to log to
    """
    # Force INFO level to disable debug logs
    logger.set_level("INFO")

    if log_file:
        logger.add_file_handler(log_file, "INFO")
