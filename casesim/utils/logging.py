"""Logging configuration for casesim."""

import logging
import sys
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

# Console for rich output
console = Console()


def setup_logging(level: str = "INFO", debug: bool = False) -> logging.Logger:
    """Set up logging with rich handler."""
    log_level = logging.DEBUG if debug else getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                show_time=True,
                show_path=debug,
                rich_tracebacks=True,
            )
        ],
    )

    # Set third-party loggers to warning
    for logger_name in ["httpx", "httpcore", "openai", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    return logging.getLogger("casesim")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(f"casesim.{name}")
