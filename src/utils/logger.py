import logging
import sys
from typing import Optional


def setup_logger(
        name: str = "borges_librarian",
        level: str = "INFO",
        format_string: Optional[str] = None
) -> logging.Logger:
    """Configure and return a logger instance."""

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(handler)

    return logger
