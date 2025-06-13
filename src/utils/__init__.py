"""
Utilities package for The Librarian.

This package contains shared utility functions including
- Logging configuration and management
- Helper functions for text processing
- Error handling utilities
- Performance monitoring helpers
"""

from .logger import setup_logger

__all__ = [
    "setup_logger",
]


# Additional utility functions that might be useful

def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to a specified length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncating

    Returns:
        str: Truncated text with suffix if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_query(query: str) -> str:
    """
    Clean and normalize user query input.

    Args:
        query: Raw user input query

    Returns:
        str: Cleaned query string
    """
    # Remove extra whitespace and normalize
    cleaned = " ".join(query.strip().split())
    return cleaned


def format_metadata(metadata: dict) -> str:
    """
    Format metadata dictionary for display.

    Args:
        metadata: Document metadata dictionary

    Returns:
        str: Formatted metadata string
    """
    if not metadata:
        return "No metadata available"

    formatted_items = []
    for key, value in metadata.items():
        if key == "story_title":
            formatted_items.append(f"Story: {value}")
        elif key == "source":
            formatted_items.append(f"Source: {value}")
        elif key == "page" or key == "chapter":
            formatted_items.append(f"{key.title()}: {value}")

    return " | ".join(formatted_items) if formatted_items else "Basic metadata"


# Add utility functions to exports
__all__.extend([
    "truncate_text",
    "clean_query",
    "format_metadata",
])

# Package-level constants
SUPPORTED_MODELS = [
    "o3-mini",
    "gpt-4o-mini"
]

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

__all__.extend([
    "SUPPORTED_MODELS",
    "DEFAULT_EMBEDDING_MODEL",
])
