"""
Configuration package for The Librarian application.

This package handles all configuration management, including environment variables,
application settings, and deployment configurations.
"""

from .settings import Settings, settings

__all__ = [
    "Settings",
    "settings",
]

__version__ = "1.0.0"
