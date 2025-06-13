"""
The Librarian: A Borges Expert RAG System

This package contains the core implementation of a Retrieval-Augmented Generation
system specialized for exploring Jorge Luis Borges' literary works.

Main Components:
- retrieval: ChromaDB integration and RAG chain implementation
- prompts: Specialized prompt templates for Borges expertise
- utils: Logging and utility functions
"""

from .prompts import BORGES_PROMPT, CONVERSATION_STARTERS
from .retrieval import BorgesVectorStore, BorgesRAGChain
from .utils import setup_logger

__all__ = [
    "BorgesVectorStore",
    "BorgesRAGChain",
    "BORGES_PROMPT",
    "CONVERSATION_STARTERS",
    "setup_logger",
]

__version__ = "1.0.0"
__author__ = "The Librarian Project - Pedro Acosta"
__description__ = "A specialized RAG system for Jorge Luis Borges literary analysis"
