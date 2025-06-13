"""
Retrieval package for The Librarian.

This package implements the core RAG functionality including:
- ChromaDB vector store integration
- Document retrieval and similarity search
- LangChain RAG chain implementation
- Query processing and response generation
"""

from .chains import BorgesRAGChain
from .vector_store import BorgesVectorStore

__all__ = [
    "BorgesVectorStore",
    "BorgesRAGChain",
]


def create_default_rag_system():
    """Create a RAG system with the default configuration."""
    vector_store = BorgesVectorStore()
    rag_chain = BorgesRAGChain(vector_store)
    return rag_chain


__all__.append("create_default_rag_system")
