from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration for ChromaDB Persistent Client."""

    # ChromaDB Persistent Client Configuration
    chroma_persist_directory: str = Field(default="./chroma_db", description="ChromaDB persistent storage directory")
    chroma_collection_name: str = Field(default="borges_stories", description="Collection name")

    # Embedding Model Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer embedding model")

    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    model_name: str = Field(default="gpt-4o-mini", description="LLM model name")

    # Retrieval Configuration
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    score_threshold: float = Field(default=0.7, description="Minimum similarity score")

    # Application Configuration
    app_title: str = Field(default="The Librarian: Borges Expert", description="App title")
    app_description: str = Field(
        default="Explore the infinite library of Jorge Luis Borges through intelligent conversation.",
        description="App description"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        protected_namespaces = ('settings_',)


settings = Settings()
