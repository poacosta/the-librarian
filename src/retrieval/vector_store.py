from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChromaVectorStore:
    """Simplified ChromaDB vector store using direct collection access."""

    def __init__(self, chroma_collection):
        """
        Initialize with a ChromaDB collection object.

        Args:
            chroma_collection: ChromaDB collection object from client.get_collection()
        """
        self.collection = chroma_collection
        self.collection_name = chroma_collection.name
        self._embedding_model = None

        logger.info(f"Initialized ChromaVectorStore for collection: {self.collection_name}")

    def _get_embedding_model(self) -> SentenceTransformer:
        """Get or create SentenceTransformer embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Initialized embedding model: {settings.embedding_model}")

        return self._embedding_model

    def search(
            self,
            query: str,
            k: int = None,
            score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search on the collection."""
        k = k or settings.top_k

        try:
            embedding_model = self._get_embedding_model()
            embedding = embedding_model.encode(query).tolist()

            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]

            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                similarity_score = 1.0 - distance if distance is not None else 0.0
                formatted_results.append({
                    "content": doc,
                    "metadata": metadata or {},
                    "score": similarity_score,
                    "distance": distance
                })

            logger.info(f"Retrieved {len(formatted_results)} documents for query: {query[:50]}...")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Failed to search collection '{self.collection_name}': {e}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()

            # Try to get a sample document to understand the structure
            sample_data = None
            if count > 0:
                try:
                    # Generate embeddings manually using the same model
                    embedding_model = self._get_embedding_model()
                    embedding = embedding_model.encode("sample").tolist()

                    sample_results = self.collection.query(
                        query_embeddings=[embedding],
                        n_results=1,
                        include=['documents', 'metadatas']
                    )
                    if sample_results['documents'] and sample_results['documents'][0]:
                        sample_doc = sample_results['documents'][0][0]
                        sample_metadata = sample_results['metadatas'][0][0] if sample_results['metadatas'][0] else {}
                        sample_data = {
                            "content_preview": sample_doc[:100] + "..." if len(sample_doc) > 100 else sample_doc,
                            "metadata_keys": list(sample_metadata.keys()) if sample_metadata else []
                        }
                except Exception as e:
                    logger.warning(f"Could not get sample data: {e}")

            return {
                "name": self.collection_name,
                "count": count,
                "status": "connected",
                "sample_data": sample_data
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "error": str(e),
                "status": "error",
                "name": self.collection_name
            }


class BorgesVectorStore:
    """Factory class for creating ChromaVectorStore with Borges collection."""

    @staticmethod
    def create(
            persist_directory: str = None,
            collection_name: str = None
    ) -> ChromaVectorStore:
        """
        Create a ChromaVectorStore for the Borges collection using Persistent Client.
        """
        persist_directory = persist_directory or settings.chroma_persist_directory
        collection_name = collection_name or settings.chroma_collection_name

        try:
            logger.info(f"Connecting to ChromaDB at {persist_directory}")

            # Create PersistentClient instead of HttpClient
            client = chromadb.PersistentClient(path=persist_directory)

            # Test connection by listing collections
            collections = client.list_collections()
            logger.info(f"ChromaDB connected successfully. Available collections: {[c.name for c in collections]}")

            # Get collection
            chroma_collection = client.get_collection(collection_name)
            logger.info(f"Successfully accessed collection: {collection_name}")

            # Create vector store
            vector_store = ChromaVectorStore(chroma_collection)

            # Log collection info
            info = vector_store.get_collection_info()
            logger.info(f"Collection contains {info['count']} documents")

            return vector_store

        except Exception as e:
            logger.error(f"Failed to create BorgesVectorStore: {e}")
            raise ConnectionError(f"Could not connect to ChromaDB persistent storage at {persist_directory} "
                                  f"or access collection '{collection_name}'. Error: {e}.")
