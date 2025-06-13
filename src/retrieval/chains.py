from typing import Dict, Any, List

from langchain_openai import ChatOpenAI

from config.settings import settings
from src.prompts.templates import BORGES_PROMPT
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BorgesRAGChain:
    """Simplified RAG chain for Borges expert queries."""

    def __init__(self, vector_store: ChromaVectorStore):
        """
        Initialize with a ChromaVectorStore.

        Args:
            vector_store: ChromaVectorStore instance
        """
        self.vector_store = vector_store
        self._llm = None

    def _get_llm(self) -> ChatOpenAI:
        """Get or create the language model."""
        if self._llm is None:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not provided")

            self._llm = ChatOpenAI(
                openai_api_key=settings.openai_api_key,
                model_name=settings.model_name,
                max_tokens=1000
            )

        return self._llm

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant passages found in the collection."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc["content"]
            metadata = doc.get("metadata", {})
            score = doc.get("score", 0)

            # Include story title if available in metadata
            story_info = ""
            if "story_title" in metadata:
                story_info = f" (from '{metadata['story_title']}')"
            elif "source" in metadata:
                story_info = f" (from {metadata['source']})"

            context_parts.append(
                f"Passage {i}{story_info} [Relevance: {score:.3f}]:\n{content}"
            )

        return "\n\n".join(context_parts)

    def query(self, question: str) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        try:
            logger.info(f"Processing query: {question[:100]}...")

            # Retrieve relevant documents
            documents = self.vector_store.search(question)

            if not documents:
                return {
                    "answer": "I couldn't find relevant passages in the Borges collection to answer your question. Could you try rephrasing or asking about a different aspect of his work?",
                    "sources": [],
                    "context_used": "",
                    "num_sources": 0
                }

            # Format context
            context = self._format_context(documents)

            # Generate response using the prompt template
            llm = self._get_llm()
            prompt_input = {
                "context": context,
                "question": question
            }

            formatted_prompt = BORGES_PROMPT.format(**prompt_input)
            response = llm.invoke(formatted_prompt)

            # Extract response content
            answer = response.content if hasattr(response, 'content') else str(response)

            # Prepare sources information
            sources = []
            for doc in documents:
                source_info = {
                    "content_preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", 0),
                    "distance": doc.get("distance", 0)
                }
                sources.append(source_info)

            logger.info(f"Successfully generated response using {len(documents)} sources")

            return {
                "answer": answer,
                "sources": sources,
                "context_used": context,
                "num_sources": len(documents)
            }

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}. Please try again.",
                "sources": [],
                "context_used": "",
                "error": str(e)
            }
