import os
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv

from config.settings import settings
from src.prompts.templates import CONVERSATION_STARTERS
from src.retrieval.chains import BorgesRAGChain
from src.retrieval.vector_store import BorgesVectorStore
from src.utils.logger import setup_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logger("gradio_app")

# Initialize components
try:
    vector_store = BorgesVectorStore.create()
    rag_chain = BorgesRAGChain(vector_store)
    logger.info("✅ Application components initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize components: {e}")
    vector_store = None
    rag_chain = None


def chat_with_librarian(message: str, history: List) -> str:
    """Main chat function for the Gradio interface."""
    if not rag_chain:
        return "❌ System not properly initialized. Please check ChromaDB connection."

    try:
        if not message.strip():
            return "Please ask me something about Jorge Luis Borges' works."

        # Process the query
        result = rag_chain.query(message)

        if "error" in result:
            return f"I apologize, but I encountered an issue: {result['error']}"

        return result["answer"]

    except Exception as e:
        logger.error(f"Chat function error: {e}")
        return f"I'm sorry, but I encountered an unexpected error. Please try again."


def get_collection_status() -> Tuple[str, str]:
    """Get status information about the ChromaDB collection."""
    if not vector_store:
        return "❌ Vector store not initialized", "error"

    try:
        info = vector_store.get_collection_info()

        if "error" in info:
            status_msg = f"❌ Collection Error: {info['error']}"
            status_type = "error"
        else:
            status_msg = (
                f"✅ Connected to '{info['name']}' collection\n"
                f"📊 Documents: {info['count']:,}\n"
                f"💾 Storage: {settings.chroma_persist_directory}\n"
                f"📄 Status: {info['status']}"
            )

            # Add sample data info if available
            if info.get('sample_data'):
                sample = info['sample_data']
                if sample.get('metadata_keys'):
                    status_msg += f"\n🏷️ Metadata: {', '.join(sample['metadata_keys'])}"

            status_type = "success"

        return status_msg, status_type

    except Exception as e:
        return f"❌ Status Check Failed: {str(e)}", "error"


def test_search_functionality() -> Tuple[str, str]:
    """Test the search functionality with a sample query."""
    if not vector_store:
        return "❌ Vector store not initialized", "error"

    try:
        # Test with a simple Borges-related query
        test_query = "Emma Zunz"
        results = vector_store.search(test_query, k=3)

        if results:
            status_msg = (
                f"✅ Search Test Successful\n"
                f"🔍 Query: '{test_query}'\n"
                f"📊 Results: {len(results)} documents found\n"
                f"🎯 Top score: {results[0]['score']:.3f}\n"
                f"📝 Preview: {results[0]['content'][:100]}..."
            )
            status_type = "success"
        else:
            status_msg = f"⚠️ Search returned no results for '{test_query}'"
            status_type = "warning"

        return status_msg, status_type

    except Exception as e:
        return f"❌ Search Test Failed: {str(e)}", "error"


def create_gradio_interface():
    """Create and configure the Gradio interface."""

    # Custom CSS for a literary aesthetic
    css = """
    .gradio-container {
        max-width: 1000px !important;
        margin: auto !important;
    }
    .chat-message {
        font-family: 'Georgia', serif !important;
        line-height: 1.6 !important;
    }
    .header {
        text-align: center;
        padding: 20px;
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .status-success {
        background-color: #d4edda !important;
        border: 1px solid #c3e6cb !important;
        color: #155724 !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }
    .status-error {
        background-color: #f8d7da !important;
        border: 1px solid #f5c6cb !important;
        color: #721c24 !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }
    .status-warning {
        background-color: #fff3cd !important;
        border: 1px solid #ffeaa7 !important;
        color: #856404 !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }
    """

    with gr.Blocks(css=css, title=settings.app_title) as app:

        # Header
        gr.HTML(f"""
        <div class="header">
            <h1>📚 {settings.app_title}</h1>
            <p><em>"{settings.app_description}"</em></p>
            <p style="font-size: 0.9em; opacity: 0.8;">Pedro Acosta • contact@pedroacosta.dev</p>
        </div>
        """)

        # Status section
        with gr.Row():
            with gr.Column(scale=2):
                status_display = gr.Textbox(
                    label="📡 Collection Status",
                    value=get_collection_status()[0],
                    interactive=False,
                    max_lines=6,
                    elem_classes=[f"status-{get_collection_status()[1]}"]
                )

            with gr.Column(scale=1):
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                    test_btn = gr.Button("🧪 Test Search", size="sm")

        # Main chat interface - only show if system is initialized
        if rag_chain:
            chatbot = gr.ChatInterface(
                fn=chat_with_librarian,
                title="",
                description="",
                examples=CONVERSATION_STARTERS,
                retry_btn="🔄 Retry",
                undo_btn="↩️ Undo",
                clear_btn="🗑️ Clear",
                submit_btn="📤 Send",
                textbox=gr.Textbox(
                    placeholder="Ask me about Borges' labyrinths, mirrors, infinite libraries, or any aspect of his work...",
                    container=False,
                    scale=7
                ),
                chatbot=gr.Chatbot(
                    height=500,
                    placeholder="<div style='text-align: center; color: #666;'><em>Welcome to the infinite library. What would you like to explore?</em></div>",
                    elem_classes=["chat-message"]
                )
            )
        else:
            gr.HTML("""
            <div style="text-align: center; padding: 40px; background-color: #f8d7da; border-radius: 10px; margin: 20px 0;">
                <h3>❌ System Not Available</h3>
                <p>The chat interface is not available because the ChromaDB connection could not be established.</p>
                <p>Please check the status above and ensure your ChromaDB server is running.</p>
            </div>
            """)

        # Configuration display
        with gr.Accordion("🔧 System Configuration", open=False):
            gr.HTML(f"""
            <div style="font-family: monospace; background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                <strong>ChromaDB Configuration:</strong><br>
                • Storage: {settings.chroma_persist_directory}<br>
                • Collection: {settings.chroma_collection_name}<br>
                • Embedding Model: {settings.embedding_model}<br>
                • LLM Model: {settings.model_name}<br>
                • Retrieval: Top-{settings.top_k}, Threshold: {settings.score_threshold}<br>
                • Architecture: Persistent Client + Local Embeddings
            </div>
            """)

        # Usage tips
        gr.HTML("""
        <div style="text-align: center; margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; color: #666;">
            <p><strong>💡 Tips for better conversations:</strong></p>
            <p>• Ask about specific stories, themes, or literary techniques</p>
            <p>• Reference characters or concepts from Borges' works</p>
            <p>• Explore connections between different stories</p>
            <p>• Inquire about philosophical themes like infinity, time, and identity</p>
        </div>
        """)

        # Event handlers
        def update_status():
            msg, msg_type = get_collection_status()
            return gr.update(value=msg, elem_classes=[f"status-{msg_type}"])

        def test_search():
            msg, msg_type = test_search_functionality()
            return gr.update(value=msg, elem_classes=[f"status-{msg_type}"])

        refresh_btn.click(fn=update_status, outputs=[status_display])
        test_btn.click(fn=test_search, outputs=[status_display])

    return app


if __name__ == "__main__":
    # Validate required environment variables
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is required")
        exit(1)

    logger.info(f"Starting {settings.app_title}")
    logger.info(f"ChromaDB Storage: {settings.chroma_persist_directory}")
    logger.info(f"Collection: {settings.chroma_collection_name}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
    logger.info(f"Architecture: Persistent Client + Local Embeddings")

    # Create and launch the application
    app = create_gradio_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )
