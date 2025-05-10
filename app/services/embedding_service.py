from sentence_transformers import SentenceTransformer
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the embedding model
_embedding_model = None

def initialize_embedding_model():
    """
    Initialize the embedding model
    """
    global _embedding_model
    try:
        logger.info("🚀 Initializing HuggingFace Sentence Transformer Embeddings")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"📱 Using device: {device}")
        
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info("✅ HuggingFace Embeddings initialized successfully")
        return _embedding_model
    except Exception as e:
        logger.error(f"❌ Failed to initialize HuggingFace Embeddings: {str(e)}")
        raise

def get_embedding_model():
    """
    Get the embedding model instance
    """
    if _embedding_model is None:
        raise RuntimeError("Embedding model not initialized. Make sure the application has started properly.")
    return _embedding_model

def cleanup_embedding_model():
    """
    Cleanup the embedding model
    """
    global _embedding_model
    logger.info("🧹 Cleaning up embedding model...")
    _embedding_model = None
    logger.info("✅ Embedding model cleanup completed") 