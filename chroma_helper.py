"""Helper wrapper for ChromaDB local collection and embeddings.

Provides init, upsert, and similarity query helpers used by the learning flows.
"""
import logging
from typing import Optional, List, Dict

from config import APP_CONFIG

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    embedding_functions = None
    CHROMADB_AVAILABLE = False
    logger.debug("ChromaDB not available.")


_client = None
_collection = None


def init_chroma():
    """Initialize Chroma client and collection using settings from config. Safe to call multiple times."""
    global _client, _collection
    if not CHROMADB_AVAILABLE:
        logger.warning("ChromaDB not installed; semantic search disabled.")
        return None

    if _client is None:
        try:
            # New, simplified client initialization for persistent storage
            _client = chromadb.PersistentClient(path=APP_CONFIG.CHROMA_PERSIST_DIRECTORY)
            logger.info("ChromaDB client initialized.")
        except Exception as e:
            logger.error(f"Failed to init Chroma client: {e}")
            return None

    if _collection is None:
        try:
            # The embedding function logic remains the same
            if embedding_functions and hasattr(embedding_functions, "SentenceTransformerEmbeddingFunction"):
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=APP_CONFIG.CHROMA_EMBEDDING_MODEL)
            else:
                ef = None
            
            _collection = _client.get_or_create_collection(name=APP_CONFIG.CHROMA_COLLECTION_NAME, embedding_function=ef)
            logger.info(f"Chroma collection '{APP_CONFIG.CHROMA_COLLECTION_NAME}' ready.")
        except Exception as e:
            logger.error(f"Failed to get/create Chroma collection: {e}")
            return None

    return _collection

def upsert_knowledge(id: str, text: str, metadata: Optional[Dict] = None):
    """Upsert a single knowledge item into Chroma collection.

    Args:
        id: Unique id (use topic name)
        text: Text to embed
        metadata: Optional metadata dict
    """
    try:
        if _collection is None:
            init_chroma()
        if _collection is None:
            return False
            
        _collection.upsert(ids=[id], documents=[text], metadatas=[metadata or {}])
        logger.debug(f"Upserted into chroma: {id}")
        
        # With PersistentClient, changes are automatically persisted.
        # The explicit _client.persist() call is no longer needed.
        
        return True
    except Exception as e:
        logger.error(f"Chroma upsert error for {id}: {e}")
        return False
        
def query_similar(text: str, n_results: int = 5):
    """Query chroma for similar documents to provided text.

    Returns list of dicts: [{'id':..., 'score':..., 'metadata':..., 'document':...}, ...]
    """
    try:
        if _collection is None:
            init_chroma()
        if _collection is None:
            return []
        results = _collection.query(query_texts=[text], n_results=n_results)
        
        docs = []
        # results come back as lists of lists because the input is a list
        for i in range(len(results.get('ids', [[]])[0])):
            docs.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': results['distances'][0][i]
            })
        return docs
    except Exception as e:
        logger.error(f"Chroma query error: {e}")
        return []


def clear_collection():
    try:
        if _collection:
            _collection.delete()
            logger.info("Chroma collection cleared")
    except Exception as e:
        logger.debug(f"Error clearing chroma collection: {e}")
