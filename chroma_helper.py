"""Helper wrapper for ChromaDB local collection and embeddings.

Provides init, upsert, and similarity query helpers used by the learning flows.
"""
import logging
from typing import Optional, List, Dict

from config import APP_CONFIG

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
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
            settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=APP_CONFIG.CHROMA_PERSIST_DIRECTORY)
            _client = chromadb.Client(settings)
            logger.info("ChromaDB client initialized.")
        except Exception as e:
            logger.error(f"Failed to init Chroma client: {e}")
            return None

    if _collection is None:
        try:
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

# ... (the rest of your chroma_helper.py functions remain the same)
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
        # Persist where applicable
        try:
            _client.persist()
        except Exception:
            pass
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
        for ids, docs_list, metas, distances in zip(results.get('ids', []), results.get('documents', []), results.get('metadatas', []), results.get('distances', [])):
            # results come back as lists of lists
            for i, did in enumerate(ids):
                docs.append({
                    'id': did,
                    'document': docs_list[i] if docs_list else None,
                    'metadata': metas[i] if metas else {},
                    'score': distances[i] if distances else None
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
