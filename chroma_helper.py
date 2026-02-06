"""Helper wrapper for ChromaDB, supporting both local and remote collections.

Provides init, upsert, and similarity query helpers. It can connect to a local,
persistent ChromaDB instance or a remote one based on environment variables.
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
    logger.warning("ChromaDB library not found. Semantic search will be disabled.")


_client = None
_collection = None

def init_chroma():
    """Initialize Chroma client and collection. Safe to call multiple times.

    Connects to a remote ChromaDB instance if CHROMA_HOST and CHROMA_PORT
    are set in the environment. Otherwise, it falls back to a local persistent client.
    """
    global _client, _collection
    if not CHROMADB_AVAILABLE:
        return None

    if _client is None:
        try:
            # --- NEW: Remote ChromaDB Connection Logic ---
            if APP_CONFIG.CHROMA_HOST and APP_CONFIG.CHROMA_PORT:
                logger.info(f"Connecting to remote ChromaDB at {APP_CONFIG.CHROMA_HOST}:{APP_CONFIG.CHROMA_PORT}")
                _client = chromadb.HttpClient(
                    host=APP_CONFIG.CHROMA_HOST, 
                    port=APP_CONFIG.CHROMA_PORT
                )
            # --- Fallback to Local Persistent Client ---
            else:
                logger.info(f"Using local persistent ChromaDB at {APP_CONFIG.CHROMA_PERSIST_DIRECTORY}")
                _client = chromadb.PersistentClient(path=APP_CONFIG.CHROMA_PERSIST_DIRECTORY)
            
            logger.info("ChromaDB client initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            return None

    if _collection is None:
        try:
            if embedding_functions and hasattr(embedding_functions, "SentenceTransformerEmbeddingFunction"):
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=APP_CONFIG.CHROMA_EMBEDDING_MODEL)
            else:
                logger.warning("SentenceTransformer embedding function not available. Embeddings will not be generated.")
                ef = None
            
            _collection = _client.get_or_create_collection(name=APP_CONFIG.CHROMA_COLLECTION_NAME, embedding_function=ef)
            logger.info(f"Chroma collection '{APP_CONFIG.CHROMA_COLLECTION_NAME}' is ready.")
        except Exception as e:
            logger.critical(f"Failed to get or create Chroma collection: {e}", exc_info=True)
            return None

    return _collection

def upsert_knowledge(id: str, text: str, metadata: Optional[Dict] = None):
    """Upsert a single knowledge item into the Chroma collection.

    Args:
        id: A unique identifier for the knowledge item (e.g., the topic name).
        text: The text content to be embedded and stored.
        metadata: An optional dictionary of metadata associated with the item.
    """
    if _collection is None:
        init_chroma()
    if _collection is None:
        logger.error("Cannot upsert knowledge: Chroma collection is not initialized.")
        return False
        
    try:
        _collection.upsert(ids=[id], documents=[text], metadatas=[metadata or {}])
        logger.debug(f"Upserted knowledge with ID: {id}")
        return True
    except Exception as e:
        logger.error(f"ChromaDB upsert error for ID '{id}': {e}", exc_info=True)
        return False

def query_similar(text: str, n_results: int = 5) -> List[Dict]:
    """Query Chroma for documents similar to the provided text.

    Returns:
        A list of dictionaries, where each dictionary contains the ID, score,
        metadata, and document of a similar item. Returns an empty list on error.
    """
    if _collection is None:
        init_chroma()
    if _collection is None:
        logger.error("Cannot query: Chroma collection is not initialized.")
        return []
        
    try:
        results = _collection.query(query_texts=[text], n_results=n_results)
        
        docs = []
        # The API returns results as a dictionary of lists, so we need to transpose it.
        ids = results.get('ids', [[]])[0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        for i in range(len(ids)):
            docs.append({
                'id': ids[i],
                'document': documents[i],
                'metadata': metadatas[i],
                'score': distances[i]
            })
        return docs
    except Exception as e:
        logger.error(f"ChromaDB query error: {e}", exc_info=True)
        return []

def clear_collection():
    """Deletes all items from the current collection. This is irreversible."""
    if _collection is None:
        init_chroma()
    if _collection is None:
        logger.error("Cannot clear collection: Chroma collection is not initialized.")
        return

    try:
        collection_name = _collection.name
        _client.delete_collection(name=collection_name)
        logger.info(f"Chroma collection '{collection_name}' cleared successfully.")
        # Force re-initialization on next call
        _collection = None
        init_chroma() 
    except Exception as e:
        logger.error(f"Error clearing Chroma collection: {e}", exc_info=True)
