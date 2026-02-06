# brain/db.py
import logging
from datetime import datetime
from services import db

logger = logging.getLogger(__name__)

KNOWN_WORDS_COLLECTION = "known_words"
UNKNOWN_WORDS_COLLECTION = "unknown_words"

def is_known(word: str) -> bool:
    """Checks if a word is in the known_words collection."""
    try:
        doc_ref = db.collection(KNOWN_WORDS_COLLECTION).document(word)
        return doc_ref.get().exists
    except Exception as e:
        logger.error(f"Error checking if word '{word}' is known: {e}", exc_info=True)
        return False

def add_word(word: str, explanation: str, is_known: bool = False, category: str = None):
    """Adds or updates a word in the database.

    If is_known is True, it adds the word to 'known_words' and deletes it from 'unknown_words'.
    If is_known is False, it adds the word to 'unknown_words'.
    """
    try:
        if is_known:
            # Add to known_words collection
            known_ref = db.collection(KNOWN_WORDS_COLLECTION).document(word)
            data = {
                "explanation": explanation,
                "learned_at": datetime.utcnow(),
            }
            if category:
                data["category"] = category
            known_ref.set(data)
            
            # Attempt to delete from unknown_words collection
            unknown_ref = db.collection(UNKNOWN_WORDS_COLLECTION).document(word)
            if unknown_ref.get().exists:
                unknown_ref.delete()
            logger.info(f"Moved word '{word}' to known words.")
        else:
            # Add to unknown_words collection
            unknown_ref = db.collection(UNKNOWN_WORDS_COLLECTION).document(word)
            unknown_ref.set({"added_at": datetime.utcnow()})
            logger.info(f"Added word '{word}' to unknown words.")
    except Exception as e:
        logger.error(f"Error adding word '{word}': {e}", exc_info=True)

def get_unknown_words() -> list:
    """Retrieves all documents from the unknown_words collection."""
    try:
        docs = db.collection(UNKNOWN_WORDS_COLLECTION).stream()
        return list(docs)
    except Exception as e:
        logger.error(f"Error getting unknown words: {e}", exc_info=True)
        return []
