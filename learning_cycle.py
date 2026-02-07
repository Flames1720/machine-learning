import os
import logging
import google.generativeai as genai
import sys
import nltk

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import APP_CONFIG
# Import db functions from the brain package
from brain.db import get_unknown_words, add_word, is_known, db
from researcher import research_new_concept
from categorizer import categorize_text
from relationship_mapper import find_and_map_relationships
from dictionary_validator import is_valid_word_api

# Get logger
logger = logging.getLogger(__name__)

# --- Gemini API Configuration ---
gemini_api_key = APP_CONFIG.GEMINI_API_KEY
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel(APP_CONFIG.GEMINI_MODEL)
        logger.info("Gemini API configured successfully for learning cycle.")
    except Exception as e:
        logger.critical(f"Failed to configure Gemini API: {e}", exc_info=True)
        gemini_model = None
else:
    logger.warning("GEMINI_API_KEY not found. Learning cycle cannot define new words.")
    gemini_model = None

# --- Learning Cycle Logic --- #
def trigger_learning_cycle():
    """This function is called to run the learning cycle.
    It can be triggered by a scheduler or run directly.
    """
    logger.info("Starting new learning cycle...")
    try:
        # Download WordNet if not already downloaded
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            logger.info("WordNet not found. Downloading...")
            nltk.download('wordnet')
            logger.info("WordNet downloaded successfully.")

        unknown_words = get_unknown_words()
        if not unknown_words:
            logger.info("No unknown words to process.")
            return

        for word_doc in unknown_words:
            word = word_doc.id

            if not is_valid_word_api(word):
                logger.warning(f"'{word}' is not a valid English word according to the API. Skipping and removing.")
                db.collection('unknown_words').document(word).delete()
                continue

            if not gemini_model:
                logger.warning(f"Skipping research for '{word}' because Gemini API is not configured.")
                continue
                
            logger.info(f"Researching new concept: {word}")
            
            explanation = research_new_concept(word, gemini_model)

            if explanation:
                logger.info(f"Successfully researched '{word}'. Categorizing and updating knowledge base.")
                category = categorize_text(explanation, gemini_model)
                add_word(word, explanation, is_known=True, category=category) 
                find_and_map_relationships(word)
            else:
                logger.warning(f"Research failed for '{word}'.")

        logger.info("Learning cycle finished.")
    except Exception as e:
        logger.critical(f"Error during learning cycle: {e}", exc_info=True)

# --- Direct Execution Block (for testing) --- #
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Setting up a direct run of the learning cycle for testing.")
    
    if not db:
        logger.critical("Database not initialized. Cannot run learning cycle test.")
    else:
        # 1. Add a test word to the database
        test_word = "autodidact"
        logger.info(f"Preparing to test with word: '{test_word}'")
        if is_known(test_word):
            logger.info(f"'{test_word}' is already known. For a clean test, it should be unknown.")
        else:
            logger.info(f"Adding '{test_word}' as an unknown word for this test run.")
            add_word(test_word, "", is_known=False)
        
        # 2. Run the learning cycle once
        trigger_learning_cycle()

        # 3. Verify the result
        logger.info(f"Verifying the result for '{test_word}'...")
        if is_known(test_word):
            logger.info(f"SUCCESS: '{test_word}' is now known. The learning cycle is working.")
        else:
            logger.error(f"FAILURE: '{test_word}' is still unknown. The learning cycle did not work as expected.")
