import os
import logging
import threading
from apscheduler.schedulers.background import BackgroundScheduler
import google.generativeai as genai
from dotenv import load_dotenv
from brain import db, detect_and_log_unknown_words
from researcher import research_new_concept

# Load environment variables and get logger
load_dotenv()
logger = logging.getLogger(__name__)

# --- Gemini API Configuration (remains the same) ---
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        logger.info("Gemini API configured successfully.")
    except Exception as e:
        logger.critical(f"Failed to configure Gemini API: {e}", exc_info=True)
else:
    logger.warning("GEMINI_API_KEY not found. Learning cycle cannot define new words.")

# --- Core Learning Functions (get_definition, learning_cycle_job - remain the same) ---
def get_definition_from_gemini(word):
    if not gemini_api_key:
        logger.warning(f"Cannot get definition for '{word}', GEMINI_API_KEY is not configured.")
        return None
    try:
        logger.info(f"Requesting definition for '{word}' from Gemini API.")
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(f"Define the word: {word}")
        logger.info(f"Successfully received definition for '{word}'.")
        return response.text
    except Exception as e:
        logger.error(f"Gemini API call failed for word '{word}': {e}", exc_info=True)
        return None

def learning_cycle_job():
    logger.info("LEARNING CYCLE JOB: Starting execution.")
    if not db:
        logger.error("LEARNING CYCLE JOB: Firestore is not initialized. Aborting.")
        return

    try:
        buffer_zone_ref = db.collection('buffer_zone')
        docs_to_process_query = buffer_zone_ref.where('mentions', '>', 1)
        words_to_process = [doc.id for doc in docs_to_process_query.stream()]

        if not words_to_process:
            logger.info("LEARNING CYCLE JOB: No words with multiple mentions found.")
            return

        logger.info(f"LEARNING CYCLE JOB: Found {len(words_to_process)} words to process: {words_to_process}")

        for word in words_to_process:
            logger.info(f"--- Processing '{word}' ---")
            definition = get_definition_from_gemini(word)

            if definition:
                solidified_knowledge_ref = db.collection('solidified_knowledge').document(word)
                solidified_knowledge_ref.set({'definition': definition, 'related_terms': []})
                logger.info(f"Solidified knowledge for '{word}'.")
                research_new_concept(word)
                detect_and_log_unknown_words(definition)
                buffer_zone_ref.document(word).delete()
                logger.info(f"Removed '{word}' from buffer_zone.")
            else:
                logger.warning(f"Failed to get a valid definition for '{word}'. It will be re-processed.")
        logger.info("LEARNING CYCLE JOB: Finished processing all words.")

    except Exception as e:
        logger.critical(f"LEARNING CYCLE JOB: An unexpected error occurred: {e}", exc_info=True)

# --- MODIFIED: Truly Non-Blocking Scheduler Start ---
def start_learning_cycle():
    
    def scheduler_task():
        try:
            scheduler = BackgroundScheduler(daemon=True)
            scheduler.add_job(learning_cycle_job, 'interval', hours=1, id='learning_job')
            scheduler.start()
            logger.info("Background learning cycle started successfully in its own thread.")
        except Exception as e:
            logger.critical(f"Failed to start the learning cycle scheduler in its thread: {e}", exc_info=True)

    # Run the scheduler in a separate thread to ensure this function returns immediately
    thread = threading.Thread(target=scheduler_task)
    thread.daemon = True
    thread.start()
    logger.info("Dispatched learning cycle scheduler to a background thread.")
