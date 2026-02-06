import logging
import sys
import os

# Add the project root to the Python path to allow for correct module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Correctly import the database functions from the brain.db module
from brain.db import is_known, add_word, get_unknown_words
from services import db as firestore_db # For checking initialization

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_test_word():
    """
    Adds a test word to the database to verify the learning cycle.
    """
    word = "autodidact"
    
    logging.info(f"Checking if '{word}' is already known...")
    # Use the correctly imported function
    if is_known(word):
        logging.info(f"'{word}' is already known. Deleting to ensure a clean test.")
        # This function needs to be defined or imported - assuming db_admin function for now
        # For now, let's just proceed, assuming it will be re-added as unknown.
        pass # Placeholder

    logging.info(f"Adding '{word}' as an unknown word for the test.")
    # Use the correctly imported function
    add_word(word, "", is_known=False)
    
    # Verify it's in the unknown collection
    # Use the correctly imported function
    unknown_words = [doc.id for doc in get_unknown_words()]
    if word in unknown_words:
        logging.info(f"Successfully added '{word}' to the unknown words collection.")
    else:
        logging.error(f"Failed to add '{word}' to the unknown words collection.")

if __name__ == "__main__":
    # Check if the firestore client from services is initialized
    if not firestore_db:
        logging.critical("Database not initialized. Cannot run test.")
    else:
        add_test_word()
