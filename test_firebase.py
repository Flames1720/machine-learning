
import os
import json
import logging
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Firebase Initialization ---
db = None
if not firebase_admin._apps:
    try:
        logger.info("Initializing Firebase Admin SDK...")
        firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS')
        if not firebase_creds_json:
            raise ValueError("The FIREBASE_CREDENTIALS environment variable is not set.")
        
        creds_dict = json.loads(firebase_creds_json)
        cred = credentials.Certificate(creds_dict)
        project_id = creds_dict.get('project_id')
        
        firebase_admin.initialize_app(cred, {'projectId': project_id})
        db = firestore.client()
        logger.info("Firebase initialization successful.")

    except Exception as e:
        db = None
        logger.critical(f"FIREBASE INITIALIZATION FAILED: {e}", exc_info=True)
        logger.warning("Database features will be disabled.")
else:
    db = firestore.client()
    logger.info("Firebase app already initialized.")

# --- Test Firestore Write and Read ---
if db:
    try:
        # 1. Write a test document
        test_collection = 'test_collection'
        test_doc_id = 'test_document'
        test_data = {'field': 'hello_world'}
        
        logger.info(f"Attempting to write to Firestore: /{test_collection}/{test_doc_id}")
        doc_ref = db.collection(test_collection).document(test_doc_id)
        doc_ref.set(test_data)
        logger.info("Write operation completed.")

        # 2. Read the document back
        logger.info(f"Attempting to read from Firestore: /{test_collection}/{test_doc_id}")
        retrieved_doc = doc_ref.get()

        if retrieved_doc.exists:
            retrieved_data = retrieved_doc.to_dict()
            logger.info(f"Successfully read data: {retrieved_data}")
            if retrieved_data == test_data:
                print("✅ SUCCESS: Firestore test passed. Write and read operations are working.")
            else:
                print("❌ FAILURE: Data mismatch. Read data does not match written data.")
        else:
            print("❌ FAILURE: Firestore test failed. Document was not found after writing.")

    except Exception as e:
        print(f"❌ FAILURE: An error occurred during the Firestore test: {e}")
        logger.critical(f"Firestore test error: {e}", exc_info=True)
else:
    print("❌ FAILURE: Firestore could not be initialized. Cannot run the test.")
