import os
import spacy
import logging
import firebase_admin
# I am using google.generativeai as instructed, though the deprecation warning will appear.
import google.generativeai as genai 
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import json

# Load environment variables and get a logger
load_dotenv()
logger = logging.getLogger(__name__)

# --- Load Models and Services ---
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded.")
except Exception as e:
    logger.critical(f"Failed to load spaCy model: {e}", exc_info=True)
    nlp = None

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

# --- Gemini Configuration ---
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    logger.info("Gemini API key found and configured.")
else:
    logger.warning("GEMINI_API_KEY not found. Response generation will be limited.")

def detect_and_log_unknown_words(text):
    logger.info(f"Detecting unknown words in text: '{text[:50]}...'")
    if not db or not nlp:
        logger.error("Firestore or spaCy not initialized. Cannot log words.")
        return
    try:
        doc = nlp(text)
        potential_words = set(token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN", 'VERB'] and token.is_alpha and not token.is_stop)
        if not potential_words:
            return
        for word in potential_words:
            known_word_ref = db.collection('solidified_knowledge').document(word)
            if not known_word_ref.get().exists:
                doc_ref = db.collection('buffer_zone').document(word)
                doc_ref.set({'mentions': firestore.Increment(1)}, merge=True)
    except Exception as e:
        logger.error(f"Error during text processing: {e}", exc_info=True)

def generate_response_from_knowledge(prompt_text):
    logger.info(f"Generating response for prompt: '{prompt_text}'")
    response_data = {
        "raw_knowledge": None,
        "synthesized_answer": None,
        "prompt": prompt_text
    }

    if not db or not nlp:
        response_data["synthesized_answer"] = "I am currently unable to process this request as my core services are offline."
        return response_data

    doc = nlp(prompt_text)
    key_concepts = {token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]}
    
    retrieved_knowledge = ""
    if key_concepts:
        for concept in key_concepts:
            doc_ref = db.collection('solidified_knowledge').document(concept)
            doc_snapshot = doc_ref.get()
            if doc_snapshot.exists:
                knowledge = doc_snapshot.to_dict()
                definition = knowledge.get('definition', '')
                retrieved_knowledge += f"- Fact about {concept}: {definition}\n"
    
    # If no knowledge is found, return None to trigger the learning flow in the UI.
    if not retrieved_knowledge:
        logger.info("No relevant information found in my knowledge base. Triggering learning flow.")
        return response_data # synthesized_answer is still None here

    response_data["raw_knowledge"] = retrieved_knowledge

    # If knowledge IS found, use the "Teacher" Gemini call to synthesize an answer.
    if not gemini_api_key:
        response_data["synthesized_answer"] = f"I don't have a complete thought on that yet, but here is the raw information I found:\n\n{retrieved_knowledge}"
        return response_data

    try:
        logger.info("Knowledge found. Engaging Teacher Mode...")
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        synthesis_prompt = (
            f"Based *only* on the following facts from my internal knowledge base, "
            f"formulate a direct and helpful answer to the user's question: '{prompt_text}'\n\n"
            f"My Knowledge:\n{retrieved_knowledge}"
        )
        
        response = model.generate_content(synthesis_prompt)
        response_data["synthesized_answer"] = response.text
        logger.info("Successfully received synthesized response from Gemini (Teacher).")
    except Exception as e:
        logger.error(f"Error during Gemini synthesis (Teacher): {e}", exc_info=True)
        response_data["synthesized_answer"] = "I found some information, but I struggled to put my thoughts together."
    
    return response_data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running brain.py directly for testing...")
    
    # Test Case 1: No knowledge exists (should trigger learning)
    test_prompt_unknown = "What is Schrodinger's Cat?"
    print(f"\n--- TEST CASE 1: UNKNOWN KNOWLEDGE ---")
    print(f"Sending test prompt: '{test_prompt_unknown}'\n")
    response = generate_response_from_knowledge(test_prompt_unknown)
    print("\n--- TEST RESULT ---")
    print(json.dumps(response, indent=2))
    if response["synthesized_answer"] is None:
        print("\nSUCCESS: AI correctly indicated no knowledge, allowing learning flow.")
    else:
        print("\nFAILURE: AI provided an answer instead of indicating no knowledge.")
    print("--- END TEST ---\n")

    # Test Case 2: Knowledge exists (should trigger teacher mode)
    # To properly test this, we would need to mock the firestore call.
    # For now, we assume the teacher mode works as verified previously.
