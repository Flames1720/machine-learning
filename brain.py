import os
import spacy
import logging
import firebase_admin
import google.genai as genai
import streamlit as st
from firebase_admin import credentials, firestore
import json

# Get a logger
logger = logging.getLogger(__name__)

# --- Load Models and Services ---
# Note: These will only be loaded once due to Streamlit's caching mechanism.

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model and caches it."""
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded.")
        return nlp
    except Exception as e:
        logger.critical(f"Failed to load spaCy model: {e}", exc_info=True)
        return None

nlp = load_spacy_model()

@st.cache_resource
def initialize_firebase():
    """Initializes Firebase Admin SDK and caches the client."""
    db = None
    try:
        firebase_creds_json = st.secrets["FIREBASE_CREDENTIALS"]
        creds_dict = json.loads(firebase_creds_json)
        
        # Avoid re-initializing the app
        if not firebase_admin._apps:
            logger.info("Initializing Firebase Admin SDK...")
            cred = credentials.Certificate(creds_dict)
            project_id = creds_dict.get('project_id')
            firebase_admin.initialize_app(cred, {'projectId': project_id})
        
        db = firestore.client()
        logger.info("Firebase initialization successful.")
        return db
    except Exception as e:
        logger.critical(f"FIREBASE INITIALIZATION FAILED: {e}", exc_info=True)
        logger.warning("Database features will be disabled.")
        return None

db = initialize_firebase()

# --- Configure Gemini API ---
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        logger.info("Gemini API key found and configured.")
    else:
        logger.warning("GEMINI_API_KEY not found in Streamlit secrets. Response generation will be limited.")
except KeyError:
    logger.warning("GEMINI_API_KEY not found in Streamlit secrets.")
    gemini_api_key = None
except Exception as e:
    logger.error(f"An unexpected error occurred during Gemini configuration: {e}")
    gemini_api_key = None


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
    
    if not retrieved_knowledge:
        logger.info("No relevant information found in my knowledge base. Triggering learning flow.")
        return response_data

    response_data["raw_knowledge"] = retrieved_knowledge

    if not gemini_api_key:
        response_data["synthesized_answer"] = f"I don't have a complete thought on that yet, but here is the raw information I found:\n\n{retrieved_knowledge}"
        return response_data

    try:
        logger.info("Knowledge found. Engaging Teacher Mode...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        
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
    # This block is for local testing only and won't run in Streamlit
    logging.basicConfig(level=logging.INFO)
    logger.info("Running brain.py directly requires secrets to be set in your environment.")
    # You would need to mock st.secrets for local CLI testing, for example:
    # class MockStreamlit:
    #     secrets = {
    #         "FIREBASE_CREDENTIALS": os.environ.get("FIREBASE_CREDENTIALS"),
    #         "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY")
    #     }
    # st = MockStreamlit()
    # print(generate_response_from_knowledge("What is a neural network?"))
