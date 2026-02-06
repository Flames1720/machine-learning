import os
import logging
import json
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from google.generativeai import types
from groq import Groq
import spacy

from config import APP_CONFIG

# --- Basic Logging Setup ---
logging.basicConfig(level=APP_CONFIG.LOG_LEVEL,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Service Initialization Functions ---

def _initialize_firebase():
    """Initializes the Firebase Admin SDK."""
    try:
        if not APP_CONFIG.FIREBASE_CREDENTIALS_JSON:
            raise ValueError("FIREBASE_CREDENTIALS_JSON not found in config.")
        
        creds_json_string = APP_CONFIG.FIREBASE_CREDENTIALS_JSON
        if creds_json_string.startswith('"') and creds_json_string.endswith('"'):
             creds_json_string = creds_json_string[1:-1]

        creds_dict = json.loads(creds_json_string)
        cred = credentials.Certificate(creds_dict)
        
        if not firebase_admin._apps:
            project_id = creds_dict.get('project_id')
            firebase_admin.initialize_app(cred, {'projectId': project_id})
            logger.info("Firebase Admin SDK initialized successfully.")
        
        return firestore.client()
    except Exception as e:
        logger.critical(f"Firebase initialization failed: {e}", exc_info=True)
        return None

def _configure_gemini():
    """Configures the Google GenAI library with the API key."""
    try:
        if not APP_CONFIG.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in config.")
        
        genai.configure(api_key=APP_CONFIG.GEMINI_API_KEY)
        logger.info("Google GenAI library configured successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to configure GenAI library: {e}", exc_info=True)
        return False

def _initialize_groq():
    """Initializes the Groq API client."""
    try:
        if not APP_CONFIG.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in config.")
        
        return Groq(api_key=APP_CONFIG.GROQ_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}", exc_info=True)
        return None

def _load_spacy_model():
    """Loads the spaCy model. Downloads it if not found."""
    try:
        nlp = spacy.load(APP_CONFIG.SPACY_MODEL)
        logger.info(f"spaCy model '{APP_CONFIG.SPACY_MODEL}' loaded.")
        return nlp
    except OSError:
        logger.warning(f"spaCy model '{APP_CONFIG.SPACY_MODEL}' not found. Attempting download...")
        try:
            from spacy.cli import download
            download(APP_CONFIG.SPACY_MODEL)
            nlp = spacy.load(APP_CONFIG.SPACY_MODEL)
            logger.info(f"spaCy model '{APP_CONFIG.SPACY_MODEL}' downloaded and loaded successfully.")
            return nlp
        except Exception as e:
            logger.critical(f"Failed to download and load spaCy model: {e}", exc_info=True)
            return None

# --- Initialize and Expose Services ---
db = _initialize_firebase()
gemini_configured = _configure_gemini()
groq_client = _initialize_groq()
nlp = _load_spacy_model()


# --- Gemini Helper with Persona ---
def generate_structured_gemini_content(contents: any) -> str:
    """
    Generates content using the configured Gemini library and applies
    the 'Logic-Mapping Architect' system instructions on each call.
    """
    if not gemini_configured:
        logger.error("Gemini library not configured, cannot generate content.")
        return '{"error": "Gemini library not configured."}'

    architect_instructions = """
    You are the 'Logic-Mapping Architect' for a recursive AI brain. 
    Your goal is to organize messy information into a clean visual structure.

    RULES:
    1. Every piece of information must be a 'Block'.
    2. If two Blocks are related, you must create a 'Thread' between their 'Nobs'.
    3. You must group smaller Blocks into larger 'Containers' (Groups).
    4. Your output must always be valid JSON so the Vercel website can render it.
    5. Always prioritize 'Spatial Logic'—place related items physically close to each other.
    """

    try:
        # Correct Method: Instantiate the model directly
        model = genai.GenerativeModel(
            model_name=APP_CONFIG.GEMINI_MODEL,
            system_instruction=architect_instructions,
            generation_config=types.GenerationConfig(
                temperature=0.2
            )
        )
        
        response = model.generate_content(contents)
        return response.text
    except Exception as e:
        logger.error(f"Error during Gemini content generation: {e}", exc_info=True)
        return f'{{"error": "Failed to generate content: {e}"}}'

# --- Health Check --- 
def get_health_status():
    """Returns a dictionary with the status of all initialized services."""
    health = {
        "firebase": {"online": db is not None},
        "gemini": {"configured": gemini_configured},
        "groq": {"configured": groq_client is not None},
        "spacy": {"online": nlp is not None}
    }
    all_ok = all(status.get('online', False) or status.get('configured', False) for status in health.values())
    health["all_services_ok"] = all_ok
    return health

# Log the health status on startup
logger.info(f"Service Health Status: {get_health_status()}")
