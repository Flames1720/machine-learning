import os
from dataclasses import dataclass
from dotenv import load_dotenv

# --- Load Environment Variables ---
# This is the crucial step. By loading .env here, we ensure that all environment
# variables are set before any other part of the application tries to access them.
load_dotenv()

# --- Logging Configuration ---
# Maps string-based log levels from environment variables to Python's logging constants.
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

@dataclass(frozen=True)
class Config:
    """
    A centralized, immutable configuration class for the application.

    This dataclass fetches its values from environment variables, providing a single,
    reliable source of truth for all configuration settings. Using a dataclass
    ensures that the configuration is type-safe and cannot be changed at runtime,
    preventing a wide class of potential bugs.

    The `frozen=True` argument makes instances of this class immutable.
    """
    
    # --- General App Settings ---
    LOG_LEVEL: int = LOG_LEVELS.get(os.environ.get("LOG_LEVEL", "INFO").upper(), 20)

    # --- External Service API Keys ---
    # Fetches API keys from environment variables. Returns None if a key is not set.
    GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY")
    GROQ_API_KEY: str | None = os.environ.get("GROQ_API_KEY")

    # --- AI Model Specifications ---
    # Defines the specific models to be used with the AI services.
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash-latest")
    GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
    
    # --- Firebase Configuration ---
    # Retrieves the full Firebase credentials JSON from an environment variable.
    FIREBASE_CREDENTIALS_JSON: str | None = os.environ.get("FIREBASE_CREDENTIALS")

    # --- NLP Model ---
    # Specifies the spaCy model for Natural Language Processing tasks.
    SPACY_MODEL: str = "en_core_web_sm"

    # --- Vector Database ---
    # Specifies the directory to persist ChromaDB data.
    CHROMA_PERSIST_DIRECTORY: str = os.environ.get("CHROMA_PERSIST_DIRECTORY", "db/chroma")
    
    # --- Remote settings for deployment (e.g., Streamlit Cloud) ---
    CHROMA_HOST: str | None = os.environ.get("CHROMA_HOST")
    CHROMA_PORT: int | None = int(os.environ.get("CHROMA_PORT")) if os.environ.get("CHROMA_PORT") else None

    CHROMA_EMBEDDING_MODEL: str = os.environ.get("CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHROMA_COLLECTION_NAME: str = os.environ.get("CHROMA_COLLECTION_NAME", "knowledge_base")

# --- Global Configuration Instance ---
# A single, immutable instance of the Config class that is imported by other modules.
# This singleton pattern ensures that configuration is consistent across the application.
APP_CONFIG = Config()
