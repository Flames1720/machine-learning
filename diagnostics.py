import json
import logging
import os

def run_diagnostics():
    results = {}

    # Check spaCy
    try:
        import spacy
        results['spacy_version'] = spacy.__version__
        try:
            spacy.load('en_core_web_sm')
            results['spacy_model'] = 'loaded'
        except Exception as me:
            results['spacy_model'] = f'error: {me}'
    except Exception as e:
        results['spacy'] = f'error: {e}'

    # Check firebase_admin SDK
    try:
        import firebase_admin
        results['firebase_admin_sdk'] = 'installed'
    except Exception as e:
        results['firebase_admin_sdk'] = f'error: {e}'

    # Check streamlit secrets (do not print secrets)
    try:
        import streamlit as st
        keys = {}
        try:
            keys['has_FIREBASE_CREDENTIALS'] = 'FIREBASE_CREDENTIALS' in st.secrets
        except Exception:
            keys['has_FIREBASE_CREDENTIALS'] = False
        try:
            keys['has_GEMINI_API_KEY'] = 'GEMINI_API_KEY' in st.secrets
        except Exception:
            keys['has_GEMINI_API_KEY'] = False
        results['streamlit_secrets'] = keys
    except Exception as e:
        results['streamlit'] = f'error: {e}'

    # Try to validate FIREBASE_CREDENTIALS JSON if present in environment variable (safe check)
    firebase_creds_env = os.environ.get('FIREBASE_CREDENTIALS')
    if firebase_creds_env:
        try:
            _ = json.loads(firebase_creds_env)
            results['FIREBASE_CREDENTIALS_env_valid_json'] = True
        except Exception as e:
            results['FIREBASE_CREDENTIALS_env_valid_json'] = f'error: {e}'

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_diagnostics()
