from flask import Flask, request, jsonify, send_from_directory
import logging

# Import the new config and services
from config import APP_CONFIG
from services import db, nlp, get_health_status

from brain.knowledge_base import generate_response_from_knowledge

# --- App Initialization and Logging ---
app = Flask(__name__, static_folder='static')
logging.basicConfig(level=APP_CONFIG.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Serve the HTML Frontend ---
@app.route('/')
def serve_index():
    """Serves the main HTML page."""
    return send_from_directory(app.static_folder, 'index.html')

# --- API Endpoint for the Chat ---
@app.route('/ask', methods=['POST'])
def ask():
    """
    Handles POST requests to the /ask endpoint.
    Receives a prompt, gets a response from the brain, and returns it.
    """
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    logger.info(f"Received prompt: {prompt}")
    
    response_data = generate_response_from_knowledge(prompt)
    assistant_response = response_data.get("synthesized_answer")

    return jsonify({
        'synthesized_answer': assistant_response,
        'raw_knowledge': response_data.get('raw_knowledge')
    })

# --- Health Check Endpoint ---
@app.route('/health')
def health_check():
    """Returns the health status of the application and its services."""
    return jsonify(get_health_status())

# --- Main entry point for running the server (for local development) ---
if __name__ == '__main__':
    app.run(debug=True, port=8080)
