from flask import Flask, request, jsonify, send_from_directory
from brain import generate_response_from_knowledge, detect_and_log_unknown_words
import logging

# --- App Initialization and Logging ---
app = Flask(__name__, static_folder='static')
logging.basicConfig(level=logging.INFO)
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
    
    # Get the AI's response
    response_data = generate_response_from_knowledge(prompt)
    assistant_response = response_data.get("synthesized_answer")

    # If the AI doesn't know, trigger the learning flow
    if not assistant_response:
        logger.info("AI has no knowledge, logging words for learning.")
        detect_and_log_unknown_words(prompt)
        # You can customize this message if you want
        assistant_response = "Thank you. I have no prior knowledge of that. I will learn from your words."

    return jsonify({
        'synthesized_answer': assistant_response,
        'raw_knowledge': response_data.get('raw_knowledge')
    })

# --- Main entry point for running the server ---
if __name__ == '__main__':
    # Note: For development, this is fine. 
    # For production, a proper WSGI server like Gunicorn would be used.
    app.run(debug=True, port=5000)
