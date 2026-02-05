from flask import Flask, request, jsonify, render_template
from brain import generate_response_from_knowledge

app = Flask(__name__, static_folder='static', template_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    response_data = generate_response_from_knowledge(prompt)
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
