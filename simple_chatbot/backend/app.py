"""
app.py
This file for handling teh backend app
routing, requests, response, etc
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import os, sys

try:
    from chatbot_model import get_chatbot
    print("Succesfully Imported get_cahtbot")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)


# Flask app init
app = Flask(__name__, static_folder='../frontend')
CORS(app) # enable CORS for all routes


# Config
app.config['JSON_SORT_KEYS'] = False  # to keep the order of JSON keys
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # 10MB limit


# Chatbot init
print("Initializing Kampus Chatbot Model...")

try:
    chatbot = get_chatbot()
    chatbot.load()
    print("Chatbot Model Ready!")
except Exception as e:
    print(f"Failed to load model: {e}")
    chatbot = None


# Routes
@app.route('/')
def index():
    # serve frontend html
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_serve(path):
    # serve static file (CSS, JS)
    return send_from_directory(app.static_folder, path)

@app.route('/api/health', methods=['GET'])
def health_check():
    status = {
        'status': 'healthy' if chatbot and chatbot.is_loaded else 'unhealthy',
        'service': 'Kampus-Chatbot-Model-API',
        'version': 'prototype-1.0',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': chatbot.is_loaded if chatbot else False,
        'intents_count': len(chatbot.tags) if chatbot and chatbot.tags else 0
    }
    return jsonify(status)

@app.route('/api/chat', methods=['POST'])
def chat():
    # Check if model is loaded
    if not chatbot or not chatbot.is_loaded:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    # get amd validate input
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({
            'success': False,
            'error': 'No message provided'
        }), 400
    
    message = data['message'].strip()
    if not message:
        return jsonify({
            'success': False,
            'error': 'Message cannot be empty'
        }), 400
    
    # process with model
    try:
        intent, confidence = chatbot.predict(message)
        confidence_level = chatbot.get_confidence_level(confidence)
        response = chatbot.get_response_with_warning(message)
        
        result = {
            'success': True,
            'message': message,
            'intent': response['intent'],
            'confidence': response['confidence'],
            'confidence_level': response['confidence_level'],
            'response': response['response'],
            'warning': response['warning'],
            'has_warning': response['has_warning'],
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"[{intent}] '{message}' -> '{response}'")
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during processing message: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/intents', methods=['GET'])
def list_intents():
    if not chatbot or not chatbot.is_loaded:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
        
    intents_list = []
    for intent in chatbot.intents:
        intents_list.append({
            'tag': intent['tag'],
            'patterns_count': len(intent['patterns']),
            'responses_count': len(intent['response']),
            'example_patterns': intent['patterns'][0] if intent['patterns'] else None
        })
    
    return jsonify({
        'success': True,
        'intents': intents_list,
        'total': len(intents_list)
    })


# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# Run Server
if __name__ == '__main__':
    print("\n","-"*50)
    print("Kampus Chatbot Model API Server")
    print("-"*50)
    print(f"Static folder: {app.static_folder}")
    print("Starting server at http://localhost:5000")
    print("-"*50, "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
    
        
        