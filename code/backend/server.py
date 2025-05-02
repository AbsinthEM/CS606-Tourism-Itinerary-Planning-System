# code/backend/server.py

import os, sys
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent “code/” folder onto PYTHONPATH so imports resolve
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from my_alns_wrapper import run_alns_with_params

app = Flask(__name__)
CORS(app)

@app.route('/')
def health():
    return "Server OK"

@app.route('/reoptimize', methods=['POST'])
def reoptimize():
    user_params = request.get_json()
    solution = run_alns_with_params(user_params)
    return jsonify(solution)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
