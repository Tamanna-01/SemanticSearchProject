# This file contains the Flask API that serves as the backend for the semantic search application.

from flask import Flask, request, jsonify
from flask_cors import CORS
from data_handler import DataLoader
from model_handler import EmbeddingModel
from search_engine import SemanticSearcher

app = Flask(__name__)
CORS(app)  # This allows your React app to communicate with Python safely

print("Initializing AI System... please wait.")
loader = DataLoader("embedding_index_3m.json")
searcher = SemanticSearcher(loader.data, loader.normalized_embeddings)
model = EmbeddingModel()
print("System Ready!")

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    # Convert text to numbers and search
    query_vec = model.encode(user_query)
    results = searcher.search(query_vector=query_vec, top_k=5)
    
    return jsonify(results)

if __name__ == '__main__':
    # Starts the API server on port 5000
    app.run(port=5000, debug=False)