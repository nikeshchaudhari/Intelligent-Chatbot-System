from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests

app = Flask(__name__)
CORS(app)

# Load FAQ data
faq_df = pd.read_csv("Ecommerce_FAQs.csv")
questions = faq_df["question"].tolist()
answers = faq_df["answer"].tolist()

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = embedder.encode(questions, convert_to_tensor=True)

@app.route('/')
def home():
    return send_file("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("user_input", "")

        user_embedding = embedder.encode(user_input, convert_to_tensor=True)
        scores = util.cos_sim(user_embedding, faq_embeddings)[0]
        top_score = scores.max().item()
        top_idx = scores.argmax().item()

        if top_score > 0.75:
            response = answers[top_idx]
        else:
            # Call Ollama local API (make sure Ollama is running)
            ollama_url = "http://localhost:11434/api/generate"
            payload = {
                "model": "mistral",
                "prompt": user_input,
                "stream": False
            }
            res = requests.post(ollama_url, json=payload)
            response = res.json().get("response", "I'm sorry, I couldn't understand that.")

        return jsonify({"bot_response": response})
    except Exception as e:
        return jsonify({"bot_response": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
