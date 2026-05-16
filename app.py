from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and tokenizer at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, "trained_models", "lstm_model.h5"))
with open(os.path.join(BASE_DIR, "trained_models", "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

MAXLEN = 100

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAXLEN)
    probability = float(model.predict(padded, verbose=0)[0][0])
    label = "Suicide" if probability > 0.5 else "Non-Suicide"

    return jsonify({
        "label": label,
        "probability": round(probability * 100, 2),
        "confidence": round((probability if probability > 0.5 else 1 - probability) * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)