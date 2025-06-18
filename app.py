from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import warnings
import re
from urllib.parse import urlparse

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

# ✅ Load model only once at startup
model = joblib.load("phishing_model.pkl")

# ✅ Custom feature extraction logic from URL
def extract_features_from_url(url):
    features = []

    # Example features — add yours accordingly
    features.append(1 if re.search(r"\d", url) else 0)  # has digits
    features.append(1 if "@" in url else 0)             # has @ symbol
    features.append(len(url))                           # URL length
    features.append(url.count('.'))                     # number of dots
    features.append(1 if "https" in url else 0)         # uses HTTPS
    features.append(1 if "-" in url else 0)             # has hyphen
    features.append(1 if re.match(r"http[s]?://\d+\.\d+\.\d+\.\d+", url) else 0)  # IP address
    
    # Pad or trim to match model’s feature length
    if len(features) < len(model.feature_names_in_):
        features += [0] * (len(model.feature_names_in_) - len(features))
    elif len(features) > len(model.feature_names_in_):
        features = features[:len(model.feature_names_in_)]

    return features

# ✅ Route to check backend status
@app.route('/')
def home():
    return "✅ Backend is live! Use POST /predict with a URL."

# ✅ Main prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get("url")

        if not url:
            return jsonify({'error': 'URL not provided'}), 400

        features = extract_features_from_url(url)
        input_df = pd.DataFrame([features], columns=model.feature_names_in_)
        prediction = model.predict(input_df)[0]

        return jsonify({'result': bool(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Production deployment using waitress
if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
