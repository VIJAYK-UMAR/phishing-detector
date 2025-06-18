from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import warnings
import re

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

# ✅ Load model once
model = joblib.load("phishing_model.pkl")

# ✅ Feature extractor — sample logic (you can replace with your real one)
def extract_features_from_url(url):
    features = []
    features.append(1 if re.search(r"\d", url) else 0)  # has digits
    features.append(1 if "@" in url else 0)
    features.append(len(url))                           # URL length
    features.append(url.count("."))                     # number of dots
    features.append(1 if "https" in url else 0)         # uses https
    features.append(1 if "-" in url else 0)             # has hyphen
    features.append(1 if re.match(r"http[s]?://\d+\.\d+\.\d+\.\d+", url) else 0)  # IP in URL

    # Pad or trim to match model input length
    if len(features) < len(model.feature_names_in_):
        features += [0] * (len(model.feature_names_in_) - len(features))
    elif len(features) > len(model.feature_names_in_):
        features = features[:len(model.feature_names_in_)]

    return features

# ✅ Health check route for Render + UptimeRobot
@app.route('/')
def home():
    return "<h2>✅ Backend is running and reachable!</h2>", 200

# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get("url")
        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        features = extract_features_from_url(url)
        input_df = pd.DataFrame([features], columns=model.feature_names_in_)
        prediction = model.predict(input_df)[0]
        return jsonify({'result': bool(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Required for Render hosting (binds to dynamic port)
if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
