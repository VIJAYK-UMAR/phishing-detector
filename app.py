from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import warnings

# Suppress user warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

# ✅ Load model globally (only once)
model = joblib.load("phishing_model.pkl")

# ✅ Simple home route to test backend status
@app.route('/')
def home():
    return "✅ Backend is live! Use POST /predict to send a URL."

# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get("features", [])

        # Ensure input is passed correctly
        if not features or not isinstance(features, list):
            return jsonify({'error': 'Invalid input. Expected list of features.'}), 400

        # ✅ Convert to DataFrame with feature names to avoid warning
        input_df = pd.DataFrame([features], columns=model.feature_names_in_)
        prediction = model.predict(input_df)[0]

        result = bool(prediction)  # True = Legit, False = Phishing
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Run app using waitress (production-safe)
if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
