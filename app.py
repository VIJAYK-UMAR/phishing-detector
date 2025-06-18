from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS
import joblib
import pandas as pd
from Phishing_Detector import extract_features
from waitress import serve


app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# Load the trained phishing detection model
model = joblib.load("phishing_model.pkl")

@app.route('/')
def home():
    return "<h2>✅ Server is live! Backend is working.</h2>", 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        url = data.get("url")
        print(f"📥 Received URL: {url}")

        # Extract features from the URL
        features = extract_features(url)
        print(f"🧠 Features extracted: {features}")

        # Create DataFrame for prediction
        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]
        print(f"🔮 Prediction result: {prediction}")

        return jsonify({"result": bool(prediction == 1)})  # True for Legitimate
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return jsonify({"error": str(e)})

#if __name__ == "__main__":
#   app.run(debug=True)


# if __name__ == "__main__":
#     from waitress import serve
#     serve(app, host="0.0.0.0", port=5000)



if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

