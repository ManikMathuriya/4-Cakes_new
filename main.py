from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

# ======================
# LOAD MODELS
# ======================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

diabetes_model = joblib.load(os.path.join(MODEL_DIR, "diabetes_model.joblib"))
stress_model = joblib.load(os.path.join(MODEL_DIR, "stress_model.joblib"))

# ======================
# HOME ROUTE
# ======================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "HealthAI Guardian Backend Running ✅"})


# ======================
# STRESS PREDICTION
# ======================
@app.route("/health/predict/stress", methods=["POST"])
def predict_stress():
    data = request.json

    features = [[
        float(data.get("anxiety_level")),
        float(data.get("depression")),
        float(data.get("sleep_quality")),
        float(data.get("peer_pressure"))
    ]]

    prediction = int(stress_model.predict(features)[0])
    result = "High Stress" if prediction == 1 else "Low Stress"

    return jsonify({
        "stress_prediction": prediction,
        "stress_level": result
    })


# ======================
# DIABETES PREDICTION
# ======================
@app.route("/health/predict/diabetes", methods=["POST"])
def predict_diabetes():
    data = request.json

    features = [[
        float(data.get("Pregnancies", 0)),
        float(data.get("Glucose")),
        float(data.get("BloodPressure")),
        float(data.get("SkinThickness", 20)),
        float(data.get("Insulin", 80)),
        float(data.get("BMI")),
        float(data.get("DiabetesPedigreeFunction", 0.5)),
        float(data.get("Age"))
    ]]

    prediction = int(diabetes_model.predict(features)[0])
    probability = diabetes_model.predict_proba(features)[0].tolist()

    result = "High Risk" if prediction == 1 else "Low Risk"

    return jsonify({
        "diabetes_prediction": prediction,
        "risk_level": result,
        "probability": probability
    })


# ======================
# FACE-BASED EMOTION / STRESS ANALYSIS
# (Lightweight – Hackathon Friendly)
# ======================
@app.route("/health/face-emotion", methods=["POST"])
def analyze_face():
    data = request.json

    image_base64 = data.get("image")

    if not image_base64:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image
    img_data = base64.b64decode(image_base64)
    np_img = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image format"}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haarcascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                         "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"emotion": "No Face Detected"}), 200

    # Simple hackathon-friendly emotion estimation
    (x, y, w, h) = faces[0]
    face_region = gray[y:y+h, x:x+w]

    brightness = np.mean(face_region)

    # Fake but impressive rules for demo
    if brightness < 80:
        emotion = "Sad / Tired"
    elif brightness < 130:
        emotion = "Neutral"
    else:
        emotion = "Happy"

    return jsonify({
        "face_detected": True,
        "estimated_emotion": emotion
    })


# ======================
# RUN SERVER
# ======================
if __name__ == "__main__":
    app.run(debug=True)
