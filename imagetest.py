import os
import json
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for
import firebase_admin
from firebase_admin import credentials, firestore
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ---------- Firebase (from env var) ----------
cred_json = os.getenv("SA_JSON")
if cred_json:
    try:
        cred = credentials.Certificate(json.loads(cred_json))
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase initialized from SA_JSON")
    except Exception as e:
        print("❌ Firebase init error:", e)
        db = None
else:
    print("⚠️ SA_JSON not found. Firebase not initialized.")
    db = None

# ---------- Model (lazy download + lazy load) ----------
MODEL_URL = (
    "https://www.dropbox.com/scl/fi/vht7jgf9lgxtqr1m0r5jd/"
    "eye_disease_model_in_use.h5?rlkey=u1afkidhs9oj7z9sooc16yczj&st=2wsmb000&dl=1"
)
MODEL_PATH = "eye_disease_model_in_use.h5"
_model = None  # lazy-loaded

def ensure_model_file(path=MODEL_PATH, url=MODEL_URL):
    """Download the model once if missing."""
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return path
    print("📥 Downloading model from Dropbox...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
    print("✅ Model downloaded:", path)
    return path

def get_model():
    """Load the Keras model on first use (keeps startup fast)."""
    global _model
    if _model is not None:
        return _model
    from tensorflow.keras.models import load_model
    ensure_model_file()
    _model = load_model(MODEL_PATH)
    print("✅ Model loaded into memory.")
    return _model

# ---------- Preprocess ----------
def preprocess_image(img_path):
    frame = cv2.imread(img_path)
    frame_resized = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_array = image.img_to_array(frame_rgb) / 255.0
    frame_array = np.expand_dims(frame_array, axis=0).astype(np.float32)
    return frame_array

# ---------- Routes ----------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Lightweight health endpoint for Koyeb/Render checks
@app.route('/health', methods=['GET'])
def health():
    return "ok", 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        user_id = request.args.get('uid')  # optional UID from query
        print("🔥 UID received:", user_id)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Predict (lazy-load model here)
        frame_array = preprocess_image(filepath)
        model = get_model()
        pred = model.predict(frame_array)
        score = float(np.ravel(pred)[0])  # ensure scalar
        label = "Myopia" if score < 0.5 else "Normal"

        # Save to Firestore if configured and UID provided
        if db and user_id:
            try:
                db.collection("users").document(user_id).collection("eye_records").add({
                    "result": label,
                    "image_name": file.filename,
                    "timestamp": firestore.SERVER_TIMESTAMP,
                })
                print(f"✅ Saved to Firestore (User: {user_id}) -> {label}")
            except Exception as e:
                print("❌ Firestore Error:", e)
        else:
            print("⚠️ Firestore not initialized or UID missing.")

        return render_template(
            'index.html',
            result=label,
            image_url=url_for('static', filename='uploads/' + file.filename)
        )

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5003, debug=True)
