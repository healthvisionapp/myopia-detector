import os
import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for
import firebase_admin
from firebase_admin import credentials, firestore
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ✅ Firebase Setup (using Render environment variable)
cred_json = os.getenv("SA_JSON")
if cred_json:
    cred = credentials.Certificate(json.loads(cred_json))
    firebase_admin.initialize_app(cred)
    db = firestore.client()
else:
    print("⚠️ SA_JSON not found. Firebase not initialized.")
    db = None

# ✅ Download model from Dropbox if not cached
MODEL_URL = "https://www.dropbox.com/scl/fi/vht7jgf9lgxtqr1m0r5jd/eye_disease_model_in_use.h5?rlkey=u1afkidhs9oj7z9sooc16yczj&st=2wsmb000&dl=1"
MODEL_PATH = "eye_disease_model_in_use.h5"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Dropbox...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Model downloaded successfully.")

# ✅ Load model
model = load_model(MODEL_PATH)

def preprocess_image(img_path):
    frame = cv2.imread(img_path)
    frame_resized = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_array = image.img_to_array(frame_rgb) / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        user_id = request.args.get('uid')  # ✅ UID from query
        print("🔥 UID received:", user_id)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # ✅ Predict
        frame_array = preprocess_image(filepath)
        prediction = model.predict(frame_array)
        label = "Myopia" if prediction[0] < 0.5 else "Normal"

        # ✅ Firestore Save
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
