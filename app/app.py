import io
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'astro_cnn.keras')
LABEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'label_map.json')
TARGET_SIZE = (128, 128)

TF_MODEL = None
LABELS = []
if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, 'r') as f:
        LABELS = json.load(f).get('classes', [])
if os.path.exists(MODEL_PATH):
    TF_MODEL = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Warning: TF model not found at", MODEL_PATH)

@app.route('/')
def front():
    return render_template('home.html')

@app.route('/start')
def home():
    return render_template('start.html')

@app.route('/predict', methods=['POST'])
def predict():
    if TF_MODEL is None:
        return jsonify({'error': 'model not loaded'}), 500
    f = request.files.get('uploadedImage')
    if not f:
        return jsonify({'error': 'no file uploaded'}), 400
    try:
        img = Image.open(io.BytesIO(f.read())).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.BILINEAR)
        arr = np.asarray(img).astype('float32') / 255.0
        x = np.expand_dims(arr, 0)
        preds = TF_MODEL.predict(x)[0]
        idx = int(np.argmax(preds))
        if float(preds[idx]) < 0.8:
            return jsonify({'error':'Photo Does Not Match One of the Classifications'}),  500
        return jsonify({'class': LABELS[idx] if LABELS else str(idx), 'confidence': float(preds[idx])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)