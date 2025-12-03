import os
import sys
import json
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'astro_cnn.keras')
LABEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'label_map.json')
TARGET_SIZE = (128, 128)  # must match training

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(f"Label map not found: {LABEL_PATH}")

with open(LABEL_PATH, 'r') as f:
    label_map = json.load(f)
CLASSES = label_map.get('classes')

model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

def preprocess_image(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(path_or_bytes)).convert('RGB')
    else:
        img = Image.open(path_or_bytes).convert('RGB')
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    arr = np.asarray(img).astype('float32') / 255.0
    return np.expand_dims(arr, 0)  # (1, H, W, 3)

def predict(path):
    x = preprocess_image(path)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    return {"class": CLASSES[idx], "confidence": float(probs[idx]), "probs": probs.tolist()}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 model/infer_tf.py /path/to/image.jpg")
        sys.exit(1)
    out = predict(sys.argv[1])
    print(json.dumps(out, indent=2))