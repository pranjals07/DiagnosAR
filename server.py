import os
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# -------------------- Configuration --------------------
MODEL_PATH = "xray_model.h5"
LABEL_MAP_TXT = "label_map.txt"
INPUT_SIZE = (128, 128)
COLOR_MODE = "L"  # grayscale
# -------------------------------------------------------

# Init Flask
app = Flask(__name__)

# Load model
print("🔄 Loading model …")
model = load_model(MODEL_PATH)
print("✅ Model loaded.")

# Load label names
if os.path.exists(LABEL_MAP_TXT):
    with open(LABEL_MAP_TXT, "r") as f:
        lines = f.readlines()
        CLASS_LABELS = [line.strip().split(",")[1] for line in sorted(lines, key=lambda x: int(x.split(",")[0]))]
else:
    CLASS_LABELS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ]
print(f"✅ Labels: {CLASS_LABELS}")

# Preprocess image
def preprocess(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert(COLOR_MODE).resize(INPUT_SIZE)
        arr = np.asarray(img, dtype="float32") / 255.0
        arr = arr.reshape((1, INPUT_SIZE[1], INPUT_SIZE[0], 1))
    return arr

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save temp file properly
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            temp_path = tmp.name
            file.save(temp_path)

        # Preprocess and predict
        img_array = preprocess(temp_path)
        preds = model.predict(img_array)[0]
        predicted_index = int(np.argmax(preds))
        predicted_class = CLASS_LABELS[predicted_index]

        # Return result
        response = {
            "class": predicted_class,
            "probabilities": [float(f"{p:.4f}") for p in preds],
            "labels": CLASS_LABELS
        }
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Ensure file is deleted
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except PermissionError:
                print(f"⚠️ File still in use, could not delete: {temp_path}")

# Start Flask app
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

