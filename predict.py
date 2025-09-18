import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("xray_model.h5")

# Load and preprocess the image
img_path = sys.argv[1]
img = Image.open(img_path).convert("L")          # Convert to grayscale
img = img.resize((128, 128))                     # Resize to match training input
img_array = np.array(img) / 255.0                # Normalize to [0,1]
img_array = img_array.reshape((1, 128, 128, 1))   # Add batch and channel dimensions

# Predict
predictions = model.predict(img_array)
class_labels = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", 
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

predicted_label = class_labels[np.argmax(predictions)]
print("Predicted Class:", predicted_label)




