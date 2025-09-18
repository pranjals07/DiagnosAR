"""
train_model.py — Custom training using labels.xlsx and flat image folder
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------- Config -----------------------
IMG_DIR = "images"
LABEL_FILE = "labels.csv"
IMG_SIZE = (128, 128)
EPOCHS = 25
BATCH_SIZE = 32
MODEL_PATH = "xray_model.h5"
LABEL_MAP_PATH = "label_map.txt"

# ---------------------- Load Labels -----------------------
df = pd.read_csv(LABEL_FILE)
df = df[["Image Index", "Finding Labels"]]
df.columns = ["filename", "label"]  # Rename for consistency

print("✅ Labels loaded:", df.shape)

# ---------------------- Load Images -----------------------
X = []
y = []

for idx, row in df.iterrows():
    fname = row["filename"]  # column must be "filename"
    label = row["label"]     # column must be "label"

    path = os.path.join(IMG_DIR, fname)
    try:
        img = Image.open(path).convert("L").resize(IMG_SIZE)
        X.append(np.array(img))
        y.append(label)
    except Exception as e:
        print(f"⚠️ Error loading {fname}: {e}")

X = np.array(X).astype("float32") / 255.0
X = np.expand_dims(X, axis=-1)
print("✅ Loaded images:", X.shape)

# ---------------------- Encode Labels -----------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
print("✅ Encoded classes:", le.classes_)

# ---------------------- Train/Test Split -----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.15, random_state=42, stratify=y_encoded
)

# ---------------------- Compute Class Weights -----------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights = dict(enumerate(class_weights))
print("⚖️ Class weights:", class_weights)

# ---------------------- Build Model -----------------------
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model((*IMG_SIZE, 1), y_categorical.shape[1])
model.summary()

# ---------------------- Train Model -----------------------
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks
)

# ---------------------- Save Model & Labels -----------------------
model.save(MODEL_PATH)
print(f"✅ Saved model to {MODEL_PATH}")

with open(LABEL_MAP_PATH, "w") as f:
    for idx, label in enumerate(le.classes_):
        f.write(f"{idx},{label}\n")
print(f"✅ Saved label map to {LABEL_MAP_PATH}")


