# DiagnosAR

This project integrates **AI-based chest X-ray diagnosis** with a **Unity-powered AR visualization tool**.
It combines a trained **Convolutional Neural Network (CNN)** for disease detection with an **interactive Unity interface**, allowing medical professionals and students to view diagnostic results in real time.

---

## 🚀 Features

* **AI Diagnosis**: CNN model trained on chest X-rays to classify medical conditions (e.g., pneumonia, tuberculosis).
* **Lightweight Model**: Converted to TensorFlow Lite for faster inference on edge devices.
* **Backend Server**: Flask API for handling X-ray image uploads and returning predictions.
* **Unity Integration**: Real-time visualization of AI predictions with interactive navigation.
* **Cross-Platform Ready**: Can be extended for Android/iOS deployment.

---

## 🏗️ Tech Stack

* **Machine Learning**: Python, TensorFlow, Keras, TensorFlow Lite
* **Backend**: Flask (REST API)
* **Frontend**: Unity (C# scripts for image upload, result display, AR visualization)
* **Dataset**: Chest X-ray images from NIH ChestX-ray8

---

## 📂 Project Workflow

1. **Dataset Preparation**: Image preprocessing, augmentation, and label mapping.
2. **Model Training**: CNN architecture built in TensorFlow/Keras (`train_model.py`).
3. **Model Optimization**: Conversion to TensorFlow Lite (`convert_model.py`).
4. **Flask API Deployment**: Model hosted via Flask (`server.py`, `predict.py`).
5. **Unity Integration**:

   * `XrayImageLoader` → loads X-rays
   * `XrayLabelLoader` → retrieves ground truth labels
   * `ImageUploader` → sends images to Flask server, fetches AI predictions
6. **Visualization**: Unity app displays static + AI-predicted diagnosis interactively.

---

## 📸 Screenshots

| Flask Server                 | Unity Visualization        | AI Prediction                        |
| ---------------------------- | -------------------------- | ------------------------------------ |
| ![Server](assets/server.png) | ![Unity](assets/unity.png) | ![Prediction](assets/prediction.png) |

*(Replace with actual screenshots from your repo)*

---

## ⚙️ Setup & Installation

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Python Backend

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask server:

```bash
python server.py
```

### 3. Unity Frontend

* Open Unity Hub → Add project folder.
* Ensure **AR Foundation / ARCore / ARKit SDKs** are installed.
* Configure build settings for **Android/iOS** as required.
* Run the project inside Unity Editor or build to device.

---

## 📦 Requirements

See [requirements.txt](requirements.txt) for Python dependencies.

Main packages:

* `tensorflow`
* `flask`
* `numpy`
* `opencv-python`

Unity:

* Tested on **Unity 6000.1.6f1**
* Requires **AR Foundation package**

---

## 🔮 Future Enhancements

* Mobile deployment (Android/iOS)
* PACS/Hospital system integration
* Offline predictions using bundled TFLite models
* Heatmaps for explainable AI visualization
* Multi-disease detection expansion
* Secure user authentication & cloud hosting

---

## 💡 Use Cases

* Assist radiologists with **preliminary diagnosis**
* Provide AI support in **rural healthcare centers**
* Serve as a **training tool** for medical students
* Enable **emergency field hospitals** to get instant AI analysis
* Act as a **second opinion system** for busy professionals

---

## 📑 Submissions

* 📹 [Demo Video](#) *(add YouTube/Drive link)*
* 📄 [Project Report](RadAI_Innovators.pdf) *(supplementary file)*

---

## 👨‍💻 Author

**Pranjal Sharma**

---

## 🏷️ Hackathon Submission

Tag your final commit as:

```bash
git tag SamsungPRISMGenAIHackathon2025
git push origin SamsungPRISMGenAIHackathon2025
```

---

## 📚 References

* [ChestX-ray8 Dataset – NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC)
* [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
* [Flask Documentation](https://flask.palletsprojects.com/)
* [Unity Scripting API](https://docs.unity3d.com/)
