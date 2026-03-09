# ✈️ AeroInspect AI

**AeroInspect AI** is a deep learning web application built with Streamlit that automates the classification of aircraft surface damage (Dent vs. Crack) and generates descriptive natural language captions for the damage. 

This project was built as the **Deep Learning Final Project** for the AI Engineer Course, completing all 10 required tasks using a pre-trained **VGG16** model for classification and a **BLIP** Transformer model for image captioning.

![AeroInspect AI UI Preview](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

---

## ✨ Features

- **Interactive Web App**: A premium dark-mode Streamlit dashboard with 5 distinct pages.
- **Automated Dataset Handling**: 1-click download and extraction of the [Roboflow Aircraft Dataset (~300MB)](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk).
- **VGG16 Classification**: Binary classification (Crack vs. Dent) using feature extraction from an ImageNet pre-trained VGG16 model with a custom classifier head.
- **Live Training Metrics**: Real-time accuracy and loss charts during model training.
- **Model Evaluation**: Visual prediction grids and test-set evaluation metrics.
- **BLIP Image Captioning**: Pure PyTorch/Transformer implementation of `Salesforce/blip-image-captioning-base` to generate AI-driven summaries and captions of the aircraft damage.

---

## 🚀 Installation & Setup

### Prerequisites
Make sure you have Python installed. The project runs locally and downloads all necessary sub-packages automatically.

### 1. Clone the Repository
```cmd
git clone https://github.com/aiprojects2060/AeroInspect-AI.git
cd AeroInspect-AI
```

### 2. Install Dependencies
Install the required packages using `pip`:
```cmd
pip install -r requirements.txt
```

*(Note: The app relies on `transformers`, `torch`, `tensorflow-cpu`, `keras`, `streamlit`, `matplotlib`, and `pillow`)*

---

## 💻 Running the Application

### Option A: Using the Launcher (Windows)
We've included a `run.bat` file for convenience. Simply double-click **`run.bat`** from the project folder.

### Option B: Using the Terminal
To run the app manually from your terminal/command prompt:
```cmd
python -m streamlit run app.py --server.port 8501
```

Once running, open your web browser and navigate to:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🛣️ How to Use the App

The sidebar on the left lets you navigate through the complete machine learning lifecycle:

1. **📦 Dataset Setup**: Click "Download Dataset" to automatically fetch and extract the images into `train/`, `valid/`, and `test/` folders.
2. **🧠 Train Classifier**: Configure your epochs and click "Start Training" to build data generators, compile the VGG16 model, and watch the live training progress.
3. **📊 Training Results**: Review the final loss and accuracy curves (Task 6).
4. **🔍 Evaluate & Predict**: Test the model against the test dataset or upload your own image to get a live prediction (Task 7).
5. **💬 Image Captioning**: Select an image from the dataset or upload your own to generate a detailed AI caption and summary using the BLIP Transformer model (Tasks 8-10).

---

## 🏗️ Project Structure

```text
AeroInspect-AI/
├── app.py                 # Main Streamlit application with UI/UX
├── train_model.py         # VGG16 model definition, data generators, and training logic
├── caption_model.py       # BLIP Transformer implementation (pure PyTorch)
├── download_dataset.py    # Utility script for fetching the Roboflow dataset
├── requirements.txt       # Project dependencies
├── run.bat                # Windows quick-launch script
└── .gitignore             # Git ignore rules (excludes dataset to save space)
```

---

## 📝 License
- **Code**: MIT License
- **Dataset**: Provided by a Roboflow user, License: CC BY 4.
