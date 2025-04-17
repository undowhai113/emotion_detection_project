
# Emotion Detection Project

This project builds a facial emotion recognition model using Transfer Learning based on the FER-2013 dataset.

## 🔗 GitHub Repository

[https://github.com/undowhai113/emotion_detection_project.git](https://github.com/undowhai113/emotion_detection_project.git)

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/undowhai113/emotion_detection_project.git
cd emotion_detection_project
```

2. **(Optional) Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate        # On Windows
```

3. **Install required dependencies:**

```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. **Run the Flask app:**

```bash
python app.py
```

2. **Open your browser and visit:**

```
http://127.0.0.1:5000
```

3. **Upload a face image and get emotion prediction.**

## 📁 Project Structure

## 📁 Project Structure

emotion_detection_project/
│
├── data/
│   ├── fer2013.csv                 # CSV file containing the FER-2013 dataset
│   └── processed/                  # Directory containing preprocessed data (optional)
│       ├── X_train.npy             # Training data (images)
│       ├── y_train.npy             # Training labels
│       ├── X_val.npy               # Validation data
│       ├── y_val.npy               # Validation labels
│       ├── X_test.npy              # Test data
│       └── y_test.npy              # Test labels
│
├── models/
│   └── transfer_learning_model.h5  # Trained model
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb # Data preprocessing notebook
│   ├── 02_model_training.ipynb     # Model training notebook
│   └── 03_model_evaluation.ipynb   # Model evaluation notebook
│
├── src/
│   ├── utils.py                    # Utility functions (data reading, preprocessing, etc.)
│   └── model.py                    # Model architecture definition
│
├── reports/
│   └── figures/                    # Directory containing charts and images
│       ├── confusion_matrix.png    # Confusion matrix
│       └── training_history.png    # Training history chart
│
├── app.py                          # Flask app for realtime prediction
|
|── templates/
|   └── index.html                  # Web interface
├── requirements.txt                # List of dependencies and versions
└── README.md                       # Project description and usage instructions



## 🧠 Notes

- Dataset used: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Model: Transfer Learning (e.g., VGG16, ResNet, etc.)
- Framework: Flask (Python web framework)
```
