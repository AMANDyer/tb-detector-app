# tb-detector-app
 # AI Tuberculosis Detector

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)

## Project Overview

This project is an end-to-end AI system for detecting Tuberculosis (TB) from chest X-ray images using deep learning. It uses transfer learning with a pre-trained MobileNetV2 model fine-tuned on the TB Chest Radiography Database. The system includes:

- A Jupyter Notebook (`cnn.ipynb`) for data preparation, model training, and evaluation.
- A FastAPI backend (`main.py`) for serving predictions via an API.
- A Streamlit frontend (`streamlit_app.py`) for an interactive web interface to upload X-rays and view results.

The model achieves high accuracy (e.g., AUC ~0.9999 on validation) by classifying images as "Normal" or "Tuberculosis" with confidence scores.

### Key Features
- **Dataset**: TB_Chest_Radiography_Database (3,500 Normal + 700 TB images from NCBI and other sources).
- **Model**: CNN with MobileNetV2 backbone, fine-tuned for binary classification.
- **Deployment**: RESTful API with FastAPI; user-friendly UI with Streamlit.
- **Preprocessing**: Image resizing to 300x300, data augmentation, class weighting for imbalance.
- **Evaluation**: Metrics include Loss, AUC, Precision, Recall.

## Dataset
The dataset is from [Kaggle's TB Chest Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset), sourced from NCBI articles and other public repositories. Metadata is provided in `Normal.metadata.xlsx` and `Tuberculosis.metadata.xlsx`.

- **Normal Images**: 3,500 PNGs (512x512).
- **TB Images**: 700 PNGs (512x512).
- Split: 80% train, 20% validation (stratified).

## Installation

1. Clone the repository:
2. Create a virtual environment (recommended):
3. Install dependencies from `requirements.txt`:


4. Download the dataset and place it in `TB_Chest_Radiography_Database/` (or update paths in `cnn.ipynb`).

5. Train the model (optional; pre-trained model is `my_model.keras`):
- Run the Jupyter Notebook `cnn.ipynb`.

## Usage

### Training the Model
1. Open `cnn.ipynb` in JupyterLab or VS Code.
2. Run cells sequentially to load data, train, and evaluate the model.
3. The trained model is saved as `my_model.keras`.

### Running the FastAPI Backend
1. Start the server:
2. API is available at `http://127.0.0.1:8000`.
3. Test via Swagger UI at `http://127.0.0.1:8000/docs`.
4. Endpoint: POST `/predict` with an image file.

### Running the Streamlit Frontend
1. Start the app:
2. Open `http://localhost:8501` in your browser.
3. Upload a chest X-ray image and click "Analyze Now" to get predictions.

### Example Prediction Response (JSON)
```json
{
"result": "TUBERCULOSIS",
"confidence": 0.9876,
"message": "TUBERCULOSIS with 98.76% confidence",
"note": "This is an AI prediction. Always consult a doctor."
}


