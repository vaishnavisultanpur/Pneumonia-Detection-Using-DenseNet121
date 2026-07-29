# Pneumonia Detection using Deep Learning

## Overview
This project uses a Deep Learning model based on DenseNet121 to classify chest X-ray images as either:
- NORMAL
- PNEUMONIA

The model is built using Transfer Learning with TensorFlow and Keras.

## Dataset
Chest X-ray Images (Pneumonia)

Dataset Structure:
- Train
- Validation
- Test

## Model
- DenseNet121 (Transfer Learning)
- Binary Classification
- Sigmoid Activation
- Binary Crossentropy Loss
- Adam Optimizer

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

## Features
- Chest X-ray image classification
- Transfer Learning using DenseNet121
- Predicts NORMAL or PNEUMONIA
- Model evaluation using Accuracy, Classification Report, and Confusion Matrix
- Streamlit web application for prediction

## Project Files

- Pneumonia_Detection.ipynb – Model training notebook
- app.py – Streamlit application
- requirements.txt – Required Python packages
- README.md – Project documentation

## Trained Model

The trained model file (`pneumonia_densenet121.keras`) is not included in this repository because of GitHub upload limitations. You can train the model using the notebook provided or place the trained model in the project folder before running the application.

## How to Run

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:

```bash
streamlit run app.py
```

## Author

**Vaishnavi Ramesh Sultanpur**
