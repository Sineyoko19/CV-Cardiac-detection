---
title: Cardiac Detection
emoji: ❤️
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Cardiac Detection App

AI-powered cardiac detection and risk assessment application using deep learning.

## Features
- Upload cardiac X-ray images for analysis
- Real-time detection and classification
- Bounding box prediction around heart areas
- Risk assessment with detailed analysis

## Project Overview
This project uses deep learning methodology to predict bounding boxes around heart areas found in X-ray images. The model is trained on chest X-ray datasets to detect and localize the heart region, which can assist in downstream cardiac/clinical image-analysis workflows.

## Technology Stack
- **Backend**: Flask
- **ML Framework**: PyTorch
- **Model**: ResNet-based cardiac detection
- **Deployment**: Hugging Face Spaces (Docker)

## Local Development

### Prerequisites
- Python 3.9+
- PyTorch
- Flask

### Installation
```bash
git clone https://github.com/Sineyoko19/CV-Cardiac-detection.git
cd CV-Cardiac-detection
python -m venv venv
source venv/bin/activate   # (or `venv\Scripts\activate` on Windows)
pip install -r requirements.txt
```

### Running Locally
```bash
python app/cardiac_detection_app.py
```

Visit `http://localhost:7860` in your browser.

## Model
The model is hosted on Hugging Face Hub and automatically downloaded on first run.