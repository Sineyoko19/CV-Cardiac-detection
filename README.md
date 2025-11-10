# CV-Cardiac-detection  
Use deep learning methodology to predict a bounding box around heart areas found in X-ray images.

## Project Overview  
This project is designed to detect and localize the heart region in chest X-ray images using a convolutional neural network. The goal is to generate bounding boxes around the heart area, 
which can assist in downstream cardiac/clinical image-analysis workflows.

## Features  
- Load and preprocess chest X-ray image dataset  
- Train a deep learning model to detect the heart region  
- Evaluate on validation set and compute performance metrics  
- Modular dataset and model code to support future extensions  

## Getting Started  

### Prerequisites  
- Python 3.9 (or compatible)  
- PyTorch  
- Additional Python packages listed in `requirements.txt`

### Installation  
```bash
git clone https://github.com/Sineyoko19/CV-Cardiac-detection.git  
cd CV-Cardiac-detection  
python -m venv venv  
source venv/bin/activate   # (or `venv\Scripts\activate` on Windows)  
pip install -r requirements.txt
```

## Model Evaluation
The evaluation function loads a specified model checkpoint (via checkpoint_file),
runs inference on the validation dataset (no gradient computation), concatenates predictions and labels, and then computes Mean Absolute Error (MAE) per output variable across all patients.
Ensure your configuration lists the correct checkpoint path(s).

