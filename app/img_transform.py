import cv2
import sys
import os
import pydicom
import torch
import pydicom
import numpy as np
import pandas as pd
from io import BytesIO
from torchvision import transforms


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def transform_predict(file) -> torch.tensor :
    """
    Tranform the dicom file given through a form into 
    resized, normalized and standardized image for prediction

    Args:
    file: dicom file read from a form

    Return:
    img: a tensor
    """

    dcm_file = pydicom.dcmread(BytesIO(file.read()))
    dcm_pixel_array = (cv2.resize(dcm_file.pixel_array, (224, 224)) / 255).astype(
            np.float32
        )
    
    df = pd.read_csv("app/static/mean_std.csv")
    
    # Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean= df["mean"].values[0] , std=df["std"].values[0])
    ])

    img_tensor = transform(dcm_pixel_array).unsqueeze(0)

    return img_tensor, dcm_pixel_array

def predict_bbox( img_tensor: torch.tensor, checkpoint_file : str, model : torch.nn.modules ) -> list:
    """
    Predict the 4 values representing the coordinates of the heart position 
    according to the image tensor given.

    Args:
    img_tensor : tensor to use for the prediction
    checkpoint_file: model file
    model : model class

    Return:
    bbox : list
    """

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)

    bbox = output.squeeze().cpu().numpy().tolist()
    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]

    return bbox
