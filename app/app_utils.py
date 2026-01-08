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
from huggingface_hub import hf_hub_download

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def transform_predict(file) -> torch.tensor :
    """
    Tranform the dicom file given through a form into 
    resized, normalized and standardized image for prediction

    Args:
    file: dicom file read from a form

    Return:
    img: a torch.Tensor
    """
    
    df = pd.read_csv("app/static/mean_std.csv")

    dcm_file = pydicom.dcmread(BytesIO(file.read()))
    dcm_pixel_array = (cv2.resize(dcm_file.pixel_array, (224, 224)) / 255).astype(
            np.float32
        )
    
    # Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean= df["mean"].values[0] , std=df["std"].values[0])
    ])
    
    img_tensor = transform(dcm_pixel_array)
    print(img_tensor.shape)
    if img_tensor.shape[0] !=1:
        transform = transforms.Compose([transforms.Grayscale()])
        img_tensor = transform(img_tensor)

    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, dcm_pixel_array

def hf_model_access(resnet_model:torch.nn.modules):
    """
    Allows access to the production model deployed on hugging face
    
    Args:
    resnet_model: Developped mdel
    type resnet_model: torch.nn.modules
    """

    # Download the model (it caches locally after first download)
    model_path = hf_hub_download(
        repo_id="AssitanSI/resnet_cardiac_detection",
        filename="epoch=148-step=5959.ckpt"
    )

    # Load your model
    model = resnet_model.load_from_checkpoint(model_path)

    return model

def predict_bbox( img_tensor: torch.Tensor, resnet_model : torch.nn.modules ) -> list:
    """
    Predict the 4 values representing the coordinates of the heart position 
    according to the image tensor given.

    Args:
    img_tensor : tensor to use for the prediction
    checkpoint_file: model file
    resnet_model : resnet model class

    Return:
    bbox : list
    """

    model = hf_model_access(resnet_model)
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)

    bbox = output.squeeze().cpu().numpy().tolist()
    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]

    return bbox

