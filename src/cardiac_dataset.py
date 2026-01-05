import torch
import random
import pandas as pd
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import calculate_mean_std


class CardiacDetectionDataset(torch.utils.data.Dataset):
    """
    This class is created to make sure that for each data augmentation applied to our chest xray images,
    the box around the heart is scaled to match.
    
    """
    def __init__(self, 
                 path_to_data_csv: str,
                 patients: str,
                 root_path: str,
                 augs: A.Compose = None):
        
        self.data_image = pd.read_csv(path_to_data_csv)
        self.patients = np.load(patients)
        self.root_path = Path(root_path)
        self.augment = augs

    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        patient_data = self.data_image[self.data_image["name"] == patient_id].iloc[0]

        bbox = [
            float(patient_data["x0"]),
            float(patient_data["y0"]),
            float(patient_data["x0"] + patient_data["w"]),
            float(patient_data["y0"] + patient_data["h"])
        ]

        file_path = self.root_path / f"{patient_id}.npy"
        img = np.load(file_path).astype(np.float32)

        # Albumentations expects HWC format; if image is grayscale (2D), expand it
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        if self.augment:
            # Set seed for reproducibility (important for DataLoader workers)
            seed = torch.randint(0,100000,(1,)).item()
            random.seed(seed)
            transformed = self.augment(
                image=img,
                bboxes=[bbox],
                labels=["heart"]
            )
            img = transformed["image"]
            bbox = transformed["bboxes"][0]

        # Standardization
        if "train" not in str(self.root_path):
            train_root_path =Path(str(self.root_path).replace("val","train"))
        else:
            train_root_path =self.root_path

        mean, std = calculate_mean_std(train_root_path)
        img = (img - mean) / std

        # Convert to tensor (C, H, W)
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        bbox = torch.tensor(bbox, dtype=torch.float32)

        return img, bbox
