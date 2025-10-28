import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from pathlib import Path
from utils import transform_data
from models.cardiac_dataset import CardiacDetectionDataset
import matplotlib.pyplot as plt

#data_hd = pd.read_csv("data/raw/rsna_heart_detection.csv")
    
#ROOT_PATH = Path("data/raw/stage_2_train_images/") 
#SAVE_PATH = Path("data/processed/Processed_heart_detection/")

# apply data preprocessing

#transform_data(data_hd, ROOT_PATH, SAVE_PATH)

# create dataset so that the bounding box around the heart is scaled according to the data augmentation applied to each xray image during the transformation needed for the training
path_to_data_csv = "data/raw/rsna_heart_detection.csv"
patients = "data/processed/Processed_heart_detection/train_subjects.npy"
train_root_path = "data/processed/Processed_heart_detection/train"
data_augmentation = A.Compose([
    A.RandomGamma(gamma_limit=(80, 120), p=1.0) ,
    A.Affine(
        scale = (0.8, 1.2),
        rotate = (-10, 10),
        translate_px = (-10,10))
],
bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


train_cardiac_data = CardiacDetectionDataset(
    path_to_data_csv,
    patients,
    train_root_path,
    data_augmentation
)

print(len(train_cardiac_data))
img, bbox = train_cardiac_data[0]
print(len(img))

fig, axis = plt.subplots(1,1,figsize = (8,8))
axis.imshow(img[0], cmap ="bone")

patch_bbox = patch.Rectangle((bbox[0],bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth = 1, edgecolor="r", facecolor="none")
axis.add_patch(patch_bbox)
plt.show()