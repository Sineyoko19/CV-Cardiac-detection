import pandas as pd
from pathlib import Path
from utils import transform_data

data_hd = pd.read_csv("data/raw/rsna_heart_detection.csv")
    
ROOT_PATH = Path("data/raw/stage_2_train_images/") 
SAVE_PATH = Path("data/processed/Processed_heart_detection/") 

transform_data(data_hd, ROOT_PATH, SAVE_PATH)