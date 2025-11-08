import torch
import json
from models.cardiac_dataset import CardiacDetectionDataset
from models.models_creator import CnnCardiacDetectorModel
from pathnames import (
    PATH_TO_DATA_CSV,
    VAL_PATIENTS,
    VAL_ROOT_PATH,
)

if __name__ == "__main__":

    with open("config.json", "r") as f:
        configs = json.load(f)

    path_to_data_csv = PATH_TO_DATA_CSV
    val_patients = VAL_PATIENTS
    val_root_path = VAL_ROOT_PATH

    val_cardiac_data = CardiacDetectionDataset(
        path_to_data_csv, val_patients, val_root_path, None
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_cardiac_data,
        batch_size = configs["batch_size"],
        num_workers=configs["num_workers"],
        shuffle=False,
    )

    checkpoint = torch.load(configs["eval"]["checkpoint_path"])
   
    model = CnnCardiacDetectorModel()  
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        preds = []
        labels = []
        for data, label in val_dataloader:
            data = data.float()
            pred = model(data)
            print(pred.shape)
            preds.append(pred)
            labels.append(label)

    preds = torch.cat(preds, dim=0)   
    labels = torch.cat(labels, dim=0) 

    # MAE per output across all patients
    mae_per_output = torch.abs(preds - labels).mean(dim=0)
    print(mae_per_output)