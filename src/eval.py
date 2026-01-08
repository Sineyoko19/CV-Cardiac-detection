import torch
import json
<<<<<<< HEAD
from .cardiac_dataset import CardiacDetectionDataset
from .pathnames import (
=======
from cardiac_dataset import CardiacDetectionDataset
from pathnames import (
>>>>>>> 810d20ac4cc7e683ef29d23edf1cf8a7656c3819
    PATH_TO_DATA_CSV,
    VAL_PATIENTS,
    VAL_ROOT_PATH,
)

def model_eval (model : torch.nn.Module , checkpoint_file:str) -> list :
    """
    Loads a trained model from a checkpoint and evaluates it on the validation set.

    Args:
        model (torch.nn.Module): Model to evaluate.
        checkpoint_file (str): Path to the saved model checkpoint.

    Returns:
        list: Mean Absolute Error (MAE) per output on the validation set.
    """

    path_to_data_csv = PATH_TO_DATA_CSV
    val_patients = VAL_PATIENTS
    val_root_path = VAL_ROOT_PATH

    val_cardiac_data = CardiacDetectionDataset(
        path_to_data_csv, val_patients, val_root_path, None
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_cardiac_data,
        batch_size = 96 ,
        num_workers = 0,
        shuffle=False,
    )

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        preds = []
        labels = []
        for data, label in val_dataloader:
            data = data.float()
            pred = model(data)
            preds.append(pred)
            labels.append(label)

    preds = torch.cat(preds, dim=0)  
    labels = torch.cat(labels, dim=0)


    # MAE per output across all patients
    mae_per_output = torch.abs(preds - labels).mean(dim=0)

    return mae_per_output
    
