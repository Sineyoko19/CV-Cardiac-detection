import torch
import json
import albumentations as A
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from cardiac_dataset import CardiacDetectionDataset
from models_creator import CnnCardiacDetectorModel
from models_creator import ResNetCardiacDetectorModel
from pathnames import (
    PATH_TO_DATA_CSV,
    TRAIN_PATIENTS,
    TRAIN_ROOT_PATH,
    VAL_PATIENTS,
    VAL_ROOT_PATH,
)

if __name__ == "__main__":

    with open("src/config.json", "r") as f:
        configs = json.load(f)

    path_to_data_csv = PATH_TO_DATA_CSV
    train_patients = TRAIN_PATIENTS
    train_root_path = TRAIN_ROOT_PATH
    val_patients = VAL_PATIENTS
    val_root_path = VAL_ROOT_PATH

    data_augmentation = A.Compose(
        [
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.Affine(scale=(0.8, 1.2), rotate=(-10, 10), translate_px=(-10, 10)),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


    train_cardiac_data = CardiacDetectionDataset(
        path_to_data_csv, train_patients, train_root_path, data_augmentation
    )
    val_cardiac_data = CardiacDetectionDataset(
        path_to_data_csv, val_patients, val_root_path, None
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_cardiac_data,
        batch_size = configs["batch_size"],
        num_workers = configs["num_workers"],
        shuffle = True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_cardiac_data,
        batch_size = configs["batch_size"],
        num_workers = configs["num_workers"],
        shuffle=False,
    )

    model = ResNetCardiacDetectorModel()
    checkpoint = ModelCheckpoint(monitor="Val Loss", save_top_k=10, mode="min")
    logger = TensorBoardLogger("lightning_logs", name = configs["logger_name"])

    trainer = pl.Trainer(
        logger = logger,
        log_every_n_steps = 1,
        callbacks = checkpoint,
        max_epochs = configs["n_epochs"],
    )
    trainer.fit(model, train_dataloader, val_dataloader)
