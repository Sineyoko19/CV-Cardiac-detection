import torch
import torchvision
import cv2
from torch import nn
import pytorch_lightning as pl
from pathlib import Path
from src.utils import calculate_mean_std
from src.pathnames import TRAIN_ROOT_PATH


class CnnCardiacDetectorModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.convlayers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=128, kernel_size=(3, 3), stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1) #TO reduce the size and fasten the training 

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, data):
        data = self.convlayers(data)
        data = self.global_pool(data)
        data = self.fc_layer(data)
        return data

    def training_step(self, batch, batch_idx):
        x_ray_image, label = batch
        label = label.float()
        pred = self(x_ray_image)
        loss = self.loss_fn(pred, label)

        self.log("Train Loss", loss)

        if batch_idx % 25 == 0:
            self.log_images(x_ray_image.cpu(), pred.cpu(), label.cpu(), "Train")

        return loss

    def validation_step(self, batch, batch_idx):
        x_ray_image, label = batch
        label = label.float()
        pred = self(x_ray_image)
        loss = self.loss_fn(pred, label)

        self.log("Val Loss", loss)

        if batch_idx % 25 == 0:
            self.log_images(x_ray_image.cpu(), pred.cpu(), label.cpu(), "Val")

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def log_images(self, x_rays, predictions, labels, name):
        results = []
        mean_, std_ = calculate_mean_std(train_path= Path(TRAIN_ROOT_PATH))
        for i in range(4):
            coords_labels = labels[i]
            coords_preds = predictions[i]

            img = ((x_rays[i] * std_) + mean_).numpy()[0]

            x0, y0 = coords_labels[0].int().item(), coords_labels[1].int().item()
            x1, y1 = coords_labels[2].int().item(), coords_labels[3].int().item()
            img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)

            x0, y0 = coords_preds[0].int().item(), coords_preds[1].int().item()
            x1, y1 = coords_preds[2].int().item(), coords_preds[3].int().item()
            img = cv2.rectangle(img, (x0, y0), (x1, y1), (1, 1, 1), 2)

            results.append(torch.tensor(img).unsqueeze(0))
        grid = torchvision.utils.make_grid(results, 2)
        self.logger.experiment.add_image(name, grid, self.global_step)

class ResNetCardiacDetectorModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained = True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=4, bias=True)

        self.loss_fn = nn.MSELoss()


    def forward(self, data):
        data = self.model(data)
        return data

    def training_step(self, batch, batch_idx):
        x_ray_image, label = batch
        label = label.float()
        pred = self(x_ray_image)
        loss = self.loss_fn(pred, label)

        self.log("Train Loss", loss)

        if batch_idx % 25 == 0:
            self.log_images(x_ray_image.cpu(), pred.cpu(), label.cpu(), "Train")

        return loss

    def validation_step(self, batch, batch_idx):
        x_ray_image, label = batch
        label = label.float()
        pred = self(x_ray_image)
        loss = self.loss_fn(pred, label)

        self.log("Val Loss", loss)

        if batch_idx % 25 == 0:
            self.log_images(x_ray_image.cpu(), pred.cpu(), label.cpu(), "Val")

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def log_images(self, x_rays, predictions, labels, name):
        results = []
        mean_, std_ = calculate_mean_std(train_path= Path(TRAIN_ROOT_PATH))
        for i in range(4):
            coords_labels = labels[i]
            coords_preds = predictions[i]

            img = ((x_rays[i] * std_) + mean_).numpy()[0]

            x0, y0 = coords_labels[0].int().item(), coords_labels[1].int().item()
            x1, y1 = coords_labels[2].int().item(), coords_labels[3].int().item()
            img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)

            x0, y0 = coords_preds[0].int().item(), coords_preds[1].int().item()
            x1, y1 = coords_preds[2].int().item(), coords_preds[3].int().item()
            img = cv2.rectangle(img, (x0, y0), (x1, y1), (1, 1, 1), 2)

            results.append(torch.tensor(img).unsqueeze(0))
        grid = torchvision.utils.make_grid(results, 2)
        self.logger.experiment.add_image(name, grid, self.global_step)

