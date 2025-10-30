import torch
import torchvision
import cv2
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class CnnCardiacDetectorModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.convlayers = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=64,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=1,
                      padding=1,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128,4)
        )

        self.loss_fn = nn.MSELoss()


    def forward(self, data):
        data = self.convlayers(data)
        data = self.fc_layer(data)
        return data
    
    def training_step(self, batch, batch_idx):
        x_ray_image, label= batch
        label = label.float()
        pred = self(x_ray_image)
        loss = self.loss_fn(pred,label)

        self.log("Train Loss", loss)

        if batch_idx % 25 == 0:
            self.log_images(x_ray_image.cpu(), pred.cpu(), label.cpu(),'Train')
        
        return loss
    
    def val_step(self, batch, batch_idx):
        x_ray_image, label= batch
        label = label.float()
        pred = self(x_ray_image)
        loss = self.loss_fn(pred,label)

        self.log("Val Loss", loss)

        if batch_idx % 25 == 0:
            self.log_images(x_ray_image.cpu(), pred.cpu(), label.cpu(),'Val')
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
        
    def log_images(self, x_rays, predictions, labels, name):
        results = []
        for i in range(4):
            coords_labels = labels[i]
            coords_preds = predictions[i]

            img = ((x_rays[i] * 0.252) + 0.494).numpy()[0]

            x0,y0 = coords_labels[0].int().itemp(),coords_labels[1].int().itemp()
            x1,y1 = coords_labels[2].int().itemp(),coords_labels[3].int().itemp()
            img = cv2.rectangle(img, (x0,y0), (x1,y1),(0,0,0),2)

            x0,y0 = coords_preds[0].int().itemp(),coords_labels[1].int().itemp()
            x1,y1 = coords_preds[2].int().itemp(),coords_labels[3].int().itemp()
            img = cv2.rectangle(img, (x0,y0), (x1,y1),(1,1,1),2)

            results.append(torch.tensor(img).unsqueeze(0))
        grid = torchvision.utils.make_grid(results,2)
        self.logger.experiment.add_image(name,grid,self.global_step)

            
