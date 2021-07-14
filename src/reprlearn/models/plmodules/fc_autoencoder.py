import torch
from torch import nn
from torch.functional import F
import pytorch_lightning as pl
from typing import Tuple

class FCAutoEncoder(pl.LightningModule):

    def __init__(self, *,
                 img_shape: Tuple[int,int,int],
                 nh1: int,
                 nh2: int,
                 dim_z: int
                 ):
        super().__init__()
        # Save init arguments as hyperparmeters to checkpt files and to the logger
        self.save_hyperparameters()

        self.dims = img_shape #CHW
        dim_in = torch.prod(torch.tensor(self.dims))

        self.encoder = nn.Sequential(
            nn.Linear(dim_in, nh1),
            nn.ReLU(),
            nn.Linear(nh1, nh2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
