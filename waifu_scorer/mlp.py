import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating', batch_norm=True):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.4),

            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer


class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, batch_norm=True, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(output_size) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)
        self.adjust_dims = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x):
        identity = self.adjust_dims(x)
        out = self.linear(x)
        out = self.relu(out)
        out = self.batch_norm(out)
        out = self.dropout(out)
        out += identity
        out = self.relu(out)
        return out


class ResMLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating', batch_norm=True):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            ResidualBlock(input_size, 2048, batch_norm, dropout_rate=0.3),
            ResidualBlock(2048, 512, batch_norm, dropout_rate=0.3),
            ResidualBlock(512, 256, batch_norm, dropout_rate=0.2),
            ResidualBlock(256, 128, batch_norm, dropout_rate=0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss
