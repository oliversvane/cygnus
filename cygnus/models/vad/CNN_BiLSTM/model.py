import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl

class CNNBiLSTM(pl.LightningModule):
    def __init__(self, output_layer_width):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ELU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(128 * 2 * 2, 128)  # Assumes input image size is (32, 32)
        self.bilstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(256, output_layer_width)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.conv(x)
        x = x.view(B * T, -1)
        x = F.elu(self.fc(x))
        x = x.view(B, T, -1)
        x = self.dropout(x)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = self.out(x)
        return torch.softmax(x, dim=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        acc = (y_hat.argmax(-1) == y.argmax(-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
