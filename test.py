from cygnus.processing.mfcc import mfcc_transformation
from cygnus.models.vad.MarbleNet.model import MarbleNet
from cygnus.processing import VADDataModule

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, DeviceStatsMonitor


torch.set_float32_matmul_precision('high')
segment_length = 10

model = MarbleNet(B=2,R=2, C=64,num_classes=segment_length*1000, lr=0.01, dropout=0.2)
datamodule = VADDataModule(train_data_path="data/vad/audioset/train", val_data_path="data/vad/audioset/val",transformation=mfcc_transformation, segment_length=segment_length, batch_size=200)

trainer = Trainer(
    profiler="simple",
    max_epochs=30,
    callbacks=[
        EarlyStopping(monitor="epoch_val_loss", patience=5, mode="min"),
        ModelCheckpoint(monitor="epoch_val_loss", save_top_k=1, mode="min"),
        DeviceStatsMonitor()
    ],
    accelerator="auto"
)

trainer.fit(model, datamodule=datamodule)