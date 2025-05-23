import os
import math
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

import lightning as pl


class VADDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        metadata: dict,
        transformation,
        segment_length: float,
        remove_empty=True,
        interval_length=1, #prediction interval in ms
        hparams: dict = {},
    ):
        super().__init__()
        self._hparams = hparams
        self._data_path = data_path
        self._segment_length = segment_length
        self._interval_length = interval_length
        self._metadata = []
        for i in metadata:
            if not len(i["timestamps"]) > 1 and remove_empty:
                pass
            else:
                for j in range(math.floor(float(i["length"]) / segment_length)):
                    self._metadata.append(
                        {
                            "path": i["path"],
                            "timestamps": i["timestamps"],
                            "start": j * segment_length,
                            "end": (j + 1) * segment_length,
                            "total_length": i["length"],
                        }
                    )
        self.len_ = len(self._metadata)
        self.transformation = transformation

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        meta = self._metadata[index]
        filepath = os.path.join(self._data_path, meta["path"])
        info = torchaudio.info(filepath)
        sr = info.sample_rate
        length = info.num_frames / sr

        target = torch.zeros(int(length * 1000/self._interval_length), dtype=torch.float)
        for timestamp in meta["timestamps"]:
            target[int(timestamp["start"] * 1000/self._interval_length) : int(timestamp["end"] * 1000/self._interval_length)] = 1.0
        target = target[int(meta["start"] * 1000/self._interval_length) : int(meta["end"] * 1000/self._interval_length)]

        waveform, sr = torchaudio.load(
            filepath,
            frame_offset=int(meta["start"] * sr),
            num_frames=int(self._segment_length * sr),
        )
        if waveform.size(0) > 1:  # multi-channel -> mono
            waveform = waveform.mean(dim=0, keepdim=True)
        feature_input = self.transformation(waveform, sr, **self._hparams)
        return feature_input, target


class VADDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_path,
        val_data_path,
        transformation,
        segment_length=1,
        batch_size=100,
    ):
        super().__init__()
        self.train_data_path = train_data_path
        with open(os.path.join(train_data_path, "strong.json"), "r") as f:
            self.train_metadata = json.load(f)
        self.val_data_path = val_data_path
        with open(os.path.join(val_data_path, "strong.json"), "r") as f:
            self.val_metadata = json.load(f)

        self.transformation = transformation
        self.batch_size = batch_size
        self.segment_length = segment_length

    def setup(self, stage=None):
        self.train_dataset = VADDataset(
            data_path=self.train_data_path,
            metadata=self.train_metadata,
            transformation=self.transformation,
            segment_length=self.segment_length,
        )
        self.val_dataset = VADDataset(
            data_path=self.val_data_path,
            metadata=self.val_metadata,
            transformation=self.transformation,
            segment_length=self.segment_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=16)
