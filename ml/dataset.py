# -*- coding:utf-8 -*-
from typing import List, Tuple, Union, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets


class DogCatDataModule(pl.LightningDataModule):
    # def __init__(self, train_transforms, val_transforms, test_transforms, dims):
    #     super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
    def __init__(self, train_data_dir: str, test_data_dir: str, batch_size: int, train_transforms: transforms.Compose, val_transforms: transforms.Compose, test_transforms: transforms.Compose, dims: Tuple):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.dims = dims
        self.batch_size = batch_size
        self.num_class = 2

    def prepare_data(self, *args, **kwargs):
        return super().prepare_data(*args, **kwargs)
    # def prepare_data(self, *args, **kwargs):
    #     pass

    def setup(self, stage: Optional[str]):
        if stage == "fit" or stage is None:
            data = datasets.ImageFolder(root=self.train_data_dir)
            train_size = int(0.8 * len(data))
            val_size = len(data) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
            self.train_dataset.dataset.transform = self.train_transforms
            self.val_dataset.dataset.transform = self.val_transforms

        if stage == "test":
            self.test_dataset = datasets.ImageFolder(root=self.test_data_dir, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)