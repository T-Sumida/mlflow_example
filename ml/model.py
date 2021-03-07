# -*- coding:utf -*-
from typing import List, Tuple, Union, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torchvision import models
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)


class ResNet(pl.LightningModule):
    def __init__(self, res_num, num_class):
        super(ResNet, self).__init__()
        try:
            base_model = getattr(
                models, "resnet"+str(res_num)
            )(pretrained=True)
        except Exception as e:
            print(e)
            exit(1)

        self.preds, self.targets = [], []

        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_class)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x).view(batch_size, -1)
        x = self.classifier(x)
        return x
    
    def calc_loss(self, x, y):
        pred = torch.argmax(self(x), dim=1)
        self.preds.append(pred.detach().cpu().numpy().astype(np.float32))
        self.targets.append(y.detach().cpu().numpy().astype(np.float32))
        loss = F.cross_entropy(self(x), y)
        return loss
    
    def calc_metrics(self, prefix: str):
        pred = np.concatenate(self.preds, axis=0)
        target = np.concatenate(self.targets, axis=0)
        acc = accuracy_score(pred, target)
        prec = precision_score(pred, target)
        recall = recall_score(pred, target)
        f1 = f1_score(pred, target)

        self.log(prefix + "_acc", acc)
        self.log(prefix + "_prec", prec)
        self.log(prefix + "_recall", recall)
        self.log(prefix + "_f1", f1)

    def training_step(self, batch, batch_nb):
        x, y = batch
        return self.calc_loss(x, y)

    def validation_step(self, batch, batch_nb):
        x, y = batch
        return self.calc_loss(x, y)

    def test_step(self, batch, batch_nb):
        x, y = batch
        return self.calc_loss(x, y)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.calc_metrics("train")
        self.preds, self.targets = [], []

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.calc_metrics("valid")
        self.preds, self.targets = [], []

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.calc_metrics("test")
        self.preds, self.targets = [], []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
