# -*- coding:utf-8 -*-
import os
import random
import datetime
import warnings
from typing import List, Tuple, Dict

import torch
import hydra
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pytorch_lightning as pl
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf

from dataset import DogCatDataModule
from model import ResNet


def get_transforms(img_size):
    return {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        'test': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    }


def init_logger():
    # return
    mlflow.set_tracking_uri("http://localhost:5000")

    # 環境変数としてオブジェクトストレージへの接続情報を指定
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio-access-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio-secret-key"


def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


@hydra.main(config_path="./config.yml")
def main(cfg: DictConfig):
    pl.seed_everything(0)
    init_logger()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform_dict = get_transforms(224)
    dogcat_model = ResNet(50, 2)
    dogcat_model = dogcat_model.to(device)
    data_module = DogCatDataModule(
        # "D:/WorkSpace/mlflow-registory-example/dataset/"
        "D:/WorkSpace/mlflow-registory-example/dataset/training_set/training_set",
        "D:/WorkSpace/mlflow-registory-example/dataset/test_set/test_set",
        64,
        transform_dict['train'],
        transform_dict['val'],
        transform_dict['test'],
        224
    )

    trainer = pl.Trainer(
        max_epochs=10, progress_bar_refresh_rate=10,
        gpus=1
    )

    mlflow.pytorch.autolog(log_models=False)
    with mlflow.start_run() as run:
        trainer.fit(dogcat_model, data_module)
        trainer.test()

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    main()
