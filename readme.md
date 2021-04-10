# mlflow-example
MLFlow Example on Docker-compose

This repository is learning and building an ML-API using FastAPI.

Build a container for ML learning, a MLFlow container, a Minio container instead of S3, and a WebAPI container in Docker-Compose.

The first step is to train and record the model on MLFlow.
Then, we use MLFlow-Registory to set a specific machine learning model to Production level, and set up a WebAPI to load it automatically.

# Environment
- Window10 home (RTX2080Ti)
- WSL2

# Requirement
- Docker
- Docker-compose

# Usage
```
$docker-compose up -d
```

## Training
```
$docker-compose exec ml /bin/bash
$cd keras
$python train.py
```
access to http://localhost:5000

## Model Settings Using MLFlow-Registory
access to http://localhost:5000

Manage models on MLFlow-Registory and set a specific version of the model to Production.

## Set up ML API
```
$docker-compose exec api /bin/bash
$uvicorn fastapi_skeleton.main:app

```

access to http://localhost:8888/docs


# Reference
https://github.com/eightBEC/fastapi-ml-skeleton


# Author
T-Sumida

<!-- # 使うもの
- MLflow
- mysql
- minio (https://openstandia.jp/oss_info/minio/)(https://vivekkaushal.com/mlflow-remote-server-setup/)
  - https://dev.classmethod.jp/articles/s3-compatible-storage-minio/
  - S3のかわり
- dataset
  - https://www.kaggle.com/tongpython/cat-and-dog

環境構築
  学習用コンテナ（https://github.com/NVIDIA/nvidia-container-runtime）
↓
kerasで学習する
↓
MLFlowに登録する
↓
Staging、Productionにする
↓
APIに登録する。 -->