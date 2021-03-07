# Environment
- Window10 home
- WSL2
- Docker
- Docker-compose

# 使うもの
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
APIに登録する。