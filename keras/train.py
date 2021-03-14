# -*- coding: utf-8 -*-
import os
import random
import datetime
import warnings
from typing import List, Tuple, Dict

import numpy as np
from numpy.lib.function_base import average
import tensorflow as tf
import keras
import yaml
import hydra
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature, infer_signature
from omegaconf import DictConfig, OmegaConf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

import models


def set_seed(seed: int = 1234):
    """init seed

    Args:
        seed (int, optional): seed value. Defaults to 1234.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def init_logger():
    # return
    mlflow.set_tracking_uri("http://localhost:5000")

    # 環境変数としてオブジェクトストレージへの接続情報を指定
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio-access-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio-secret-key"


def get_generator(gen_cfg: DictConfig, target_size: int) -> Tuple[keras.preprocessing.image.DirectoryIterator, keras.preprocessing.image.DirectoryIterator, keras.preprocessing.image.DirectoryIterator]:
    """create generator

    Args:
        gen_cfg (DictConfig): config
        target_size (int): image size

    Returns:
        Tuple[keras.preprocessing.image.DirectoryIterator, keras.preprocessing.image.DirectoryIterator, keras.preprocessing.image.DirectoryIterator]: train, valid, test generator
    """
    train_datagen = ImageDataGenerator(**gen_cfg['train_gen'])
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        gen_cfg['train_path'],
        target_size=(target_size, target_size),
        batch_size=gen_cfg['batch_size'],
        class_mode='categorical',
        subset='training'
    )
    valid_generator = train_datagen.flow_from_directory(
        gen_cfg['train_path'],
        target_size=(target_size, target_size),
        batch_size=gen_cfg['batch_size'],
        class_mode='categorical',
        subset='validation'
    )
    test_generator = test_datagen.flow_from_directory(
        gen_cfg['test_path'],
        target_size=(target_size, target_size),
        batch_size=1,
        class_mode='categorical'
    )
    return train_generator, valid_generator, test_generator


def get_callbacks(epochs: int, tmp_dir: str) -> List:
    """create train callbacks

    Args:
        epochs (int): number of epoch
        tmp_dir (str): tmp file directory path

    Returns:
        List: callback list
    """
    def scheduler(epoch):
        lr = 1e-3
        if epoch >= epochs // 2:
            lr = 1e-4
        if epoch >= epochs // 4 * 3:
            lr = 1e-5
        return lr
    return [
        EarlyStopping(
            monitor='val_loss', patience=epochs//4*3
        ),
        ModelCheckpoint(
            os.path.join(tmp_dir, "model.h5"),
            monitor='val_loss', save_best_only=True,
            save_weights_only=True, mode='auto',
            verbose=1
        ),
        CSVLogger(
            os.path.join(tmp_dir, "history.log")
        ),
        LearningRateScheduler(scheduler)
    ]


def get_signature(generator: keras.preprocessing.image.DirectoryIterator, model: keras.Model) -> ModelSignature:
    """create mlflow signature

    Args:
        generator (keras.preprocessing.image.DirectoryIterator): dataset generator
        model (keras.Model): prediction model

    Returns:
        ModelSignature: Signature
    """
    x_data, _ = generator.__getitem__(0)
    return infer_signature(x_data, model.predict(x_data))


def evaluate(generator: keras.preprocessing.image.DirectoryIterator, model: keras.Model) -> Tuple[float, float, float, float]:
    """evaluate model created

    Args:
        generator (keras.preprocessing.image.DirectoryIterator): dataset generator
        model (keras.Model): prediction model

    Returns:
        Tuple[float, float, float, float]: accuracy, precision, recall, f1
    """
    targets, preds = [], []
    
    for i in range(generator.__len__()):
        x_data, y_data = generator.__getitem__(i)
        for x, y in zip(x_data, y_data):
            pred = model.predict(np.expand_dims(x, axis=0), batch_size=1)
            targets.append(np.argmax(y))
            preds.append(np.argmax(pred))
    return accuracy_score(targets, preds), precision_score(targets, preds, average='macro'), recall_score(targets, preds, average='macro'), f1_score(targets, preds, average='macro')


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
    """Main

    Args:
        cfg (DictConfig): config info
    """
    set_seed()
    init_logger()

    cfg = yaml.load(OmegaConf.to_yaml(cfg))
    target_size = cfg['main']['target_size']
    class_num = 2
    model_name = "ResNet50"
    train_generator, val_generator, test_generator = get_generator(cfg['generator'], cfg['main']['target_size'])
    try:
        model = getattr(models, "get_"+model_name)(target_size, class_num)
    except Exception as e:
        print(e)
        exit(1)
    
    if not os.path.exists(cfg['main']['tmp_path']):
        os.makedirs(cfg['main']['tmp_path'])

    callbacks = get_callbacks(cfg['main']['epoch'], cfg['main']['tmp_path'])

    opt = keras.optimizers.Adam(**cfg['optimizer'])
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=opt
    )

    mlflow.keras.autolog(log_models=False)
    with mlflow.start_run() as run:
        model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.__len__(),
            epochs=cfg['main']['epoch'],
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=val_generator.__len__()
        )

        model.load_weights(os.path.join(cfg['main']['tmp_path'], 'model.h5'))

        signature = get_signature(test_generator, model)
        mlflow.keras.log_model(model, "models", signature=signature)
        _, val_generator, test_generator = get_generator(cfg['generator'], cfg['main']['target_size'])
        val_acc, val_prec, val_recall, val_f1 = evaluate(val_generator, model)
        test_acc, test_prec, test_recall, test_f1 = evaluate(test_generator, model)
        mlflow.log_metrics({'val_acc': val_acc, 'val_prec': val_prec, 'val_recall': val_recall, 'val_f1': val_f1})
        mlflow.log_metrics({'test_acc': test_acc, 'test_prec': test_prec, 'test_recall': test_recall, 'test_f1': test_f1})

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
  

if __name__ == "__main__":
    main()
