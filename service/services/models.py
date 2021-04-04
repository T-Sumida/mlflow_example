# -*- coding:utf-8 -*-
import os
from typing import BinaryIO

import cv2
import mlflow
import mlflow.keras
import numpy as np
import keras.backend.tensorflow_backend as tb
from fastapi import File, UploadFile

from service.models.prediction import DogCatPredictionResult, RESULT_DICT

class DogCatModel(object):
    def __init__(self, tracking_url: str, model_name: str, size: int) -> None:
        """init

        Args:
            tracking_url (str): mlflow uri
            model_name (str): model name of mlflow
            size (int): target image size
        """
        self.tracking_url = tracking_url
        self.model_name = model_name
        self.size = size
        self._load_model()

    def _load_model(self) -> None:
        """load model"""
        # mlflow.set_tracking_uri(self.tracking_url)
        # os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
        mlflow.set_tracking_uri("http://mlflow:5000")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
        os.environ["AWS_ACCESS_KEY_ID"] = "minio-access-key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minio-secret-key"
        stage = 'Production'

        self.model = mlflow.keras.load_model(
            model_uri=f"models:/{self.model_name}/{stage}"
        )

    def _pre_process(self, bin_data: BinaryIO) -> np.array:
        """preprocess

        Args:
            bin_data (BinaryIO): binary image data

        Returns:
            np.array: image data
        """
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        return img


    def _post_process(self, prediction: np.array) -> DogCatPredictionResult:
        """post process

        Args:
            prediction (np.array): result of predict

        Returns:
            DogCatPredictionResult: prediction
        """
        result_idx = np.argmax(prediction)
        dcp = DogCatPredictionResult(name=RESULT_DICT[result_idx])
        return dcp

    def _predict(self, img: np.array) -> np.array:
        """predict

        Args:
            img (np.array): image data

        Returns:
            np.array: prediction
        """
        tb._SYMBOLIC_SCOPE.value = True #TODO https://github.com/keras-team/keras/issues/13353#issuecomment-545459472
        prediction_result = self.model.predict(np.expand_dims(img, axis=0))
        return prediction_result

    def predict(self, bin_data: BinaryIO) -> DogCatPredictionResult:
        """predict method

        Args:
            bin_data (BinaryIO): binary image data

        Returns:
            DogCatPredictionResult: prediction
        """
        img = self._pre_process(bin_data)
        result = self._predict(img)
        post_processed_result = self._post_process(result)
        return post_processed_result
        