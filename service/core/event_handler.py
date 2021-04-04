# -*- coding:utf-8 -*-

from typing import Callable

from fastapi import FastAPI
from loguru import logger

from service.core.config import MODEL_NAME, TRACKING_URI, IMG_SIZE
from service.services.models import DogCatModel

def _startup_model(app: FastAPI) -> None:
    model_instalce = DogCatModel(TRACKING_URI, MODEL_NAME, IMG_SIZE)
    app.state.model = model_instalce


def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        logger.info("Running app start handler.")
        _startup_model(app)
    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        _shutdown_model(app)
    return shutdown
