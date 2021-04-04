# -*- coding:utf-8 -*-
from typing import cast
from starlette.config import Config
from starlette.datastructures import Secret

APP_VERSION = "0.0.1"
APP_NAME = "Dog Cat Classifier Example"
API_PREFIX = "/api"

config = Config(".env")

API_KEY: Secret = config("API_KEY", cast=Secret)
IS_DEBUG: bool = config("IS_DEBUG", cast=bool, default=False)

TRACKING_URI: str = config("TRACKING_URI", cast=str)
MODEL_NAME: str = config("MODEL_NAME", cast=str)
IMG_SIZE: int = 224


# DEFAULT_MODEL_PATH: str = config("DEFAULT_MODEL_PATH")