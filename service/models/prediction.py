# -*- coding:utf-8 -*-

from pydantic import BaseModel

RESULT_DICT = {0: "cat", 1: "dog"}

class DogCatPredictionResult(BaseModel):
    name: str
