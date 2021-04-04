# -*- coding:utf-8 -*-
import io
from typing import List
from fastapi import APIRouter, Depends, File, UploadFile
from starlette.requests import Request

from service.models.prediction import DogCatPredictionResult
from service.core import security
from service.services.models import DogCatModel

router = APIRouter()


@router.post("/predict", response_model=DogCatPredictionResult, name="predict")
def post_predict(
    request: Request,
    files: List[UploadFile] = File(...),
    authenticated: bool = Depends(security.validate_request),
) -> DogCatPredictionResult:
    model: DogCatModel = request.app.state.model
    prediction: DogCatPredictionResult = model.predict(io.BytesIO(files[0].file.read()))
    return prediction
