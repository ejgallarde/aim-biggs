import os

import mlflow.pyfunc
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel


def get_model():
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(mlflow_uri)
        model_name = "iris_classifier"
        model_version = "latest"
        model_uri = f"models:/{model_name}/{model_version}"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception:
        raise HTTPException(
            status_code=500, detail="Model not found or failed to load."
        )


app = FastAPI()


class InputData(BaseModel):
    data: list[list[float]]  # Ensures the input is a list of lists of floats


class PredictionResponse(BaseModel):
    predictions: list[int]  # Ensures the output is a list of integers


@app.post("/predict", response_model=PredictionResponse)
def predict(
    input_data: InputData, model=Depends(get_model)
):  # Inject model for testing
    x = np.array(input_data.data)
    predictions = model.predict(x).tolist()
    return PredictionResponse(predictions=predictions)
