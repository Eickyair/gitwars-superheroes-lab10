from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import joblib
from typing import List,Dict
import os
import importlib.util


MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_final.pkl")
bundle = joblib.load(MODEL_PATH)
MODULE_PATH_PREPROCESSOR = os.path.join(os.path.dirname(__file__),"..","src","utils.py")
spec = importlib.util.spec_from_file_location("utils", MODULE_PATH_PREPROCESSOR)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

model = bundle["model"]
scaler = bundle["scaler"]
parameters = bundle["params"]


class PredictionInput(BaseModel):
    data: Dict

app = FastAPI(
    title="API Bomba en el IIMAS",
    description="Practica 10: Optimización Bayesiana",
    version="1.0.0"
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/info")
async def info():
    return {
        "Equipo": "Bomba en el IIMAS",
        "Tipo de Modelo": str(type(model).__name__),
        "hyperparameters": parameters,
        "preprocessing": "StandardScaler normalization. Best model found via Bayesian Optimization: Gaussian Process with RBF kernel guided hyperparameter search using UCB acquisition function, minimizing RMSE across 3 model types (SVR, RF, MLP) over multiple iterations."
        }

@app.post("/predict")
async def make_predict(input: PredictionInput):
    try:
        data = input.data
        
        if not data:
            return {
                "error": "Input data is empty. Please provide superhero information.",
                "power_prediction": None
            }
        
        superhero = utils.process_basic_hero(data)
        superhero_features = superhero.values
        superhero_scaled = scaler.transform(superhero_features)

        prediction = model.predict(superhero_scaled)[0]
        return {
            "power_prediction": int(prediction)
        }
    except KeyError as e:
        return {
            "error": f"Missing required field: {str(e)}. Verify that all necessary fields are present.",
            "power_prediction": None
        }
    except ValueError as e:
        return {
            "error": f"Invalid value in data: {str(e)}. Ensure values are of the correct type.",
            "power_prediction": None
        }
    except AttributeError as e:
        return {
            "error": f"Attribute error: {str(e)}. Check the input data structure.",
            "power_prediction": None
        }
    except Exception as e:
        return {
            "error": f"Unexpected error processing prediction: {str(e)}. Contact administrator if problem persists.",
            "power_prediction": None
        }

if __name__ == "__main__":
     port = int(os.getenv("PORT",8000))
     uvicorn.run(app, host='0.0.0.0', port=port)