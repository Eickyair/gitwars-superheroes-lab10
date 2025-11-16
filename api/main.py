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
        "Preprocesamiento": "StandardScaler",
        "Archivo Modelo": MODEL_PATH
    }

@app.post("/predict")
async def make_predict(input: PredictionInput):

    data = input.data
    hero = utils.process_raw_hero(data)

    data_scaled = scaler.transform(hero)

    pred = model.predict(data_scaled)[0]
    return {
        "power_prediction": int(pred)
    }

if __name__ == "__main__":
     port = int(os.getenv("PORT",8000))
     uvicorn.run(app, host='0.0.0.0', port=port)