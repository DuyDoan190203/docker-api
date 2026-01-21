from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("wine_quality_model.joblib")

# Define API
app = FastAPI(title="Wine Quality Prediction API")

# Input data model
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
async def root():
    return { "msg": "Welcome to the Wine Prediction API!" }

@app.post("/predict")
def predict_quality(features: WineFeatures):
    data = np.array([[
        features.fixed_acidity,
        features.volatile_acidity,
        features.citric_acid,
        features.residual_sugar,
        features.chlorides,
        features.free_sulfur_dioxide,
        features.total_sulfur_dioxide,
        features.density,
        features.pH,
        features.sulphates,
        features.alcohol
    ]])
    prediction = model.predict(data)[0]
    return {"predicted_quality": round(prediction, 2)}
    