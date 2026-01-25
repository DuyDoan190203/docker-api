from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel
import joblib
import numpy as np
from dotenv import load_dotenv
import os

# ────────────────────────────────────────────────
# Load environment variables (local + deployed)
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set!")

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

async def get_api_key(
    header_key: str = Depends(api_key_header),
    query_key: str = Depends(api_key_query)
):
    if header_key == API_KEY or query_key == API_KEY:
        return API_KEY
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key"
    )


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


model = joblib.load("wine_quality_model.joblib")

app = FastAPI(title="Wine Quality Prediction API")

@app.get("/")
async def root():
    return {"msg": "Welcome to the Wine Prediction API!"}

@app.post("/predict")
def predict_quality(features: WineFeatures, api_key: str = Depends(get_api_key)):
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
    return {"predicted_quality": round(float(prediction), 2)}