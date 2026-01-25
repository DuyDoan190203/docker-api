from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel
import joblib
import numpy as np
from dotenv import load_dotenv
import os

# Load env vars (for local dev)
load_dotenv()

# Get API key from env (works local and deployed)
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY not set in environment")

# Security schemes: header (preferred) or query param as fallback
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

# Dependency to check key
async def get_api_key(
    header_key: str = Depends(api_key_header),
    query_key: str = Depends(api_key_query)
):
    if header_key == API_KEY or query_key == API_KEY:
        return API_KEY
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )

# Load model
model = joblib.load("wine_quality_model.joblib")  # Assume this file is in the repo or Docker image

# Define API
app = FastAPI(title="Wine Quality Prediction API")

# Public endpoint
@app.get("/")
async def root():
    return {"msg": "Welcome to the Wine Prediction API!"}

# Protected endpoint
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
    return {"predicted_quality": round(prediction, 2)}

# Input data model (keep this)
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