from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel
import joblib
import numpy as np
from dotenv import load_dotenv
import os
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timezone

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


JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable is not set!")

ALGORITHM = "HS256"

MESSAGES = [
    {"id": 1, "user_id": 1, "text": "Welcome to the platform!"},
    {"id": 2, "user_id": 2, "text": "Your report is ready for download."},
    {"id": 3, "user_id": 1, "text": "You have a new notification."},
    {"id": 4, "user_id": 3, "text": "Password will expire in 5 days."},
    {"id": 5, "user_id": 2, "text": "New login detected from a new device."},
    {"id": 6, "user_id": 3, "text": "Your subscription has been updated."},
]

# OAuth2 scheme for Bearer token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # placeholder, no real login needed

# Dependency: verify JWT and extract user info
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        user_id: int = int(payload.get("sub"))  # make sure it's int
        role: str = payload.get("role", "user")
        if user_id is None:
            raise credentials_exception
    except (JWTError, ValueError):
        raise credentials_exception
    
    return {"user_id": user_id, "role": role}

# New protected endpoint
@app.get("/messages")
async def get_messages(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    role = current_user["role"]
    
    if role == "admin":
        return MESSAGES  # admin sees all
    
    # regular user sees only their messages
    user_messages = [msg for msg in MESSAGES if msg["user_id"] == user_id]
    return user_messages if user_messages else {"message": "No messages for you"}