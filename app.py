import os
import joblib
import pandas as pd
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="NYC Yellow Taxi Prediction API")

# Enable CORS for local development & frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load machine learning models safely from the models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FARE_MODEL_PATH = os.path.join(BASE_DIR, "models", "fare_model.pkl")
PAYMENT_MODEL_PATH = os.path.join(BASE_DIR, "models", "payment_model.pkl")

try:
    fare_model = joblib.load(FARE_MODEL_PATH)
except Exception as e:
    fare_model = None
    print(f"Warning: Could not load fare model from {FARE_MODEL_PATH}: {e}")

try:
    payment_model = joblib.load(PAYMENT_MODEL_PATH)
except Exception as e:
    payment_model = None
    print(f"Warning: Could not load payment model from {PAYMENT_MODEL_PATH}: {e}")


class TripInput(BaseModel):
    trip_distance: float
    passenger_count: int
    pickup_hour: int
    trip_duration: float
    ratecode_id: int


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serves the index.html dashboard directly from the root route."""
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Frontend index.html not found in root directory.</h3>"


@app.post("/predict")
def predict_trip(data: TripInput):
    if not fare_model or not payment_model:
        raise HTTPException(status_code=500, detail="Models are not loaded on the server.")
    
    try:
        # 1. Fare model feature order: ['trip_distance', 'trip_duration', 'pickup_hour', 'passenger_count', 'RatecodeID']
        fare_features_df = pd.DataFrame([[
            data.trip_distance,
            data.trip_duration,
            data.pickup_hour,
            data.passenger_count,
            data.ratecode_id
        ]], columns=["trip_distance", "trip_duration", "pickup_hour", "passenger_count", "RatecodeID"])
        
        fare_pred = fare_model.predict(fare_features_df)[0]
        
        # 2. Payment model feature order: ['trip_distance', 'trip_duration', 'pickup_hour', 'fare_amount', 'passenger_count']
        payment_features_df = pd.DataFrame([[
            data.trip_distance,
            data.trip_duration,
            data.pickup_hour,
            fare_pred,
            data.passenger_count
        ]], columns=["trip_distance", "trip_duration", "pickup_hour", "fare_amount", "passenger_count"])
        
        payment_pred = payment_model.predict(payment_features_df)[0]
        
        return {
            "predicted_fare": float(fare_pred),
            "predicted_payment_type": int(payment_pred)
        }
    except Exception as e:
        full_traceback = traceback.format_exc()
        print("\n--- PREDICTION ERROR TRACEBACK ---")
        print(full_traceback)
        print("----------------------------------\n")
        raise HTTPException(status_code=400, detail=str(e))