from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import numpy as np

# import mlflow
# import json
import joblib
import os
from dotenv import load_dotenv
load_dotenv()
STREAMLIT_API_URL = os.getenv("STREAMLIT_API_URL")
# from mlflow import MlflowClient
from sklearn import set_config

# from scripts.data_cleaning_script import perform_data_cleaning
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache

set_config(transform_output="pandas")

# import dagshub
# import mlflow.client

# dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_USER_TOKEN"))
# dagshub.init(repo_owner="rabin20-04", repo_name="delivery_time_prediction", mlflow=True)  # type: ignore

# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))  # type: ignore


class Data(BaseModel):
    age: float
    ratings: float
    pickup_time_minutes: float
    weather: str
    traffic: str
    vehicle_condition: int
    type_of_order: str
    type_of_vehicle: str
    multiple_deliveries: float
    festival: str
    city_type: str
    distance: float
    is_weekend: int
    order_time_of_day: str
    distance_type: str


# def load_model_information(file_path):
#     with open(file_path) as f:
#         run_info = json.load(f)
#     return run_info


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer


# ------> columns to preprocess in data <-------
num_cols = ["age", "ratings", "pickup_time_minutes", "distance"]

nominal_cat_cols = [
    "weather",
    "type_of_order",
    "type_of_vehicle",
    "festival",
    "city_type",
    "is_weekend",
    "order_time_of_day",
]

ordinal_cat_cols = ["traffic", "distance_type"]


@lru_cache(maxsize=1)
def load_model_and_preprocessor():
    try:
        model = joblib.load("models/model.joblib")
        preprocessor = joblib.load("models/preprocessor.joblib")
        model_pipe = Pipeline(
            steps=[("preprocess", preprocessor), ("regressor", model)]
        )
        print("Loaded latest local model and preprocessor")
        return model_pipe
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


app = FastAPI()

model_pipe = load_model_and_preprocessor()


@app.get(path="/")
async def home():
    return " Welcome to the Delivery time Precition App"


@app.get("/ping")
def ping():
    return {"status": "alive"}


# predict endpoint
@app.post(path="/predict")
async def perform_prediction(data: Data):
    pred_data = pd.DataFrame(
        {
            "age": data.age,
            "ratings": data.ratings,
            "pickup_time_minutes": data.pickup_time_minutes,
            "weather": data.weather,
            "traffic": data.traffic,
            "vehicle_condition": data.vehicle_condition,
            "type_of_order": data.type_of_order,
            "type_of_vehicle": data.type_of_vehicle,
            "multiple_deliveries": data.multiple_deliveries,
            "festival": data.festival,
            "city_type": data.city_type,
            "distance": data.distance,
            "is_weekend": data.is_weekend,
            "order_time_of_day": data.order_time_of_day,
            "distance_type": data.distance_type,
        },
        index=[0],
    )

    # cleaned_data = perform_data_cleaning(pred_data)  
    predictions = model_pipe.predict(pred_data)[0]
    return predictions


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"{STREAMLIT_API_URL}",
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    uvicorn.run(
        app="app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
