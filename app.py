from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import numpy as np
import mlflow
import json
import joblib
import os
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn import set_config
from scripts.data_cleaning_script import perform_data_cleaning
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache

load_dotenv()
set_config(transform_output="pandas")

import dagshub
import mlflow.client

dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_USER_TOKEN"))
dagshub.init(repo_owner="rabin20-04", repo_name="delivery_time_prediction", mlflow=True)  # type: ignore

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))  # type: ignore


class Data(BaseModel):
    # # ID: str
    # # Delivery_person_ID: str
    age: float
    ratings: float
    # # Restaurant_latitude: float
    # # Restaurant_longitude: float
    # # Delivery_location_latitude: float
    # # Delivery_location_longitude: float
    # # Order_Date: str
    # Time_Orderd: str
    # # Time_Order_picked: str
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


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info


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
    client = MlflowClient()
    run_info = json.load(open("run_information.json"))
    model_name = run_info["model_name"]
    stage = "Staging"
    latest_model_ver = client.get_latest_versions(name=model_name, stages=[stage])[
        0
    ].version
    print(f"Latest model in production is version {latest_model_ver}")

    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    preprocessor = joblib.load("models/preprocessor.joblib")
    model_pipe = Pipeline(steps=[("preprocess", preprocessor), ("regressor", model)])
    return model_pipe


app = FastAPI()

# Load model and preprocessor once at startup
model_pipe = load_model_and_preprocessor()


@app.get(path="/")
async def home():
    return " Welcome to the Delivery time Precition App"


# predict endpoint
@app.post(path="/predict")
async def perform_prediction(data: Data):
    pred_data = pd.DataFrame(
        {
            # "ID": data.ID,
            # "Delivery_person_ID": data.Delivery_person_ID,
            "age": data.age,
            "ratings": data.ratings,
            # "Restaurant_latitude": data.Restaurant_latitude,
            # "Restaurant_longitude": data.Restaurant_longitude,
            # "Delivery_location_latitude": data.Delivery_location_latitude,
            # "Delivery_location_longitude": data.Delivery_location_longitude,
            "pickup_time_minutes": data.pickup_time_minutes,
            # "Order_Date": data.Order_Date,
            # "Time_Orderd": data.Time_Orderd,
            # "Time_Order_picked": data.Time_Order_picked,
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

    # clean the raw input data
    # cleaned_data = perform_data_cleaning(pred_data)  # type: ignore
    predictions = model_pipe.predict(pred_data)[0]
    return predictions



app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://deliverytimeprediction-2r2xwb66xapjc4ywf5f3cq.streamlit.app", "https://*.streamlit.app","http://localhost:8501"],
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
