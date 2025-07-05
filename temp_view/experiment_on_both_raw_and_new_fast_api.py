from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import numpy as np
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
from scripts.data_cleaning_script import perform_data_cleaning

set_config(transform_output="pandas")

import dagshub
import mlflow.client

dagshub.init(repo_owner="rabin20-04", repo_name="delivery_time_prediction", mlflow=True)  # type: ignore

mlflow.set_tracking_uri(
    "https://dagshub.com/rabin20-04/delivery_time_prediction.mlflow"
)


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
    multiple_deliveries: str
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


client = MlflowClient()

model_name = load_model_information("run_information.json")["model_name"]

stage = "Staging"

# get the latest model version
latest_model_ver = client.get_latest_versions(name=model_name, stages=[stage])
print(f"Latest model in production is version {latest_model_ver[0].version}")


model_path = f"models:/{model_name}/{stage}"

model = mlflow.sklearn.load_model(model_path)

preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)


model_pipe = Pipeline(steps=[("preprocess", preprocessor), ("regressor", model)])

app = FastAPI()


@app.get(path="/")
def home():
    return " Welcome to the Delivery time Precition App"


# predict endpoint
@app.post(path="/predict")
def perform_prediction(data: Data):
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
    cleaned_data = perform_data_cleaning(pred_data)  # type: ignore
    predictions = model_pipe.predict(cleaned_data)[0]
    return predictions


if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000, reload=True)










# ----------------> Running on total raw data as of dataset

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import numpy as np
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
from scripts.data_cleaning_script import perform_data_cleaning

set_config(transform_output="pandas")

import dagshub
import mlflow.client

dagshub.init(repo_owner="rabin20-04", repo_name="delivery_time_prediction", mlflow=True)  # type: ignore

mlflow.set_tracking_uri(
    "https://dagshub.com/rabin20-04/delivery_time_prediction.mlflow"
)


class Data(BaseModel):
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str


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


client = MlflowClient()

model_name = load_model_information("run_information.json")["model_name"]

stage = "Staging"

# get the latest model version
latest_model_ver = client.get_latest_versions(name=model_name, stages=[stage])
print(f"Latest model in production is version {latest_model_ver[0].version}")


model_path = f"models:/{model_name}/{stage}"

model = mlflow.sklearn.load_model(model_path)

preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)


model_pipe = Pipeline(steps=[("preprocess", preprocessor), ("regressor", model)])

app = FastAPI()


@app.get(path="/")
def home():
    return " Welcome to the Delivery time Precition App"


# predict endpoint
@app.post(path="/predict")
def perform_prediction(data: Data):
    pred_data = pd.DataFrame(
        {
            "ID": data.ID,
            "Delivery_person_ID": data.Delivery_person_ID,
            "Delivery_person_Age": data.Delivery_person_Age,
            "Delivery_person_Ratings": data.Delivery_person_Ratings,
            "Restaurant_latitude": data.Restaurant_latitude,
            "Restaurant_longitude": data.Restaurant_longitude,
            "Delivery_location_latitude": data.Delivery_location_latitude,
            "Delivery_location_longitude": data.Delivery_location_longitude,
            "Order_Date": data.Order_Date,
            "Time_Orderd": data.Time_Orderd,
            "Time_Order_picked": data.Time_Order_picked,
            "Weatherconditions": data.Weatherconditions,
            "Road_traffic_density": data.Road_traffic_density,
            "Vehicle_condition": data.Vehicle_condition,
            "Type_of_order": data.Type_of_order,
            "Type_of_vehicle": data.Type_of_vehicle,
            "multiple_deliveries": data.multiple_deliveries,
            "Festival": data.Festival,
            "City": data.City,
        },
        index=[0],
    )

    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data) # type: ignore
    predictions = model_pipe.predict(cleaned_data)[0]
    return predictions


if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000, reload=True)
