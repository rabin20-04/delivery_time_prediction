import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
API_URL = os.getenv("API_URL")

st.title("Delivery Time Prediction")

weather_options = ["SUNNY", "CLOUDY","Sandstorms", "FOG",  "STORMY", "WINDY"]
traffic_options = ["LOW", "MEDIUM", "HIGH", "JAM"]
vehicle_condition_options = [0, 1, 2, 3]
type_of_order_options = ["SNACK", "MEAL", "DRINKS", "BUFFET"]
type_of_vehicle_options = ["MOTORCYCLE", "SCOOTER", "ELECTRIC_SCOOTER", "BICYCLE"]
festival_options = ["NO", "YES"]
city_type_options = ["URBAN", "METROPOLITAN", "SEMI-URBAN"]
is_weekend_options = [0, 1]
order_time_of_day_options = ["MORNING", "AFTERNOON", "EVENING", "NIGHT"]
distance_type_options = ["SHORT", "MEDIUM", "LONG"]

with st.form("prediction_form"):
    age = st.number_input("Delivery Person Age", min_value=18.0, max_value=60.0, value=30.0, step=1.0)
    ratings = st.number_input("Delivery Person Ratings", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
    pickup_time_minutes = st.number_input("Pickup Time (Minutes)", min_value=0.0, max_value=60.0, value=10.0, step=1.0)
    weather = st.selectbox("Weather", weather_options)
    traffic = st.selectbox("Traffic Condition", traffic_options)
    vehicle_condition = st.selectbox("Vehicle Condition", vehicle_condition_options)
    type_of_order = st.selectbox("Type of Order", type_of_order_options)
    type_of_vehicle = st.selectbox("Type of Vehicle", type_of_vehicle_options)
    multiple_deliveries = st.number_input("Multiple Deliveries", min_value=0.0, max_value=5.0, value=1.0, step=1.0)
    festival = st.selectbox("Festival", festival_options)
    city_type = st.selectbox("City Type", city_type_options)
    distance = st.number_input("Distance (km)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    is_weekend = st.selectbox("Is Weekend", is_weekend_options)
    order_time_of_day = st.selectbox("Order Time of Day", order_time_of_day_options)
    distance_type = st.selectbox("Distance Type", distance_type_options)

    submitted = st.form_submit_button("Predict Delivery Time")

if submitted:
    data = {
        "age": age,
        "ratings": ratings,
        "pickup_time_minutes": pickup_time_minutes,
        "weather": weather.lower(),
        "traffic": traffic.lower(),
        "vehicle_condition": vehicle_condition,
        "type_of_order": type_of_order.lower(),
        "type_of_vehicle": type_of_vehicle.lower(),
        "multiple_deliveries": multiple_deliveries,
        "festival": festival.lower(),
        "city_type": city_type.lower(),
        "distance": distance,
        "is_weekend": is_weekend,
        "order_time_of_day": order_time_of_day.lower(),
        "distance_type": distance_type.lower()
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
        prediction = response.json()
        st.success(f"Predicted Delivery Time: {prediction:.2f} minutes")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the API: {e}. The backend may be starting up.")