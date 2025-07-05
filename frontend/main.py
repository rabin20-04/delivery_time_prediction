import streamlit as st
import requests
import pandas as pd

st.title("Delivery Time Prediction")

# Define options for categorical variables
weather_options = ["sunny", "Cloudy", "Fog", "Rainy", "Stormy", "Windy"]
traffic_options = ["low", "Medium", "High", "Jam"]
vehicle_condition_options = [0, 1, 2, 3]
type_of_order_options = ["snack", "Meal", "Drinks", "Buffet"]
type_of_vehicle_options = ["motorcycle", "scooter", "electric_scooter", "bicycle"]
festival_options = ["no", "Yes"]
city_type_options = ["urban", "Metropolitan", "Semi-Urban"]
is_weekend_options = [0, 1]
order_time_of_day_options = ["morning", "Afternoon", "Evening", "Night"]
distance_type_options = ["short", "Medium", "Long"]

# Create input fields
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

    # Submit button
    submitted = st.form_submit_button("Predict Delivery Time")

# Handle form submission
if submitted:
    # Prepare data for API
    data = {
        "age": age,
        "ratings": ratings,
        "pickup_time_minutes": pickup_time_minutes,
        "weather": weather,
        "traffic": traffic,
        "vehicle_condition": vehicle_condition,
        "type_of_order": type_of_order,
        "type_of_vehicle": type_of_vehicle,
        "multiple_deliveries": multiple_deliveries,
        "festival": festival,
        "city_type": city_type,
        "distance": distance,
        "is_weekend": is_weekend,
        "order_time_of_day": order_time_of_day,
        "distance_type": distance_type
    }

    # Send request to FastAPI endpoint
    try:
        response = requests.post("http://localhost:8000/predict", json=data)
        if response.status_code == 200:
            prediction = response.json()
            st.success(f"Predicted Delivery Time: {prediction:.2f} minutes")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the API: {e}")