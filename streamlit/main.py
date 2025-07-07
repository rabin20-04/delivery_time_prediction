import streamlit as st
import requests
import os
import time
from dotenv import load_dotenv

# Load environment variable
load_dotenv()
API_URL = os.getenv("API_URL")
log_messages = []
# Wake up render
try:
    BACKEND_URL = f"{API_URL}/ping"
    requests.get(BACKEND_URL, timeout=10)
except:
    st.text("")

# Page config
st.set_page_config(page_title="🚚 Delivery Time Predictor", layout="wide")

city_type_options = ["URBAN", "METROPOLITAN", "SEMI-URBAN"]
weather_options = ["SUNNY", "CLOUDY", "SANDSTORMS", "FOG", "STORMY", "WINDY"]
traffic_options = ["LOW", "MEDIUM", "HIGH", "JAM"]
type_of_order_options = ["SNACK", "MEAL", "DRINKS", "BUFFET"]
type_of_vehicle_options = ["MOTORCYCLE", "SCOOTER", "ELECTRIC_SCOOTER", "BICYCLE"]
festival_options = ["YES", "NO"]
is_weekend_options = ["Yes", "No"]
order_time_of_day_options = ["MORNING", "AFTERNOON", "EVENING", "NIGHT"]
distance_type_options = ["SHORT", "MEDIUM", "LONG"]

page = st.sidebar.radio("", ["Description", "Prediction"], index=1)

st.title("📦 Delivery Time Prediction App")
# DESCRIPTION PAGE
if page == "Description":
    st.markdown(
        ':gray[For detailed explanation of this project :] <a href="https://medium.com/@poudelrabin2004/predicting-online-delivery-time-with-machine-learning-5d92c0c76bcc" target="_blank">Blog</a>',
        unsafe_allow_html=True
    )

    st.markdown("#### Powered by Streamlit & FastAPI")
    st.markdown(
        """
        In today’s fast-moving world, where online shopping and food deliveries are part of daily life, knowing **exactly when** your order will arrive is a game-changer. My project, the **Delivery Time Prediction App**, makes this happen by giving you accurate delivery time estimates with a simple, user-friendly interface. After experimenting with different approaches, I’ve built a tool that delivers reliable predictions to make life easier for everyone involved.

        ---

        ### Why I Built This
        Accurate delivery times matter to everyone:
        - **Customers:** Plan your day without the stress of wondering when your order will show up—whether it’s a quick snack or a big meal.
        - **Riders:** Save time by choosing the best routes, with predictions that consider real-world factors like traffic or weather.
        - **Businesses:** Keep customers happy, reduce complaints, and run smoother operations with dependable ETAs.


        ---

        ### How It Works
        This app is powered with advanced machine learning models, carefully chosen after testing different approaches to ensure decent accuracy:

        1. **Core Models:**
           - **Random Forest:** Spots patterns in complex delivery data to make solid predictions.
           - **LightGBM:** Delivers fast and accurate results, even with tricky conditions.

        2. **Final Touch:**
           - **Linear Regression:** Combines the strengths of both models for a precise, polished prediction.

         The final model is a **Stacking Regressor** a method I chose after experimenting to balance speed and accuracy tuned through optuna, where the base models were Random Forest and LightGBM, and the meta-model is Linear Regression.

          The result? Predictions you can count on, accurate within ±3 minutes.

       ---

       
         :gray[**Thank you for checking out this project.**  
        Feel free to explore the prediction page and see how everything works in action. ] 


        **Suggestions are always appreciated!**
       <a href="https://www.linkedin.com/in/rabin-poudel-770842277/" target="_blank">            
       <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30" height="30">
       </a>
       <a href="https://github.com/rabin20-04/delivery_time_prediction" target="_blank">
       <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30" height="30">
       </a>

        """,
        unsafe_allow_html=True,
    )

# PREDICTION PAGE
else:
    # Main form
    with st.form(key="prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Delivery Person Age", min_value=10, max_value=60, value=25
            )
            st.divider()
            ratings = st.slider(
                "⭐ Delivery Person Ratings",
                min_value=0.0,
                max_value=5.0,
                value=4.0,
                step=0.1,
            )
            st.divider()
            pickup_time = st.number_input(
                " Pickup Time (minutes)",
                min_value=0,
                max_value=60,
                value=10,
                help="Time until the delivery person arrives at the restaurant",
            )
            st.divider()
            multiple_deliveries = st.number_input(
                " Multiple Deliveries", min_value=0, max_value=5, value=0
            )
            st.divider()

        with col2:
            weather = st.selectbox(" Weather", weather_options)
            st.divider()
            traffic = st.selectbox(" Traffic", traffic_options)
            st.divider()
            city_type = st.selectbox("City Type", city_type_options)
            st.divider()
            vehicle_condition = st.radio(
                " Vehicle Condition",
                ["Excellent (3)", "Good (2)", "Fair (1)", "Poor (0)"],
                index=0,
            )
            st.divider()
            distance = st.number_input(
                " Distance (km)", min_value=0.0, max_value=100.0, value=5.0, step=0.1
            )

        with col3:
            order_type = st.selectbox(" Type of Order", type_of_order_options)
            st.divider()
            vehicle_type = st.selectbox(" Type of Vehicle", type_of_vehicle_options)
            st.divider()
            festival = st.radio(" Festival Day?", festival_options, horizontal=True)
            st.divider()
            is_weekend = st.radio("Is Weekend?", is_weekend_options, horizontal=True)
            st.divider()
            time_of_day = st.selectbox(" Time of Day", order_time_of_day_options)
            st.divider()
            distance_type = st.selectbox("Distance Category", distance_type_options)
            st.divider()

        submit = st.form_submit_button("Predict Delivery Time")

   
    if submit:
       
        payload = {
            "age": age,
            "ratings": ratings,
            "pickup_time_minutes": pickup_time,
            "weather": weather.lower(),
            "traffic": traffic.lower(),
            "vehicle_condition": int(vehicle_condition.split()[1].strip("()")),
            "type_of_order": order_type.lower(),
            "type_of_vehicle": vehicle_type.lower().replace(" ", "_"),
            "multiple_deliveries": multiple_deliveries,
            "festival": festival,
            "city_type": city_type.lower(),
            "distance": distance,
            "is_weekend": 1 if is_weekend else 0,
            "order_time_of_day": time_of_day.lower(),
            "distance_type": distance_type.lower(),
        }

        # API call
        with st.spinner("Calculating optimal delivery time..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                response.raise_for_status()
                prediction = response.json()
                time.sleep(1)  # Simulate processing delay
            except requests.exceptions.RequestException:
                st.error(
                    "Failed to connect to the API. The backend may be starting up."
                )
                st.error("Please reload!")
                prediction = None

        #  just fun animation display
        if prediction is not None:
            display = st.empty()
            for i in range(0, int(prediction) + 1):
                display.markdown(
                    f"<h1 style='text-align:center; color:#FF6F61; font-size:72px;'>{i} <span style='font-size:24px;'>min</span></h1>",
                    unsafe_allow_html=True,
                )
                time.sleep(0.02)
            display.markdown(
                f"<h1 style='text-align:center; color:#FF6F61; font-size:72px;'>Total delivery time: {prediction:.2f} <span style='font-size:24px;'>minutes</span></h1>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f":gray[The total Estimated Delivery Time: {prediction:.2f} ± 3 minutes]"
            )
            st.divider()
            st.image("https://res.cloudinary.com/dz4tg6vyg/image/upload/v1751882333/image_mk5iwm.png")
        else:
            st.error("Failed to get a prediction.")

st.divider()
