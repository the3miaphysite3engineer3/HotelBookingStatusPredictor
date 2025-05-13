import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
from io import BytesIO

st.set_page_config(page_title="Hotel Booking Predictor", layout="centered")

st.title("🏨 Hotel Booking Status Predictor")
st.markdown("Predict whether a hotel booking will be **Confirmed** or **Canceled** based on the input features.")

# ----------------------------------
# Step 1: Model Selection or URL Input
# ----------------------------------

st.markdown("### Step 1: Choose or Provide a Model")

preset_urls = {
    "Random Forest (Google Drive)": "https://drive.usercontent.google.com/u/0/uc?id=1N36N2qn3egwmW6iwFOO4_5c_gSrrsODB&export=download",
    "Extra Trees (Google Drive)": "https://drive.usercontent.google.com/u/0/uc?id=1vKWd-GA6eMB7g0LRCitsEzQjNHIDBDG0&export=download"
}

selected_preset = st.selectbox("Select a pre-trained model", list(preset_urls.keys()))
custom_model_url = st.text_input("Or paste your own model URL (.pkl file)", value="")

# Determine model URL
model_url = custom_model_url.strip() if custom_model_url.strip() else preset_urls[selected_preset]

# ----------------------------------
# Step 2: Load Model
# ----------------------------------

@st.cache_resource
def load_model(url):
    try:
        with st.spinner("Loading model (this may take a moment)..."):
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            model = joblib.load(BytesIO(response.content))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_url)
if model is None:
    st.stop()

# ----------------------------------
# Step 3: User Inputs
# ----------------------------------

st.markdown("### Step 2: Enter Booking Details")

no_of_adults = st.number_input("Number of Adults", value=2, step=1)
no_of_children = st.number_input("Number of Children", value=0, step=1)
no_of_weekend_nights = st.number_input("Weekend Nights", value=1, step=1)
no_of_week_nights = st.number_input("Week Nights", value=2, step=1)
type_of_meal_plan = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Not Selected", "Meal Plan 3"])
required_car_parking_space = st.selectbox("Car Parking Required", [0, 1])
room_type_reserved = st.selectbox("Room Type Reserved", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4"])
lead_time = st.number_input("Lead Time (days)", value=0, step=1)
arrival_year = st.selectbox("Arrival Year", list(range(1900, 2101)))
arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
arrival_date = st.number_input("Arrival Date", value=15, step=1)
market_segment_type = st.selectbox("Market Segment", ["Offline", "Online", "Corporate", "Complementary", "Aviation"])
repeated_guest = st.selectbox("Is Repeated Guest?", [0, 1])
no_of_previous_cancellations = st.number_input("Previous Cancellations", value=0, step=1)
no_of_previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", value=0, step=1)
avg_price_per_room = st.number_input("Average Price per Room", value=100.0, step=1.0)
no_of_special_requests = st.number_input("Special Requests", value=0, step=1)

# ----------------------------------
# Step 4: Feature Engineering
# ----------------------------------

input_data = {
    "no_of_adults": no_of_adults,
    "no_of_children": no_of_children,
    "no_of_weekend_nights": no_of_weekend_nights,
    "no_of_week_nights": no_of_week_nights,
    "type_of_meal_plan": type_of_meal_plan,
    "required_car_parking_space": required_car_parking_space,
    "room_type_reserved": room_type_reserved,
    "lead_time": lead_time,
    "arrival_year": arrival_year,
    "arrival_month": arrival_month,
    "arrival_date": arrival_date,
    "market_segment_type": market_segment_type,
    "repeated_guest": repeated_guest,
    "no_of_previous_cancellations": no_of_previous_cancellations,
    "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
    "avg_price_per_room": avg_price_per_room,
    "no_of_special_requests": no_of_special_requests,
}

df = pd.DataFrame([input_data])
df["quarter"] = pd.to_datetime(df["arrival_month"], format='%m').dt.to_period("Q").astype(str)

# One-hot encode
categorical_columns = ["type_of_meal_plan", "room_type_reserved", "market_segment_type", "quarter"]
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Align with model training features
expected_columns = model.feature_names_in_
for col in expected_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
df_encoded = df_encoded[expected_columns]

# ----------------------------------
# Step 5: Prediction
# ----------------------------------

st.markdown("### Step 3: Prediction")

if st.button("Predict Booking Status"):
    try:
        prediction = model.predict(df_encoded)[0]
        status = "✅ Confirmed" if prediction == 1 else "❌ Canceled"
        st.success(f"**Predicted Booking Status: {status}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ----------------------------------
# Step 6: Power BI Integration
# ----------------------------------

st.markdown("### Step 4: Power BI Report Integration")

powerbi_url = st.text_input("Enter Power BI Report URL", value="")

if powerbi_url:
    st.markdown(f'<iframe width="800" height="600" src="{powerbi_url}" frameborder="0" allowFullScreen="true"></iframe>', unsafe_allow_html=True)
