import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()

st.title("Hotel Booking Status Predictor")

st.markdown("Enter booking details:")

# User input features
no_of_adults = st.number_input("Number of adults", value=2, step=1)
no_of_children = st.number_input("Number of children", value=0, step=1)
no_of_weekend_nights = st.number_input("Weekend nights", value=1, step=1)
no_of_week_nights = st.number_input("Week nights", value=2, step=1)
type_of_meal_plan = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Not Selected", "Meal Plan 3"])
required_car_parking_space = st.selectbox("Car Parking", [0, 1])
room_type_reserved = st.selectbox("Room Type", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4"])
lead_time = st.number_input("Lead time", value=0, step=1)
arrival_year = st.selectbox("Arrival Year", [2017, 2018])
arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
arrival_date = st.number_input("Arrival Date", value=15, step=1)
market_segment_type = st.selectbox("Market Segment", ["Offline", "Online", "Corporate", "Complementary", "Aviation"])
repeated_guest = st.selectbox("Repeated Guest", [0, 1])
no_of_previous_cancellations = st.number_input("Previous Cancellations", value=0, step=1)
no_of_previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", value=0, step=1)
avg_price_per_room = st.number_input("Average Price per Room", value=100.0, step=1.0)
no_of_special_requests = st.number_input("Number of Special Requests", value=0, step=1)

# Combine into DataFrame
input_dict = {
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

df = pd.DataFrame([input_dict])

# Derive 'quarter' like in training
df["quarter"] = pd.to_datetime(df["arrival_month"], format='%m').dt.to_period("Q").astype(str)

# One-hot encode categorical columns like in training
categorical_cols = ["type_of_meal_plan", "room_type_reserved", "market_segment_type", "quarter"]
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Align to training feature columns
expected_features = model.feature_names_in_
for col in expected_features:
    if col not in df_encoded.columns:
        df_encoded[col] = 0  # Add missing column with 0
df_encoded = df_encoded[expected_features]  # Ensure order matches

# Prediction
if st.button("Predict Booking Status"):
    prediction = model.predict(df_encoded)[0]
    status = "Confirmed ✅" if prediction == 1 else "Canceled ❌"
    st.success(f"Predicted Booking Status: **{status}**")
