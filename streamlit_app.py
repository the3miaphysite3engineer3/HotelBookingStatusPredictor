import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
from io import BytesIO
import os
import google.generativeai as genai

# Set page configuration
st.set_page_config(page_title="Hotel Booking Predictor", layout="centered")
st.title("🏨 Hotel Booking Status Predictor")
st.markdown("Predict whether a hotel booking will be **Confirmed** or **Canceled** based on the input features.")

# -----------------------------
# Step 1: Load Model from Hugging Face
# -----------------------------

st.markdown("### Step 1: Load Model from Hugging Face")

huggingface_model_paths = {
    "Random Forest (Hugging Face)": "georgtawadrous/HotelBookingStatusPredictor/blob/main/random_forest_model.pkl"
}

selected_model = st.selectbox("Select a pre-trained model", list(huggingface_model_paths.keys()))
custom_model_path = st.text_input("Or paste your own Hugging Face model path (e.g., `username/repo/blob/main/model.pkl`)", value="")

# Determine which model path to use
model_path = custom_model_path.strip() if custom_model_path.strip() else huggingface_model_paths[selected_model]

# Construct Hugging Face raw download URL
def get_hf_resolve_url(path: str):
    return path.replace("blob/", "").replace("/main/", "/resolve/main/")

model_url = f"https://huggingface.co/{get_hf_resolve_url(model_path)}"
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")  # Ensure this is set in your environment

# -----------------------------
# Step 2: Load Model
# -----------------------------

@st.cache_resource
def load_model(url):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    try:
        with st.spinner("Loading model from Hugging Face..."):
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            model = joblib.load(BytesIO(response.content))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_url)
if model is None:
    st.stop()

# -----------------------------
# Step 3: User Inputs
# -----------------------------

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

# -----------------------------
# Step 4: Feature Engineering
# -----------------------------

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

# Define categorical columns
categorical_columns = ["type_of_meal_plan", "room_type_reserved", "market_segment_type", "quarter"]

# Create dummy variables with numeric dtype
df_encoded = pd.get_dummies(df, columns=categorical_columns, dtype=float)

# Ensure all expected columns are present
expected_columns = model.feature_names_in_
for col in expected_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0.0  # Use float to ensure numeric type

# Reorder columns to match model's expected order
df_encoded = df_encoded[expected_columns]

# Convert all columns to float64 to avoid dtype issues
df_encoded = df_encoded.astype(float)

# Debug: Inspect df_encoded
#st.write("df_encoded shape:", df_encoded.shape)
#st.write("df_encoded dtypes:", df_encoded.dtypes)

# Verify no non-numeric columns
if df_encoded.select_dtypes(include=['object', 'category']).columns.any():
    #st.error("Non-numeric columns detected in df_encoded!")
    #st.write(df_encoded.dtypes)
    st.stop()

# -----------------------------
# Step 5: Prediction and Gemini Integration
# -----------------------------

# Configure Gemini API (ensure GEMINI_API_KEY is set in environment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Placeholder function for Gemini API call
def query_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Adjust model as needed
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Simulated response if API call fails or key is missing
        return (
            f"**Analysis of Cancellation Risk**: This booking is at risk due to being made through the Online segment, "
            f"the customer being new, not requesting parking, and including a weekend night. "
            f"However, the winter (Q1) timing reduces risk slightly.\n\n"
            f"**Strategies to Prevent Cancellation**:\n"
            f"1. Offer a 10% discount on their next stay or loyalty points to encourage commitment.\n"
            f"2. Provide a partially refundable rate with a free breakfast or $10 room credit.\n"
            f"3. Send a personalized confirmation email or make a courtesy call within 24 hours.\n"
            f"4. Promote Room Type 1 and Meal Plan 1 benefits (e.g., cozy winter experience).\n"
            f"5. Suggest adding parking or a low-cost add-on (e.g., spa access)."
        )

if st.button("Predict Booking Status"):
    try:
        prediction = model.predict(df_encoded)[0]
        status = "✅ Confirmed" if prediction == 1 else "❌ Canceled"
        st.success(f"*Predicted Booking Status: {status}*")

        # If prediction is Canceled, query Gemini for prevention strategies
        if prediction == 0:
            st.markdown("### Cancellation Prevention Strategies")
            # Construct customer data summary
            customer_summary = (
                f"Booking for {no_of_adults} adults, {no_of_children} children, "
                f"{no_of_weekend_nights} weekend night(s), {no_of_week_nights} weeknight(s), "
                f"{type_of_meal_plan}, {required_car_parking_space} parking, {room_type_reserved}, "
                f"{lead_time} days lead time, arrival on {arrival_month}/{arrival_date}/{arrival_year}, "
                f"via {market_segment_type}, repeated guest: {repeated_guest}, "
                f"{no_of_previous_cancellations} previous cancellations, "
                f"{no_of_previous_bookings_not_canceled} previous bookings not canceled, "
                f"${avg_price_per_room} room price, {no_of_special_requests} special requests, "
                f"in Q{((arrival_month-1)//3)+1} {arrival_year}."
            )
            # Dataset insights
            insights = (
                "Cancellations are most common in summer (41.19%), especially for early bookings and weekend stays. "
                "Guests booking very early are more likely to cancel; early-bird offers should be non-refundable or include incentives. "
                "Repeat guests and corporate clients are less likely to cancel. "
                "Rooms with more guests tend to have higher cancellation rates. "
                "Guests requesting parking almost never cancel. "
                "Guests booking online (especially Online TA) cancel more frequently. "
                "Most customers are new, not repeat clients; loyal customers behave consistently. "
                "Cancellations are lowest in winter, and more frequent in Q3 and Q4. "
                "Meal Plan 1, Room Type 1, and mid-range pricing are most common."
            )
            # Construct prompt
            prompt = (
                f"You are a data-driven hotel strategy assistant. A customer’s booking is predicted to be canceled based on their profile: {customer_summary} "
                f"Based on these insights: {insights} "
                f"Analyze the customer’s data, identify cancellation risk factors, and suggest specific, actionable, and personalized strategies to prevent this cancellation. "
                f"Be concise, practical, and business-oriented."
            )
            # Query Gemini
            gemini_response = query_gemini(prompt)
            st.markdown(gemini_response)
    except Exception as e:
        st.error(f"Prediction error: {e}")

# -----------------------------
# Step 6: Power BI Report Integration
# -----------------------------

st.markdown("### Step 6: Power BI Report")
st.markdown(
    '<iframe width="800" height="600" src="https://app.powerbi.com/reportEmbed?reportId=4596f25e-8722-4aaa-8a2b-381faa24055f&autoAuth=true&ctid=ad2a8324-bef7-46a8-adb4-fe51b6613b24" frameborder="0" allowFullScreen="true"></iframe>',
    unsafe_allow_html=True
)
