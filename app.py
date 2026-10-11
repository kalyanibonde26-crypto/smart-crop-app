import streamlit as st
import numpy as np
import joblib

# Load saved models
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🌾 Smart Agriculture Monitoring & Crop Recommendation")

st.sidebar.header("Group Details")
st.sidebar.text("1. Gaurang")
st.sidebar.text("2. Nikhil")
st.sidebar.text("3. Kalyani")

# User input features
n = st.slider("Nitrogen (N)", 0, 140, 50)
p = st.slider("Phosphorus (P)", 0, 145, 50)
k = st.slider("Potassium (K)", 0, 205, 50)
temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
ph = st.number_input("pH Level", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)

if st.button("Predict Crop"):
    input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    crop_name = le.inverse_transform(prediction)[0]
    st.success(f"🌱 Recommended Crop: **{crop_name.upper()}**")