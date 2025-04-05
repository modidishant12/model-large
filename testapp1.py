import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Set the title
st.title("📦 Order Delivery Time Prediction")

# Define model file path
model_path = "order_delivery_model.pkl"

# Check if the model file exists
if not os.path.exists(model_path):
    st.error("Model file not found! Please place 'order_delivery_model.pkl' in the working directory.")
    st.stop()

# Load the model
try:
    rf, xgb = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define feature names
features = [
    "purchase_dow", "purchase_month", "year", "product_size_cm3", "product_weight_g", 
    "geolocation_state_customer", "geolocation_state_seller", "distance"
]

# Ensemble prediction function
def ensemble_predict(X):
    X_df = pd.DataFrame(X, columns=features)
    rf_pred = rf.predict(X_df)
    xgb_pred = xgb.predict(X_df)
    return (rf_pred + xgb_pred) / 2

# Sidebar inputs
st.sidebar.header("🔢 Input Parameters")

# Optional image display
image_path = "supply_chain_optimisation.jpg"
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.sidebar.image(image, caption="Supply Chain Optimization", use_container_width=True)

# User inputs
purchase_dow = st.sidebar.number_input("Purchased Day of the Week", 0, 6, 3)
purchase_month = st.sidebar.number_input("Purchased Month", 1, 12, 1)
year = st.sidebar.number_input("Purchased Year", 2018, 2025, 2018)
product_size_cm3 = st.sidebar.number_input("Product Size (cm³)", 100, 50000, 9328)
product_weight_g = st.sidebar.number_input("Product Weight (g)", 100, 50000, 1800)
geolocation_state_customer = st.sidebar.number_input("Customer State", 1, 50, 10)
geolocation_state_seller = st.sidebar.number_input("Seller State", 1, 50, 20)
distance = st.sidebar.number_input("Distance (km)", 0.0, 5000.0, 475.35)

# Prediction logic
def predict_wait_time():
    input_data = [[
        purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
        geolocation_state_customer, geolocation_state_seller, distance
    ]]
    prediction = ensemble_predict(input_data)
    return round(prediction[0])

# Predict button
if st.sidebar.button("🚀 Predict Wait Time"):
    with st.spinner("Predicting..."):
        result = predict_wait_time()
    st.sidebar.success(f"### ⏳ Predicted Delivery Time: **{result} days**")
