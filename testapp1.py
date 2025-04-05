import streamlit as st
import pandas as pd
import numpy as np
import gdown
import joblib
import os
from PIL import Image

# Define model file path
model_path = "order_delivery_model.pkl"

# Check if the model exists, else download
if not os.path.exists(model_path):
    file_id = "1cEhD4f1e9tryBVqlEAs0ZIwa_gcaAYIn"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the model
rf, xgb = joblib.load(model_path)

# Define feature names
features = [
    "purchase_dow", "purchase_month", "year", "product_size_cm3", "product_weight_g", 
    "geolocation_state_customer", "geolocation_state_seller", "distance"
]

# Define the ensemble prediction function
def ensemble_predict(X):
    X_df = pd.DataFrame(X, columns=features)  # Ensure correct feature names
    rf_pred = rf.predict(X_df)
    xgb_pred = xgb.predict(X_df)
    return (rf_pred + xgb_pred) / 2

# Streamlit UI
st.title("📦 Order Delivery Time Prediction")

# Sidebar UI
st.sidebar.header("Supply Chain Optimization")
image_path = "supply_chain_optimisation.jpg"  # Ensure image is in 'assets' folder
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.sidebar.image(image, caption="Supply Chain Optimization", use_container_width=True)

st.sidebar.header("🔢 Input Parameters")

# Input fields with reasonable ranges
purchase_dow = st.sidebar.number_input("Purchased Day of the Week", 0, 6, 3)
purchase_month = st.sidebar.number_input("Purchased Month", 1, 12, 1)
year = st.sidebar.number_input("Purchased Year", 2018, 2025, 2018)
product_size_cm3 = st.sidebar.number_input("Product Size (cm³)", 100, 50000, 9328)
product_weight_g = st.sidebar.number_input("Product Weight (g)", 100, 50000, 1800)
geolocation_state_customer = st.sidebar.number_input("Customer State", 1, 50, 10)
geolocation_state_seller = st.sidebar.number_input("Seller State", 1, 50, 20)
distance = st.sidebar.number_input("Distance (km)", 0.0, 5000.0, 475.35)

# Prediction function
def predict_wait_time():
    input_data = [[  # Convert input into a list of lists
        purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
        geolocation_state_customer, geolocation_state_seller, distance
    ]]
    prediction = ensemble_predict(input_data)
    return round(prediction[0])

# Button to trigger prediction
if st.sidebar.button("🚀 Predict Wait Time"):
    with st.spinner("Predicting..."):
        result = predict_wait_time()
    st.sidebar.success(f"### ⏳ Predicted Delivery Time: **{result} days**")