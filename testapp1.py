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

# Check and load model
if not os.path.exists(model_path):
    st.error("Model file not found! Please upload 'order_delivery_model.pkl'.")
    st.stop()

try:
    rf, xgb = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
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

# Optional image
image_path = "supply_chain_optimisation.jpg"
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.sidebar.image(image, caption="Supply Chain Optimization", use_container_width=True)

# Input fields with safe limits
purchase_dow = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)
purchase_month = st.sidebar.slider("Month", 1, 12, 6)
year = st.sidebar.selectbox("Year", [2018, 2019, 2020, 2021])
product_size_cm3 = st.sidebar.slider("Product Size (cm³)", 100, 30000, 9328)
product_weight_g = st.sidebar.slider("Product Weight (g)", 100, 30000, 1800)
geolocation_state_customer = st.sidebar.slider("Customer State Code", 1, 27, 10)
geolocation_state_seller = st.sidebar.slider("Seller State Code", 1, 27, 20)
distance = st.sidebar.slider("Distance (km)", 0.0, 3000.0, 475.35)

# Prediction function
def predict_wait_time():
    try:
        input_data = [[
            purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
            geolocation_state_customer, geolocation_state_seller, distance
        ]]

        # Optional warnings for out-of-distribution inputs
        if distance > 2000:
            st.warning("⚠️ Distance is unusually high — prediction may be less accurate.")
        if product_weight_g > 20000:
            st.warning("⚠️ Product is very heavy — might not generalize well.")
        if year not in [2018, 2019, 2020, 2021]:
            st.warning("⚠️ Year is outside training data range.")

        prediction = ensemble_predict(input_data)
        return round(prediction[0])

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        return None

# Predict button
if st.sidebar.button("🚀 Predict Wait Time"):
    with st.spinner("Predicting..."):
        result = predict_wait_time()
    if result is not None:
        st.sidebar.success(f"### ⏳ Predicted Delivery Time: **{result} days**")
