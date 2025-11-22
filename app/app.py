import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("models/best_model.joblib")

st.title("🚗 Engine Condition Prediction Dashboard")
st.write("Use the sliders below to simulate engine sensor values and predict the condition.")

# Sensor Ranges (from your dataset)
custom_ranges = {
    'Engine rpm': (61.0, 2239.0),
    'Lub oil pressure': (0.003384, 7.265566),
    'Fuel pressure': (0.003187, 21.138326),
    'Coolant pressure': (0.002483, 7.478505),
    'lub oil temp': (71.321974, 89.580796),
    'Coolant temp': (61.673325, 195.527912),
    'Temperature_difference': (-22.669427, 119.008526)
}

# Sidebar info
st.sidebar.title("ℹ Sensor Descriptions")
st.sidebar.write("""
- **Engine RPM:** Engine rotation per minute  
- **Oil Pressure:** Lubricant pressure  
- **Fuel Pressure:** Fuel injection pressure  
- **Coolant Pressure/Temp:** Engine cooling system  
- **Temperature Difference:** Coolant - Oil  
""")

# Input sliders
engine_rpm = st.slider("Engine RPM", *custom_ranges['Engine rpm'])
lub_oil_pressure = st.slider("Lub Oil Pressure", *custom_ranges['Lub oil pressure'])
fuel_pressure = st.slider("Fuel Pressure", *custom_ranges['Fuel pressure'])
coolant_pressure = st.slider("Coolant Pressure", *custom_ranges['Coolant pressure'])
lub_oil_temp = st.slider("Lub Oil Temperature", *custom_ranges['lub oil temp'])
coolant_temp = st.slider("Coolant Temperature", *custom_ranges['Coolant temp'])

# Derived Feature
temp_difference = coolant_temp - lub_oil_temp
st.write(f"**Temperature Difference:** {temp_difference:.2f}")

# Engine Power Feature
engine_power = engine_rpm * lub_oil_pressure
st.write(f"**Engine Power:** {engine_power:.2f}")

# Predict button
if st.button("🔍 Predict Engine Condition"):
    input_data = pd.DataFrame([{
        'Engine rpm': engine_rpm,
        'Lub oil pressure': lub_oil_pressure,
        'Fuel pressure': fuel_pressure,
        'Coolant pressure': coolant_pressure,
        'lub oil temp': lub_oil_temp,
        'Coolant temp': coolant_temp,
        'Temperature_difference': temp_difference,
        'Engine_power': engine_power
    }])
    # FEATURES IN THE EXACT ORDER USED DURING TRAINING
    feature_order = [
    'Engine rpm',
    'Lub oil pressure',
    'Fuel pressure',
    'Coolant pressure',
    'lub oil temp',
    'Coolant temp',
    'Engine_power',
    'Temperature_difference'
]


    # Reorder columns to match training
    input_data = input_data[feature_order]


    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][1]

    if prediction == 0:
        st.success(f"✔ Engine is NORMAL | Confidence: {1 - confidence:.2%}")
    else:
        st.error(f"⚠ Warning: Engine condition ABNORMAL | Confidence: {confidence:.2%}")

