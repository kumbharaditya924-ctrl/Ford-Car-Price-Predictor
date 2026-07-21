import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.set_page_config(page_title="Ford Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Ford Car Price Prediction App")
st.write("---")

try:
    
    model = joblib.load('LR_model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    st.success("✅ All ML Models & Scalers loaded successfully!")

    
    st.subheader("Enter Car Details:")
    year = st.number_input("Year", min_value=2000, max_value=2026, value=2019)
    mileage = st.number_input("Mileage (kms/miles)", min_value=0, value=25000)
    tax = st.number_input("Tax", min_value=0, value=145)
    mpg = st.number_input("MPG", min_value=0.0, value=55.0)
    engineSize = st.number_input("Engine Size", min_value=0.0, value=1.5)

    
    if st.button("Predict Price"):
        
        num_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
        num_data = pd.DataFrame([[year, mileage, tax, mpg, engineSize]], columns=num_cols)
        
        
        scaled_num_data = scaler.transform(num_data)
        
        
        final_input = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
        
        
        for col, val in zip(num_cols, scaled_num_data[0]):
            if col in final_input.columns:
                final_input[col] = val
        
        
        prediction = model.predict(final_input)
        final_price = round(float(prediction[0]), 2)
        
        
        st.write("---")
        st.balloons() 
        st.success(f"💰 **Predicted Price for this Ford Car is: ${final_price:,}**")

except Exception as e:
    st.error(f"❌ Error during prediction: {e}")