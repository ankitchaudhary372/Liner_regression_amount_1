import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("linear_model.pkl")

# Streamlit UI Setup
st.set_page_config(page_title="Day vs Amount Predictor", page_icon="📈")
st.title("📊 Linear Regression Prediction App")
st.write("Enter a day to predict the corresponding amount using our trained ML model.")

# User input for day
day = st.number_input("Enter Day:", min_value=1, max_value=500, step=1)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(np.array([[day]]))[0]
    st.success(f"Predicted Amount for Day {day} is: {prediction:.2f}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Ankit Kumar**")

