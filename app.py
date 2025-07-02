import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load model and columns
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

st.title("Water Pollutants Predictor")
st.write("Predict the levels of major pollutants based on Year and Station ID.")

# Fix: CSS must be wrapped in triple quotes
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e0f7fa;
        font-family: 'Segoe UI', sans-serif;
    }

    h1 {
        color: #006064;
        text-align: center;
    }

    .stMarkdown h3 {
        color: #00796b;
    }

    input {
        border-radius: 5px;
        padding: 5px;
        border: 1px solid #80cbc4;
    }

    .stButton>button {
        background-color: #0097a7;
        color: white;
        border-radius: 10px;
        padding: 0.5em 2em;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #006064;
        color: #e0f2f1;
    }

    .stMetric {
        font-weight: bold;
        color: #004d40;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

if st.button('Predict'):
    if not station_id:
        st.warning('Please enter a Station ID.')
    else:
        # Prepare and encode input
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align with model columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        try:
            predicted_pollutants = model.predict(input_encoded)[0]
            pollutants = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4','PO4', 'CL']

            st.subheader(f"Predicted Pollutant Levels for Station '{station_id}' in {year_input}:")
            for p, val in zip(pollutants, predicted_pollutants):
                st.metric(label=p, value=f"{val:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
