import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler



# Load the trained model
model = tf.keras.models.load_model("nhl_model.h5")

# Page title
st.title("NHL Game Outcome Predictor")

# SELECTBOX component:
team = st.selectbox("Select Your Team", ["Toronto Maple Leafs", "Ottawa Senators", "Boston Bruins", "Montreal Canadiens"])
st.write(f"You selected: {team}")

st.write("Enter team stats below to predict if the team will WIN or LOSE the game.")

# Streamlit Inputs
goals = st.slider("Goals For", 0, 10, 3)
shots = st.slider("Shots For", 0, 50, 25)
fenwick_for = st.slider("Fenwick For (FF)", 0, 60, 30) # Shots without blocked shots
corsi_for = st.slider("Corsi For (CF)", 0, 80, 40) # All shots

f_percent = st.number_input("Fenwick % (FF%)", min_value=0.0, max_value=100.0, value=50.0)
s_percent = st.number_input("Shots % (SF%)", min_value=0.0, max_value=100.0, value=50.0)

# Button & Prediction:
if st.button("Predict Game Result"):

    # Preparing the input
    user_input = [[goals, shots, fenwick_for, corsi_for, f_percent, s_percent]]

    # Convert to NumPy array
    user_input = np.array(user_input, dtype=float)

    # Prediction
    prediction = model.predict(user_input)
    outcome = "WIN" if prediction[0][0] > 0.5 else "LOSS"

    # Metric:
    st.metric(label="Predicted Outcome", value=outcome)
    st.success("This prediction is based on offensive performance.")