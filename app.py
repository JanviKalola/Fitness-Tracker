import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Set Page Configuration
st.set_page_config(page_title="🏋️ Fitness Tracker", layout="wide")

# Model File Path
MODEL_FILE = "fitness_model.pkl"

# ✅ Function to Load or Train Model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    else:
        X = np.random.rand(100, 3)  # Random synthetic data
        y = np.random.randint(1, 10, size=100)  # Random fitness scores
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save model using pickle
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)

        return model

# Load Model
model = load_model()

# Initialize Session State for Storing Predictions
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame(columns=["🏃 Steps", "❤️ Heart Rate (BPM)", "🔥 Calories Burned", "🎯 Predicted Score"])

#  Home Page UI
def show_home():
    st.title("🏋️ Fitness Tracker")
    st.subheader("📊 Prediction History")

    # Sidebar Input
    st.sidebar.header("🏃 Enter Your Activity Data")

    steps = st.sidebar.number_input("🚶 Steps", min_value=0, max_value=20000, value=5000, step=100)
    heart_rate = st.sidebar.number_input("❤️ Heart Rate", min_value=40, max_value=200, value=75, step=1)
    calories = st.sidebar.number_input("🔥 Calories Burned", min_value=0, max_value=5000, value=200, step=10)

    # Predict Button
    if st.sidebar.button("🔍 Predict Fitness Score"):
        input_features = np.array([[steps, heart_rate, calories]])
        prediction = model.predict(input_features)
        st.sidebar.success(f"🏆 Estimated Fitness Score: {prediction[0]:.2f}")

        # Store the Prediction in Session State
        new_data = pd.DataFrame({
            "🏃 Steps": [steps],
            "❤️ Heart Rate (BPM)": [heart_rate],
            "🔥 Calories Burned": [calories],
            "🎯 Predicted Score": [round(prediction[0], 2)]
        })

        # Append new row to the top, keeping only the latest 10 predictions
        st.session_state.prediction_history = pd.concat([new_data, st.session_state.prediction_history], ignore_index=True).head(10)

    # Display Prediction Table
    st.dataframe(st.session_state.prediction_history, height=250, use_container_width=True)

    # Visualization: Prediction Trend Graph
    if not st.session_state.prediction_history.empty:
        st.subheader("📈 Fitness Score Trend")
        plt.style.use("dark_background")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(st.session_state.prediction_history.index, 
                st.session_state.prediction_history["🎯 Predicted Score"], 
                marker="o", linestyle="-", color="cyan", markersize=8, linewidth=2)

        ax.set_facecolor("#121212")
        fig.patch.set_facecolor("#121212")

        ax.set_title("Predicted Fitness Score Over Time", color="white", fontsize=14, fontweight="bold")
        ax.set_xlabel("Prediction Instance", color="lightgray", fontsize=12)
        ax.set_ylabel("Fitness Score", color="lightgray", fontsize=12)
        ax.tick_params(colors="lightgray", size=10)

        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

        st.pyplot(fig)

# About Page
def show_about():
    st.title("ℹ️ About Us")
    st.write("""
    ## 🏋️ Fitness Tracking App  
    This app predicts your fitness score based on **steps**, **heart rate**, and **calories burned**.  
    - 🔍 Uses **Machine Learning** (Random Forest) to estimate fitness scores.  
    - 📊 Displays past predictions & fitness trends.  
    - 🚀 Built with **Streamlit** & **Scikit-Learn**.  

    **Developed By:**  
    - Janvi Kalola  
    - 📧 Contact: janvikalola1703@gmail.com  
    """)

# Navigation System
if "page" not in st.session_state:
    st.session_state.page = "Home"

col1, col2 = st.columns([1, 12])

with col1:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"
with col2:
    if st.button("ℹ️ About Us"):
        st.session_state.page = "About Us"

# Page Navigation Logic
if st.session_state.page == "Home":
   show_home()
elif st.session_state.page == "About Us":
    show_about()
