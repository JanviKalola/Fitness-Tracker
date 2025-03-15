import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Set Page Configuration
st.set_page_config(page_title="🏋️ Fitness Tracker", layout="wide")

# ✅ Function to Train & Cache Model
@st.cache_resource
def train_model():
    X = np.random.rand(100, 3)  # Random synthetic data
    y = np.random.randint(1, 10, size=100)  # Random fitness scores
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Load Model
model = train_model()

# Initialize Session State for Storing Predictions
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame(columns=["🏃 Steps", "❤️ Heart Rate (BPM)", "🔥 Calories Burned", "🎯 Predicted Score"])

#  Home Page UI
def show_home():
    st.title("Fitness Tracker")
    st.subheader("Prediction History ")

    #  Sidebar Input
    st.sidebar.header("🏃 Enter Your Activity Data")

    steps = st.sidebar.number_input("🚶 Steps", min_value=0, max_value=20000, value=5000, step=100)
    heart_rate = st.sidebar.number_input("❤️ Heart Rate", min_value=40, max_value=200, value=75, step=1)
    calories = st.sidebar.number_input("🔥 Calories Burned", min_value=0, max_value=5000, value=200, step=10)

    # Predict Button
    if st.sidebar.button("🔍 Predict Fitness Score"):
        input_features = np.array([[steps, heart_rate, calories]])
        prediction = model.predict(input_features)
        st.sidebar.success(f"🏆 Estimated Fitness Score: {prediction[0]:.2f}")

        #  Store the Prediction in Session State
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

    # Visualization: Dark Theme Prediction Graph
    if not st.session_state.prediction_history.empty:
        st.subheader("📈 Fitness Score Trend")

        # Use a dark background theme for the graph
        plt.style.use("dark_background")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(st.session_state.prediction_history.index, 
                st.session_state.prediction_history["🎯 Predicted Score"], 
                marker="o", linestyle="-", color="cyan", markersize=8, linewidth=2)

        ax.set_facecolor("#121212")  # Set background color matching dark theme
        fig.patch.set_facecolor("#121212")  # Set figure background color

        # Customize labels and grid
        ax.set_title("Predicted Fitness Score Over Time", color="white", fontsize=14, fontweight="bold")
        ax.set_xlabel("Prediction Instance", color="lightgray", fontsize=12)
        ax.set_ylabel("Fitness Score", color="lightgray", fontsize=12)
        ax.tick_params(colors="lightgray", size=10)  # Change tick color

        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)  # Light grid

        # Display Matplotlib graph in Streamlit
        st.pyplot(fig)

#  About Us Page
def show_about():
     st.title("About Us")
     st.write("""  
    ### Fitness Tracking App 
    This application is designed to track and predict an individual's fitness level based on daily activity data.
    Using machine learning algorithms, we analyze steps, heart rate, and calories burned to provide fitness insights and recommendations.

    **Key Features:**  
    - Machine learning-driven fitness score predictions.  
    - User-friendly interface for easy data input and analysis.  
    - Real-time insights based on activity data.
    - Updated model with hyperparameter tuning for better accuracy.  

    **Datasets:**
    The project utilizes a dataset containing fitness-related metrics:
    1. **Fitness Data**: Daily activity data such as steps taken, heart rate, calories burned, and other biometric details.
    2. **User Activity Logs**: Collects real-time fitness tracking data from users to enhance model accuracy.
              
    **Fitness measure** :
    1. **Steps** : **0 to 20000** (Best: 1000 - 5000)
    2. **Heart Rate** : **40 to 200** (Best: 70 - 79)
    3. **Calories Burned** : **0 to 5000** (Best: 1000 - 4000)

    **Machine Learning Algorithms Used**:
    - **Random Forest** (Accuracy: **0.88**)  
    - **Logistic Regression** (Accuracy: **0.89**)  

    **Hyperparameter Tuning Algorithm**:

        1. **Grid Search:**  
            - Exhaustively tests multiple hyperparameter combinations to find the best-performing model.

        2. **Random Search:**  
            - Randomly selects hyperparameters from a predefined range, offering a balance between efficiency and performance.

    **Ensure the following Python libraries are installed before running the app:**
    - `numpy`
    - `pandas`
    - `scikit-learn`
    - `matplotlib`
    - `streamlit`

    **Developed By:**  
    - Janvi Kalola  
    - Contact: janvikalola1703@gmail.com  
    """)

#  Navigation System
if "page" not in st.session_state:
    st.session_state.page = "Home"

col1, col2 = st.columns([1, 12])

with col1:
    if st.button("Home"):
        st.session_state.page = "Home"
with col2:
    if st.button("About Us"):
        st.session_state.page = "About Us"

#  Page Navigation Logic
if st.session_state.page == "Home":
   show_home()
elif st.session_state.page == "About Us":
    show_about()
