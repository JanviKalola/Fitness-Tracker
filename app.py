import streamlit as st
import numpy as np
import pandas as pd
#import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Set Page Configuration
st.set_page_config(page_title="🏋️ Fitness Tracker", layout="wide")

#  Model Filei
MODEL_FILE = "fitness_model.pkl"

# ✅ Function to Load or Train Model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        X = np.random.rand(100, 3)  # Random synthetic data
        y = np.random.randint(1, 10, size=100)  # Random fitness scores
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
        return model

#  Load Model
model = load_model()

# Sample Training Data (Replace with actual dataset)
data = pd.DataFrame({
    "steps": np.random.randint(1000, 20000, 100),
    "heart_rate": np.random.randint(40, 180, 100),
    "calories_burned": np.random.randint(50, 5000, 100),
    "fitness_score": np.random.uniform(1, 10, 100)
})

# Features & Target
X = data[["steps", "heart_rate", "calories_burned"]]
y = data["fitness_score"]

# Train Model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

# Save Model
joblib.dump(model, "fitness_model.pkl")

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

    #Visualization: Dark Theme Prediction Graph
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
    1. **Fitness Data (fitness_data.csv)**: Includes daily activity data such as steps taken, heart rate, calories burned, and other biometric details. (900 Data)
    2. **User Activity Logs**: Collects real-time fitness tracking data from users to enhance model accuracy.
              
    **Fitness mesure** :
    1. **Stpes** : **0 t0 20000** (Best : 1000 - 5000)
    2. **Heart Rate** : **40 to 200** (Best : 70 - 79)
    3. **Calories Burned** : **0 t0 5000** (Best : 1000 - 4000)

    **Machine learning Algorithms Accuracy**:
              
    1. **Decision Tree :
        Accuracy : 0.74
                     
    2. **Naïve Bayes** :
        Accuracy : 0.60

    3. **Gradient Boosting** :
        Accuracy : 0.77
                        
    3. **Random Forest** :
        Accuracy : 0.88  (This Algorithm is used)

    6. **Logistic Regression:
        Accuracy : 0.89  (This Algorithm is used)

    **Two machine learning algorithms were employed:**
    1. **Logistic Regression**:
    - A statistical model that uses a logistic function to predict a fitness category (e.g., low, moderate, high).
    - Useful for binary or categorical classification.

    2. **Random Forest Classifier**:
    - An ensemble learning method that constructs multiple decision trees to predict fitness scores.
    - Handles complex relationships in data and improves prediction accuracy.

    **Hyperparameter Tuning Algorithms Accuracy** :

    1. **Decision Tree + Naïve Bayes** :
        Accuracy : 0.79

    2. **Gradient Boosting + Logistic Regression** :
        Accuarcy : 0.66

    3. **Random Forest + Gradient Boosting** :
        Accuracy : 0.69

    4. **Random Forest + Logistic Regression** :
        Accuracy : 0.899 (This Algorithm is used)

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
    - `seaborn`
    - `jupyter`
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
