import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 📌 File Paths
DATA_FILE = "fitness_data.csv"
MODEL_FILE = "fitness_model.pkl"

# ✅ Function to Load Dataset
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        # Simulate dataset
        np.random.seed(42)
        num_samples = 500
        data = pd.DataFrame({
            'steps': np.random.randint(1000, 20000, num_samples),
            'heart_rate': np.random.randint(50, 180, num_samples),
            'calories_burned': np.random.randint(50, 5000, num_samples),
        })
        data['fitness_score'] = np.random.randint(1, 10, num_samples)
        return data

# ✅ Hyperparameter Tuning Function
def tune_hyperparameters(method="grid"):
    df = load_data()
    X = df[['steps', 'heart_rate', 'calories_burned']]
    y = df['fitness_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    }

    model = RandomForestRegressor(random_state=42)

    if method == "grid":
        search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    else:
        search = RandomizedSearchCV(model, param_grid, cv=3, n_jobs=-1, n_iter=5)

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    joblib.dump(best_model, MODEL_FILE)

    # Evaluate Model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Best Hyperparameters:", search.best_params_)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return best_model, search.best_params_, mse, r2

# ✅ Train Model
def train_model():
    print("Starting Hyperparameter Tuning...")
    best_model, best_params, mse, r2 = tune_hyperparameters("grid")
    print("Model training and evaluation completed!")

if _name_ == "_main_":
    train_model()
