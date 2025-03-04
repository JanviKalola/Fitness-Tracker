# Fitness Tracker App


This project focuses on tracking and predicting an individual's fitness level based on daily activity data. Using machine learning techniques, specifically Logistic Regression and Random Forest classifiers, the project provides fitness insights and recommendations that can be valuable for health-conscious individuals, trainers, and researchers.

## Table of Contents
- [Overview](#overview)
- [Datasets](#datasets)
- [Features](#features)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Dependencies](#dependencies)
- [Future Work](#future-work)

## Overview

Fitness tracking is essential for maintaining a healthy lifestyle. This project leverages machine learning to analyze activity metrics such as steps taken, heart rate, and calories burned to estimate an individual's fitness score. The model provides insights that can help users improve their fitness levels.

## Datasets

The project utilizes two primary datasets:

1. **Fitness Data (`fitness_data.csv`)**: Contains daily activity data such as steps taken, heart rate, and calories burned.
2. **User Activity Logs**: Collects real-time fitness tracking data to enhance model accuracy.

Both datasets are crucial in training the machine learning models for accurate fitness score predictions.

## Features

Key features engineered and utilized in the modeling process include:

- **Steps Taken**: The number of steps walked in a day.
- **Heart Rate**: The average heart rate (bpm) throughout the day.
- **Calories Burned**: The total calories burned based on activity level.
- **Fitness Score**: A computed score representing overall fitness based on activity data.

## Machine Learning Models

Three machine learning approaches were implemented:

1. **Logistic Regression**:
   - A classification model that predicts fitness categories (e.g., low, moderate, high).
   - Suitable for categorical classification tasks.

2. **Random Forest Regressor**:
   - An ensemble learning method that constructs multiple decision trees.
   - Provides a continuous fitness score prediction based on activity data.

3. **Hyperparameter Tuning**:
   - **Grid Search**: Systematically evaluates multiple hyperparameter combinations.
   - **Random Search**: Selects hyperparameters randomly for a more efficient search.

Both models were trained and optimized using hyperparameter tuning techniques to improve prediction accuracy.

## Results

- **Accuracy**:
  - The Random Forest model outperformed Logistic Regression in predicting fitness scores.

- **Evaluation Metrics**:
  - **Mean Squared Error (MSE)**: Used to measure prediction error.
  - **R² Score**: Assesses how well the model explains the variance in the fitness score.

- **Hyperparameter Tuning Results**:
  - Best parameters were selected using Grid Search and Random Search for the Random Forest model, leading to improved accuracy.

## Dependencies

Ensure the following Python libraries are installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyter`
- `streamlit`
- `os`
- `joblib`

You can install these dependencies using `pip install -r requirements.txt`.

## Future Work

### Incorporate Wearable Device Data:
Integrate real-time data from smartwatches and fitness bands.

### Expand Feature Set:
Include additional metrics like sleep patterns, hydration levels, and stress levels.

### Advanced Modeling Techniques:
Explore deep learning models like LSTMs and neural networks for time-series fitness data.

### Mobile App Integration:
Develop a mobile application for users to track fitness scores conveniently.

This project provides an essential step toward personalized fitness tracking using machine learning. 🚀
