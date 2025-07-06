# 🏀 NBA Game Outcome Prediction

This project applies machine learning to predict NBA game outcomes based on historical player performance data. We explore both **classification** (predicting win/loss) and **regression** (predicting point difference) tasks using various algorithms to determine the most effective approach for NBA game prediction.

## 📊 Dataset

The data was sourced from [Kaggle - NBA Games Dataset](https://www.kaggle.com/datasets/nathanlauga/nba-games).

**Dataset Details:**
- **Source**: Kaggle NBA Games Dataset by Nathan Lauga
- **File Used**: `players.csv` (contains individual player statistics)
- **Data Scope**: Historical NBA player performance metrics
- **Format**: CSV with player-level game statistics

## 🧹 Data Preprocessing

All data cleaning and feature engineering steps are implemented in the preprocessing notebook:

### `BetterPreprocessing.ipynb`

**Key Preprocessing Steps:**
- **Data Cleaning**: Handles missing values, outliers, and inconsistent data formats
- **Feature Engineering**: Creates rolling averages and performance metrics
- **Temporal Analysis**: Generates 21 different datasets based on varying lookback windows (1-21 previous games)
- **Data Aggregation**: Transforms player-level data into team-level features for game prediction
- **Output**: 21 preprocessed datasets stored separately for model training and evaluation

**Preprocessing Pipeline:**
1. Load raw player statistics from `players.csv`
2. Clean and validate data integrity
3. Calculate rolling statistics for different time windows
4. Create team-level aggregated features
5. Generate target variables (win/loss classification and point difference regression)
6. Export processed datasets for model training

### `SVMHyper.ipynb`

**Hyperparameter Tuning for SVM:**
- **Purpose**: Optimizes SVM performance by finding the best hyperparameters
- **Method**: Uses best-performing season data with fixed optimal gamma and kernel parameters
- **Search Strategy**: Extensive grid search on regularization parameter (C) values
- **Evaluation**: Compares train vs test F1 scores to identify optimal C value and detect overfitting
- **Visualization**: Plots performance curves to visualize hyperparameter impact on model performance

## 🤖 Machine Learning Models

We implemented and evaluated **6 different algorithms** for both classification and regression tasks:

### Classification Models (Win/Loss Prediction)
- **Naive Bayes**
- **k-Nearest Neighbors (kNN)**
- **Decision Tree**
- **Random Forest**
- **XGBoost**
- **Support Vector Machine (SVM)**

### Regression Models (Point Difference Prediction)
- **Linear Regression**
- **kNN Regression**
- **Decision Tree Regression**
- **Random Forest Regression**
- **XGBoost Regression**
- **SVM Regression**

🛠️ Installation & Setup
Development Environment

Neural Networks & XGBoost: Implemented in Deepnote for faster execution and GPU acceleration
Other Models: Developed in Visual Studio Code for local development and testing

## 📈 Model Evaluation

**Training Setup:**
- **Train-Test Split**: 80% training, 20% testing
- **Cross-Validation**: Implemented for robust model evaluation
- **Random State**: Fixed at 42 for reproducible results

**Classification Metrics:**
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix Analysis

**Regression Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)
