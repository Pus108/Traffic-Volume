# 🚦 Traffic Volume Prediction Using Machine Learning

A Machine Learning project that analyzes historical highway traffic data and predicts hourly traffic volume using weather conditions and time-based features.

The project compares multiple regression algorithms to identify the best-performing model for traffic volume prediction.

## 📌 Project Overview

Traffic congestion is a major challenge in urban transportation. Predicting traffic volume can help understand traffic patterns and support better transportation planning.

This project uses historical traffic data from the **Metro Interstate Highway (I-94)** to predict the number of vehicles passing through a highway segment per hour.

The main objective is to build and compare different Machine Learning regression models and evaluate their prediction performance.

## 🎯 Objectives

- Analyze historical traffic volume patterns.
- Understand the relationship between weather conditions and traffic.
- Perform data cleaning and preprocessing.
- Apply feature engineering to extract useful time-based information.
- Train multiple Machine Learning regression models.
- Compare model performance using evaluation metrics.
- Identify the best-performing model.

## 📂 Dataset

**Dataset:** Metro Interstate Traffic Volume

**Source:** [UCI Machine Learning Repository – Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)

Dataset details:

- 48,204 hourly traffic observations.
- Location: Westbound I-94, Minneapolis–St. Paul, Minnesota.
- Time period: October 2012 – September 2018.
- Target variable: `traffic_volume`.

### Features Used

| Feature | Description |
|---|---|
| `temp` | Temperature |
| `rain_1h` | Rainfall in the last hour |
| `snow_1h` | Snowfall in the last hour |
| `clouds_all` | Cloud coverage |
| `hour` | Hour of the day |
| `day` | Day of the month |
| `month` | Month |
| `day_of_week` | Day of the week |
| `year` | Year |

**Target Variable:** `traffic_volume` — Number of vehicles passing through the highway per hour.

## ⚙️ Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Boosting Algorithm:** XGBoost

## 🔄 Project Workflow

1. **Data Collection:** Load the Metro Interstate Traffic Volume dataset.
2. **Data Preprocessing:** Handle data types, inspect missing values, and prepare the dataset.
3. **Exploratory Data Analysis:** Analyze traffic patterns, weather conditions, and correlations.
4. **Feature Engineering:** Extract hour, day, month, day of the week, and year from the date-time column.
5. **Model Training:** Train three regression models.
6. **Model Evaluation:** Compare predictions using MAE, MSE, RMSE, and R² Score.
7. **Model Selection:** Identify the best-performing model based on evaluation results.

## 🤖 Machine Learning Algorithms

This project implements three regression algorithms:

- **Linear Regression:** Establishes a baseline relationship between input features and traffic volume.
- **Random Forest Regressor:** Uses multiple decision trees to capture complex patterns.
- **XGBoost Regressor:** Uses gradient boosting to improve predictive performance.

## 📊 Model Performance Comparison

The models were evaluated using an 80:20 training and testing split.

| Evaluation Metric | Linear Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| MAE | 1603.26 | 207.08 | 205.00 |
| MSE | 3,318,076.68 | 140,881.03 | 116,231.91 |
| RMSE | 1821.56 | 375.34 | 340.93 |
| R² Score | 16.07% | 96.44% | **97.06%** |

### 🏆 Best Performing Model: XGBoost Regressor

XGBoost achieved the best results among the three evaluated models:

- **R² Score:** 97.06%
- **RMSE:** 340.93
- **MAE:** 205.00

It achieved the lowest prediction error and the highest R² score on the test set.

## 📈 Key Learnings

- Understanding real-world traffic data and its characteristics.
- Performing exploratory data analysis on transportation datasets.
- Applying feature engineering to time-based data.
- Understanding the differences between linear and ensemble-based regression models.
- Evaluating Machine Learning models using multiple performance metrics.
- Comparing models to select an appropriate predictive approach.

## 🚀 Future Improvements

- Use chronological train-test splitting for more realistic future forecasting evaluation.
- Add lag-based and rolling traffic features.
- Explore additional weather and calendar-related features.
- Perform hyperparameter tuning.
- Experiment with LightGBM, CatBoost, and other forecasting approaches.
- Develop an interactive dashboard for traffic volume predictions and model comparison.
- Explore deployment for real-world usage.

## ⚠️ Important Note

The reported model performance is based on a random 80:20 train-test split. Since traffic observations are time-dependent, this evaluation may give optimistic results for future forecasting. A chronological split should be used to assess performance on genuinely unseen future periods.

