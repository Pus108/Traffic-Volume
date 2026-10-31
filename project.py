import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv(r"E:\data science\Predictive Analysis\Metro_Interstate_Traffic_Volume.csv")
print(df)
print(df.info())
print(df.describe())
print(df.isnull().sum().sum())
df.columns = df.columns.str.strip()
# Convert date_time into datetime format
df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

#Convert numeric columns
numeric_cols = ["temp", "rain_1h", "snow_1h","clouds_all", "traffic_volume"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows without target or valid time
df.dropna(subset=["traffic_volume", "date_time"], inplace=True)
# Create hour feature
df["hour"] = df["date_time"].dt.hour
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(df["traffic_volume"], bins=30, kde=True)
plt.title("Distribution of Traffic Volume")
plt.xlabel("Traffic Volume (vehicles/hour)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df["traffic_volume"])
plt.title("Traffic Volume - Outlier Detection")
plt.xlabel("Traffic Volume (vehicles/hour)")
plt.tight_layout()
plt.show()

hourly_traffic = df.groupby("hour")["traffic_volume"].mean()

plt.figure(figsize=(9, 5))
sns.lineplot(x=hourly_traffic.index, y=hourly_traffic.values,
             marker="o")
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Traffic Volume")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()


corr_cols =["temp", "rain_1h", "snow_1h","clouds_all", "hour", "traffic_volume"]
corr=df[corr_cols].corr()
plt.figure(figsize=(9, 6))
sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f",vmin=-1,vmax=1)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
target_corr = corr["traffic_volume"].drop("traffic_volume").sort_values()

plt.figure(figsize=(8, 5))
target_corr.plot(kind="barh")
plt.title("Feature Correlation with Traffic Volume")
plt.xlabel("Pearson Correlation")
plt.ylabel("Feature")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()

# Print correlation values
print("\nCorrelation with traffic_volume:")
print(target_corr.sort_values(ascending=False))


# Machine Learning
df["day"] = df["date_time"].dt.day
df["month"] = df["date_time"].dt.month
df["day_of_week"] = df["date_time"].dt.dayofweek
df["year"] = df["date_time"].dt.year

x=df[['temp','rain_1h','snow_1h','clouds_all','hour','day','month','day_of_week','year']]
y=df['traffic_volume']
x=x.fillna(x.median())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Linear Regression Result: ")
mae=mean_absolute_error(y_test,y_pred)
print("MAE of LR: ",mae)
mse=mean_squared_error(y_test,y_pred)
print("MSE of LR: ",mse)
rmse=np.sqrt(mse)
print("RMSE of LR: ",rmse)
r2=r2_score(y_test,y_pred)
print("R2 of LR: ",r2)



#RandomForest

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100,max_depth=15,random_state=42,n_jobs=-1)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)
print("Random Forest Results")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

#XGBoost

from xgboost import XGBRegressor
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
    n_jobs=-1
)
xgb_model.fit(x_train, y_train)
xgb_pred = xgb_model.predict(x_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(xgb_mse)
xgb_r2 = r2_score(y_test, xgb_pred)
print("XGBoost Results")
print("MAE:", xgb_mae)
print("MSE:", xgb_mse)
print("RMSE:", xgb_rmse)
print("R2 Score:", xgb_r2)

#Model Compression

results = pd.DataFrame({"Model": ["Linear Regression","Random Forest","XGBoost"],
    "MAE": [
        mae,
        rf_mae,
        xgb_mae
    ],
    "MSE": [
        mse,
        rf_mse,
        xgb_mse
    ],
    "RMSE": [
        rmse,
        rf_rmse,
        xgb_rmse
    ],
    "R2 Score": [
        r2,
        rf_r2,
        xgb_r2
    ]
})

print(results)
best_model = results.loc[results["RMSE"].idxmin()]
print("\nBest Model:", best_model["Model"])
print("Best RMSE:", best_model["RMSE"])
print("Best R2 Score:", best_model["R2 Score"])

plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label="Actual", marker="o")
plt.plot(y_pred[:100], label="Linear Regression")
plt.plot(rf_pred[:100], label="Random Forest")
plt.plot(xgb_pred[:100], label="XGBoost")
plt.title("Actual vs Predicted Traffic Volume")
plt.xlabel("Test Samples")
plt.ylabel("Traffic Volume")
plt.legend()
plt.tight_layout()
plt.show()



