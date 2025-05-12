import pandas as pd
import holidays
from prophet import Prophet
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.prophet import log_model
import glob

# Find the spark df csv file
file_path = "output/stock_data/part-00000*"

# Use glob to find files matching the pattern
matching_files = glob.glob(file_path)

# Load the data
df = pd.read_csv(matching_files[0])

# Prepare data for Prophet
df_pandas = df.rename(columns={
    "Date": "ds", 
    "Close_AAPL": "y",
    "Volume_AAPL": "Volume_AAPL",
    "volatility_7_days": "volatility_7_days"
})

# Convert "ds" to datetime
df_pandas["ds"] = pd.to_datetime(df_pandas["ds"]).dt.tz_localize(None)

# Include US holidays
us_holidays = holidays.US(years=[2023, 2024, 2025])
holidays_df = pd.DataFrame({
    'ds': pd.to_datetime(list(us_holidays.keys())).tz_localize(None),
    'holiday': list(us_holidays.values())
})

# Start MLflow experiment
mlflow.set_experiment("apple_stock_forecasting")

with mlflow.start_run():
    # Initialize Prophet model
    model = Prophet(
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        uncertainty_samples=5000
    )

    # Add additional regressors
    model.add_regressor("Volume_AAPL")
    model.add_regressor("volatility_7_days")

    # Train the model
    model.fit(df_pandas)

    # Define future dataframe
    future = model.make_future_dataframe(periods=90)
    future["Volume_AAPL"] = df_pandas["Volume_AAPL"].mean()  # simple way of getting some future values
    future["volatility_7_days"] = df_pandas["volatility_7_days"].mean() # just mean of older, works bad for long forecast

    # Make predictions
    forecast = model.predict(future)

    # Log the Prophet model with MLflow
    mlflow.prophet.log_model(model, "prophet_model")

    # Log parameters
    mlflow.log_param("yearly_seasonality", True)
    mlflow.log_param("weekly_seasonality", True)
    mlflow.log_param("daily_seasonality", True)
    mlflow.log_param("uncertainty_samples", 5000)

    # Compute RMSE as an evaluation metric
    rmse = forecast['yhat'].sub(df_pandas['y']).pow(2).mean() ** 0.5
    mlflow.log_metric("rmse", rmse)

    # Save and log predictions
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("/app/output/predictions/prophet_predictions.csv", index=False)
    mlflow.log_artifact("/app/output/predictions/prophet_predictions.csv")
    
