import pandas as pd
from prophet import Prophet
import os
import numpy as np
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.holtwinters import ExponentialSmoothing

def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "clothing_sales.csv")
    df = pd.read_csv(file_path, parse_dates=["order_date"])
    df = df.groupby(["order_date", "sku"]).agg({
        "revenue": "sum",
        "quantity": "sum"
    }).reset_index()
    return df


def forecast_prophet(df, sku, target="revenue", days=30):
    df_sku = df[df["sku"] == sku].rename(columns={"order_date": "ds", target: "y"})
    if df_sku.shape[0] < 10:
        return {"error": "Not enough data for this SKU."}

    model = Prophet()
    model.fit(df_sku)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)
    return forecast.to_dict("records")


def forecast_custom(df, sku, target="revenue", days=30, window=30):
    df_sku = df[df["sku"] == sku].sort_values("order_date")
    if df_sku.shape[0] < window:
        return {"error": f"Not enough data for this SKU (need at least {window} points)."}

    # Select last `window` points
    y = df_sku[target].tail(window).values
    X = np.arange(len(y)).reshape(-1, 1)

    # Fit simple linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Forecast future days
    future_X = np.arange(len(y), len(y) + days).reshape(-1, 1)
    y_pred = model.predict(future_X)

    future_dates = pd.date_range(start=df_sku["order_date"].max() + pd.Timedelta(days=1), periods=days)
    forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat": y_pred,
        "yhat_lower": y_pred * 0.9,
        "yhat_upper": y_pred * 1.1
    })
    return forecast.to_dict("records")

def holt_winters_forecast(df, periods=30, seasonal_periods=7):
    df = df.set_index("ds")
    
    model = ExponentialSmoothing(
        df["y"],
        trend="add", 
        seasonal="add", 
        seasonal_periods=seasonal_periods
    )
    fit = model.fit()
    
    forecast = fit.forecast(periods)
    
    result = pd.DataFrame({
        "ds": pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D"),
        "yhat": forecast.values
    })
    return result


def forecast_sku(df, sku, target="revenue", days=30):
    prophet_result = forecast_prophet(df, sku, target, days)
    custom_result = forecast_custom(df, sku, target, days)

    return {
        "prophet": prophet_result,
        "custom": custom_result
    }



