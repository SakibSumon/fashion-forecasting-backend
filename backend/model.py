import pandas as pd
from prophet import Prophet
import os
import numpy as np

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


def forecast_custom(df, sku, target="revenue", days=30, window=7):
    """Simple Moving Average Forecast"""
    df_sku = df[df["sku"] == sku].sort_values("order_date")
    if df_sku.shape[0] < window:
        return {"error": f"Not enough data for this SKU (need at least {window} points)."}

    # Compute rolling mean of last window
    last_value = df_sku[target].rolling(window=window).mean().iloc[-1]

    future_dates = pd.date_range(start=df_sku["order_date"].max() + pd.Timedelta(days=1), periods=days)
    forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat": [last_value] * days,
        "yhat_lower": [last_value * 0.9] * days,
        "yhat_upper": [last_value * 1.1] * days,
    })
    return forecast.to_dict("records")


def forecast_sku(df, sku, target="revenue", days=30):
    prophet_result = forecast_prophet(df, sku, target, days)
    custom_result = forecast_custom(df, sku, target, days)

    return {
        "prophet": prophet_result,
        "custom": custom_result
    }
