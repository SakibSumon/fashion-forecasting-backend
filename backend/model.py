
import pandas as pd
from prophet import Prophet
import os

def load_data():
    # go one directory up to find dataset/
    file_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "clothing_sales.csv")
    df = pd.read_csv(file_path, parse_dates=["order_date"])
    # df = df.groupby(["order_date", "sku"])["revenue"].sum().reset_index()
    df = df.groupby(["order_date", "sku"]).agg({
        "revenue": "sum",
        "quantity": "sum"  # assuming the column is called "quantity"
    }).reset_index()
    return df

def forecast_sku(df, sku, target="revenue", days=30):
    df_sku = df[df["sku"] == sku].rename(columns={"order_date": "ds", target: "y"})
    
    if df_sku.shape[0] < 10:
        return {"error": "Not enough data for this SKU."}
    
    model = Prophet()
    model.fit(df_sku)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)
    return forecast.to_dict("records")

