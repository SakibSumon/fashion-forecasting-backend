

import pandas as pd
from prophet import Prophet
import os

def load_data():
    # go one directory up to find dataset/
    file_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "warehouse_sales.csv")
    df = pd.read_csv(file_path)

    # create a datetime column from YEAR and MONTH (default day=1)
    df['date'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))

    # ensure numeric columns
    df['RETAIL SALES'] = pd.to_numeric(df['RETAIL SALES'], errors='coerce').fillna(0)

    # aggregate by date and SKU
    df = df.groupby(['date', 'SKU']).agg({
        'RETAIL SALES': 'sum'
    }).reset_index()

    return df


def forecast_sku(df, sku, target="RETAIL SALES", periods=6):
    """
    Forecast target metric for a given SKU.
    periods: number of future months to predict.
    """
    df_sku = df[df['SKU'] == sku].rename(columns={'date': 'ds', target: 'y'})

    if df_sku.shape[0] < 6:  # need enough history for Prophet
        return {"error": "Not enough data for this SKU."}

    # train model
    model = Prophet()
    model.fit(df_sku)

    # create future dataframe for monthly periods
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)

    # return only the forecast part (future months)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    return forecast.to_dict('records')
