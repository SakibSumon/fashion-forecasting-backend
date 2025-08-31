import pandas as pd
from prophet import Prophet
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')  
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.holtwinters import ExponentialSmoothing

def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "fashion_sales_dataset.csv")
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

def holt_winters_forecast(df, sku, target="revenue", periods=30, seasonal_periods=7):
    df_sku = df[df["sku"] == sku].rename(columns={"order_date": "ds", target: "y"})
    df_sku = df_sku.sort_values("ds")

    if df_sku.shape[0] < seasonal_periods * 2:  # need enough data
        return {"error": f"Not enough data for Holt-Winters (need at least {seasonal_periods*2} points)."}

    df_sku = df_sku.set_index("ds")

    model = ExponentialSmoothing(
        df_sku["y"],
        trend="add", 
        seasonal="add", 
        seasonal_periods=seasonal_periods
    )
    fit = model.fit()

    forecast = fit.forecast(periods)

    result = pd.DataFrame({
        "ds": pd.date_range(df_sku.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D"),
        "yhat": forecast.values,
        "yhat_lower": forecast.values * 0.9,
        "yhat_upper": forecast.values * 1.1
    })
    return result.to_dict("records")

def forecast_arima(df, sku, target="revenue", days=30):
    df_sku = df[df["sku"] == sku].sort_values("order_date")
    if df_sku.shape[0] < 30:
        return {"error": f"Not enough data for ARIMA (need at least 30 points)."}

    y = df_sku.set_index("order_date")[target]

    try:
        # Fit ARIMA model
        model = ARIMA(y, order=(1, 1, 1))
        model_fit = model.fit()

        # Use get_forecast to obtain predictions + confidence intervals
        forecast_obj = model_fit.get_forecast(steps=days)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=0.05)  # 95% confidence interval

        future_dates = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=days)

        result = []
        for i in range(days):
            result.append({
                "ds": future_dates[i],
                "yhat": forecast.iloc[i],
                "yhat_lower": conf_int.iloc[i, 0],
                "yhat_upper": conf_int.iloc[i, 1]
            })

        return result
    except Exception as e:
        return {"error": f"ARIMA model failed: {str(e)}"}

def forecast_sarima(df, sku, target="revenue", days=30, seasonal_periods=7):
    df_sku = df[df["sku"] == sku].sort_values("order_date")
    if df_sku.shape[0] < 2 * seasonal_periods:
        return {"error": f"Not enough data for SARIMA (need at least {2*seasonal_periods} points)."}

    y = df_sku.set_index("order_date")[target]

    try:
        model = SARIMAX(
            y,
            order=(1, 1, 1),                 # ARIMA part
            seasonal_order=(1, 1, 1, seasonal_periods),  # seasonal part
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)

        forecast_obj = model_fit.get_forecast(steps=days)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=0.05)

        future_dates = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=days)

        result = []
        for i in range(days):
            result.append({
                "ds": future_dates[i],
                "yhat": forecast.iloc[i],
                "yhat_lower": conf_int.iloc[i, 0],
                "yhat_upper": conf_int.iloc[i, 1]
            })

        return result
    except Exception as e:
        return {"error": f"SARIMA model failed: {str(e)}"}

def evaluate_models(df, sku, target="revenue", test_size=30):
    df_sku = df[df["sku"] == sku].sort_values("order_date")
    if df_sku.shape[0] < test_size + 30:
        return {"error": f"Not enough data for evaluation (need at least {test_size + 30} points)."}
    
    # Split data into train and test
    train = df_sku.iloc[:-test_size]
    test = df_sku.iloc[-test_size:]
    
    results = {}
    
    # Evaluate Prophet
    try:
        prophet_forecast = forecast_prophet(train, sku, target, test_size)
        if "error" not in prophet_forecast:
            actuals = test[target].values
            predicted = [f["yhat"] for f in prophet_forecast]
            results["prophet"] = calculate_metrics(actuals, predicted)
    except:
        results["prophet"] = {"error": "Evaluation failed"}
    
    # Evaluate Custom model
    try:
        custom_forecast = forecast_custom(train, sku, target, test_size)
        if "error" not in custom_forecast:
            actuals = test[target].values
            predicted = [f["yhat"] for f in custom_forecast]
            results["custom"] = calculate_metrics(actuals, predicted)
    except:
        results["custom"] = {"error": "Evaluation failed"}
    
    # Evaluate Holt-Winters
    try:
        hw_forecast = holt_winters_forecast(train, sku, target, test_size)
        if "error" not in hw_forecast:
            actuals = test[target].values
            predicted = [f["yhat"] for f in hw_forecast]
            results["holt_winters"] = calculate_metrics(actuals, predicted)
    except:
        results["holt_winters"] = {"error": "Evaluation failed"}

     # Evaluate ARIMA
    try:
        arima_forecast = forecast_arima(train, sku, target, test_size)
        if "error" not in arima_forecast:
            actuals = test[target].values
            predicted = [f["yhat"] for f in arima_forecast]
            results["arima"] = calculate_metrics(actuals, predicted)
    except:
        results["arima"] = {"error": "Evaluation failed"}


     # SARIMA
    try:
        sarima_forecast = forecast_sarima(train, sku, target, test_size)
        if "error" not in sarima_forecast:
            actuals = test[target].values
            predicted = [f["yhat"] for f in sarima_forecast]
            results["sarima"] = calculate_metrics(actuals, predicted)
    except:
        results["sarima"] = {"error": "Evaluation failed"}
    
    return results

   


def calculate_metrics(actual, predicted):
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": math.sqrt(mean_squared_error(actual, predicted)),
        "mape": np.mean(np.abs((actual - predicted) / actual)) * 100
    }

def forecast_sku(df, sku, target="revenue", days=30):
    prophet_result = forecast_prophet(df, sku, target, days)
    custom_result = forecast_custom(df, sku, target, days)
    hw_result = holt_winters_forecast(df, sku, target, days)
    arima_result = forecast_arima(df, sku, target, days)
    sarima_result = forecast_sarima(df, sku, target, days)

    return {
        "prophet": prophet_result,
        "custom": custom_result,
        "holt_winters": hw_result,
        "arima": arima_result,
        "sarima": sarima_result
    }




