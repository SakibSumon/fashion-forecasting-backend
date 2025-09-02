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
from statsmodels.tsa.stattools import adfuller

def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "fashion_sales_dataset.csv")
    df = pd.read_csv(file_path, parse_dates=["order_date"])
    df = df.groupby(["order_date", "sku"]).agg({
        "revenue": "sum",
        "quantity": "sum"
    }).reset_index()
    return df

def fast_prophet_hyperparameters(df_sku):
    """Fast Prophet parameter tuning with sensible defaults"""
    # Quick heuristic-based tuning instead of grid search
    volatility = df_sku['y'].pct_change().std()
    
    if volatility > 0.3:  # High volatility
        return {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0}
    elif volatility > 0.1:  # Medium volatility
        return {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 1.0}
    else:  # Low volatility
        return {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.1}

def forecast_prophet(df, sku, target="revenue", days=30):
    df_sku = df[df["sku"] == sku].rename(columns={"order_date": "ds", target: "y"})
    if df_sku.shape[0] < 10:
        return {"error": "Not enough data for this SKU."}

    # Fast parameter tuning
    best_params = fast_prophet_hyperparameters(df_sku)
    
    model = Prophet(**best_params, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_sku)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)
    return forecast.to_dict("records")

def forecast_custom(df, sku, target="revenue", days=30):
    """Fast custom model with fixed window"""
    df_sku = df[df["sku"] == sku].sort_values("order_date")
    
    # Use fixed window size for speed
    window = min(30, len(df_sku) - 1)
    if df_sku.shape[0] < window:
        return {"error": f"Not enough data for this SKU (need at least {window} points)."}

    y = df_sku[target].tail(window).values
    X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

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

def fast_holt_winters_parameters(df_sku):
    """Fast Holt-Winters parameter selection"""
    y = df_sku.set_index("ds")["y"]
    
    # Simple heuristic: if data shows clear seasonality, use multiplicative
    # Otherwise use additive
    if len(y) > 14:  # Need enough data to check seasonality
        seasonal_strength = (y[-14:].std() / y.std()) if y.std() > 0 else 0
        if seasonal_strength > 1.2:
            return {'trend': 'add', 'seasonal': 'mul'}
    
    return {'trend': 'add', 'seasonal': 'add'}

def holt_winters_forecast(df, sku, target="revenue", periods=30, seasonal_periods=7):
    df_sku = df[df["sku"] == sku].rename(columns={"order_date": "ds", target: "y"})
    df_sku = df_sku.sort_values("ds")

    if df_sku.shape[0] < seasonal_periods * 2:
        return {"error": f"Not enough data for Holt-Winters (need at least {seasonal_periods*2} points)."}

    # Fast parameter selection
    best_params = fast_holt_winters_parameters(df_sku)
    
    df_sku = df_sku.set_index("ds")
    model = ExponentialSmoothing(
        df_sku["y"],
        trend=best_params['trend'],
        seasonal=best_params['seasonal'],
        seasonal_periods=seasonal_periods
    )
    fit = model.fit(optimized=True)  # Use optimized fitting

    forecast = fit.forecast(periods)

    result = pd.DataFrame({
        "ds": pd.date_range(df_sku.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D"),
        "yhat": forecast.values,
        "yhat_lower": forecast.values * 0.9,
        "yhat_upper": forecast.values * 1.1
    })
    return result.to_dict("records")

def fast_arima_params(y):
    """Fast ARIMA parameter selection using simple rules"""
    # Check stationarity
    try:
        adf_result = adfuller(y)
        is_stationary = adf_result[1] <= 0.05
    except:
        is_stationary = False
    
    # Simple rules-based parameter selection
    if is_stationary:
        return (1, 0, 1)  # Stationary data
    else:
        return (1, 1, 1)  # Non-stationary data

def forecast_arima(df, sku, target="revenue", days=30):
    """Fast ARIMA with simple parameter selection"""
    df_sku = df[df["sku"] == sku].sort_values("order_date")
    if df_sku.shape[0] < 30:
        return {"error": f"Not enough data for ARIMA (need at least 30 points)."}

    y = df_sku.set_index("order_date")[target]

    try:
        # Fast parameter selection
        best_order = fast_arima_params(y)
        
        model = ARIMA(y, order=best_order)
        model_fit = model.fit()

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
        return {"error": f"ARIMA model failed: {str(e)}"}

def forecast_sarima(df, sku, target="revenue", days=30, seasonal_periods=7):
    """Fast SARIMA with fixed parameters"""
    df_sku = df[df["sku"] == sku].sort_values("order_date")
    if df_sku.shape[0] < 2 * seasonal_periods:
        return {"error": f"Not enough data for SARIMA (need at least {2*seasonal_periods} points)."}

    y = df_sku.set_index("order_date")[target]

    try:
        # Use fixed parameters for speed - (1,1,1) x (1,1,1,7) works well for many cases
        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal_periods),
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

def evaluate_models_fast(df, sku, target="revenue", test_size=30):
    """Fast evaluation with limited model testing"""
    df_sku = df[df["sku"] == sku].sort_values("order_date")
    if df_sku.shape[0] < test_size + 30:
        return {"error": f"Not enough data for evaluation (need at least {test_size + 30} points)."}
    
    train = df_sku.iloc[:-test_size]
    test = df_sku.iloc[-test_size:]
    
    results = {}
    
    # Only evaluate 2-3 fastest models for speed
    models_to_test = ['prophet', 'custom', 'holt_winters']
    
    if 'prophet' in models_to_test:
        try:
            prophet_forecast = forecast_prophet(train, sku, target, test_size)
            if "error" not in prophet_forecast:
                actuals = test[target].values
                predicted = [f["yhat"] for f in prophet_forecast]
                results["prophet"] = calculate_metrics(actuals, predicted)
        except:
            pass
    
    if 'custom' in models_to_test:
        try:
            custom_forecast = forecast_custom(train, sku, target, test_size)
            if "error" not in custom_forecast:
                actuals = test[target].values
                predicted = [f["yhat"] for f in custom_forecast]
                results["custom"] = calculate_metrics(actuals, predicted)
        except:
            pass
    
    if 'holt_winters' in models_to_test:
        try:
            hw_forecast = holt_winters_forecast(train, sku, target, test_size)
            if "error" not in hw_forecast:
                actuals = test[target].values
                predicted = [f["yhat"] for f in hw_forecast]
                results["holt_winters"] = calculate_metrics(actuals, predicted)
        except:
            pass
    
    return results

def calculate_metrics(actual, predicted):
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        mape = np.nan if np.isinf(mape) else mape
    
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": math.sqrt(mean_squared_error(actual, predicted)),
        "mape": mape
    }

def forecast_sku_fast(df, sku, target="revenue", days=30):
    """Fast forecasting - only use fastest models"""
    # Use only the fastest models for production
    prophet_result = forecast_prophet(df, sku, target, days)
    custom_result = forecast_custom(df, sku, target, days)
    hw_result = holt_winters_forecast(df, sku, target, days)
    
    return {
        "prophet": prophet_result,
        "custom": custom_result,
        "holt_winters": hw_result
    }

def forecast_sku_comprehensive(df, sku, target="revenue", days=30):
    """Comprehensive forecasting for when speed is less critical"""
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