import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "fashion_sales_dataset.csv")
    df = pd.read_csv(file_path, parse_dates=["order_date"])
    df = df.groupby(["order_date", "sku"]).agg({
        "revenue": "sum",
        "quantity": "sum"
    }).reset_index()
    return df

def tune_prophet_hyperparameters(df_sku, horizon=30, initial='180 days', period='30 days'):
    """Tune Prophet hyperparameters using grid search"""
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    mapes = []
    
    # Use historical data for parameter tuning
    cutoff_date = df_sku['ds'].max() - pd.Timedelta(days=horizon)
    train_df = df_sku[df_sku['ds'] <= cutoff_date]
    
    if len(train_df) < 60:  # Need sufficient data for tuning
        return {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'}
    
    for params in all_params:
        try:
            m = Prophet(**params)
            m.fit(train_df)
            
            # Cross-validation
            df_cv = cross_validation(m, initial=initial, period=period, horizon=f'{horizon} days')
            df_p = performance_metrics(df_cv)
            mapes.append(df_p['mape'].values[0])
        except:
            mapes.append(float('inf'))
    
    # Return best parameters
    best_params = all_params[np.argmin(mapes)]
    return best_params

def forecast_prophet(df, sku, target="revenue", days=30):
    df_sku = df[df["sku"] == sku].rename(columns={"order_date": "ds", target: "y"})
    if df_sku.shape[0] < 10:
        return {"error": "Not enough data for this SKU."}

    # Tune hyperparameters
    best_params = tune_prophet_hyperparameters(df_sku, horizon=min(30, days))
    
    model = Prophet(**best_params)
    model.fit(df_sku)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)
    return forecast.to_dict("records")

def forecast_custom(df, sku, target="revenue", days=30):
    """Enhanced custom model with automatic window size selection"""
    df_sku = df[df["sku"] == sku].sort_values("order_date")
    
    # Find optimal window size using recent data for validation
    if df_sku.shape[0] >= 60:
        # Use last 30 days for validation to find best window size
        val_size = 30
        train_val = df_sku.iloc[:-val_size]
        val_data = df_sku.iloc[-val_size:]
        
        best_window = 30  # default
        best_mae = float('inf')
        
        for window in [15, 30, 60, 90]:
            if len(train_val) >= window + val_size:
                # Train on the last 'window' points before validation period
                train_window = train_val.tail(window)
                y = train_window[target].values
                X = np.arange(len(y)).reshape(-1, 1)
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict validation period
                val_X = np.arange(len(y), len(y) + val_size).reshape(-1, 1)
                val_pred = model.predict(val_X)
                
                mae = mean_absolute_error(val_data[target].values, val_pred)
                if mae < best_mae:
                    best_mae = mae
                    best_window = window
    else:
        best_window = min(30, len(df_sku) - days) if len(df_sku) > days else len(df_sku) // 2
    
    if df_sku.shape[0] < best_window:
        return {"error": f"Not enough data for this SKU (need at least {best_window} points)."}

    # Use optimal window for final forecast
    y = df_sku[target].tail(best_window).values
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

def tune_holt_winters_parameters(df_sku, target="y", seasonal_periods=7):
    """Find best Holt-Winters parameters"""
    y = df_sku.set_index("ds")[target]
    
    best_score = float('inf')
    best_params = {}
    
    # Test different combinations
    for trend in ['add', 'mul', None]:
        for seasonal in ['add', 'mul', None]:
            if seasonal and len(y) < seasonal_periods * 2:
                continue
                
            try:
                model = ExponentialSmoothing(
                    y,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods if seasonal else None
                )
                fit = model.fit()
                # Use AIC as selection criterion
                if fit.aic < best_score:
                    best_score = fit.aic
                    best_params = {'trend': trend, 'seasonal': seasonal}
            except:
                continue
    
    return best_params if best_params else {'trend': 'add', 'seasonal': 'add'}

def holt_winters_forecast(df, sku, target="revenue", periods=30, seasonal_periods=7):
    df_sku = df[df["sku"] == sku].rename(columns={"order_date": "ds", target: "y"})
    df_sku = df_sku.sort_values("ds")

    if df_sku.shape[0] < seasonal_periods * 2:
        return {"error": f"Not enough data for Holt-Winters (need at least {seasonal_periods*2} points)."}

    # Tune parameters
    best_params = tune_holt_winters_parameters(df_sku, seasonal_periods=seasonal_periods)
    
    df_sku = df_sku.set_index("ds")
    model = ExponentialSmoothing(
        df_sku["y"],
        trend=best_params['trend'],
        seasonal=best_params['seasonal'],
        seasonal_periods=seasonal_periods if best_params['seasonal'] else None
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

def find_best_arima_params(y, max_p=3, max_d=2, max_q=3):
    """Find best ARIMA parameters using grid search with BIC"""
    best_bic = float('inf')
    best_order = (1, 1, 1)  # default
    
    # Test for stationarity to determine d
    result = adfuller(y)
    if result[1] > 0.05:
        d_range = [1, 2]  # needs differencing
    else:
        d_range = [0, 1]
    
    for p in range(max_p + 1):
        for d in d_range:
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue  # Skip invalid orders
                
                try:
                    model = ARIMA(y, order=(p, d, q))
                    model_fit = model.fit()
                    bic = model_fit.bic
                    
                    if bic < best_bic:
                        best_bic = bic
                        best_order = (p, d, q)
                except:
                    continue
    
    return best_order

def forecast_arima(df, sku, target="revenue", days=30):
    """ARIMA with automatic parameter selection"""
    df_sku = df[df["sku"] == sku].sort_values("order_date")
    if df_sku.shape[0] < 30:
        return {"error": f"Not enough data for ARIMA (need at least 30 points)."}

    y = df_sku.set_index("order_date")[target]

    try:
        # Find optimal parameters
        best_order = find_best_arima_params(y)
        
        # Fit ARIMA model with best parameters
        model = ARIMA(y, order=best_order)
        model_fit = model.fit()

        # Use get_forecast to obtain predictions + confidence intervals
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

def find_best_sarima_params(y, seasonal_periods=7, max_p=2, max_d=1, max_q=2, max_P=1, max_D=1, max_Q=1):
    """Find best SARIMA parameters using grid search with BIC"""
    best_bic = float('inf')
    best_order = (1, 1, 1)
    best_seasonal_order = (1, 1, 1, seasonal_periods)
    
    # Limited grid search for performance
    for p in [0, 1, 2]:
        for d in [0, 1]:
            for q in [0, 1, 2]:
                for P in [0, 1]:
                    for D in [0, 1]:
                        for Q in [0, 1]:
                            if p == 0 and q == 0 and P == 0 and Q == 0:
                                continue
                            
                            try:
                                model = SARIMAX(
                                    y,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, seasonal_periods),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                model_fit = model.fit(disp=False)
                                bic = model_fit.bic
                                
                                if bic < best_bic:
                                    best_bic = bic
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P, D, Q, seasonal_periods)
                            except:
                                continue
    
    return best_order, best_seasonal_order

def forecast_sarima(df, sku, target="revenue", days=30, seasonal_periods=7):
    """SARIMA with automatic parameter selection"""
    df_sku = df[df["sku"] == sku].sort_values("order_date")
    if df_sku.shape[0] < 2 * seasonal_periods:
        return {"error": f"Not enough data for SARIMA (need at least {2*seasonal_periods} points)."}

    y = df_sku.set_index("order_date")[target]

    try:
        # Find optimal parameters
        best_order, best_seasonal_order = find_best_sarima_params(y, seasonal_periods)
        
        model = SARIMAX(
            y,
            order=best_order,
            seasonal_order=best_seasonal_order,
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
        else:
            results["prophet"] = {"error": prophet_forecast["error"]}
    except Exception as e:
        results["prophet"] = {"error": f"Evaluation failed: {str(e)}"}
    
    # Evaluate Custom model
    try:
        custom_forecast = forecast_custom(train, sku, target, test_size)
        if "error" not in custom_forecast:
            actuals = test[target].values
            predicted = [f["yhat"] for f in custom_forecast]
            results["custom"] = calculate_metrics(actuals, predicted)
        else:
            results["custom"] = {"error": custom_forecast["error"]}
    except Exception as e:
        results["custom"] = {"error": f"Evaluation failed: {str(e)}"}
    
    # Evaluate Holt-Winters
    try:
        hw_forecast = holt_winters_forecast(train, sku, target, test_size)
        if "error" not in hw_forecast:
            actuals = test[target].values
            predicted = [f["yhat"] for f in hw_forecast]
            results["holt_winters"] = calculate_metrics(actuals, predicted)
        else:
            results["holt_winters"] = {"error": hw_forecast["error"]}
    except Exception as e:
        results["holt_winters"] = {"error": f"Evaluation failed: {str(e)}"}

    # Evaluate ARIMA
    try:
        arima_forecast = forecast_arima(train, sku, target, test_size)
        if "error" not in arima_forecast:
            actuals = test[target].values
            predicted = [f["yhat"] for f in arima_forecast]
            results["arima"] = calculate_metrics(actuals, predicted)
        else:
            results["arima"] = {"error": arima_forecast["error"]}
    except Exception as e:
        results["arima"] = {"error": f"Evaluation failed: {str(e)}"}

    # Evaluate SARIMA
    try:
        sarima_forecast = forecast_sarima(train, sku, target, test_size)
        if "error" not in sarima_forecast:
            actuals = test[target].values
            predicted = [f["yhat"] for f in sarima_forecast]
            results["sarima"] = calculate_metrics(actuals, predicted)
        else:
            results["sarima"] = {"error": sarima_forecast["error"]}
    except Exception as e:
        results["sarima"] = {"error": f"Evaluation failed: {str(e)}"}
    
    return results

def calculate_metrics(actual, predicted):
    # Handle division by zero in MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        mape = np.nan if np.isinf(mape) else mape
    
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": math.sqrt(mean_squared_error(actual, predicted)),
        "mape": mape
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