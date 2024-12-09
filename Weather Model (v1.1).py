import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import plotly.express as px
from retry_requests import retry
from sklearn.metrics import mean_absolute_error, r2_score, brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
import datetime
import threading
import sys
import time


# Constants
K = 5  # Number of Fourier terms
CACHE_EXPIRY = -1  # Cache expiration for requests
file_path = "predictions_with_metrics.csv"  # Path to save metrics and predictions
N_ITER = 50  # Number of iterations for Bayesian optimization


########################################################################################################################
# Utility Functions
########################################################################################################################

def get_location(zip_code):
    """Retrieve latitude, longitude, and timezone from a ZIP code."""
    geolocator = Nominatim(user_agent="B.Ritt")
    location = geolocator.geocode(f'{zip_code}, USA')
    if not location:
        raise ValueError("Invalid ZIP code.")
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lng=location.longitude, lat=location.latitude)
    utc_offset = None
    if timezone_str:
        local_tz = ZoneInfo(timezone_str)
        utc_time = datetime.datetime.utcnow()
        local_time = utc_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(local_tz)
        utc_offset = local_time.utcoffset().total_seconds() / 3600
    return location.latitude, location.longitude, timezone_str, utc_offset


def setup_openmeteo_client():
    """Initialize the Open-Meteo API client with caching and retry capabilities."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=CACHE_EXPIRY)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def process_weather_data(response, timezone_str):
    """Process raw weather data into a pandas DataFrame."""
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature": hourly_temperature_2m
    }
    df = pd.DataFrame(data)
    df['date'] = df['date'].dt.tz_convert(timezone_str)
    return df.dropna()


def add_fourier_features(df, timestamp_col, k=K):
    for i in range(1, k + 1):
        df[f'sin_{i}'] = np.sin(2 * np.pi * i * df[timestamp_col] / (24 * 3600 * 365.25))
        df[f'cos_{i}'] = np.cos(2 * np.pi * i * df[timestamp_col] / (24 * 3600 * 365.25))
    return df


def add_features(df, k=K):
    """Add cyclical features (Fourier transforms) and time-based variables."""
    df['hour'] = df['date'].dt.hour
    df['day_of_year'] = df['date'].dt.dayofyear
    df['timestamp'] = df['date'].apply(lambda x: x.timestamp())
    df = add_fourier_features(df, 'timestamp', k)
    return df


def show_thinking_indicator():
    """Display a 'thinking' indicator with three dots cycling on the same line."""
    symbols = ['', '.', '..', '...']
    while not stop_indicator.is_set():
        for symbol in symbols:
            if stop_indicator.is_set():
                break
            sys.stdout.write(f'\rProcessing{symbol}')  # Overwrite the current line
            sys.stdout.flush()
            time.sleep(1)
    sys.stdout.write('\r')  # Clear the line at the end
    sys.stdout.flush()


def log_metrics(metrics, model_name):
    print(f"Metrics for {model_name}:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")


def save_predictions_with_metrics(metrics, future_df, file_path):
    """Save model metrics and future predictions to a CSV file."""
    # Convert metrics to a DataFrame
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

    # Add predictions DataFrame (future_df)
    future_df = future_df.rename(columns={'final_pred': 'Prediction'})
    future_df = future_df[['date', 'Prediction']]

    # Align metrics and predictions by filling blanks in the middle column
    max_rows = max(len(metrics_df), len(future_df))
    metrics_df = metrics_df.reindex(range(max_rows)).fillna("")
    future_df = future_df.reindex(range(max_rows)).fillna("")
    blank_col = pd.DataFrame([""] * max_rows, columns=[""])

    # Combine into a single DataFrame
    combined_df = pd.concat([metrics_df, blank_col, future_df], axis=1)
    combined_df.columns = ["Metric", "Value", "", "date", "Prediction"]

    # Convert numeric columns back to numeric (ensures proper formatting)
    combined_df["Value"] = pd.to_numeric(combined_df["Value"], errors="coerce")
    combined_df["Prediction"] = pd.to_numeric(combined_df["Prediction"], errors="coerce")

    # Save to CSV with UTF-8 encoding
    combined_df.to_csv(file_path, index=False, float_format="%.2f", encoding="utf-8")
    print(f"CSV file saved to: {file_path}")

########################################################################################################################
# Modeling Functions
########################################################################################################################
def train_xgboost_model(X, y, param_space, cv_splits, n_iter=N_ITER):
    """Train an XGBoost model with Bayesian hyperparameter optimization."""
    print(f"Starting Bayesian Optimization for {n_iter} iterations.")

    # Start the thinking indicator in a separate thread
    global stop_indicator
    stop_indicator = threading.Event()
    indicator_thread = threading.Thread(target=show_thinking_indicator)
    indicator_thread.start()

    try:
        opt = BayesSearchCV(
            estimator=XGBRegressor(random_state=42),
            search_spaces=param_space,
            n_iter=n_iter,
            scoring='neg_mean_squared_error',
            cv=cv_splits,
            n_jobs=-1,
            return_train_score=True,
        )

        opt.fit(X, y)

    finally:
        # Stop the thinking indicator
        stop_indicator.set()
        indicator_thread.join()

    print("Optimization complete.")
    return opt.best_params_, opt.best_estimator_


def train_logistic_regression(filtered_df, xg_features, pred_features, bottom_threshold, upper_threshold,
                              final_xg_model, final_xg_extreme_model):
    """Train logistic regression to identify extreme temperatures and generate weighted predictions."""
    # Create extreme temperature labels
    filtered_df['low'] = (filtered_df['temperature'] <= bottom_threshold).astype(int)
    filtered_df['high'] = (filtered_df['temperature'] >= upper_threshold).astype(int)
    filtered_df['extreme_temp'] = filtered_df['high'] + filtered_df['low']

    # Train logistic regression model
    logistic_model = LogisticRegression()  # Use consistent variable name
    logistic_model.fit(xg_features.iloc[:, 2:], filtered_df['extreme_temp'])

    # Evaluate model performance
    ex_probabilities = logistic_model.predict_proba(xg_features.iloc[:, 2:])[:, 1]
    final_auc = roc_auc_score(filtered_df['extreme_temp'], ex_probabilities)
    final_brier = brier_score_loss(filtered_df['extreme_temp'], ex_probabilities)
    metrics = {
        "AUC-ROC": final_auc,
        "Brier Score": final_brier
    }
    log_metrics(metrics, "Logistic Regression")

    # Generate predictions for future data
    extreme_probabilities = logistic_model.predict_proba(pred_features.iloc[:, 2:])[:, 1]

    # Weighted average of regular and extreme predictions
    xg_y_future_pred = final_xg_model.predict(pred_features)
    xg_y_extreme_future_pred = final_xg_extreme_model.predict(pred_features)
    weighted_predictions = (extreme_probabilities * xg_y_extreme_future_pred +
                            (1 - extreme_probabilities) * xg_y_future_pred)

    # Return results and the logistic model
    return weighted_predictions, metrics, logistic_model

########################################################################################################################
# Main Workflow
########################################################################################################################

def main():
    # 1. Set up parameters and retrieve location data
    zip_code = input('ZIP code for US city: ')
    latitude, longitude, timezone_str, utc_offset = get_location(zip_code)

    # 2. Initialize Open-Meteo client and fetch weather data
    client = setup_openmeteo_client()
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2010-01-01",
        "end_date": datetime.date.today().isoformat(),
        "hourly": ["temperature_2m"],
        "temperature_unit": "fahrenheit"
    }
    try:
        responses = client.weather_api(url="https://archive-api.open-meteo.com/v1/archive", params=params)
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return

    filtered_df = process_weather_data(responses[0], timezone_str)

    # 3. Add features to the data
    filtered_df = add_features(filtered_df)

    # 4. Train primary XGBoost model
    split_num = (filtered_df['date'].dt.year.max() - filtered_df['date'].dt.year.min()) - 1
    tscv = TimeSeriesSplit(n_splits=split_num)

    xg_features = pd.concat([
        filtered_df[['hour', 'day_of_year']],
        filtered_df.filter(like='sin_'),
        filtered_df.filter(like='cos_')
    ], axis=1)

    xg_X = xg_features.values
    xg_y = filtered_df['temperature'].values

    xg_param_space = {
        'n_estimators': list(range(100, 201, 25)),
        'max_depth': list(range(3, 16, 2)),
        'learning_rate': (0.01, 0.3, 'log-uniform'),
        'subsample': (0.5, 1.0, 'uniform'),
        'colsample_bytree': (0.5, 1.0, 'uniform'),
        'gamma': (0, 5, 'uniform'),
        'min_child_weight': (1, 10, 'uniform'),
        'reg_alpha': (1e-5, 1.0, 'log-uniform'),
        'reg_lambda': (1e-5, 1.0, 'log-uniform')
    }

    print("Training Primary XGBoost model")
    xg_best_params, final_xg_model = train_xgboost_model(xg_X, xg_y, xg_param_space, tscv)

    # Evaluate the model
    xg_y_pred = final_xg_model.predict(xg_X)
    final_xg_r2 = r2_score(xg_y, xg_y_pred)
    final_xg_mae = mean_absolute_error(xg_y, xg_y_pred)
    print("Best Parameters:", xg_best_params)
    primary_xgb_metrics = {
        "R²": final_xg_r2,
        "MAE": final_xg_mae
    }
    log_metrics(primary_xgb_metrics, "Primary XGBoost")

    # 5. Train extreme XGBoost model
    bottom_threshold = np.percentile(filtered_df['temperature'], 2.5)
    temp_extremes_low = filtered_df[filtered_df['temperature'] <= bottom_threshold]

    upper_threshold = np.percentile(filtered_df['temperature'], 97.5)
    temp_extremes_high = filtered_df[filtered_df['temperature'] >= upper_threshold]

    temp_extremes_df = pd.concat([
        temp_extremes_low,
        temp_extremes_high,
    ], axis=0)

    temp_extremes_df = temp_extremes_df.sort_values(by='date').reset_index(drop=True)

    xg_features_extreme = pd.concat([
        temp_extremes_df[['hour', 'day_of_year']],
        temp_extremes_df.filter(like='sin_'),
        temp_extremes_df.filter(like='cos_')
    ], axis=1)

    xg_X_extreme = xg_features_extreme.values
    xg_y_extreme = temp_extremes_df['temperature'].values

    print("Training Extreme XGBoost model")
    xg_extreme_best_params, final_xg_extreme_model = train_xgboost_model(xg_X_extreme, xg_y_extreme, xg_param_space, tscv)

    # Evaluate the model
    xg_y_extreme_pred = final_xg_extreme_model.predict(xg_X_extreme)
    final_xg_extreme_r2 = r2_score(xg_y_extreme, xg_y_extreme_pred)
    final_xg_extreme_mae = mean_absolute_error(xg_y_extreme, xg_y_extreme_pred)
    print("Best Parameters:", xg_extreme_best_params)
    extreme_xgb_metrics = {
        "R²": final_xg_extreme_r2,
        "MAE": final_xg_extreme_mae
    }
    log_metrics(extreme_xgb_metrics, "Extreme XGBoost")

    # 6. Train Logistic Regression Meta Model
    start_prediction_date = filtered_df['date'].max() + pd.Timedelta(hours=1)
    future_dates = pd.date_range(start=start_prediction_date, periods=168, freq='h', tz=filtered_df['date'].dt.tz)
    future_df = pd.DataFrame({'date': future_dates})
    future_df['hour'] = future_df['date'].dt.hour
    future_df['day_of_year'] = future_df['date'].dt.day_of_year
    future_df['timestamp'] = future_df['date'].apply(lambda x: x.timestamp())
    future_df = add_fourier_features(future_df, 'timestamp', K)
    pred_features = pd.concat([
        future_df[['hour', 'day_of_year']],
        future_df.filter(like='sin_'),
        future_df.filter(like='cos_')
    ], axis=1)

    weighted_predictions, logit_metrics, logistic_model = train_logistic_regression(
        filtered_df=filtered_df,
        xg_features=xg_features,
        pred_features=pred_features,
        bottom_threshold=bottom_threshold,
        upper_threshold=upper_threshold,
        final_xg_model=final_xg_model,
        final_xg_extreme_model=final_xg_extreme_model
    )

    future_df['final_pred'] = weighted_predictions

    # 8. Aggregate all metrics
    all_metrics = {
        "Primary XGBoost": primary_xgb_metrics,
        "Extreme XGBoost": extreme_xgb_metrics,
        "Logistic Regression": logit_metrics
    }
    aggregated_metrics = {f"{model} - {metric}": value for model, model_metrics in all_metrics.items() for metric, value in model_metrics.items()}

    # Save metrics and predictions
    save_predictions_with_metrics(aggregated_metrics, future_df, file_path)

    # 9. Plot the final predictions
    fig = px.line(future_df, x='date', y='final_pred',
                  title=f"Next Week's Temperature Predictions ({zip_code})",
                  labels={'final_pred': 'Temperature (°F)', 'date': 'Date'})
    fig.show()


if __name__ == "__main__":
    main()
