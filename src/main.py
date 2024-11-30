import os
import logging
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Price Prediction",
    description="Machine learning project for stock price prediction",
    version="1.0.0",
)

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates and static files
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def preprocess_data(df):
    """
    Preprocess the dataframe by calculating technical indicators and target variables.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date", drop=True)
    # Calculate Relative Strength Index (RSI) with a 15-day window
    df["RSI"] = ta.rsi(df.Close, length=15)

    # Calculate Exponential Moving Averages (EMAs) with different windows
    df["EMAF"] = ta.ema(df.Close, length=20)  # Fast EMA
    df["EMAM"] = ta.ema(df.Close, length=100)  # Medium EMA
    df["EMAS"] = ta.ema(df.Close, length=150)  # Slow EMA

    # Compute target as the difference between close and open prices
    df["Target"] = df["Close"] - df["Open"]

    # Shift the target column up by one row for future prediction
    df["Target"] = df["Target"].shift(-1)

    # Create a binary class based on whether the target is positive or not
    df["TargetClass"] = [1 if df.Target[i] > 0 else 0 for i in range(len(df))]

    # Shift the close prices up by one row to get the next day's close
    df["TargetNextClose"] = df["Close"].shift(-1)

    # Remove Volume and % Change
    df = df.drop(["Volume", "Change %"], axis=1)

    # Remove rows with missing values and remove duplicates
    df = df.dropna().drop_duplicates()

    return df


def generate_sequence(df, sequence_length=30):
    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    X = []
    for j in range(9):
        X.append([])
        for i in range(sequence_length, df_scaled.shape[0]):
            X[j].append(df_scaled[i - sequence_length : i, j])

    # Move Axis
    X = np.moveaxis(X, [0], [2])
    X, yi = np.array(X), np.array(df_scaled[sequence_length:, -1])
    y = np.reshape(yi, (len(yi), 1))

    return X, y, scaler


def load_ml_model(model_path):
    """
    Load machine learning model with robust error handling.

    Args:
        model_path (str): Path to the saved model

    Returns:
        Loaded TensorFlow model or None
    """
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        model.summary()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


# Load Model
model = load_ml_model(os.getcwd() + "/models/stock_model.h5")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the main index page for stock price prediction.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=JSONResponse)
async def predict_stock_price(file: UploadFile = File(...)):
    """
    Main prediction endpoint with comprehensive error handling.
    """
    if not model:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    try:
        contents = await file.read()
        filepath = StringIO(contents.decode("utf-8"))

        test_df = pd.read_csv(filepath)
        processed_df = preprocess_data(test_df)

        X_test, y_test, scaler = generate_sequence(processed_df)
        print(f"Scaler: {scaler}")

        # Make predictions
        y_pred = model.predict(X_test)

        test_dates = test_df["Date"].iloc[-len(y_test) :]

        # Plot the predictions
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, y_test, color="black", label="Actual")
        plt.plot(test_dates, y_pred, color="green", label="Predicted")
        plt.title("Stock Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()

        # Actual and Predicted Prices
        def get_unscaled_prices(y_test_scaled, y_pred_scaled):
            # Print Shapes
            print("y_test shape:", y_test.shape)
            print("y_pred shape:", y_pred.shape)
            print("scaler.min_ shape:", scaler.min_.shape)
            print("scaler.scale_ shape:", scaler.scale_.shape)

            # Reshaping in the format: Shape(n_samples, 1)
            y_test_reshaped = y_test.reshape(-1, 1)
            y_pred_reshaped = y_pred.reshape(-1, 1)

            # No. of features
            n_features = scaler.scale_.shape[0]

            # Create a dummy array of zeros with shape (n_samples, n_features)
            dummy_input = np.zeros((len(y_pred_reshaped), n_features))
            dummy_input[:, -1] = (
                y_pred_reshaped.flatten()
            )  # Set the last column to the predictions
            dummy_input_test = np.zeros((len(y_test_reshaped), n_features))
            dummy_input_test[:, -1] = y_test_reshaped.flatten()

            # Get only the last column (target)
            y_pred_actual = scaler.inverse_transform(dummy_input)[:, -1]
            y_test_actual = scaler.inverse_transform(dummy_input_test)[:, -1]

            return y_pred_actual, y_test_actual

        y_pred_actual, y_test_actual = get_unscaled_prices(y_test, y_pred)

        def evaluate_model(y_test_actual, y_pred_actual):
            # Print some values to verify
            for x, y in zip(y_test_actual[-20:], y_pred_actual[-20:]):
                print(f"Actual: [{x.round(2)}] | Predicted: [{y.round(2)}]")
            print("=" * 100)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2score = r2_score(y_test, y_pred)

            return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2score}

        # Evaluate model performance
        metrics = evaluate_model(y_test_actual, y_pred_actual)

        test_dates_str = test_dates.dt.strftime("%Y-%m-%d").tolist()

        # Prepare data to send back to frontend
        response_data = {
            "predictions": {
                "dates": test_dates_str,
                "actual_prices": y_test_actual.tolist(),
                "predicted_prices": y_pred_actual.tolist(),
            },
            "metrics": metrics,
        }

        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
