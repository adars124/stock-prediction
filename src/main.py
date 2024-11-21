from io import StringIO
from pathlib import Path

import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, Query

# Configure logging
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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the main index page for stock price prediction.
    """
    return templates.TemplateResponse("index.html", {"request": request})


def calculate_rsi(data, periods=14):
    """
    Calculate Relative Strength Index (RSI) with more robust error handling.

    Args:
        data (pd.Series): Price data series
        periods (int): Number of periods for RSI calculation

    Returns:
        pd.Series: RSI values
    """
    try:
        delta = data.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)

        roll_up = up.ewm(com=periods - 1, adjust=False).mean()
        roll_down = down.ewm(com=periods - 1, adjust=False).mean()

        rs = roll_up / (roll_down + 1e-10)  # Prevent division by zero
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi
    except Exception as e:
        logger.error(f"Error in RSI calculation: {e}")
        raise


def calculate_ema(data, span):
    """
    Calculate Exponential Moving Average with error handling.

    Args:
        data (pd.Series): Input data series
        span (int): Span for EMA calculation

    Returns:
        pd.Series: EMA series
    """
    try:
        return data.ewm(span=span, adjust=False).mean()
    except Exception as e:
        logger.error(f"Error in EMA calculation: {e}")
        raise


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


# Load model with better path handling
model_path = Path.home() / "Downloads" / "project-docs" / "models" / "model.h5"
model = load_ml_model(str(model_path))


def read_csv(filepath, start_date=None, end_date=None):
    """
    Read CSV file with improved date filtering and error handling.

    Args:
        filepath (str or io.StringIO): Input file path or StringIO object
        start_date (str, optional): Start date for filtering
        end_date (str, optional): End date for filtering

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    try:
        df = pd.read_csv(filepath, parse_dates=["Date"])
        df.set_index("Date", inplace=True)

        if start_date and end_date:
            df = df.loc[start_date:end_date]

        return df
    except Exception as e:
        logger.error(f"CSV reading error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


def preprocess_data(df, sequence_length=30):
    """
    Enhanced data preprocessing with more feature engineering.

    Args:
        df (pd.DataFrame): Input dataframe
        sequence_length (int): Sequence length for sliding window

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Calculate technical indicators
    df["RSI"] = calculate_rsi(df["Close"], periods=15)
    df["EMAF"] = calculate_ema(df["Close"], span=20)
    df["EMAM"] = calculate_ema(df["Close"], span=100)
    df["EMAS"] = calculate_ema(df["Close"], span=150)

    # Target calculation
    df["Target"] = df["Close"].shift(-sequence_length) - df["Close"]
    df["TargetClass"] = (df["Target"] > 0).astype(int)

    return df.dropna()


def generate_sequence(df, sequence_length=30, test_size=0.2):
    """
    Generate sequences with train/test split.

    Args:
        df (pd.DataFrame): Preprocessed dataframe
        sequence_length (int): Length of input sequences
        test_size (float): Proportion of test data

    Returns:
        Tuple of training and testing data with scalers
    """
    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "RSI",
        "EMAF",
        "EMAM",
        "EMAS",
        "Target",
    ]

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i : i + sequence_length])
        y.append(df[features[-1]][i + sequence_length])

    X, y = np.array(X), np.array(y)

    # Split data
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler


@app.post("/predict", response_class=JSONResponse)
async def predict_stock_price(
    file: UploadFile = File(...),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    """
    Main prediction endpoint with comprehensive error handling.
    """
    if not model:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    try:
        contents = await file.read()
        csv_data = StringIO(contents.decode("utf-8"))

        # Read and preprocess data
        df = read_csv(csv_data, start_date, end_date)
        preprocessed_df = preprocess_data(df)

        # Generate sequences
        X_train, X_test, y_train, y_test, scaler = generate_sequence(preprocessed_df)

        # Optional: Retrain or fine-tune model here if needed
        # model.fit(X_train, y_train, epochs=10, validation_split=0.2)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            "predictions": y_pred.tolist(),
            "actual_values": y_test.tolist(),
            "metrics": {
                "mean_squared_error": float(mse),
                "mean_absolute_error": float(mae),
                "r2_score": float(r2),
            },
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def get_model_info():
    """
    Endpoint to retrieve model information.
    """
    if not model:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    return {
        "model_type": str(type(model)),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
    }
