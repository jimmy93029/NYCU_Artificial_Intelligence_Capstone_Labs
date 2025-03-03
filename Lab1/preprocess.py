# preprocess.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import os


class Config:
    """
    Configuration class to centralize all hyperparameters and settings.
    """
    # API configuration
    BINANCE_BASE_URL = "https://fapi.binance.com/fapi"
    API_VERSION = "v1"
    API_ENDPOINT = "klines"
    API_LIMIT = 1500

    # Pandas display options
    DISPLAY_MAX_ROWS = None
    DISPLAY_MAX_COLUMNS = None
    DISPLAY_WIDTH = None

    # Default data parameters
    DEFAULT_SYMBOLS = ["BTCUSDT"]
    DEFAULT_INTERVAL = "15m"
    DEFAULT_START_DATE = "2024-12-01"
    DEFAULT_END_DATE = "2025-02-01"
    DEFAULT_FILENAME = "klines_BTC.csv"

    # Date format
    DATE_FORMAT = "%Y-%m-%d"


# Configure pandas display options
pd.set_option("display.max_rows", Config.DISPLAY_MAX_ROWS)
pd.set_option("display.max_columns", Config.DISPLAY_MAX_COLUMNS)
pd.set_option("display.width", Config.DISPLAY_WIDTH)


def fetch_kline_price_data(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches K-line (candlestick) data from Binance API for a given trading pair.

    :param symbol: Trading pair (e.g., 'BTCUSDT')
    :param interval: Time interval (e.g., '1h')
    :param start_date: Start date (YYYY-MM-DD)
    :param end_date: End date (YYYY-MM-DD)
    :return: Processed DataFrame containing OHLC and taker buy volume data
    """
    api_endpoint = f"{Config.API_VERSION}/{Config.API_ENDPOINT}"
    start_time = datetime.strptime(start_date, Config.DATE_FORMAT)
    end_time = datetime.strptime(end_date, Config.DATE_FORMAT)

    price_data = []

    while start_time < end_time:
        start_time_ms = int(start_time.timestamp() * 1000)
        url = f"{Config.BINANCE_BASE_URL}/{api_endpoint}?symbol={symbol}&interval={interval}&limit={Config.API_LIMIT}&startTime={start_time_ms}"

        response = requests.get(url)
        data = response.json()

        if not data:
            break

        price_data.extend(data)

        # Update start_time to the close time of the last retrieved data entry
        last_entry = data[-1]
        last_close_time = last_entry[6]  # Close time (7th element in response data)

        start_time = datetime.fromtimestamp(last_close_time / 1000.0)

    return process_price_data(price_data, start_date, end_date)


def process_price_data(price_data: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Processes raw price data from Binance API into a structured DataFrame.

    :param price_data: Raw data retrieved from API
    :param start_date: Start date for filtering
    :param end_date: End date for filtering
    :return: Processed DataFrame with selected OHLC and taker buy volume columns
    """
    df = pd.DataFrame(price_data, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "num_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore"
    ])

    # Convert data types and remove invalid entries
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    # Convert timestamp columns to human-readable format
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    # Extract relevant columns
    df = df[["open_time", "open", "high", "low", "close", "volume",
             "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]]

    # Filter data within the specified date range
    df = df[(df["open_time"] >= pd.to_datetime(start_date)) &
            (df["open_time"] < pd.to_datetime(end_date) + timedelta(days=1))]

    return df


def save_to_drive(df: pd.DataFrame, filename: str):
    """
    Saves the given DataFrame to a CSV file.

    :param df: DataFrame to be saved
    :param filename: Name of the CSV file
    """
    # Define save path
    drive_path = f"{filename}"

    # Save DataFrame
    df.to_csv(drive_path, index=False)
    print(f"File saved to {drive_path}")


if __name__ == "__main__":
    # Load parameters from configuration
    symbols = Config.DEFAULT_SYMBOLS
    interval = Config.DEFAULT_INTERVAL
    start_date = Config.DEFAULT_START_DATE
    end_date = Config.DEFAULT_END_DATE
    filename = Config.DEFAULT_FILENAME

    # Fetch and merge data
    all_data = [fetch_kline_price_data(symbol, interval, start_date, end_date)
                for symbol in symbols]

    # Combine all symbol data
    final_df = pd.concat(all_data, ignore_index=True)

    # Save to file
    save_to_drive(final_df, filename)