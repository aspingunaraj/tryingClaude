import os
import csv
import json
from datetime import datetime, timedelta

import kite_service

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "backtest_data")
STOCKS_CONFIG = os.path.join(BASE_DIR, "backtest_stocks.json")

os.makedirs(DATA_DIR, exist_ok=True)

_DEFAULT_STOCKS = [
    {"symbol": "INFY",     "exchange": "NSE", "instrument_token": 408065,  "name": "Infosys Ltd"},
    {"symbol": "HDFCBANK", "exchange": "NSE", "instrument_token": 341249,  "name": "HDFC Bank Ltd"},
    {"symbol": "ITC",      "exchange": "NSE", "instrument_token": 424961,  "name": "ITC Ltd"},
]


def load_stocks():
    if not os.path.exists(STOCKS_CONFIG):
        config = {"stocks": _DEFAULT_STOCKS}
        save_stocks(config)
        return config
    with open(STOCKS_CONFIG) as f:
        return json.load(f)


def save_stocks(config):
    with open(STOCKS_CONFIG, "w") as f:
        json.dump(config, f, indent=2)


def _data_path(symbol, exchange):
    return os.path.join(DATA_DIR, f"{exchange}_{symbol}_1min.csv")


def get_data_status(symbol, exchange):
    path = _data_path(symbol, exchange)
    if not os.path.exists(path):
        return {"exists": False, "rows": 0, "from_date": None, "to_date": None}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        rows = list(reader)
    if not rows:
        return {"exists": True, "rows": 0, "from_date": None, "to_date": None}
    return {
        "exists": True,
        "rows": len(rows),
        "from_date": rows[0][0][:10],
        "to_date":   rows[-1][0][:10],
    }


def fetch_stock_data(access_token, instrument_token, symbol, exchange, days=120):
    """
    Fetch `days` calendar days of 1-minute candles from Kite in ≤60-day chunks
    (Kite limits minute-interval requests to 60 days each) and save to CSV.
    Returns total candle count written.
    """
    now = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
    chunk_size = 60
    candles = []

    for offset in range(0, days, chunk_size):
        to_dt   = now - timedelta(days=offset)
        from_dt = now - timedelta(days=min(offset + chunk_size, days))
        chunk = kite_service.get_historical_data(
            access_token,
            instrument_token,
            from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            to_dt.strftime("%Y-%m-%d %H:%M:%S"),
            interval="minute",
        )
        # prepend so final list is chronological (oldest → newest)
        candles = chunk + candles

    path = _data_path(symbol, exchange)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerows(candles)

    return len(candles)
