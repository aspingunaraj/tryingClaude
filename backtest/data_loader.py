"""Load and prepare 1-minute OHLCV data from CSV files produced by backtest_fetcher."""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest_data")


def load_csv(symbol: str, exchange: str = "NSE") -> pd.DataFrame:
    """Load CSV, parse datetime, sort, add date column."""
    path = os.path.join(DATA_DIR, f"{exchange}_{symbol}_1min.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data file at {path}. Fetch data first.")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.rename(columns={"timestamp": "datetime"})
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # Normalise timezone: strip tz info so all arithmetic is tz-naive
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    df["date"] = df["datetime"].dt.date
    return df


def split_train_test(df: pd.DataFrame, train_days: int = 80, test_days: int = 40):
    """
    Split data into non-overlapping train / test sets by trading day.
    If fewer than train_days + test_days unique days are available the
    dataset is split in half with no data leakage.
    """
    unique_days = sorted(df["date"].unique())
    total = len(unique_days)

    if total < train_days + test_days:
        train_days = total // 2
        test_days  = total - train_days

    train_days_list = unique_days[:train_days]
    test_days_list  = unique_days[train_days: train_days + test_days]

    train_df = df[df["date"].isin(train_days_list)].reset_index(drop=True)
    test_df  = df[df["date"].isin(test_days_list)].reset_index(drop=True)

    return train_df, test_df, train_days_list, test_days_list
