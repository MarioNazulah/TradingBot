import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


def load_and_preprocess_data(csv_path: str):
    """
    Load FX data and add scale-aware technical features.

    The input feed is expected to contain a GMT/UTC timestamp column plus OHLCV.
    Timestamps are parsed as UTC explicitly so session features are not shifted by
    local machine timezone settings.
    """
    df = pd.read_csv(
        csv_path,
        dayfirst=True,
    )

    df.columns = df.columns.str.strip()
    time_col = next((c for c in df.columns if "time" in c.lower()), None)
    if time_col is None:
        raise ValueError(f"No time column found. Columns are: {list(df.columns)}")
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, utc=True)
    df = df.set_index(time_col)

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Technicals ----
    # RSI and ATR (already scale-invariant-ish)
    df["rsi_14"] = RSIIndicator(df["Close"], window=14).rsi()
    df["atr_14"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    df["ma_20"]  = df["Close"].rolling(20).mean()
    df["ma_50"]  = df["Close"].rolling(50).mean()

    # Slopes of the MAs
    df["ma_20_slope"] = df["ma_20"].diff() / df["atr_14"]
    df["ma_50_slope"] = df["ma_50"].diff() / df["atr_14"]

    # Distance of price from each MA (ATR-normalized)
    df["close_ma20_diff"] = (df["Close"] - df["ma_20"]) / df["atr_14"]
    df["close_ma50_diff"] = (df["Close"] - df["ma_50"]) / df["atr_14"]

    # MA divergence: MA20 vs MA50 (ATR-normalized)
    df["ma_spread"] = (df["ma_20"] - df["ma_50"]) / df["atr_14"]
    df["ma_spread_slope"] = df["ma_spread"].diff()

    hour = df.index.tz_convert("UTC").hour
    df["vol_ratio"] = df["atr_14"] / df["atr_14"].rolling(200).mean()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_london"]  = ((hour >= 8)  & (hour < 16)).astype(float)
    df["is_ny"]      = ((hour >= 13) & (hour < 21)).astype(float)
    df["is_overlap"] = ((hour >= 13) & (hour < 16)).astype(float)  # London/NY overlap, highest volume
    # Drop initial NaNs from indicators
    df.dropna(inplace=True)

    # Columns the AGENT should see (no raw price levels / raw MAs)
    feature_cols = [
        "rsi_14",
        "atr_14",
        "ma_20_slope",
        "ma_50_slope",
        "close_ma20_diff",
        "close_ma50_diff",
        "ma_spread",
        "ma_spread_slope",
        "vol_ratio",
        "hour_sin",
        "hour_cos",
        "is_london",
        "is_ny",
        "is_overlap",
    ]

    return df, feature_cols
