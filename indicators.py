from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

from data_qc import run_data_qc


# ---------------------------------------------------------------------------
# Phase 2.1 helpers — multi-timeframe context
# ---------------------------------------------------------------------------

def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample H1 OHLC to a higher timeframe.

    label/closed="left" means a bar is stamped with the start of its window and
    carries data for that whole window. Empty windows (weekends) become NaN and
    are dropped so they never bleed into the shift/ffill alignment below.
    """
    agg = df.resample(rule, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    )
    return agg.dropna(how="any")


def _align_to_h1(htf_features: pd.DataFrame, h1_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Project higher-timeframe features back onto the H1 index, lookahead-safe.

    The HTF series is shifted by one bar before forward-filling so each H1 bar
    only ever sees the most recently *completed* higher-timeframe bar. Without
    the shift, an H1 bar early in a day would see D1/H4 indicators computed from
    that same still-open day — a lookahead leak that would invalidate every
    downstream result.
    """
    return htf_features.shift(1).reindex(h1_index, method="ffill")


def _h4_features(df: pd.DataFrame) -> pd.DataFrame:
    h4 = _resample_ohlc(df, "4h")
    out = pd.DataFrame(index=h4.index)
    out["rsi_14_h4"] = RSIIndicator(h4["Close"], window=14).rsi()
    return out


def _d1_features(df: pd.DataFrame) -> pd.DataFrame:
    d1 = _resample_ohlc(df, "1D")
    out = pd.DataFrame(index=d1.index)
    out["rsi_14_d1"] = RSIIndicator(d1["Close"], window=14).rsi()

    ma20 = d1["Close"].rolling(20).mean()
    ma50 = d1["Close"].rolling(50).mean()
    atr_d1 = AverageTrueRange(d1["High"], d1["Low"], d1["Close"], window=14).average_true_range()
    out["ma_spread_d1"] = (ma20 - ma50) / atr_d1
    # Carry the raw D1 ATR and the rolling 20-day high so the H1-scale distance
    # feature can be built after alignment (current H1 close vs prior 20d high).
    out["d1_high_20"] = d1["High"].rolling(20).max()
    out["d1_atr"] = atr_d1
    return out


def load_and_preprocess_data(
    csv_path: str | Path,
    fingerprint_path: str | Path | None = None,
    run_qc: bool = True,
):
    """
    Load FX data and add scale-aware technical features.

    The input feed is expected to contain a GMT/UTC timestamp column plus OHLCV.
    Timestamps are parsed as UTC explicitly so session features are not shifted by
    local machine timezone settings.

    Phase 2.3: a fail-fast data-QC pass runs on the raw OHLCV before any feature
    engineering and a dataset fingerprint is persisted next to the source file
    (override with ``fingerprint_path``).
    """
    csv_path = Path(csv_path)
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

    # Phase 2.3: data QC + fingerprint on the raw feed (replaces the previous
    # inline OHLC check). Fail-fast on hard violations before we spend time on
    # feature engineering. Volume==0 from a tick-volume feed is logged, not fatal.
    if run_qc:
        if fingerprint_path is None:
            fingerprint_path = csv_path.with_name(f"{csv_path.stem}_fingerprint.json")
        run_data_qc(df, fingerprint_path=fingerprint_path)

    # ---- Technicals (H1) ----
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

    # ---- Phase 2.1: multi-timeframe context (H4 + D1), aligned lookahead-safe ----
    h4 = _align_to_h1(_h4_features(df), df.index)
    d1 = _align_to_h1(_d1_features(df), df.index)
    df["rsi_14_h4"] = h4["rsi_14_h4"]
    df["rsi_14_d1"] = d1["rsi_14_d1"]
    df["ma_spread_d1"] = d1["ma_spread_d1"]
    # Current H1 close vs the prior completed 20-day high, in D1-ATR units.
    df["dist_to_d1_high_20"] = (df["Close"] - d1["d1_high_20"]) / d1["d1_atr"]

    # ---- Phase 2.2: regime tags ----
    df["adx_14"] = ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()
    # Percentile rank (0-1) of current ATR within a rolling 500-bar window.
    df["vol_regime"] = df["atr_14"].rolling(500).rank(pct=True)
    # Ranging market: weak trend AND below-median volatility.
    range_regime = ((df["adx_14"] < 20) & (df["vol_regime"] < 0.5)).astype(float)
    range_regime[df["adx_14"].isna() | df["vol_regime"].isna()] = np.nan
    df["range_regime"] = range_regime

    # ATR-normalized features divide by atr_14, which can be 0 -> Inf.
    # dropna() alone won't remove Inf, so coerce it to NaN first (review #6).
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop initial NaNs from indicators (incl. multi-TF warmup) before training.
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
        # Phase 2.1 — multi-timeframe context
        "rsi_14_h4",
        "rsi_14_d1",
        "ma_spread_d1",
        "dist_to_d1_high_20",
        # Phase 2.2 — regime tags
        "adx_14",
        "vol_regime",
        "range_regime",
    ]

    return df, feature_cols
