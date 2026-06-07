# tests/test_phase2.py
# Phase 2 coverage: multi-timeframe features (2.1), regime tags (2.2),
# data QC + fingerprint (2.3), and feature-schema persistence (2.4).
# Run with: python -m pytest tests/test_phase2.py -v

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    assert_feature_schema,
    build_bot_config,
    load_feature_columns,
    save_run_config,
)
from data_qc import DataQCError, compute_fingerprint, run_data_qc
from indicators import load_and_preprocess_data

PIP = 0.0001

PHASE2_FEATURES = [
    "rsi_14_h4",
    "rsi_14_d1",
    "ma_spread_d1",
    "dist_to_d1_high_20",
    "adx_14",
    "vol_regime",
    "range_regime",
]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _synthetic_h1(n=4000, start="2021-01-04 00:00", seed=0):
    """Weekday-only H1 OHLCV with a gentle random walk (valid OHLC)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    idx = idx[idx.dayofweek < 5]  # drop weekends like a real FX feed
    n = len(idx)
    steps = rng.normal(0, 5 * PIP, size=n).cumsum()
    close = 1.1000 + steps
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + rng.uniform(0, 5 * PIP, size=n)
    low = np.minimum(open_, close) - rng.uniform(0, 5 * PIP, size=n)
    vol = rng.uniform(100, 5000, size=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


@pytest.fixture(scope="module")
def real_df_and_cols():
    config = build_bot_config()
    if not config.data.dataset_path.exists():
        pytest.skip(f"Local dataset not present: {config.data.dataset_path}")
    df, cols = load_and_preprocess_data(config.data.dataset_path)
    return df, cols


# ---------------------------------------------------------------------------
# 2.1 — Multi-timeframe features
# ---------------------------------------------------------------------------

def test_multi_tf_columns_present_and_finite(real_df_and_cols):
    df, cols = real_df_and_cols
    for c in ("rsi_14_h4", "rsi_14_d1", "ma_spread_d1", "dist_to_d1_high_20"):
        assert c in cols, f"{c} missing from feature_cols"
        assert c in df.columns
        assert np.isfinite(df[c].to_numpy()).all(), f"{c} has NaN/Inf after dropna"


def test_h4_d1_rsi_in_valid_range(real_df_and_cols):
    df, _ = real_df_and_cols
    for c in ("rsi_14_h4", "rsi_14_d1"):
        assert df[c].between(0, 100).all(), f"{c} outside [0,100]"


def test_multi_tf_no_lookahead():
    """Truncating future bars must not change a past bar's HTF features.

    If any H4/D1 feature peeked at future data, recomputing on a shorter series
    would change earlier values. We assert byte-stability instead.
    """
    df = _synthetic_h1(n=6000)
    # Feed the raw frame through the indicator pipeline without QC fingerprint IO.
    import indicators

    full, _ = _preprocess_inmemory(df.copy())
    cut = int(len(df) * 0.8)
    truncated, _ = _preprocess_inmemory(df.iloc[:cut].copy())

    common = full.index.intersection(truncated.index)
    # Compare a safely interior slice (skip the very last few bars of the
    # truncated frame, whose trailing ffill naturally has less right-context).
    common = common[:-5]
    assert len(common) > 100
    for c in ("rsi_14_h4", "rsi_14_d1", "ma_spread_d1", "dist_to_d1_high_20"):
        a = full.loc[common, c].to_numpy()
        b = truncated.loc[common, c].to_numpy()
        assert np.allclose(a, b, atol=1e-9, equal_nan=True), f"{c} changed when future bars removed (lookahead!)"


def _preprocess_inmemory(df):
    """Run the indicator pipeline on an in-memory frame (no CSV/QC IO).

    Mirrors load_and_preprocess_data's feature engineering by reusing its
    helpers so the no-lookahead test exercises the real alignment code.
    """
    import indicators as ind
    from ta.momentum import RSIIndicator
    from ta.trend import ADXIndicator
    from ta.volatility import AverageTrueRange

    df["rsi_14"] = RSIIndicator(df["Close"], window=14).rsi()
    df["atr_14"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    h4 = ind._align_to_h1(ind._h4_features(df), df.index)
    d1 = ind._align_to_h1(ind._d1_features(df), df.index)
    df["rsi_14_h4"] = h4["rsi_14_h4"]
    df["rsi_14_d1"] = d1["rsi_14_d1"]
    df["ma_spread_d1"] = d1["ma_spread_d1"]
    df["dist_to_d1_high_20"] = (df["Close"] - d1["d1_high_20"]) / d1["d1_atr"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    return df, None


# ---------------------------------------------------------------------------
# 2.2 — Regime tags
# ---------------------------------------------------------------------------

def test_regime_columns_present_and_bounded(real_df_and_cols):
    df, cols = real_df_and_cols
    for c in ("adx_14", "vol_regime", "range_regime"):
        assert c in cols and c in df.columns
        assert np.isfinite(df[c].to_numpy()).all()
    assert df["vol_regime"].between(0, 1).all()
    assert set(np.unique(df["range_regime"].to_numpy())).issubset({0.0, 1.0})
    assert (df["adx_14"] >= 0).all()


def test_range_regime_logic(real_df_and_cols):
    df, _ = real_df_and_cols
    expected = ((df["adx_14"] < 20) & (df["vol_regime"] < 0.5)).astype(float)
    assert (df["range_regime"].to_numpy() == expected.to_numpy()).all()


# ---------------------------------------------------------------------------
# 2.3 — Data QC + fingerprint
# ---------------------------------------------------------------------------

def test_qc_rejects_duplicate_timestamps():
    df = _synthetic_h1(n=200)
    bad = pd.concat([df, df.iloc[[100]]]).sort_index()
    with pytest.raises(DataQCError, match="Duplicate"):
        run_data_qc(bad)


def test_qc_rejects_non_monotonic_index():
    df = _synthetic_h1(n=200)
    reordered = df.iloc[list(range(50)) + [80] + list(range(50, 80)) + list(range(81, len(df)))]
    with pytest.raises(DataQCError, match="monotonic"):
        run_data_qc(reordered)


def test_qc_rejects_bad_ohlc():
    df = _synthetic_h1(n=200)
    df.iloc[10, df.columns.get_loc("High")] = df.iloc[10]["Low"] - 1.0  # High < Low
    with pytest.raises(DataQCError, match="OHLC"):
        run_data_qc(df)


def test_qc_flags_weekday_gap_softly(caplog):
    df = _synthetic_h1(n=300)
    # Drop a midday weekday block to create a >4h hole.
    drop = df.index[(df.index.dayofweek == 2) & (df.index.hour >= 9) & (df.index.hour <= 16)][:6]
    holed = df.drop(index=drop)
    report = run_data_qc(holed)  # must NOT raise
    assert len(report["weekday_gaps"]) >= 1


def test_qc_does_not_flag_normal_weekend_gap():
    # Synthetic feed already drops weekends, so every Fri->Mon gap (~49h) must
    # be excused, not flagged as a weekday hole.
    df = _synthetic_h1(n=800)
    report = run_data_qc(df)
    assert report["weekday_gaps"] == [], f"weekend gaps wrongly flagged: {report['weekday_gaps'][:3]}"


def test_qc_logs_nonpositive_volume_without_failing():
    df = _synthetic_h1(n=200)
    df.iloc[5:10, df.columns.get_loc("Volume")] = 0.0
    report = run_data_qc(df)
    assert report["volume"]["non_positive_count"] == 5
    assert report["volume"]["non_positive_pct"] == pytest.approx(100 * 5 / len(df), rel=1e-3)


def test_fingerprint_is_deterministic_and_sensitive():
    df = _synthetic_h1(n=300)
    fp1 = compute_fingerprint(df)
    fp2 = compute_fingerprint(df.copy())
    assert fp1["price_sha256"] == fp2["price_sha256"]
    assert fp1["row_count"] == len(df)

    changed = df.copy()
    changed.iloc[0, changed.columns.get_loc("Close")] += 1e-6
    assert compute_fingerprint(changed)["price_sha256"] != fp1["price_sha256"]


def test_fingerprint_persisted_to_disk(tmp_path):
    df = _synthetic_h1(n=300)
    path = tmp_path / "dataset_fingerprint.json"
    report = run_data_qc(df, fingerprint_path=path)
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["price_sha256"] == report["fingerprint"]["price_sha256"]
    assert {"row_count", "first_timestamp", "last_timestamp", "price_sha256"} <= set(on_disk)


# ---------------------------------------------------------------------------
# 2.4 — Feature-schema persistence
# ---------------------------------------------------------------------------

def test_schema_roundtrip(tmp_path):
    config = build_bot_config(output_dir=tmp_path)
    cols = ["rsi_14", "atr_14", "adx_14"]
    path = tmp_path / "model_config.json"
    save_run_config(config, path, feature_columns=cols)
    assert load_feature_columns(path) == cols


def test_schema_absent_returns_none(tmp_path):
    config = build_bot_config(output_dir=tmp_path)
    path = tmp_path / "no_schema.json"
    save_run_config(config, path)  # no feature_columns
    assert load_feature_columns(path) is None


def test_assert_feature_schema_passes_on_match():
    cols = ["a", "b", "c"]
    assert_feature_schema(cols, list(cols))  # no raise
    assert_feature_schema(None, cols)        # backward compat: no saved schema


def test_assert_feature_schema_raises_on_drift():
    with pytest.raises(ValueError, match="schema mismatch"):
        assert_feature_schema(["a", "b", "c"], ["a", "c", "b"])  # reordered
    with pytest.raises(ValueError, match="schema mismatch"):
        assert_feature_schema(["a", "b"], ["a", "b", "c"])       # added column


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
