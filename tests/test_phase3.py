# tests/test_phase3.py
# Phase 3 coverage for the parts that do NOT need the heavy RL stack:
#   3.3 baselines (run through the real ForexTradingEnv cost model),
#   3.6 statistical tests (t-test, bootstrap Sharpe CI, deflated Sharpe),
#   3.7/3.8 date slicing used by holdout / regime evaluators.
# The SB3-dependent runners (train_agent EvalCallback, tune, bakeoff) are
# verified separately by py_compile; importing SB3 here would make conftest skip
# the whole module on a data-only checkout.
# Run with: python -m pytest tests/test_phase3.py -v

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baselines
import evaluation as ev
from config import build_bot_config
from model_eval import slice_by_date

PIP = 0.0001


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_featured_df(n=400, trend_pips_per_bar=0.5):
    """Trending OHLC frame with the columns the env + baselines need."""
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.1000 + np.arange(n) * trend_pips_per_bar * PIP
    df = pd.DataFrame({
        "Open": close,
        "High": close + 5 * PIP,
        "Low": close - 5 * PIP,
        "Close": close,
        "Volume": np.ones(n),
        "rsi_14": np.full(n, 55.0),
        "atr_14": np.full(n, 10 * PIP),
    }, index=idx)
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()
    return df.dropna()


def small_config():
    cfg = build_bot_config()
    # Shrink the window so a few-hundred-bar synthetic frame is plenty.
    from dataclasses import replace
    return replace(cfg, env=replace(cfg.env, window_size=10, min_episode_steps=50))


FEATURE_COLS = ["rsi_14", "atr_14"]


# ---------------------------------------------------------------------------
# 3.6 — statistical tests
# ---------------------------------------------------------------------------

def test_returns_ttest_detects_positive_drift():
    rng = np.random.default_rng(0)
    ret = rng.normal(0.001, 0.005, 600)
    eq = 10000 * np.cumprod(1 + ret)
    res = ev.returns_ttest(list(eq))
    assert res["t_stat"] > 0
    assert res["p_value"] is not None and res["p_value"] < 0.05
    assert res["n"] == 599


def test_returns_ttest_degenerate_inputs():
    assert ev.returns_ttest([10000.0])["p_value"] is None
    assert ev.returns_ttest([10000.0, 10000.0, 10000.0])["p_value"] is None  # zero variance


def test_bootstrap_sharpe_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    r = list(rng.normal(0.3, 1.0, 200))
    out = ev.bootstrap_sharpe_ci(r, n_resamples=2000, seed=7)
    assert out["ci_low"] <= out["sharpe"] <= out["ci_high"]
    assert out["n_trades"] == 200
    # too few points -> CI undefined, not a crash
    assert ev.bootstrap_sharpe_ci([0.1, 0.2])["ci_low"] is None


def test_bootstrap_sharpe_ci_reproducible():
    r = list(np.random.default_rng(2).normal(0.2, 1.0, 150))
    a = ev.bootstrap_sharpe_ci(r, n_resamples=1500, seed=99)
    b = ev.bootstrap_sharpe_ci(r, n_resamples=1500, seed=99)
    assert a == b


def test_deflated_sharpe_decreases_with_more_trials():
    rng = np.random.default_rng(3)
    r = list(rng.normal(0.25, 1.0, 250))
    dsr1 = ev.deflated_sharpe_ratio(r, n_trials=1)["dsr"]
    dsr50 = ev.deflated_sharpe_ratio(r, n_trials=50)["dsr"]
    # More configurations tried -> a higher bar -> lower deflated Sharpe.
    assert dsr1 >= dsr50
    assert 0.0 <= dsr50 <= 1.0


def test_statistical_report_bundles_all_three():
    rng = np.random.default_rng(4)
    eq = 10000 * np.cumprod(1 + rng.normal(0.0005, 0.005, 300))
    trades = [{"r_multiple": float(x), "net_pips": float(x) * 20} for x in rng.normal(0.2, 1.0, 80)]
    rep = ev.statistical_report(list(eq), trades, n_trials=10, n_resamples=500)
    assert set(rep) >= {"returns_ttest", "bootstrap_sharpe_ci", "deflated_sharpe", "n_trials"}
    assert rep["deflated_sharpe"]["n_trials"] == 10


def test_r_multiples_fallback_to_net_pips():
    trades = [{"net_pips": 10.0}, {"r_multiple": 0.5, "net_pips": 9.0}]
    assert ev.r_multiples_from_trades(trades) == [10.0, 0.5]


# ---------------------------------------------------------------------------
# 3.3 — baselines through the real env
# ---------------------------------------------------------------------------

def test_action_helpers_multidiscrete_and_discrete():
    from dataclasses import replace
    cfg = small_config()
    df = make_featured_df(120)
    env_md = baselines.make_env(df, FEATURE_COLS, cfg, random_start=False)
    a = baselines.open_action(env_md, direction=1, sl_idx=0, tp_idx=2)
    assert list(a) == [1, 1, 0, 2]
    assert list(baselines.hold_action(env_md)) == [0, 0, 0, 0]
    assert list(baselines.close_action(env_md)) == [2, 0, 0, 0]

    cfg_d = replace(cfg, env=replace(cfg.env, action_space_mode="discrete"))
    env_d = baselines.make_env(df, FEATURE_COLS, cfg_d, random_start=False)
    idx = baselines.open_action(env_d, direction=1, sl_idx=0, tp_idx=2)
    # Decoding that index back via the action_map must round-trip to OPEN long.
    kind, direction, sl, tp = env_d.action_map[idx]
    assert kind == "OPEN" and direction == 1
    assert sl == float(env_d.sl_options[0]) and tp == float(env_d.tp_options[2])


def test_evaluate_baselines_returns_all_policies():
    cfg = small_config()
    df = make_featured_df(400)
    results = baselines.evaluate_baselines(df, FEATURE_COLS, cfg, seed=42)
    assert set(results) == {"random", "buy_and_hold", "ma_crossover"}
    for name, m in results.items():
        assert {"sharpe", "n_trades", "final_equity", "max_dd"} <= set(m)
        assert m["n_trades"] >= 0


def test_buy_and_hold_takes_at_least_one_trade_on_trend():
    cfg = small_config()
    df = make_featured_df(400, trend_pips_per_bar=1.0)
    results = baselines.evaluate_baselines(df, FEATURE_COLS, cfg, seed=1)
    assert results["buy_and_hold"]["n_trades"] >= 1


def test_random_baseline_reproducible():
    cfg = small_config()
    df = make_featured_df(300)
    env1 = baselines.make_env(df, FEATURE_COLS, cfg, random_start=False)
    env2 = baselines.make_env(df, FEATURE_COLS, cfg, random_start=False)
    out1 = baselines.run_baseline(baselines.random_policy, env1, df, seed=5)
    out2 = baselines.run_baseline(baselines.random_policy, env2, df, seed=5)
    assert out1[1] == out2[1]  # identical final equity


# ---------------------------------------------------------------------------
# 3.7 / 3.8 — date slicing
# ---------------------------------------------------------------------------

def test_slice_by_date_inclusive_start_exclusive_end():
    idx = pd.date_range("2024-01-01", periods=240, freq="1h", tz="UTC")
    df = pd.DataFrame({"Close": np.arange(240)}, index=idx)
    sl = slice_by_date(df, start="2024-01-05", end="2024-01-06")
    assert sl.index.min() >= pd.Timestamp("2024-01-05", tz="UTC")
    assert sl.index.max() < pd.Timestamp("2024-01-06", tz="UTC")
    assert len(sl) == 24


def test_slice_by_date_open_bounds_and_requires_datetimeindex():
    idx = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    df = pd.DataFrame({"Close": np.arange(48)}, index=idx)
    assert len(slice_by_date(df, start="2024-01-02")) == 24
    assert len(slice_by_date(df)) == 48
    with pytest.raises(ValueError):
        slice_by_date(df.reset_index(drop=True), start="2024-01-02")
