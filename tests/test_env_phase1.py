# tests/test_env_phase1.py
# Phase 1 coverage: MultiDiscrete action space, action masking, realistic
# execution costs (commission/swap/variable spread), same-bar SL/TP fill
# ordering, and the R-multiple reward variant.
# Run with: python -m pytest tests/test_env_phase1.py -v

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_env import ForexTradingEnv

PIP = 0.0001


def flat_df(n=120, start=1.1000):
    prices = np.full(n, start, dtype=np.float64)
    return pd.DataFrame({
        "Open": prices,
        "High": prices + 5 * PIP,
        "Low": prices - 5 * PIP,
        "Close": prices,
        "Volume": np.ones(n),
        "rsi_14": np.full(n, 50.0),
        "atr_14": np.full(n, 10 * PIP),
    })


def jump_df(n=100, start=1.1000, move=20, jump_at=13):
    prices = np.full(n, start, dtype=np.float64)
    prices[jump_at:] = start + move * PIP
    return pd.DataFrame({
        "Open": prices,
        "High": prices + 2 * PIP,
        "Low": prices - 2 * PIP,
        "Close": prices,
        "Volume": np.ones(n),
        "rsi_14": np.full(n, 50.0),
        "atr_14": np.full(n, 10 * PIP),
    })


def make_env(df, **kwargs):
    defaults = dict(
        window_size=10,
        sl_options=[20, 30],
        tp_options=[20, 30],
        feature_columns=["rsi_14", "atr_14"],
        spread_pips=0.0,
        commission_pips=0.0,
        max_slippage_pips=0.0,
        random_start=False,
        episode_max_steps=None,
        hold_reward_weight=0.0,
        open_penalty_pips=0.0,
        time_penalty_pips=0.0,
        unrealized_delta_weight=0.0,
        # Phase 1 cost knobs default to inert so each test turns on exactly one.
        commission_per_lot_usd=0.0,
        swap_long_pips_per_day=0.0,
        swap_short_pips_per_day=0.0,
        variable_spread=None,
    )
    defaults.update(kwargs)
    return ForexTradingEnv(df=df, **defaults)


def _discrete_open_long(env, sl, tp):
    return next(
        i for i, (a, d, s, t) in enumerate(env.action_map)
        if a == "OPEN" and d == 1 and s == sl and t == tp
    )


# ---------------------------------------------------------------------------
# 1.1 — MultiDiscrete action space
# ---------------------------------------------------------------------------

def test_multidiscrete_action_space_shape():
    env = make_env(flat_df(), action_space_mode="multidiscrete",
                   sl_options=[20, 30], tp_options=[20, 30])
    assert list(env.action_space.nvec) == [3, 2, 2, 2]


def test_multidiscrete_open_hold_close_cycle():
    env = make_env(flat_df(), action_space_mode="multidiscrete",
                   sl_options=[20, 30], tp_options=[20, 30])
    env.reset(seed=0)

    # HOLD keeps us flat
    env.step([0, 0, 0, 0])
    assert env.position == 0

    # OPEN long (direction idx 1), sl_idx 0, tp_idx 1
    env.step([1, 1, 0, 1])
    assert env.position == 1

    # CLOSE
    env.step([2, 0, 0, 0])
    assert env.position == 0


# ---------------------------------------------------------------------------
# 1.2 — Action masking
# ---------------------------------------------------------------------------

def test_action_masks_multidiscrete_flat_vs_in_position():
    env = make_env(flat_df(), action_space_mode="multidiscrete",
                   sl_options=[20, 30], tp_options=[20, 30])
    env.reset(seed=0)

    m = env.action_masks()
    assert len(m) == 3 + 2 + 2 + 2
    # structural sub-space [HOLD, OPEN, CLOSE]
    assert m[0] and m[1] and not m[2], "flat: HOLD+OPEN valid, CLOSE invalid"

    env.step([1, 1, 0, 0])  # open long
    assert env.position == 1
    m2 = env.action_masks()
    assert m2[0], "HOLD always valid"
    assert not m2[1], "OPEN invalid while in a position (no flip)"
    assert m2[2], "CLOSE valid while in a position"


def test_action_masks_discrete():
    env = make_env(flat_df(), action_space_mode="discrete")
    env.reset(seed=0)
    m = env.action_masks()
    assert m[0]          # HOLD
    assert not m[1]      # CLOSE invalid while flat
    assert m[2:].any()   # some OPEN valid while flat


# ---------------------------------------------------------------------------
# 1.3 — Realistic execution costs
# ---------------------------------------------------------------------------

def test_commission_charged_at_close():
    df = jump_df()
    base = make_env(df, action_space_mode="discrete", sl_options=[20], tp_options=[30],
                    commission_per_lot_usd=0.0)
    comm = make_env(df, action_space_mode="discrete", sl_options=[20], tp_options=[30],
                    commission_per_lot_usd=7.0)

    for env in (base, comm):
        env.reset()
        a = _discrete_open_long(env, 20, 30)
        env.step(a); env.step(0); env.step(0); env.step(1)  # open, hold, hold, close

    lots = comm.lot_size_units / 100_000.0
    expected_commission = 7.0 * lots
    diff = (base.equity_usd - comm.equity_usd)
    assert diff == pytest.approx(expected_commission, rel=1e-6), (
        f"commission delta {diff} != expected {expected_commission}"
    )


def test_swap_reduces_equity_on_long_hold():
    df = flat_df(n=200)
    swap = make_env(df, action_space_mode="discrete", sl_options=[60], tp_options=[120],
                    swap_long_pips_per_day=-2.4, bar_hours=1.0)
    swap.reset()
    a = _discrete_open_long(swap, 60, 120)
    swap.step(a)
    for _ in range(48):  # hold ~2 days of H1 bars
        swap.step(0)
    swap.step(1)  # manual close

    ti = swap.last_trade_info
    assert ti["reason"] == "MANUAL_CLOSE"
    assert ti["swap_pips"] < 0, "long swap should be a financing cost here"
    # Flat price -> realized ~0; net should be dominated by negative swap.
    assert ti["net_pips"] < 0


def test_swap_uses_utc_rollover_when_timestamps_present():
    n = 100
    idx = pd.date_range("2023-01-02 18:00", periods=n, freq="h", tz="UTC")
    prices = np.full(n, 1.1000)
    df = pd.DataFrame({
        "Open": prices, "High": prices + 5 * PIP, "Low": prices - 5 * PIP,
        "Close": prices, "Volume": np.ones(n),
        "rsi_14": np.full(n, 50.0), "atr_14": np.full(n, 10 * PIP),
    }, index=idx)

    env = make_env(df, action_space_mode="discrete", sl_options=[80], tp_options=[160],
                   swap_long_pips_per_day=-3.0, window_size=10)
    env.reset()
    a = _discrete_open_long(env, 80, 160)
    env.step(a)
    entry_step = env.entry_step
    for _ in range(30):
        env.step(0)
    env.step(1)  # close
    ti = env.last_trade_info
    exit_step = ti["step"]

    hours = pd.DatetimeIndex(idx[entry_step + 1: exit_step + 1]).hour
    expected_rollovers = int((hours == 22).sum())
    assert expected_rollovers >= 1, "test should span at least one 22:00 UTC rollover"
    assert ti["swap_pips"] == pytest.approx(-3.0 * expected_rollovers)


def test_variable_spread_sampled_in_range_and_reproducible():
    df = flat_df()
    e1 = make_env(df, action_space_mode="discrete", sl_options=[20], tp_options=[30],
                  variable_spread=(0.8, 2.5), spread_pips=0.0)
    e2 = make_env(df, action_space_mode="discrete", sl_options=[20], tp_options=[30],
                  variable_spread=(0.8, 2.5), spread_pips=0.0)
    e1.reset(seed=5); e2.reset(seed=5)
    a1 = _discrete_open_long(e1, 20, 30)
    a2 = _discrete_open_long(e2, 20, 30)
    e1.step(a1); e2.step(a2)

    assert 0.8 <= e1.current_spread_pips <= 2.5
    assert e1.current_spread_pips == pytest.approx(e2.current_spread_pips), \
        "same seed must reproduce the sampled spread"


# ---------------------------------------------------------------------------
# 1.4 — Same-bar SL/TP fill ordering (closer level first)
# ---------------------------------------------------------------------------

def _both_hit_df(n=100, start=1.1000, extreme_pips=60, bar=11):
    prices = np.full(n, start, dtype=np.float64)
    high = prices + 2 * PIP
    low = prices - 2 * PIP
    high[bar] = start + extreme_pips * PIP
    low[bar] = start - extreme_pips * PIP
    return pd.DataFrame({
        "Open": prices, "High": high, "Low": low, "Close": prices,
        "Volume": np.ones(n),
        "rsi_14": np.full(n, 50.0), "atr_14": np.full(n, 10 * PIP),
    })


def test_tiebreak_closer_stop_fills_first():
    # sl=10 (close), tp=50 (far) -> SL should fill first deterministically.
    df = _both_hit_df()
    env = make_env(df, action_space_mode="discrete", sl_options=[10, 50], tp_options=[10, 50])
    env.reset()
    a = _discrete_open_long(env, 10, 50)
    env.step(a)  # opens at step 10; SL/TP check uses bar 11 (the extreme bar)
    assert env.position == 0
    assert "SL_FIRST" in env.last_trade_info["reason"]
    assert env.last_trade_info["realized_pips"] < 0


def test_tiebreak_closer_target_fills_first():
    # sl=50 (far), tp=10 (close) -> TP should fill first deterministically.
    df = _both_hit_df()
    env = make_env(df, action_space_mode="discrete", sl_options=[10, 50], tp_options=[10, 50])
    env.reset()
    a = _discrete_open_long(env, 50, 10)
    env.step(a)
    assert env.position == 0
    assert "TP_FIRST" in env.last_trade_info["reason"]
    assert env.last_trade_info["realized_pips"] > 0


# ---------------------------------------------------------------------------
# 1.5 — R-multiple reward variant
# ---------------------------------------------------------------------------

def test_r_multiple_reward_on_close():
    df = jump_df(move=20)
    env = make_env(df, action_space_mode="discrete", sl_options=[20], tp_options=[30],
                   reward_mode="r_multiple")
    env.reset()
    a = _discrete_open_long(env, 20, 30)
    env.step(a); env.step(0); env.step(0)
    out = env.step(1)  # close: +20 pips realized, sl=20 -> R = +1.0
    reward = out[1]
    assert reward == pytest.approx(1.0, abs=0.05)


def test_pnl_reward_mode_is_in_pips():
    df = jump_df(move=20)
    env = make_env(df, action_space_mode="discrete", sl_options=[20], tp_options=[30],
                   reward_mode="pnl")
    env.reset()
    a = _discrete_open_long(env, 20, 30)
    env.step(a); env.step(0); env.step(0)
    out = env.step(1)  # close: +20 pips -> reward ~ 20 (pips)
    assert out[1] == pytest.approx(20.0, abs=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
