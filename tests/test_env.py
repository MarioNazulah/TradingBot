# tests/test_env.py
# Run with: python -m pytest tests/test_env.py -v

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_env import ForexTradingEnv


# -----------------------------------------------
# Minimal synthetic DataFrame factory
# -----------------------------------------------

def make_df(n=500, start_price=1.1000, pip_value=0.0001):
    """Flat price DataFrame — easy to reason about expected PnL."""
    prices = np.full(n, start_price, dtype=np.float64)
    df = pd.DataFrame({
        "Open":   prices,
        "High":   prices + 5 * pip_value,
        "Low":    prices - 5 * pip_value,
        "Close":  prices,
        "Volume": np.ones(n),
        "rsi_14": np.full(n, 50.0),
        "atr_14": np.full(n, 10 * pip_value),
    })
    return df


def make_trending_df(n=500, start_price=1.1000, pip_per_bar=1, pip_value=0.0001):
    """Trending price DataFrame — price rises pip_per_bar pips each bar."""
    prices = start_price + np.arange(n) * pip_per_bar * pip_value
    df = pd.DataFrame({
        "Open":   prices,
        "High":   prices + 5 * pip_value,
        "Low":    prices - 2 * pip_value,
        "Close":  prices,
        "Volume": np.ones(n),
        "rsi_14": np.full(n, 55.0),
        "atr_14": np.full(n, 10 * pip_value),
    })
    return df


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
    )
    defaults.update(kwargs)
    return ForexTradingEnv(df=df, **defaults)


# -----------------------------------------------
# TEST 1: Observation shape matches observation_space
# -----------------------------------------------

def test_observation_shape():
    df = make_df()
    env = make_env(df)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    assert obs.shape == env.observation_space.shape, (
        f"obs shape {obs.shape} != observation_space {env.observation_space.shape}"
    )


# -----------------------------------------------
# TEST 2: No lookahead — observation at step N
#         must only contain rows < N
# -----------------------------------------------

def test_no_lookahead():
    n = 200
    # Each bar has a unique close price so we can identify which bars are in the obs
    prices = np.arange(1, n + 1, dtype=np.float64) * 0.0001 + 1.0
    df = pd.DataFrame({
        "Open": prices, "High": prices + 0.0005,
        "Low": prices - 0.0005, "Close": prices,
        "Volume": np.ones(n),
        "rsi_14": np.arange(n, dtype=np.float64),
        "atr_14": np.full(n, 0.001),
    })

    env = make_env(df, window_size=10, feature_columns=["rsi_14", "atr_14"])
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    step = env.current_step  # should be window_size = 10

    # rsi_14 values in obs (first feature, every other value in flat obs)
    # obs = [feat0_t-10, feat1_t-10, feat0_t-9, feat1_t-9, ..., state0, state1, state2]
    n_base = env.base_num_features  # 2
    window = env.window_size        # 10
    window_part = obs[:window * n_base].reshape(window, n_base)
    rsi_in_obs = window_part[:, 0]  # rsi_14 is first feature

    # rsi values in obs should all be < step (no future data)
    # rsi_14[i] = i, so all values in obs should be < step
    for val in rsi_in_obs:
        assert val < step, f"Lookahead detected: rsi value {val} >= current_step {step}"


# -----------------------------------------------
# TEST 3: PnL accounting — long 20 pips, 0 spread → net = 20 pips
# -----------------------------------------------

def test_pnl_long_no_spread():
    pip = 0.0001
    start = 1.1000
    move = 20

    n = 100
    prices = np.full(n, start, dtype=np.float64)
    prices[13:] = start + move * pip  # jump at bar 13, not 12

    df = pd.DataFrame({
        "Open":   prices,
        "High":   prices + 2 * pip,
        "Low":    prices - 2 * pip,
        "Close":  prices,
        "Volume": np.ones(n),
        "rsi_14": np.full(n, 50.0),
        "atr_14": np.full(n, 10 * pip),
    })

    env = make_env(df, spread_pips=0.0, commission_pips=0.0,
                   max_slippage_pips=0.0, open_penalty_pips=0.0,
                   time_penalty_pips=0.0)
    env.reset()  # current_step = 10
    initial_equity = env.equity_usd

    # Use tp=30 so TP doesn't auto-trigger on the 20 pip move
    open_long_action = next(
        i for i, (a, d, sl, tp) in enumerate(env.action_map)
        if a == "OPEN" and d == 1 and sl == 20 and tp == 30
    )

    # Open at step 10 (price = 1.1000)
    env.step(open_long_action)
    assert env.position == 1
    assert env.entry_price is not None

    # Hold step 11 and 12 (price still 1.1000, no SL/TP hit)
    env.step(0)
    env.step(0)

    # Now at step 13 — close price = 1.1020
    env.step(1)  # CLOSE

    realized = env.equity_usd - initial_equity
    expected_pips = move
    expected_usd  = expected_pips * env.usd_per_pip

    assert abs(realized - expected_usd) < 0.01, (
        f"PnL mismatch: got ${realized:.4f}, expected ${expected_usd:.4f}"
    )


def test_pnl_long_with_spread():
    pip = 0.0001
    start = 1.1000
    move = 20

    n = 100
    prices = np.full(n, start, dtype=np.float64)
    prices[13:] = start + move * pip  # jump at bar 13

    df = pd.DataFrame({
        "Open":   prices,
        "High":   prices + 2 * pip,
        "Low":    prices - 2 * pip,
        "Close":  prices,
        "Volume": np.ones(n),
        "rsi_14": np.full(n, 50.0),
        "atr_14": np.full(n, 10 * pip),
    })

    env = make_env(df, spread_pips=1.0, commission_pips=0.0,
                   max_slippage_pips=0.0, open_penalty_pips=0.0,
                   time_penalty_pips=0.0)
    env.reset()
    initial_equity = env.equity_usd

    open_long_action = next(
        i for i, (a, d, sl, tp) in enumerate(env.action_map)
        if a == "OPEN" and d == 1 and sl == 20 and tp == 30
    )

    env.step(open_long_action)  # open at step 10
    env.step(0)                 # hold step 11
    env.step(0)                 # hold step 12
    env.step(1)                 # close at step 13

    realized = env.equity_usd - initial_equity
    expected_pips = move - 1.0  # 19 pips after spread
    expected_usd  = expected_pips * env.usd_per_pip

    assert abs(realized - expected_usd) < 0.01, (
        f"Spread PnL mismatch: got ${realized:.4f}, expected ${expected_usd:.4f}"
    )

# -----------------------------------------------
# TEST 5: SL hit — conservative worst case
# -----------------------------------------------

def test_sl_hit_long():
    pip = 0.0001
    start = 1.1000
    sl_pips = 20

    n = 100
    prices = np.full(n, start, dtype=np.float64)

    # Make next bar dip below SL
    sl_price = start - sl_pips * pip
    df = pd.DataFrame({
        "Open":  prices,
        "High":  prices + 2 * pip,
        "Low":   np.where(np.arange(n) == 12, sl_price - pip, prices - 2 * pip),
        "Close": prices,
        "Volume": np.ones(n),
        "rsi_14": np.full(n, 50.0),
        "atr_14": np.full(n, 10 * pip),
    })

    env = make_env(df, spread_pips=0.0, open_penalty_pips=0.0, time_penalty_pips=0.0)
    env.reset()
    initial_equity = env.equity_usd

    open_long_action = next(
        i for i, (a, d, sl, tp) in enumerate(env.action_map)
        if a == "OPEN" and d == 1 and sl == 20
    )

    # Advance to step 11 (next bar = 12 where low dips below SL)
    for _ in range(1):
        env.step(0)

    env.step(open_long_action)
    # One HOLD step — SL check happens on next bar (bar 12)
    env.step(0)

    # Position should be closed by SL
    assert env.position == 0, "Position should have been closed by SL"
    assert env.last_trade_info["reason"] == "SL_HIT", (
        f"Expected SL_HIT, got {env.last_trade_info['reason']}"
    )

    realized_pips = env.last_trade_info["net_pips"]
    assert realized_pips < 0, f"SL hit should result in negative PnL, got {realized_pips}"
    assert abs(realized_pips + sl_pips) < 2.0, (
        f"SL PnL should be ~-{sl_pips} pips, got {realized_pips}"
    )


# -----------------------------------------------
# TEST 6: Drawdown termination
# -----------------------------------------------

def test_drawdown_termination():
    pip = 0.0001
    start = 1.1000

    # Make price crash so many SLs get hit
    n = 500
    prices = start - np.arange(n) * 5 * pip  # price falls 5 pips per bar
    prices = np.clip(prices, 0.5, 2.0)

    df = pd.DataFrame({
        "Open":   prices,
        "High":   prices + pip,
        "Low":    prices - 100 * pip,  # huge low to ensure SLs hit
        "Close":  prices,
        "Volume": np.ones(n),
        "rsi_14": np.full(n, 30.0),
        "atr_14": np.full(n, 10 * pip),
    })

    env = make_env(df, spread_pips=1.0, open_penalty_pips=0.0, time_penalty_pips=0.0)
    env.reset()

    open_long_action = next(
        i for i, (a, d, sl, tp) in enumerate(env.action_map)
        if a == "OPEN" and d == 1 and sl == 20
    )

    done = False
    steps = 0
    while not done:
        action = open_long_action if env.position == 0 else 0
        step_out = env.step(action)
        if len(step_out) == 4:
            _, _, done, _ = step_out
        else:
            _, _, term, trunc, _ = step_out
            done = term or trunc
        steps += 1
        if steps > 10000:
            break

    # Episode should have terminated early due to drawdown
    min_equity = env.initial_equity_usd * 0.70
    assert env.equity_usd <= min_equity or env.terminated, (
        "Drawdown termination did not trigger — episode ran to completion without killing"
    )


# -----------------------------------------------
# TEST 7: Observation is always within observation_space bounds
# -----------------------------------------------

def test_observation_within_bounds():
    df = make_trending_df(n=300)
    env = make_env(df)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    for _ in range(100):
        action = env.action_space.sample()
        step_out = env.step(action)
        if len(step_out) == 4:
            obs, _, done, _ = step_out
        else:
            obs, _, term, trunc, _ = step_out
            done = term or trunc

        assert obs.shape == env.observation_space.shape, \
            f"Obs shape mismatch at step {env.current_step}"
        assert not np.any(np.isnan(obs)), \
            f"NaN in observation at step {env.current_step}"
        assert not np.any(np.isinf(obs)), \
            f"Inf in observation at step {env.current_step}"

        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]


# -----------------------------------------------
# TEST 8: High water mark — reward only on new equity highs
# -----------------------------------------------

def test_hold_reward_only_on_new_highs():
    pip = 0.0001
    start = 1.1000

    # Price goes up 5 pips, then back down 3 pips, then up 10 pips
    n = 200
    prices = np.full(n, start, dtype=np.float64)
    prices[12:17] = start + 5 * pip
    prices[17:22] = start + 2 * pip   # pullback
    prices[22:]   = start + 10 * pip  # new high

    df = pd.DataFrame({
        "Open":  prices, "High": prices + 2 * pip,
        "Low":   prices - 2 * pip, "Close": prices,
        "Volume": np.ones(n),
        "rsi_14": np.full(n, 50.0),
        "atr_14": np.full(n, 10 * pip),
    })

    env = make_env(df, hold_reward_weight=1.0, open_penalty_pips=0.0, time_penalty_pips=0.0)
    env.reset()

    open_long = next(
        i for i, (a, d, sl, tp) in enumerate(env.action_map)
        if a == "OPEN" and d == 1 and sl == 20
    )

    for _ in range(2):
        env.step(0)

    env.step(open_long)

    rewards_during_pullback = []
    for step in range(15):
        step_out = env.step(0)  # hold
        reward = step_out[1]
        current_unreal = env._compute_unrealized_pips()
        hwm = env.trade_high_water_mark

        # During pullback (price below high water mark), hold reward should be 0
        if current_unreal < hwm:
            rewards_during_pullback.append(reward)

    # All rewards during pullback should be <= 0 (no hold bonus, only time penalty if any)
    for r in rewards_during_pullback:
        assert r <= 0.001, f"Got hold reward {r} during pullback — high water mark not working"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])