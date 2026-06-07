"""Phase 3.3 — reference baseline policies.

A Sharpe of 1.16 from the PPO policy is meaningless in isolation; it only earns
the right to be reported once it beats a dumb policy run through the *same*
environment with the *same* cost model. This module implements three reference
policies and scores them on the identical test slice the agent is evaluated on:

  * random        — sample a valid action every bar (respects action masks),
  * buy_and_hold  — hold maximum long exposure for the whole episode,
  * ma_crossover  — long on MA20-crosses-above-MA50, flat on the opposite cross.

All three run inside a raw ``ForexTradingEnv`` (built via ``env_factory.make_env``
so spread / slippage / commission / swap are exactly the agent's), and their
results are reported with the same ``compute_metrics`` used everywhere else.

Run standalone:
    python baselines.py --no-plot
    python baselines.py --dataset-path data/your_file.csv
"""

from __future__ import annotations

import argparse

import numpy as np

from config import build_bot_config, ensure_output_dirs, save_json, split_train_val_test
from env_factory import make_env
from evaluation import compute_metrics
from indicators import load_and_preprocess_data


# ---------------------------------------------------------------------------
# Action construction — mode-agnostic (MultiDiscrete default, Discrete legacy)
# ---------------------------------------------------------------------------

def _discrete_open_index(env, direction: int, sl_idx: int, tp_idx: int) -> int:
    """Index into action_map for an OPEN(direction, sl_idx, tp_idx).

    action_map layout: [HOLD, CLOSE, then for direction in (0,1): for sl: for tp].
    """
    n_tp = len(env.tp_options)
    n_sl = len(env.sl_options)
    return 2 + direction * (n_sl * n_tp) + sl_idx * n_tp + tp_idx


def hold_action(env):
    return 0 if env.action_space_mode == "discrete" else np.array([0, 0, 0, 0], dtype=np.int64)


def close_action(env):
    return 1 if env.action_space_mode == "discrete" else np.array([2, 0, 0, 0], dtype=np.int64)


def open_action(env, direction: int, sl_idx: int, tp_idx: int):
    if env.action_space_mode == "discrete":
        return _discrete_open_index(env, direction, sl_idx, tp_idx)
    return np.array([1, int(direction), int(sl_idx), int(tp_idx)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Policies — each is policy(env, ctx) -> action, where ctx carries the rng and
# the raw (un-reset-indexed) frames the policy needs.
# ---------------------------------------------------------------------------

def random_policy(env, ctx):
    """Uniformly sample a *valid* action using the env's own mask."""
    rng = ctx["rng"]
    flat = env.position == 0
    if env.action_space_mode == "discrete":
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        return int(rng.choice(valid))
    # MultiDiscrete: pick a valid structural move, then random sub-choices.
    structural_valid = [s for s, ok in enumerate((True, flat or env.allow_flip, not flat)) if ok]
    structural = int(rng.choice(structural_valid))
    direction = int(rng.integers(0, 2))
    sl_idx = int(rng.integers(0, len(env.sl_options)))
    tp_idx = int(rng.integers(0, len(env.tp_options)))
    return np.array([structural, direction, sl_idx, tp_idx], dtype=np.int64)


def buy_and_hold_policy(env, ctx):
    """Stay maximally long. Open long (widest SL/TP) whenever flat, never close
    voluntarily. If a stop/target closes the position, re-enter next bar so the
    strategy keeps continuous long exposure for the whole episode."""
    if env.position == 0:
        sl_idx = len(env.sl_options) - 1   # widest stop
        tp_idx = len(env.tp_options) - 1   # widest target
        return open_action(env, direction=1, sl_idx=sl_idx, tp_idx=tp_idx)
    return hold_action(env)


def ma_crossover_policy(env, ctx):
    """Long-only MA20/MA50 crossover.

    Bullish cross (MA20 crosses above MA50) while flat -> open long.
    Bearish cross while in a position -> close. Widest SL/TP so the crossover
    rule, not an arbitrary bracket, drives the exits.
    """
    ma20 = ctx["ma20"]
    ma50 = ctx["ma50"]
    t = env.current_step
    if t < 1:
        return hold_action(env)
    prev_diff = ma20[t - 1] - ma50[t - 1]
    cur_diff = ma20[t] - ma50[t]
    if np.isnan(prev_diff) or np.isnan(cur_diff):
        return hold_action(env)

    bullish_cross = prev_diff <= 0.0 < cur_diff
    bearish_cross = prev_diff >= 0.0 > cur_diff

    if env.position == 0 and bullish_cross:
        sl_idx = len(env.sl_options) - 1
        tp_idx = len(env.tp_options) - 1
        return open_action(env, direction=1, sl_idx=sl_idx, tp_idx=tp_idx)
    if env.position != 0 and bearish_cross:
        return close_action(env)
    return hold_action(env)


POLICIES = {
    "random": random_policy,
    "buy_and_hold": buy_and_hold_policy,
    "ma_crossover": ma_crossover_policy,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_baseline(policy_fn, env, df_slice, seed: int = 0):
    """Step a raw ForexTradingEnv under ``policy_fn`` for one full episode.

    Returns (equity_curve, final_equity, closed_trades). The env is reset with a
    fixed seed so the random baseline (and any sampled spread/slippage) is
    reproducible.
    """
    rng = np.random.default_rng(seed)
    # MA-crossover reads the raw MA columns aligned to env.current_step. The env
    # reset_index(drop=True)s the frame, so positional .values match env steps.
    ctx = {
        "rng": rng,
        "ma20": df_slice["ma_20"].to_numpy() if "ma_20" in df_slice.columns else None,
        "ma50": df_slice["ma_50"].to_numpy() if "ma_50" in df_slice.columns else None,
    }

    env.reset(seed=seed)
    equity_curve: list[float] = []
    closed_trades: list[dict] = []

    while True:
        action = policy_fn(env, ctx)
        step_out = env.step(action)
        if len(step_out) == 5:
            _obs, _r, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            _obs, _r, done, info = step_out

        equity_curve.append(float(info.get("equity_usd", env.equity_usd)))
        trade_info = info.get("last_trade_info")
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)
        if done:
            break

    final_equity = equity_curve[-1] if equity_curve else float(env.equity_usd)
    return equity_curve, final_equity, closed_trades


def evaluate_baselines(df_slice, feature_cols, config, seed: int = 0):
    """Run all baseline policies on ``df_slice`` and return {name: metrics}."""
    bar_hours = config.env.bar_hours
    results: dict[str, dict] = {}
    for name, policy_fn in POLICIES.items():
        # Fresh env per policy: random_start=False so the whole slice is traded
        # end to end, episode_max_steps=None so nothing truncates early.
        env = make_env(df_slice, feature_cols, config, random_start=False, episode_max_steps=None)
        if name == "ma_crossover" and ("ma_20" not in df_slice.columns or "ma_50" not in df_slice.columns):
            print("[baselines] skipping ma_crossover: ma_20/ma_50 columns absent")
            continue
        equity_curve, final_equity, closed_trades = run_baseline(policy_fn, env, df_slice, seed=seed)
        metrics = compute_metrics(closed_trades, equity_curve, bar_hours=bar_hours)
        metrics["final_equity"] = round(final_equity, 2)
        results[name] = metrics
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline trading policies on the test slice.")
    parser.add_argument("--dataset-path", type=str, help="Override the default dataset path.")
    parser.add_argument("--output-dir", type=str, help="Directory for the baselines report.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random baseline and env sampling.")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_bot_config(dataset_path=args.dataset_path, output_dir=args.output_dir, seed=args.seed)
    ensure_output_dirs(config)

    df, feature_cols = load_and_preprocess_data(config.data.dataset_path)
    _, _, test_df = split_train_val_test(df, config.data.train_ratio, config.data.val_ratio)

    print(f"Dataset    : {config.data.dataset_path}")
    print(f"Test bars  : {len(test_df)}")

    results = evaluate_baselines(test_df, feature_cols, config, seed=args.seed)

    print("\n" + "=" * 60)
    print("BASELINE RESULTS (test slice, same cost model as the agent)")
    print("=" * 60)
    for name, m in results.items():
        print(f"  {name:14s} | Sharpe={m['sharpe']:>7.3f} | PF={m['profit_factor']:>6.3f} | "
              f"WR={m['win_rate']:>5.3f} | Trades={m['n_trades']:>4d} | "
              f"MaxDD={m['max_dd']:>5.3f} | FinalEq={m['final_equity']:.2f}")

    report_path = config.output.reports_dir / "baselines_metrics.json"
    save_json({"dataset": str(config.data.dataset_path), "seed": args.seed,
               "test_bars": len(test_df), "baselines": results}, report_path)
    print(f"\nReport saved: {report_path}")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        for name, policy_fn in POLICIES.items():
            if name not in results:
                continue
            env = make_env(test_df, feature_cols, config, random_start=False, episode_max_steps=None)
            curve, _, _ = run_baseline(policy_fn, env, test_df, seed=args.seed)
            plt.plot(curve, label=name)
        plt.title("Baseline Equity Curves (Test)")
        plt.xlabel("Steps")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
