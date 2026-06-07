"""Phase 3.5 — algorithm bake-off.

Train and evaluate several RL algorithms on the *same* env, the *same* test
split and the *same* seed pool, then rank them by out-of-sample Sharpe across
seeds (not final equity — equity is path-dependent and a noisier signal).

Algorithms:
  * PPO           — on-policy baseline (MultiDiscrete, no masking).
  * MaskablePPO   — PPO + invalid-action masking (the current champion path).
  * RecurrentPPO  — LSTM policy, a natural fit for time-series state.
  * QR-DQN        — distributional DQN; needs a Discrete action space, so it uses
                    the legacy Discrete(20) env path (action_space_mode="discrete").

Only MaskablePPO consumes action masks; the others see the raw action space and
rely on the env treating structurally-invalid actions (CLOSE while flat, OPEN
while in a position) as no-ops, which it does.

    python bakeoff.py --algos PPO MaskablePPO RecurrentPPO QR-DQN --seeds 5 --timesteps 60000
    python bakeoff.py --algos MaskablePPO --seeds 1 --timesteps 4000   # quick check
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from config import build_bot_config, ensure_output_dirs, save_json, split_train_val_test
from env_factory import make_eval_env, make_train_env
from evaluation import compute_metrics
from indicators import load_and_preprocess_data

NET_ARCH = (256, 256)


# ---------------------------------------------------------------------------
# Algorithm registry — each entry knows its action mode and eval semantics.
# ---------------------------------------------------------------------------

def _algo_specs():
    """Built lazily so importing this module never requires SB3."""
    from sb3_contrib import MaskablePPO, QRDQN, RecurrentPPO
    from stable_baselines3 import PPO

    def build_ppo(train_vec, config, seed):
        return PPO(
            "MlpPolicy", env=train_vec, verbose=0, seed=seed,
            n_steps=config.training.n_steps, batch_size=config.training.batch_size,
            ent_coef=config.training.ent_coef, clip_range=config.training.clip_range,
            gamma=config.training.gamma, gae_lambda=config.training.gae_lambda,
            policy_kwargs={"net_arch": list(config.training.net_arch)},
        )

    def build_maskable(train_vec, config, seed):
        return MaskablePPO(
            "MlpPolicy", env=train_vec, verbose=0, seed=seed,
            n_steps=config.training.n_steps, batch_size=config.training.batch_size,
            ent_coef=config.training.ent_coef, clip_range=config.training.clip_range,
            gamma=config.training.gamma, gae_lambda=config.training.gae_lambda,
            policy_kwargs={"net_arch": list(config.training.net_arch)},
        )

    def build_recurrent(train_vec, config, seed):
        return RecurrentPPO(
            "MlpLstmPolicy", env=train_vec, verbose=0, seed=seed,
            n_steps=config.training.n_steps, batch_size=config.training.batch_size,
            ent_coef=config.training.ent_coef, clip_range=config.training.clip_range,
            gamma=config.training.gamma, gae_lambda=config.training.gae_lambda,
            policy_kwargs={"net_arch": list(config.training.net_arch)},
        )

    def build_qrdqn(train_vec, config, seed):
        # DQN-family hyperparameters; off-policy, so no n_steps/clip_range.
        return QRDQN(
            "MlpPolicy", env=train_vec, verbose=0, seed=seed,
            learning_rate=1e-4, buffer_size=100_000, learning_starts=1_000,
            batch_size=128, gamma=config.training.gamma, train_freq=4,
            target_update_interval=1_000, exploration_fraction=0.2,
            policy_kwargs={"net_arch": list(config.training.net_arch)},
        )

    return {
        "PPO": {"build": build_ppo, "action_mode": "multidiscrete", "maskable": False, "recurrent": False},
        "MaskablePPO": {"build": build_maskable, "action_mode": "multidiscrete", "maskable": True, "recurrent": False},
        "RecurrentPPO": {"build": build_recurrent, "action_mode": "multidiscrete", "maskable": False, "recurrent": True},
        "QR-DQN": {"build": build_qrdqn, "action_mode": "discrete", "maskable": False, "recurrent": False},
    }


# ---------------------------------------------------------------------------
# Generic single-episode runner (handles masks + recurrent state)
# ---------------------------------------------------------------------------

def run_episode_generic(model, vec_env, maskable: bool, recurrent: bool):
    """One deterministic episode; returns (equity_curve, final_equity, trades).

    Mirrors evaluation.run_one_episode but branches on whether the model takes
    action masks (MaskablePPO) and/or carries LSTM state (RecurrentPPO).
    """
    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    equity_curve, closed_trades = [], []

    while True:
        if maskable:
            action_masks = np.asarray(vec_env.env_method("action_masks"))
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        elif recurrent:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
        else:
            action, _ = model.predict(obs, deterministic=True)

        step_out = vec_env.step(action)
        if len(step_out) == 4:
            obs, _, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, _, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
        episode_starts = np.array([done], dtype=bool)

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        equity_curve.append(float(info.get("equity_usd", vec_env.get_attr("equity_usd")[0])))
        trade_info = info.get("last_trade_info")
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)
        if done:
            break

    final_equity = equity_curve[-1] if equity_curve else float(vec_env.get_attr("equity_usd")[0])
    return equity_curve, final_equity, closed_trades


def _config_for_mode(base_config, action_mode):
    return replace(base_config, env=replace(base_config.env, action_space_mode=action_mode))


def run_algo_seed(spec, name, base_config, train_df, test_df, feature_cols, seed, total_timesteps):
    """Train one algo for one seed and evaluate on test; return metrics dict."""
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed)
    config = _config_for_mode(base_config, spec["action_mode"])
    # Per-algo/seed artifact dir so VecNormalize stats don't collide.
    run_root = base_config.output.root_dir / f"{name}_seed{seed}"
    config = replace(config, output=replace(config.output, root_dir=run_root))
    ensure_output_dirs(config)

    train_vec = make_train_env(train_df, feature_cols, config, config.env.train_episode_max_steps)
    model = spec["build"](train_vec, config, seed)
    model.learn(total_timesteps=total_timesteps)

    train_vec.save(str(config.output.vecnormalize_path))
    test_vec = make_eval_env(test_df, feature_cols, config, config.output.vecnormalize_path)

    equity_curve, final_equity, trades = run_episode_generic(
        model, test_vec, maskable=spec["maskable"], recurrent=spec["recurrent"]
    )
    metrics = compute_metrics(trades, equity_curve, bar_hours=config.env.bar_hours)
    metrics["final_equity"] = round(final_equity, 2)
    metrics["seed"] = seed
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Algorithm bake-off on a shared env/test/seed pool (Phase 3.5).")
    parser.add_argument("--dataset-path", type=str, help="Override the default training dataset path.")
    parser.add_argument("--output-dir", type=str, help="Directory for bake-off artifacts.")
    parser.add_argument("--algos", nargs="*",
                        default=["PPO", "MaskablePPO", "RecurrentPPO", "QR-DQN"],
                        help="Subset of algorithms to run.")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per algorithm.")
    parser.add_argument("--timesteps", type=int, default=60_000, help="Train timesteps per seed.")
    parser.add_argument("--base-seed", type=int, default=42, help="First seed; pool is base..base+seeds-1.")
    return parser.parse_args()


def main():
    args = parse_args()
    base_config = build_bot_config(
        dataset_path=args.dataset_path, output_dir=args.output_dir, total_timesteps=args.timesteps,
        seed=args.base_seed,
    )
    ensure_output_dirs(base_config)

    specs = _algo_specs()
    unknown = [a for a in args.algos if a not in specs]
    if unknown:
        raise ValueError(f"Unknown algos {unknown}; choose from {list(specs)}")

    df, feature_cols = load_and_preprocess_data(base_config.data.dataset_path)
    train_df, _, test_df = split_train_val_test(df, base_config.data.train_ratio, base_config.data.val_ratio)
    seeds = [args.base_seed + i for i in range(args.seeds)]

    print(f"Dataset   : {base_config.data.dataset_path}")
    print(f"Algos     : {args.algos}")
    print(f"Seeds     : {seeds} | timesteps/seed: {args.timesteps}")

    all_results = {}
    for name in args.algos:
        spec = specs[name]
        per_seed = []
        for seed in seeds:
            print(f"\n--- {name} | seed {seed} ---")
            metrics = run_algo_seed(spec, name, base_config, train_df, test_df, feature_cols, seed, args.timesteps)
            per_seed.append(metrics)
            print(f"    test Sharpe={metrics['sharpe']} FinalEq={metrics['final_equity']:.2f} "
                  f"Trades={metrics['n_trades']}")
        sharpes = np.array([m["sharpe"] for m in per_seed], dtype=np.float64)
        all_results[name] = {
            "sharpe_mean": round(float(sharpes.mean()), 4),
            "sharpe_std": round(float(sharpes.std(ddof=1)) if sharpes.size > 1 else 0.0, 4),
            "sharpe_min": round(float(sharpes.min()), 4),
            "sharpe_max": round(float(sharpes.max()), 4),
            "per_seed": per_seed,
        }

    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["sharpe_mean"], reverse=True)
    print("\n" + "=" * 64)
    print("ALGORITHM BAKE-OFF — ranked by out-of-sample Sharpe across seeds")
    print("=" * 64)
    for name, r in ranked:
        print(f"  {name:14s} | Sharpe mean={r['sharpe_mean']:>7.3f} ± {r['sharpe_std']:<7.3f} "
              f"[min={r['sharpe_min']}, max={r['sharpe_max']}]")

    report_path = base_config.output.reports_dir / "bakeoff_metrics.json"
    save_json({
        "dataset": str(base_config.data.dataset_path),
        "seeds": seeds,
        "timesteps_per_seed": args.timesteps,
        "ranking": [name for name, _ in ranked],
        "results": all_results,
    }, report_path)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
