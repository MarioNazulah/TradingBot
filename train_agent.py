from __future__ import annotations

import argparse
import shutil
from dataclasses import replace

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.utils import set_random_seed

from config import build_bot_config, ensure_output_dirs, save_json, save_run_config, split_train_val_test
from env_factory import make_callback_eval_env, make_eval_env, make_train_env
from evaluation import compute_metrics, run_one_episode, score_model, statistical_report
from indicators import load_and_preprocess_data
from utils.experiment_logger import log_experiment_to_obsidian


def parse_args():
    parser = argparse.ArgumentParser(description="Train the MaskablePPO forex trading bot.")
    parser.add_argument("--dataset-path", type=str, help="Override the default training dataset path.")
    parser.add_argument("--output-dir", type=str, help="Directory for checkpoints, reports, and the saved model.")
    parser.add_argument("--timesteps", type=int, help="Override total PPO training timesteps.")
    parser.add_argument("--seed", type=int, help="Base random seed (seeds are base, base+1, ... base+n_seeds-1).")
    parser.add_argument("--n-seeds", type=int, dest="n_seeds",
                        help="Phase 3.2: number of independent seeds to train and aggregate (default from config, 5).")
    parser.add_argument("--eval-freq", type=int, dest="eval_freq",
                        help="Phase 3.1: env steps between validation evaluations (default from config).")
    parser.add_argument("--n-eval-episodes", type=int, dest="n_eval_episodes", default=5,
                        help="Episodes per validation evaluation inside EvalCallback.")
    parser.add_argument(
        "--enable-obsidian-log",
        action="store_true",
        dest="enable_obsidian_log",
        help="Write experiment notes into the local experiments folder.",
    )
    parser.add_argument(
        "--no-experiment-log",
        action="store_false",
        dest="enable_obsidian_log",
        help="Disable automatic markdown experiment logging for this run.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots.")
    parser.set_defaults(enable_obsidian_log=True)
    return parser.parse_args()


def run_single_seed(seed, df, train_df, val_df, test_df, feature_cols, base_config, n_eval_episodes):
    """Train one model for one seed and evaluate it once on the test split.

    Phase 3.1: best-model selection is done online by MaskableEvalCallback
    (evaluating on a normalization-synced val env every eval_freq steps and
    saving best_model.zip by mean validation reward) — no more post-hoc
    checkpoint sweep, and TensorBoard gets the eval curve for free.

    Returns (test_metrics, seed_config, val_sharpe).
    """
    seed_root = base_config.output.root_dir / f"seed_{seed}"
    config = replace(
        base_config,
        output=replace(base_config.output, root_dir=seed_root),
        training=replace(base_config.training, seed=seed),
    )
    ensure_output_dirs(config)
    set_random_seed(seed)

    train_vec_env = make_train_env(train_df, feature_cols, config, config.env.train_episode_max_steps)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=train_vec_env,
        verbose=config.training.verbose,
        seed=seed,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        ent_coef=config.training.ent_coef,
        clip_range=config.training.clip_range,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        policy_kwargs={"net_arch": list(config.training.net_arch)},
        tensorboard_log=str(config.output.tensorboard_dir),
    )

    # Phase 3.1: EvalCallback on a VecNormalize-wrapped val env. The callback
    # syncs the training observation stats into the eval env before each eval,
    # saves the best model by validation reward, and logs evaluations/<seed> to
    # TensorBoard. eval_freq is per-env-step (single env here).
    best_dir = config.output.root_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    eval_env = make_callback_eval_env(val_df, feature_cols, config)
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(best_dir),
        eval_freq=max(1, config.training.eval_freq),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=config.training.total_timesteps, callback=eval_callback)

    # Persist the training normalization stats; eval envs reuse them.
    train_vec_env.save(str(config.output.vecnormalize_path))
    bar_hours = config.env.bar_hours

    # Prefer the callback's best model; fall back to the final model if no eval
    # ever beat the initial -inf (e.g. eval_freq > total_timesteps).
    best_zip = best_dir / "best_model.zip"
    if best_zip.exists():
        best_model = MaskablePPO.load(str(best_zip), env=train_vec_env)
        print(f"[seed {seed}] using EvalCallback best model: {best_zip}")
    else:
        best_model = model
        print(f"[seed {seed}] no best_model.zip found; using final model")

    best_model.save(str(config.output.model_base_path))
    save_run_config(config, config.output.model_config_path, feature_columns=feature_cols)
    save_run_config(config, config.output.run_config_path, feature_columns=feature_cols)

    stats = config.output.vecnormalize_path
    val_eval_env = make_eval_env(val_df, feature_cols, config, stats)
    test_eval_env = make_eval_env(test_df, feature_cols, config, stats)

    val_sharpe, val_final_equity, _, _ = score_model(best_model, val_eval_env, bar_hours)
    equity_curve_test, final_equity_test, trades_test = run_one_episode(best_model, test_eval_env)

    metrics = compute_metrics(trades_test, equity_curve_test, bar_hours=bar_hours)
    metrics["final_equity_test"] = round(final_equity_test, 2)
    metrics["val_sharpe"] = round(val_sharpe, 3)
    metrics["val_final_equity"] = round(val_final_equity, 2)
    metrics["seed"] = seed
    # Phase 3.6: attach the statistical battery for this seed's test episode.
    metrics["statistics"] = statistical_report(equity_curve_test, trades_test, n_trials=1)

    save_json(metrics, config.output.metrics_path)
    print(f"[seed {seed}] TEST Sharpe={metrics['sharpe']} PF={metrics['profit_factor']} "
          f"WR={metrics['win_rate']} Trades={metrics['n_trades']} MaxDD={metrics['max_dd']} "
          f"FinalEq={final_equity_test:.2f}")

    return metrics, config, val_sharpe


def aggregate_seed_metrics(per_seed_metrics, noise_threshold=0.30):
    """Phase 3.2: mean ± stddev and [min, max] across seeds for every metric.

    A result whose stddev exceeds ``noise_threshold`` * |mean| is flagged as
    noise — a single seed is an anecdote, and a result that swings that much
    across seeds is not one you can take to a paper-trading decision.
    """
    numeric_keys = ["sharpe", "calmar", "profit_factor", "win_rate", "avg_trade_pips",
                    "n_trades", "max_dd", "final_equity_test", "val_sharpe"]
    aggregate = {}
    for key in numeric_keys:
        values = [float(m[key]) for m in per_seed_metrics if key in m]
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        noisy = bool(abs(mean) > 1e-9 and std > noise_threshold * abs(mean))
        aggregate[key] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "per_seed": [round(v, 4) for v in values],
            "noisy": noisy,
        }
    return aggregate


def main():
    args = parse_args()
    config = build_bot_config(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
        seed=args.seed,
        enable_obsidian_log=args.enable_obsidian_log,
    )
    if args.n_seeds is not None:
        config = replace(config, training=replace(config.training, n_seeds=args.n_seeds))
    if args.eval_freq is not None:
        config = replace(config, training=replace(config.training, eval_freq=args.eval_freq))

    ensure_output_dirs(config)

    df, feature_cols = load_and_preprocess_data(config.data.dataset_path)
    train_df, val_df, test_df = split_train_val_test(df, config.data.train_ratio, config.data.val_ratio)

    base_seed = config.training.seed
    n_seeds = max(1, config.training.n_seeds)
    seeds = [base_seed + i for i in range(n_seeds)]

    print(f"Dataset         : {config.data.dataset_path}")
    print(f"Training bars   : {len(train_df)}")
    print(f"Validation bars : {len(val_df)}")
    print(f"Testing bars    : {len(test_df)}")
    print(f"Artifacts dir   : {config.output.root_dir}")
    print(f"Seeds           : {seeds}")
    print(f"Eval freq       : {config.training.eval_freq}")

    per_seed_metrics = []
    best_overall = {"val_sharpe": -np.inf, "config": None}
    for seed in seeds:
        print(f"\n{'=' * 60}\nSEED {seed}\n{'=' * 60}")
        metrics, seed_config, val_sharpe = run_single_seed(
            seed, df, train_df, val_df, test_df, feature_cols, config, args.n_eval_episodes
        )
        per_seed_metrics.append(metrics)
        if val_sharpe > best_overall["val_sharpe"]:
            best_overall = {"val_sharpe": val_sharpe, "config": seed_config}

    # Promote the best-by-validation-Sharpe seed's model to the top-level
    # artifact paths so eval_agent.py finds artifacts/model_eurusd_best.zip.
    if best_overall["config"] is not None:
        src = best_overall["config"].output
        dst = config.output
        for src_path, dst_path in (
            (src.model_file_path, dst.model_file_path),
            (src.model_config_path, dst.model_config_path),
            (src.vecnormalize_path, dst.vecnormalize_path),
            (src.run_config_path, dst.run_config_path),
        ):
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
        print(f"\nPromoted best seed (val Sharpe {best_overall['val_sharpe']:.3f}) -> {dst.model_file_path}")

    aggregate = aggregate_seed_metrics(per_seed_metrics)
    report = {
        "dataset": str(config.data.dataset_path),
        "seeds": seeds,
        "n_seeds": n_seeds,
        "aggregate": aggregate,
        "per_seed": per_seed_metrics,
    }
    agg_path = config.output.reports_dir / "multiseed_metrics.json"
    save_json(report, agg_path)

    print(f"\n{'=' * 60}\nMULTI-SEED AGGREGATE (n={n_seeds})\n{'=' * 60}")
    for key, stat in aggregate.items():
        flag = "  <-- NOISY (std > 30% of mean)" if stat["noisy"] else ""
        print(f"  {key:18s} mean={stat['mean']:>10.4f} std={stat['std']:>9.4f} "
              f"[min={stat['min']}, max={stat['max']}]{flag}")
    print(f"\nAggregate report saved: {agg_path}")

    if config.output.enable_obsidian_log:
        sharpe_stat = aggregate.get("sharpe", {})
        log_experiment_to_obsidian(
            config={
                "sl_opts": list(config.env.sl_options),
                "tp_opts": list(config.env.tp_options),
                "window_size": config.env.window_size,
                "timesteps": config.training.total_timesteps,
                "dataset": str(config.data.dataset_path),
                "n_seeds": n_seeds,
                "seeds": seeds,
                "eval_freq": config.training.eval_freq,
                "method": "multiseed_evalcallback",
            },
            results={"aggregate_sharpe": sharpe_stat, "aggregate": aggregate},
            notes=f"Per-seed: {per_seed_metrics}",
            vault_path=config.output.obsidian_vault_dir,
        )

    if not args.no_plot:
        import matplotlib.pyplot as plt
        if best_overall["config"] is not None:
            bc = best_overall["config"]
            stats = bc.output.vecnormalize_path
            best_model = MaskablePPO.load(str(bc.output.model_file_path))
            test_eval_env = make_eval_env(test_df, feature_cols, bc, stats)
            curve, _, _ = run_one_episode(best_model, test_eval_env)
            plt.figure(figsize=(12, 6))
            plt.plot(curve, label="Best seed — Test (out-of-sample)")
            plt.title("Equity Curve: Best-Seed Test")
            plt.xlabel("Steps")
            plt.ylabel("Equity ($)")
            plt.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
