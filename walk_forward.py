from __future__ import annotations

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from config import PROJECT_ROOT, build_bot_config, ensure_output_dirs, save_json
from env_factory import make_eval_env, make_train_env
from evaluation import compute_metrics, run_one_episode, score_model
from indicators import load_and_preprocess_data
from utils.experiment_logger import log_experiment_to_obsidian


def parse_args():
    parser = argparse.ArgumentParser(description="Run walk-forward validation for the PPO forex trading bot.")
    parser.add_argument("--dataset-path", type=str, help="Override the default training dataset path.")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "artifacts" / "walk_forward"), help="Directory for walk-forward artifacts.")
    parser.add_argument("--timesteps", type=int, help="Override total PPO training timesteps per fold.")
    parser.add_argument("--seed", type=int, help="Override the base random seed.")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds.")
    parser.add_argument(
        "--enable-obsidian-log",
        action="store_true",
        dest="enable_obsidian_log",
        help="Write aggregate experiment notes into the local experiments folder.",
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


def get_walk_forward_splits(df, n_folds=5, val_ratio=0.15, test_ratio=0.15):
    n = len(df)
    # Size val/test windows relative to each fold's TRAINING segment, not the
    # whole dataset (review #15). Otherwise early folds get val/test windows
    # larger than the data they trained on.
    step = max(1, n // (n_folds + 2))

    splits = []
    for fold in range(n_folds):
        train_end = step * (fold + 1)
        val_size = max(1, int(train_end * val_ratio))
        test_size = max(1, int(train_end * test_ratio))
        val_start = train_end
        test_start = val_start + val_size
        test_end = test_start + test_size

        if test_end > n:
            break
        if train_end <= 0:
            continue

        splits.append({
            "fold": fold + 1,
            "train": df.iloc[:train_end].copy(),
            "val": df.iloc[val_start:test_start].copy(),
            "test": df.iloc[test_start:test_end].copy(),
        })

    return splits


def run_walk_forward(df, feature_cols, config, n_folds=5, plot_results: bool = True):
    splits = get_walk_forward_splits(
        df,
        n_folds=n_folds,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
    )
    if not splits:
        raise ValueError("No valid walk-forward splits were generated.")

    print(f"Walk-forward folds: {len(splits)}")

    all_fold_metrics = []
    all_test_equity_curves = []

    for split in splits:
        fold = split["fold"]
        fold_seed = config.training.seed + fold
        set_random_seed(fold_seed)

        train_df = split["train"]
        val_df = split["val"]
        test_df = split["test"]

        print(f"\n{'=' * 50}")
        print(f"FOLD {fold} | train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        print(f"{'=' * 50}")

        fold_root = config.output.root_dir / f"fold_{fold}"
        fold_config = replace(config, output=replace(config.output, root_dir=fold_root))
        ensure_output_dirs(fold_config)

        # env_factory binds df by argument (no loop-variable closure, review #9)
        # and wraps train in VecNormalize (review #14).
        train_vec = make_train_env(train_df, feature_cols, fold_config, fold_config.env.train_episode_max_steps)

        model = MaskablePPO(
            policy="MlpPolicy",
            env=train_vec,
            verbose=0,
            seed=fold_seed,
            n_steps=fold_config.training.n_steps,
            batch_size=fold_config.training.batch_size,
            ent_coef=fold_config.training.ent_coef,
            clip_range=fold_config.training.clip_range,
            policy_kwargs={"net_arch": list(fold_config.training.net_arch)},
            tensorboard_log=str(fold_config.output.tensorboard_dir),
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=fold_config.training.checkpoint_freq,
            save_path=str(fold_config.output.checkpoints_dir),
            name_prefix=f"{fold_config.output.model_name}_fold{fold}",
        )

        model.learn(total_timesteps=fold_config.training.total_timesteps, callback=checkpoint_callback)

        # Save normalization stats; build eval envs that reuse them.
        train_vec.save(str(fold_config.output.vecnormalize_path))
        bar_hours = fold_config.env.bar_hours
        stats = fold_config.output.vecnormalize_path
        val_vec = make_eval_env(val_df, feature_cols, fold_config, stats)
        test_vec = make_eval_env(test_df, feature_cols, fold_config, stats)

        # Select by Sharpe, not final equity (review #10).
        best_score, best_val_equity, _, _ = score_model(model, val_vec, bar_hours)
        best_model = model
        best_path = None

        checkpoints = sorted(
            fold_config.output.checkpoints_dir.glob(f"{fold_config.output.model_name}_fold{fold}*.zip"),
            key=lambda item: item.stat().st_mtime,
        )

        for checkpoint in checkpoints:
            try:
                candidate = MaskablePPO.load(str(checkpoint), env=val_vec)
                sharpe, final_equity, _, _ = score_model(candidate, val_vec, bar_hours)
                if sharpe > best_score:
                    best_score = sharpe
                    best_val_equity = final_equity
                    best_path = checkpoint
            except Exception as exc:
                print(f"[Skip] {checkpoint.name}: {exc}")

        if best_path is not None:
            best_model = MaskablePPO.load(str(best_path), env=train_vec)
            print(f"Fold {fold}: best checkpoint {best_path} (validation Sharpe: {best_score:.3f})")
        else:
            print(f"Fold {fold}: using last model (validation Sharpe: {best_score:.3f})")

        equity_curve, final_equity, closed_trades = run_one_episode(best_model, test_vec)
        metrics = compute_metrics(closed_trades, equity_curve, bar_hours=bar_hours)
        metrics["fold"] = fold
        metrics["final_equity"] = round(final_equity, 2)
        metrics["best_validation_equity"] = round(best_val_equity, 2)

        all_fold_metrics.append(metrics)
        all_test_equity_curves.append(equity_curve)

        best_model.save(str(fold_config.output.model_base_path))
        save_json(metrics, fold_config.output.metrics_path)

        print(f"Fold {fold} TEST | Sharpe={metrics['sharpe']} PF={metrics['profit_factor']} "
              f"WR={metrics['win_rate']} Trades={metrics['n_trades']} MaxDD={metrics['max_dd']} "
              f"FinalEq={final_equity:.2f}")

    aggregate = {}
    metric_keys = [
        "sharpe",
        "calmar",
        "profit_factor",
        "win_rate",
        "avg_trade_pips",
        "n_trades",
        "max_dd",
        "final_equity",
        "best_validation_equity",
    ]
    # Trade counts are integers — averaging them as floats and rounding to
    # 3 decimals is meaningless. Report sum + per-fold spread instead.
    INT_KEYS = {"n_trades"}

    print(f"\n{'=' * 50}")
    print("WALK-FORWARD AGGREGATE RESULTS")
    print(f"{'=' * 50}")
    for key in metric_keys:
        values = [fold_metrics[key] for fold_metrics in all_fold_metrics]
        if key in INT_KEYS:
            int_values = [int(v) for v in values]
            aggregate[key] = {
                "sum": int(sum(int_values)),
                "min": int(min(int_values)),
                "max": int(max(int_values)),
                "per_fold": int_values,
            }
            print(f"  {key}: sum={aggregate[key]['sum']} "
                  f"[min={aggregate[key]['min']}, max={aggregate[key]['max']}] "
                  f"per-fold={int_values}")
        else:
            aggregate[key] = round(float(np.mean(values)), 3)
            print(f"  {key}: mean={aggregate[key]:.3f} per-fold={values}")

    save_json(
        {
            "config_dataset": str(config.data.dataset_path),
            "n_folds": len(splits),
            "aggregate": aggregate,
            "per_fold": all_fold_metrics,
        },
        config.output.reports_dir / "walk_forward_metrics.json",
    )

    if config.output.enable_obsidian_log:
        log_experiment_to_obsidian(
            config={
                "n_actions": len(config.env.sl_options) * len(config.env.tp_options) * 2 + 2,
                "sl_opts": list(config.env.sl_options),
                "tp_opts": list(config.env.tp_options),
                "window_size": config.env.window_size,
                "timesteps": config.training.total_timesteps,
                "dataset": str(config.data.dataset_path),
                "hold_reward_weight": config.env.hold_reward_weight,
                "open_penalty_pips": config.env.open_penalty_pips,
                "time_penalty_pips": config.env.time_penalty_pips,
                "n_folds": len(splits),
                "seed": config.training.seed,
                "method": "walk_forward",
            },
            results=aggregate,
            notes=f"Per-fold results: {all_fold_metrics}",
            vault_path=config.output.obsidian_vault_dir,
        )

    if plot_results:
        plt.figure(figsize=(14, 6))
        for idx, curve in enumerate(all_test_equity_curves, start=1):
            plt.plot(curve, label=f"Fold {idx} Test")
        plt.title("Walk-Forward Test Equity Curves")
        plt.xlabel("Steps")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return all_fold_metrics, aggregate


def main():
    args = parse_args()
    config = build_bot_config(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
        seed=args.seed,
        enable_obsidian_log=args.enable_obsidian_log,
    )
    ensure_output_dirs(config)
    set_random_seed(config.training.seed)

    df, feature_cols = load_and_preprocess_data(config.data.dataset_path)
    run_walk_forward(df, feature_cols, config, n_folds=args.folds, plot_results=not args.no_plot)


if __name__ == "__main__":
    main()
