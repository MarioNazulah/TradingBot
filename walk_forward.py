from __future__ import annotations

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from config import PROJECT_ROOT, build_bot_config, ensure_output_dirs, save_json
from evaluation import compute_metrics, run_one_episode
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv
from utils.experiment_logger import log_experiment_to_obsidian


def parse_args():
    parser = argparse.ArgumentParser(description="Run walk-forward validation for the PPO forex trading bot.")
    parser.add_argument("--dataset-path", type=str, help="Override the default training dataset path.")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "artifacts" / "walk_forward"), help="Directory for walk-forward artifacts.")
    parser.add_argument("--timesteps", type=int, help="Override total PPO training timesteps per fold.")
    parser.add_argument("--seed", type=int, help="Override the base random seed.")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds.")
    parser.add_argument("--enable-obsidian-log", action="store_true", help="Write aggregate experiment notes into the local Obsidian vault.")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots.")
    return parser.parse_args()


def make_env(df, feature_cols, config, random_start: bool, episode_max_steps: int | None):
    env_cfg = config.env
    return ForexTradingEnv(
        df=df,
        window_size=env_cfg.window_size,
        sl_options=env_cfg.sl_options,
        tp_options=env_cfg.tp_options,
        spread_pips=env_cfg.spread_pips,
        commission_pips=env_cfg.commission_pips,
        max_slippage_pips=env_cfg.max_slippage_pips,
        random_start=random_start,
        min_episode_steps=env_cfg.min_episode_steps,
        episode_max_steps=episode_max_steps,
        feature_columns=feature_cols,
        hold_reward_weight=env_cfg.hold_reward_weight,
        open_penalty_pips=env_cfg.open_penalty_pips,
        time_penalty_pips=env_cfg.time_penalty_pips,
        unrealized_delta_weight=env_cfg.unrealized_delta_weight,
        allow_flip=env_cfg.allow_flip,
    )


def get_walk_forward_splits(df, n_folds=5, val_ratio=0.15, test_ratio=0.15):
    n = len(df)
    val_size = int(n * val_ratio)
    test_size = int(n * test_ratio)
    step = max(1, (n - val_size - test_size) // n_folds)

    splits = []
    for fold in range(n_folds):
        train_end = step * (fold + 1)
        val_start = train_end
        test_start = val_start + val_size
        test_end = test_start + test_size

        if test_end > n:
            break
        if min(train_end, val_size, test_size) <= 0:
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

        train_vec = DummyVecEnv([
            lambda: make_env(train_df, feature_cols, fold_config, True, fold_config.env.train_episode_max_steps)
        ])
        val_vec = DummyVecEnv([
            lambda: make_env(val_df, feature_cols, fold_config, False, None)
        ])
        test_vec = DummyVecEnv([
            lambda: make_env(test_df, feature_cols, fold_config, False, None)
        ])

        model = PPO(
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

        _, best_val_equity, _ = run_one_episode(model, val_vec)
        best_model = model
        best_path = None

        checkpoints = sorted(
            fold_config.output.checkpoints_dir.glob(f"{fold_config.output.model_name}_fold{fold}*.zip"),
            key=lambda item: item.stat().st_mtime,
        )

        for checkpoint in checkpoints:
            try:
                candidate = PPO.load(str(checkpoint), env=val_vec)
                _, final_equity, _ = run_one_episode(candidate, val_vec)
                if final_equity > best_val_equity:
                    best_val_equity = final_equity
                    best_path = checkpoint
            except Exception as exc:
                print(f"[Skip] {checkpoint.name}: {exc}")

        if best_path is not None:
            best_model = PPO.load(str(best_path), env=train_vec)
            print(f"Fold {fold}: best checkpoint {best_path} (validation equity: {best_val_equity:.2f})")
        else:
            print(f"Fold {fold}: using last model (validation equity: {best_val_equity:.2f})")

        equity_curve, final_equity, closed_trades = run_one_episode(best_model, test_vec)
        metrics = compute_metrics(closed_trades, equity_curve)
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

    print(f"\n{'=' * 50}")
    print("WALK-FORWARD AGGREGATE RESULTS")
    print(f"{'=' * 50}")
    for key in metric_keys:
        values = [fold_metrics[key] for fold_metrics in all_fold_metrics]
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
