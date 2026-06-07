"""Phase 3.4 — Optuna hyperparameter sweep.

TPE sampler + MedianPruner over the MaskablePPO search space below. Each trial
trains the model on the train split for a small budget across several seeds and
scores on the validation split; the objective is the *mean validation final
equity across seeds* so a config that only works for one lucky seed is not
rewarded. Trials are pruned at 25% of the timestep budget when their early
validation equity is below the running median.

Search space (per UPGRADES.md 3.4):
  learning_rate  log-uniform [1e-5, 1e-3]
  n_steps        {1024, 2048, 4096}
  batch_size     {64, 128, 256, 512}   (must divide n_steps)
  ent_coef       log-uniform [1e-4, 1e-1]
  clip_range     {0.1, 0.2, 0.3}
  gamma          {0.95, 0.99, 0.999}
  gae_lambda     {0.9, 0.95, 0.98}
  net_arch       {[64,64], [128,128], [256,256], [256,256,128]}

    python tune.py --trials 50 --n-jobs 6 --seeds 3 --timesteps 60000
    python tune.py --trials 5 --n-jobs 1 --seeds 1 --timesteps 4000   # quick check

NOTE: each trial trains ``seeds`` models, so wall-clock ~= trials * seeds *
train-time / n_jobs. Start small to validate the pipeline before a real sweep.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from config import build_bot_config, ensure_output_dirs, save_json, split_train_val_test
from env_factory import make_eval_env, make_train_env
from evaluation import score_model
from indicators import load_and_preprocess_data

NET_ARCH_CHOICES = {
    "64_64": (64, 64),
    "128_128": (128, 128),
    "256_256": (256, 256),
    "256_256_128": (256, 256, 128),
}


def _build_trial_config(base_config, params):
    """Return a BotConfig with the trial's hyperparameters applied."""
    training = replace(
        base_config.training,
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        ent_coef=params["ent_coef"],
        clip_range=params["clip_range"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        net_arch=params["net_arch"],
    )
    return replace(base_config, training=training)


def _train_eval_one_seed(trial, params, base_config, train_df, val_df, feature_cols,
                         seed, total_timesteps, report_offset):
    """Train one model for one seed, reporting an intermediate value for pruning.

    Returns the validation final equity. The intermediate report at 25% of the
    budget lets MedianPruner kill clearly-bad trials early. ``report_offset``
    spaces the per-seed intermediate reports onto distinct Optuna steps.
    """
    import optuna
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed)
    config = _build_trial_config(base_config, params)
    train_vec = make_train_env(train_df, feature_cols, config, config.env.train_episode_max_steps)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=train_vec,
        verbose=0,
        seed=seed,
        learning_rate=params["learning_rate"],
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        ent_coef=config.training.ent_coef,
        clip_range=config.training.clip_range,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        policy_kwargs={"net_arch": list(config.training.net_arch)},
    )

    prune_at = max(1, int(total_timesteps * 0.25))

    class _PruneCallback(BaseCallback):
        """Evaluate val equity once at 25% budget; report to Optuna and prune."""

        def __init__(self):
            super().__init__()
            self._reported = False

        def _on_step(self) -> bool:
            if not self._reported and self.num_timesteps >= prune_at:
                self._reported = True
                train_vec.save(str(config.output.vecnormalize_path))
                val_vec = make_eval_env(val_df, feature_cols, config, config.output.vecnormalize_path)
                _, val_equity, _, _ = score_model(self.model, val_vec, config.env.bar_hours)
                trial.report(float(val_equity), step=report_offset)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return True

    model.learn(total_timesteps=total_timesteps, callback=_PruneCallback())

    train_vec.save(str(config.output.vecnormalize_path))
    val_vec = make_eval_env(val_df, feature_cols, config, config.output.vecnormalize_path)
    _, val_equity, _, _ = score_model(model, val_vec, config.env.bar_hours)
    return float(val_equity)


def make_objective(base_config, train_df, val_df, feature_cols, n_seeds, total_timesteps):
    def objective(trial):
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        # batch_size must divide n_steps. All listed batch sizes divide all listed
        # n_steps, but guard anyway so the constraint is explicit and future-proof.
        if n_steps % batch_size != 0:
            import optuna
            raise optuna.TrialPruned()

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": n_steps,
            "batch_size": batch_size,
            "ent_coef": trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),
            "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
            "gamma": trial.suggest_categorical("gamma", [0.95, 0.99, 0.999]),
            "gae_lambda": trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98]),
            "net_arch": NET_ARCH_CHOICES[
                trial.suggest_categorical("net_arch", list(NET_ARCH_CHOICES.keys()))
            ],
        }

        seed_equities = []
        for i in range(n_seeds):
            seed = base_config.training.seed + i
            val_equity = _train_eval_one_seed(
                trial, params, base_config, train_df, val_df, feature_cols,
                seed=seed, total_timesteps=total_timesteps, report_offset=i,
            )
            seed_equities.append(val_equity)

        mean_equity = float(np.mean(seed_equities))
        trial.set_user_attr("seed_equities", seed_equities)
        trial.set_user_attr("std_equity", float(np.std(seed_equities)))
        return mean_equity

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep for MaskablePPO (Phase 3.4).")
    parser.add_argument("--dataset-path", type=str, help="Override the default training dataset path.")
    parser.add_argument("--output-dir", type=str, help="Directory for sweep artifacts.")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--n-jobs", type=int, default=6, help="Parallel trial workers.")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds trained+averaged per trial.")
    parser.add_argument("--timesteps", type=int, default=60_000, help="Train timesteps per seed per trial.")
    parser.add_argument("--study-name", type=str, default="maskableppo_tune", help="Optuna study name.")
    parser.add_argument("--storage", type=str, help="Optuna storage URL (e.g. sqlite:///tune.db) for resumable studies.")
    return parser.parse_args()


def main():
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    args = parse_args()
    base_config = build_bot_config(
        dataset_path=args.dataset_path, output_dir=args.output_dir, total_timesteps=args.timesteps,
    )
    ensure_output_dirs(base_config)

    df, feature_cols = load_and_preprocess_data(base_config.data.dataset_path)
    train_df, val_df, _ = split_train_val_test(df, base_config.data.train_ratio, base_config.data.val_ratio)

    print(f"Dataset   : {base_config.data.dataset_path}")
    print(f"Train/Val : {len(train_df)}/{len(val_df)} bars")
    print(f"Trials    : {args.trials} | n_jobs: {args.n_jobs} | seeds/trial: {args.seeds} "
          f"| timesteps/seed: {args.timesteps}")

    sampler = TPESampler(seed=base_config.training.seed)
    # Prune against the running median once a quarter of trials have started.
    pruner = MedianPruner(n_startup_trials=max(5, args.trials // 10), n_warmup_steps=0)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(base_config, train_df, val_df, feature_cols, args.seeds, args.timesteps)
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)

    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"  value (mean val equity): {study.best_value:.2f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    report_path = base_config.output.reports_dir / "tune_best.json"
    save_json({
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state.name == "COMPLETE"]),
        "n_pruned": len([t for t in study.trials if t.state.name == "PRUNED"]),
        "seeds_per_trial": args.seeds,
        "timesteps_per_seed": args.timesteps,
    }, report_path)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
