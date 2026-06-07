# ReinforcementTrading

`ReinforcementTrading` is a research-grade PPO trading bot for EUR/USD data. It builds technical features from OHLCV candles, trains a reinforcement-learning policy in a custom Gymnasium environment, and evaluates the result on held-out data.

This repository is intentionally code-only. Local datasets, trained models, checkpoints, TensorBoard logs, and personal experiment notes are excluded from git.

## What The Project Does

- Preprocesses FX candles into relative features such as RSI, ATR-normalized moving-average spreads, and session features.
- Trains a PPO policy against a custom `ForexTradingEnv`.
- Evaluates the saved model on the out-of-sample test split using the exact same saved run configuration.
- Supports walk-forward validation for more realistic robustness checks.

## Repository Layout

- `train_agent.py` trains the model (multi-seed, EvalCallback best-model selection) and saves outputs into `artifacts/`.
- `eval_agent.py` evaluates a saved model using the matching saved config sidecar.
- `walk_forward.py` runs expanding-window walk-forward validation.
- `trading_env.py` contains the trading environment.
- `indicators.py` loads CSV data and engineers features.
- `config.py` is the shared source of truth for dataset paths, environment settings, and output paths.
- `evaluation.py` contains shared episode/metric logic plus the statistical battery (t-test, bootstrap Sharpe CI, deflated Sharpe).
- `baselines.py` scores random / buy-and-hold / MA-crossover reference policies on the same test slice and cost model.
- `tune.py` runs the Optuna hyperparameter sweep (TPE + MedianPruner).
- `bakeoff.py` trains and ranks PPO / MaskablePPO / RecurrentPPO / QR-DQN by out-of-sample Sharpe.
- `model_eval.py` shared helpers for evaluating a saved model on a date slice.
- `holdout_eval.py` runs the one-time true-holdout evaluation on the strictly-newer second dataset.
- `regime_eval.py` reports per-regime metrics over named date windows.

## Data Format

The loader expects a CSV with:

- A GMT/UTC timestamp column containing `time` in the header name.
- `Open`, `High`, `Low`, `Close`, `Volume` columns.

Timestamps are parsed as UTC explicitly before London/New York session features are created.

## Setup

Use a clean virtual environment, then install the runtime or dev dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

## Train

The default training dataset path is configured in `config.py`. Outputs go to `artifacts/`.

```powershell
python train_agent.py
```

Useful overrides:

```powershell
python train_agent.py --dataset-path data\your_file.csv --timesteps 20000 --seed 123 --no-plot
python train_agent.py --no-experiment-log
```

Each training run also writes a human-readable markdown experiment note to
`brain/tradingbot-brain/experiments` by default.

Training writes:

- `artifacts/model_eurusd_best.zip`
- `artifacts/model_eurusd_best_config.json`
- `artifacts/latest_run_config.json`
- `artifacts/reports/latest_metrics.json`

## Evaluate

Evaluation loads the saved model and, when present, the matching saved config sidecar so the action map and environment settings stay aligned.

```powershell
python eval_agent.py --no-plot
```

Optional overrides:

```powershell
python eval_agent.py --model-path artifacts\model_eurusd_best.zip
python eval_agent.py --dataset-path data\your_file.csv --output-csv artifacts\reports\trades.csv
```

## Walk-Forward Validation

```powershell
python walk_forward.py --folds 5 --no-plot
```

Walk-forward runs also write a markdown experiment note by default. Use
`--no-experiment-log` to suppress it for a specific run.

## Tests

```powershell
python -m pytest -q
```

Some smoke tests skip automatically when local datasets or saved models are not present, which keeps the code-only repository runnable after cloning.

## Phase 3 — Training & Evaluation Rigor

Training selects the best model online with SB3's `MaskableEvalCallback` (no
post-hoc checkpoint sweep) and runs multiple independent seeds by default so a
reported number is an aggregate, not an anecdote.

```powershell
# Train 5 seeds (default), evaluating on validation every 10k steps:
python train_agent.py --n-seeds 5 --eval-freq 10000 --no-plot
```

The aggregate report (`artifacts/reports/multiseed_metrics.json`) gives
mean ± stddev and `[min, max]` for every metric and flags any whose stddev
exceeds 30% of the mean as noise.

Reference baselines (random, buy-and-hold, MA-crossover) run through the same
env and cost model so the agent's Sharpe is comparable:

```powershell
python baselines.py --no-plot
```

Hyperparameter sweep (Optuna TPE + MedianPruner, objective = mean validation
final equity across seeds, pruned at 25% of the budget):

```powershell
python tune.py --trials 50 --n-jobs 6 --seeds 3 --timesteps 60000
```

Algorithm bake-off, ranked by out-of-sample Sharpe across a shared seed pool:

```powershell
python bakeoff.py --algos PPO MaskablePPO RecurrentPPO QR-DQN --seeds 5
```

Statistical reporting (`evaluation.py`): every test episode is reported with a
one-sample t-test that the mean per-bar return is positive, a bootstrapped 95%
CI for the per-trade Sharpe (10k resamples on R-multiples), and the Deflated
Sharpe Ratio, which discounts the Sharpe for the number of configurations tried
(pass `--n-trials` from the sweep). Do not quote a Sharpe without its CI.

True holdout (Phase 3.7) — the strictly-newer slice of the second dataset,
used exactly once (a marker file guards against accidental reuse):

```powershell
python holdout_eval.py --model-path artifacts\model_eurusd_best.zip --n-trials 50
```

Regime-stratified evaluation (Phase 3.8) — per-window metrics so a regime-
specific blow-up is visible:

```powershell
python regime_eval.py --dataset-path data\test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv
```

## Current Limitations

- This is research code, not production trading infrastructure.
- Execution modeling is simplified: no live broker integration, no order-book data, and no realistic latency model.
- Performance metrics are only as good as the local dataset quality and split policy.
- Large local artifacts are intentionally excluded from version control; a fresh clone will need its own dataset and training run.

## GitHub Push Checklist

Inside `C:\MyFirstProject\ReinforcementTrading`:

```powershell
git init
git add .
git commit -m "Initial trading bot cleanup"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

If GitHub rejects the push because of a large tracked file, remove it from the index, confirm `.gitignore` covers it, commit again, and retry the push.
