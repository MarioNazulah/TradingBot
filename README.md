# ReinforcementTrading

`ReinforcementTrading` is a research-grade PPO trading bot for EUR/USD data. It builds technical features from OHLCV candles, trains a reinforcement-learning policy in a custom Gymnasium environment, and evaluates the result on held-out data.

This repository is intentionally code-only. Local datasets, trained models, checkpoints, TensorBoard logs, and personal experiment notes are excluded from git.

## What The Project Does

- Preprocesses FX candles into relative features such as RSI, ATR-normalized moving-average spreads, and session features.
- Trains a PPO policy against a custom `ForexTradingEnv`.
- Evaluates the saved model on the out-of-sample test split using the exact same saved run configuration.
- Supports walk-forward validation for more realistic robustness checks.

## Repository Layout

- `train_agent.py` trains the model and saves outputs into `artifacts/`.
- `test_agent.py` evaluates a saved model using the matching saved config sidecar.
- `walk_forward.py` runs expanding-window walk-forward validation.
- `trading_env.py` contains the trading environment.
- `indicators.py` loads CSV data and engineers features.
- `config.py` is the shared source of truth for dataset paths, environment settings, and output paths.
- `evaluation.py` contains shared episode and metric logic.

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
python train_agent.py --enable-obsidian-log
```

Training writes:

- `artifacts/model_eurusd_best.zip`
- `artifacts/model_eurusd_best_config.json`
- `artifacts/latest_run_config.json`
- `artifacts/reports/latest_metrics.json`

## Evaluate

Evaluation loads the saved model and, when present, the matching saved config sidecar so the action map and environment settings stay aligned.

```powershell
python test_agent.py --no-plot
```

Optional overrides:

```powershell
python test_agent.py --model-path artifacts\model_eurusd_best.zip
python test_agent.py --dataset-path data\your_file.csv --output-csv artifacts\reports\trades.csv
```

## Walk-Forward Validation

```powershell
python walk_forward.py --folds 5 --no-plot
```

## Tests

```powershell
python -m pytest -q
```

Some smoke tests skip automatically when local datasets or saved models are not present, which keeps the code-only repository runnable after cloning.

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
