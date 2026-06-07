import os
from datetime import datetime
from pathlib import Path


def log_experiment_to_obsidian(config: dict, results: dict, notes: str = "", vault_path: str | Path = "./brain/tradingbot-brain/experiments"):
    vault_path = Path(vault_path)
    os.makedirs(vault_path, exist_ok=True)

    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M")
    filename = vault_path / f"{run_id}.md"

    content = f"""---
date: {now.strftime("%Y-%m-%d")}
run_id: {run_id}
status: done
---

## Config
- action_space: {config.get('n_actions', 'N/A')}
- sl_opts: {config.get('sl_opts', 'N/A')}
- tp_opts: {config.get('tp_opts', 'N/A')}
- window_size: {config.get('window_size', 'N/A')}
- timesteps: {config.get('timesteps', 'N/A')}
- dataset: {config.get('dataset', 'N/A')}
- hold_reward_weight: {config.get('hold_reward_weight', 'N/A')}
- open_penalty_pips: {config.get('open_penalty_pips', 'N/A')}
- time_penalty_pips: {config.get('time_penalty_pips', 'N/A')}

## Results
- Train Final Equity: {results.get('final_equity_train', 'N/A')}
- Val Final Equity: {results.get('final_equity_val', 'N/A')}
- OOS Final Equity: {results.get('final_equity_test', 'N/A')}
- OOS Sharpe: {results.get('sharpe', 'N/A')}
- OOS Calmar: {results.get('calmar', 'N/A')}
- Profit Factor: {results.get('profit_factor', 'N/A')}
- Win Rate: {results.get('win_rate', 'N/A')}
- Avg trade (pips): {results.get('avg_trade_pips', 'N/A')}
- # trades: {results.get('n_trades', 'N/A')}
- Max DD: {results.get('max_dd', 'N/A')}

## Notes
{notes}

## What to try next

"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Experiment logged: {filename}")
