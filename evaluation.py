from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def run_one_episode(model, vec_env, deterministic: bool = True):
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = vec_env.step(action)

        if len(step_out) == 4:
            obs, _, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, _, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        equity = info.get("equity_usd", vec_env.get_attr("equity_usd")[0])
        equity_curve.append(float(equity))

        trade_info = info.get("last_trade_info")
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)

        if done:
            break

    final_equity = float(equity_curve[-1]) if equity_curve else float(vec_env.get_attr("equity_usd")[0])
    return equity_curve, final_equity, closed_trades


def compute_metrics(closed_trades: list, equity_curve: list, initial_equity: float = 10000.0):
    if not closed_trades or not equity_curve:
        return {
            "sharpe": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "avg_trade_pips": 0.0,
            "n_trades": 0,
            "max_dd": 0.0,
        }

    net_pips = [trade["net_pips"] for trade in closed_trades]
    wins = [pips for pips in net_pips if pips > 0]
    losses = [pips for pips in net_pips if pips <= 0]

    win_rate = len(wins) / len(net_pips)
    avg_trade_pips = float(np.mean(net_pips))
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_wins / gross_losses

    equity_arr = np.array(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(equity_arr)
    drawdowns = (peak - equity_arr) / np.maximum(peak, 1e-9)
    max_dd = float(np.max(drawdowns))

    returns = np.diff(equity_arr) / np.maximum(equity_arr[:-1], 1e-9)
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 24))

    total_return = (equity_arr[-1] - initial_equity) / initial_equity
    calmar = float(total_return / max_dd) if max_dd > 0 else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "calmar": round(calmar, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate, 3),
        "avg_trade_pips": round(avg_trade_pips, 3),
        "n_trades": len(net_pips),
        "max_dd": round(max_dd, 3),
    }


def save_trade_history(closed_trades: list[dict], output_path: str | Path) -> Path | None:
    if not closed_trades:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(closed_trades).to_csv(output_path, index=False)
    return output_path
