"""Phase 3.8 — regime-stratified evaluation.

An overall Sharpe hides regime risk: a policy that averages out positive but
loses badly in trending markets is not paper-trade ready. This evaluator runs
the chosen model over named date windows and reports per-regime metrics so a
blow-up in any single regime is visible.

Default windows (override with --windows name:start:end ...):
  * 2022_trending  2022-01-01 .. 2023-01-01  (ECB tightening, strong trend)
  * 2023Q1_choppy  2023-01-01 .. 2023-04-01  (banking-crisis chop)
  * 2024H2_lowvol  2024-07-01 .. 2025-01-01  (low-vol drift)

Windows that fall outside the supplied dataset are skipped with a note rather
than failing — the training file and the holdout file cover different spans, so
which windows resolve depends on --dataset-path.

    python regime_eval.py --model-path artifacts/model_eurusd_best.zip \
        --dataset-path data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import build_bot_config, ensure_output_dirs, save_json
from indicators import load_and_preprocess_data
from model_eval import assert_schema_or_raise, evaluate_on_slice, load_model_and_config, slice_by_date


# name -> (start_inclusive, end_exclusive)
DEFAULT_WINDOWS = {
    "2022_trending": ("2022-01-01", "2023-01-01"),
    "2023Q1_choppy": ("2023-01-01", "2023-04-01"),
    "2024H2_lowvol": ("2024-07-01", "2025-01-01"),
}


def parse_windows(spec_list):
    """Parse --windows name:start:end tokens into the {name: (start, end)} dict."""
    if not spec_list:
        return dict(DEFAULT_WINDOWS)
    windows = {}
    for spec in spec_list:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad window spec {spec!r}; expected name:start:end (e.g. 2022:2022-01-01:2023-01-01)")
        name, start, end = parts
        windows[name] = (start, end)
    return windows


def parse_args():
    parser = argparse.ArgumentParser(description="Regime-stratified evaluation of a saved model (Phase 3.8).")
    parser.add_argument("--model-path", type=str, help="Path to the saved model .zip (defaults to artifacts model).")
    parser.add_argument("--dataset-path", type=str, help="Dataset to slice regimes from (defaults to training dataset).")
    parser.add_argument("--windows", nargs="*", help="Custom windows as name:start:end tokens.")
    parser.add_argument("--n-trials", type=int, default=1, help="Trials before selection (feeds Deflated Sharpe).")
    return parser.parse_args()


def evaluate_regimes(model, df, feature_cols, config, windows, n_trials=1):
    """Run the model over each named date window; return {name: result}."""
    data_start, data_end = df.index.min(), df.index.max()
    results = {}
    for name, (start, end) in windows.items():
        window_df = slice_by_date(df, start=start, end=end)
        if len(window_df) <= config.env.window_size + 2:
            print(f"[regime] skip {name} ({start}..{end}): "
                  f"{len(window_df)} bars in dataset span {data_start.date()}..{data_end.date()}")
            results[name] = {"skipped": True, "reason": "no/insufficient data in window",
                             "start": start, "end": end, "bars": int(len(window_df))}
            continue
        result = evaluate_on_slice(model, window_df, feature_cols, config,
                                   config.output.vecnormalize_path, n_trials=n_trials)
        result["start"], result["end"] = start, end
        results[name] = result
    return results


def main():
    args = parse_args()

    model_path = Path(args.model_path).resolve() if args.model_path else build_bot_config().output.model_file_path
    model, config, config_path = load_model_and_config(model_path)
    ensure_output_dirs(config)

    dataset_path = Path(args.dataset_path).resolve() if args.dataset_path else config.data.dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    windows = parse_windows(args.windows)

    df, feature_cols = load_and_preprocess_data(dataset_path)
    assert_schema_or_raise(config_path, feature_cols)

    print(f"Model     : {model_path}")
    print(f"Dataset   : {dataset_path}  ({df.index.min().date()} .. {df.index.max().date()})")
    print(f"Windows   : {list(windows.keys())}")

    results = evaluate_regimes(model, df, feature_cols, config, windows, n_trials=args.n_trials)

    print("\n" + "=" * 72)
    print("REGIME-STRATIFIED RESULTS")
    print("=" * 72)
    for name, r in results.items():
        if r.get("skipped"):
            print(f"  {name:16s} | SKIPPED ({r['reason']}, {r['bars']} bars)")
            continue
        m = r["metrics"]
        ci = r["statistics"]["bootstrap_sharpe_ci"]
        print(f"  {name:16s} | Sharpe={m['sharpe']:>7.3f} (CI[{ci['ci_low']}, {ci['ci_high']}]) "
              f"| PF={m['profit_factor']:>6.3f} | Trades={m['n_trades']:>4d} "
              f"| MaxDD={m['max_dd']:>5.3f} | FinalEq={r['final_equity']:.2f}")

    report_path = config.output.reports_dir / "regime_metrics.json"
    save_json({
        "model": str(model_path),
        "dataset": str(dataset_path),
        "n_trials": args.n_trials,
        "windows": {k: {"start": v[0], "end": v[1]} for k, v in windows.items()},
        "results": results,
    }, report_path)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
