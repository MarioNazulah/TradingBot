"""Phase 2.3 — Data quality control.

Fail-fast checks run before any feature engineering or training so that a
corrupted feed cannot silently poison a model. Hard violations (duplicate
timestamps, non-monotonic index, OHLC inconsistency) raise ``DataQCError``.
Soft observations (weekday gaps, non-positive volume) are reported and logged
but do not abort, because they are common in real FX feeds (holidays, broker
quirks) and are better surfaced than fatal.

A ``dataset_fingerprint.json`` is persisted so a saved model can later be tied
back to the exact bytes of price data it was trained on.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRICE_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close")
# FX trades ~24h on weekdays; a gap larger than this on a weekday is a hole in
# the feed worth flagging. Weekend gaps (Fri close -> Sun/Mon open) are normal
# and explicitly allowed.
DEFAULT_MAX_WEEKDAY_GAP = pd.Timedelta(hours=4)


class DataQCError(ValueError):
    """Raised when the dataset fails a hard data-quality invariant."""


def _price_sha256(df: pd.DataFrame, price_cols: Sequence[str]) -> str:
    """Deterministic SHA-256 over the index and price columns.

    Hashing the raw float64 bytes plus the int64-nanosecond index makes the
    fingerprint sensitive to any change in price data or timestamps while being
    independent of which derived features happen to be attached.
    """
    digest = hashlib.sha256()
    # Index as int64 nanoseconds (works for tz-aware DatetimeIndex via asi8).
    idx_ns = np.ascontiguousarray(df.index.asi8, dtype=np.int64)
    digest.update(idx_ns.tobytes())
    prices = np.ascontiguousarray(df[list(price_cols)].to_numpy(dtype=np.float64))
    digest.update(prices.tobytes())
    return digest.hexdigest()


def compute_fingerprint(df: pd.DataFrame, price_cols: Sequence[str] = PRICE_COLUMNS) -> dict[str, Any]:
    """Build a dataset fingerprint dict (row count, time bounds, price hash)."""
    return {
        "row_count": int(len(df)),
        "first_timestamp": df.index[0].isoformat() if len(df) else None,
        "last_timestamp": df.index[-1].isoformat() if len(df) else None,
        "price_columns": list(price_cols),
        "price_sha256": _price_sha256(df, price_cols),
    }


def run_data_qc(
    df: pd.DataFrame,
    *,
    price_cols: Sequence[str] = PRICE_COLUMNS,
    volume_col: str | None = "Volume",
    max_weekday_gap: pd.Timedelta = DEFAULT_MAX_WEEKDAY_GAP,
    fingerprint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an OHLCV DataFrame indexed by a UTC DatetimeIndex.

    Hard failures raise ``DataQCError``:
      * non-DatetimeIndex
      * duplicate timestamps
      * non-monotonic-increasing index
      * OHLC inconsistency (High < Low, High < max(Open, Close),
        Low > min(Open, Close))

    Soft observations are logged and returned in the report (not fatal):
      * weekday gaps greater than ``max_weekday_gap``
      * non-positive volume rows (percentage logged)

    Returns a report dict that always includes the dataset fingerprint. If
    ``fingerprint_path`` is given the fingerprint is also written there as JSON.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataQCError(f"Expected a DatetimeIndex, got {type(df.index).__name__}.")
    if len(df) == 0:
        raise DataQCError("Dataset is empty.")

    missing = [c for c in price_cols if c not in df.columns]
    if missing:
        raise DataQCError(f"Missing required price columns: {missing}")

    # --- Duplicate timestamps (hard) ---
    dup_mask = df.index.duplicated(keep=False)
    if dup_mask.any():
        first_dup = df.index[dup_mask][0]
        raise DataQCError(
            f"Duplicate timestamps detected on {int(dup_mask.sum())} row(s); "
            f"first duplicate index: {first_dup}"
        )

    # --- Monotonic, increasing index (hard) ---
    if not df.index.is_monotonic_increasing:
        raise DataQCError("Index is not monotonically increasing.")

    # --- OHLC consistency (hard) ---
    ohlc = df[list(price_cols)].dropna()
    bad = (
        (ohlc["High"] < ohlc["Low"])
        | (ohlc["High"] < ohlc[["Open", "Close"]].max(axis=1))
        | (ohlc["Low"] > ohlc[["Open", "Close"]].min(axis=1))
    )
    if bad.any():
        raise DataQCError(
            f"OHLC consistency check failed on {int(bad.sum())} row(s); "
            f"first bad index: {ohlc.index[bad][0]}"
        )

    # --- Weekday gaps (soft) ---
    # Flag holes larger than max_weekday_gap, but excuse the normal FX weekend
    # close: a gap is allowed if the span between the two bars crosses a
    # Saturday or Sunday (this also covers Friday-evening -> Monday-open).
    index_series = df.index.to_series()
    deltas = index_series.diff()
    prev_ts = index_series.shift(1)
    candidates = np.flatnonzero((deltas > max_weekday_gap).to_numpy())
    weekday_gaps = []
    for i in candidates:
        start = prev_ts.iloc[i]
        end = df.index[i]
        span_days = pd.date_range(start.normalize(), end.normalize(), freq="D")
        if (span_days.dayofweek >= 5).any():
            continue  # weekend gap — allowed
        weekday_gaps.append({
            "after": start.isoformat(),
            "gap_hours": round(deltas.iloc[i].total_seconds() / 3600.0, 3),
        })
    if weekday_gaps:
        logger.warning(
            "Data QC: %d weekday gap(s) > %s found; first after %s (%.2fh).",
            len(weekday_gaps), max_weekday_gap, weekday_gaps[0]["after"], weekday_gaps[0]["gap_hours"],
        )

    # --- Non-positive volume (soft, log percentage) ---
    volume_report: dict[str, Any] = {"checked": False}
    if volume_col is not None and volume_col in df.columns:
        vol = pd.to_numeric(df[volume_col], errors="coerce")
        nonpos = int((vol <= 0).sum())
        pct = round(100.0 * nonpos / len(df), 4)
        volume_report = {"checked": True, "non_positive_count": nonpos, "non_positive_pct": pct}
        if nonpos:
            logger.warning(
                "Data QC: %d/%d rows (%.4f%%) have non-positive %s.",
                nonpos, len(df), pct, volume_col,
            )

    fingerprint = compute_fingerprint(df, price_cols)

    report = {
        "fingerprint": fingerprint,
        "weekday_gaps": weekday_gaps,
        "volume": volume_report,
        "max_weekday_gap_hours": max_weekday_gap.total_seconds() / 3600.0,
    }

    if fingerprint_path is not None:
        fingerprint_path = Path(fingerprint_path)
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        with fingerprint_path.open("w", encoding="utf-8") as handle:
            json.dump(fingerprint, handle, indent=2, sort_keys=True)
        report["fingerprint_path"] = str(fingerprint_path)
        logger.info("Data QC: dataset fingerprint written to %s", fingerprint_path)

    return report
