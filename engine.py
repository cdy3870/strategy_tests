"""engine.py — Equity curve builder and walk-forward split utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_equity(strat_dfs: list[pd.DataFrame], date_index: pd.DatetimeIndex,
                 initial: float, r_unit: float) -> tuple[pd.Series, pd.DataFrame]:
    alloc = initial / len(strat_dfs)
    caps  = [alloc] * len(strat_dfs)
    rows  = []
    for si, df in enumerate(strat_dfs):
        for _, r in df.iterrows():
            rows.append({**r.to_dict(), "_si": si})
    rows.sort(key=lambda x: x["exit_date"])

    eq_pts, trade_log = [(date_index[0], initial)], []
    for r in rows:
        si   = r["_si"]
        size = r.get("size", 1.0)
        dpnl = caps[si] * size * r["net_ret"]
        caps[si] *= (1 + size * r["net_ret"])
        eq_pts.append((r["exit_date"], sum(caps)))
        trade_log.append({"strat": r["strat"], "entry_date": r["entry_date"],
                          "exit_date": r["exit_date"], "net_ret": r["net_ret"],
                          "size": size, "dollar_pnl": dpnl, "r": dpnl / r_unit})

    raw = pd.Series([v for _, v in eq_pts], index=[d for d, _ in eq_pts])
    raw = raw[~raw.index.duplicated(keep="last")]
    eq  = raw.reindex(date_index, method="ffill").fillna(method="bfill")
    tlog = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(
        columns=["strat", "entry_date", "exit_date", "net_ret", "dollar_pnl", "r"])
    return eq, tlog


def purged_walk_forward_splits(
    index: pd.DatetimeIndex,
    n_splits: int,
    test_years: int,
    embargo_days: int,
) -> list[dict]:
    """
    Expanding-window walk-forward splits with embargo gap.
    Returns list of dicts: {fold, train_end, embargo_end, test_start, test_end}
    Train always starts at index[0]; each fold expands the train window by test_years.
    The embargo gap prevents leakage at the train/test boundary.
    """
    start = index[0]
    end   = index[-1]
    test_td    = pd.DateOffset(years=test_years)
    embargo_td = pd.DateOffset(days=embargo_days)

    total_years = (end - start).days / 365.25
    min_train   = int(max(1, total_years - n_splits * test_years))

    folds = []
    for i in range(n_splits):
        train_end   = start + pd.DateOffset(years=min_train + i * test_years)
        embargo_end = train_end + embargo_td
        test_start  = embargo_end
        test_end    = test_start + test_td
        if test_end > end:
            test_end = end
        if test_start >= end:
            break
        folds.append(dict(
            fold        = i + 1,
            train_end   = train_end,
            embargo_end = embargo_end,
            test_start  = test_start,
            test_end    = test_end,
        ))
    return folds


def fold_stats(tlog: pd.DataFrame, fold: dict,
               equity: pd.Series, r_unit: float) -> dict:
    """Metrics for a single OOS fold."""
    mask = (tlog["entry_date"] >= fold["test_start"]) & \
           (tlog["entry_date"] <  fold["test_end"])
    sub  = tlog[mask]
    eq   = equity[fold["test_start"]:fold["test_end"]]
    if sub.empty or len(eq) < 2:
        return dict(fold=fold["fold"], n_trades=0, sharpe=np.nan,
                    win_rate=np.nan, net_pnl=np.nan, pf=np.nan,
                    test_start=fold["test_start"], test_end=fold["test_end"])
    rets   = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    wins   = sub.loc[sub["dollar_pnl"] > 0, "dollar_pnl"]
    losses = sub.loc[sub["dollar_pnl"] < 0, "dollar_pnl"]
    pf     = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf
    return dict(
        fold       = fold["fold"],
        test_start = fold["test_start"],
        test_end   = fold["test_end"],
        n_trades   = len(sub),
        sharpe     = round(sharpe, 3),
        win_rate   = round((sub["dollar_pnl"] > 0).mean(), 3),
        net_pnl    = round(sub["dollar_pnl"].sum(), 2),
        r_total    = round(sub["r"].sum(), 2),
        pf         = round(pf, 2),
    )
