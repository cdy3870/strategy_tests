"""metrics.py — Portfolio statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def permutation_pvalue(tlog: pd.DataFrame, price: pd.Series,
                       n_iter: int = 1000, rng_seed: int = 42) -> dict:
    """
    Null hypothesis: random entry achieves the same Sharpe as the strategy.

    For each of n_iter iterations, randomly pick the same number of entry dates
    from the price series (within the same date range as tlog) and hold for the
    strategy's average hold period.  p-value = fraction of random Sharpes that
    meet or beat the actual Sharpe.  A value ≤ 0.05 means the strategy's timing
    is unlikely to be luck.
    """
    if len(tlog) < 5:
        return dict(pvalue=np.nan, actual_sharpe=np.nan,
                    null_mean=np.nan, null_p95=np.nan)

    n_trades = len(tlog)
    # actual per-trade Sharpe (mean/std of trade returns)
    actual_rets   = tlog["net_ret"].values
    actual_sharpe = (float(actual_rets.mean() / actual_rets.std())
                     if actual_rets.std() > 0 else 0.0)

    # average hold in trading days, clamped to at least 1
    avg_hold = max(1, int(
        (tlog["exit_date"] - tlog["entry_date"]).dt.days.mean()
    ))

    # restrict random entries to the same date window as tlog
    window_start = tlog["entry_date"].min()
    window_end   = tlog["entry_date"].max()
    valid_idx = price.index[
        (price.index >= window_start) &
        (price.index <= window_end - pd.Timedelta(days=avg_hold))
    ]
    if len(valid_idx) < n_trades:
        return dict(pvalue=np.nan, actual_sharpe=actual_sharpe,
                    null_mean=np.nan, null_p95=np.nan)

    price_arr   = price.values
    price_index = price.index
    rng         = np.random.default_rng(rng_seed)
    null_sharpes: list[float] = []

    for _ in range(n_iter):
        chosen = rng.choice(len(valid_idx), size=n_trades, replace=True)
        rand_rets = []
        for ci in chosen:
            ei = price_index.get_loc(valid_idx[ci])
            xi = min(ei + avg_hold, len(price_arr) - 1)
            rand_rets.append(price_arr[xi] / price_arr[ei + 1] - 1)
        r = np.array(rand_rets)
        null_sharpes.append(float(r.mean() / r.std()) if r.std() > 0 else 0.0)

    null = np.array(null_sharpes)
    return dict(
        pvalue        = float((null >= actual_sharpe).mean()),
        actual_sharpe = actual_sharpe,
        null_mean     = float(null.mean()),
        null_p95      = float(np.percentile(null, 95)),
    )


def compute_stats(equity: pd.Series, tlog: pd.DataFrame,
                  initial: float, r_unit: float) -> dict:
    if tlog.empty:
        return {}
    rets    = equity.pct_change().dropna()
    peak    = equity.cummax()
    max_dd  = ((equity - peak) / peak).min()
    net_pnl = equity.iloc[-1] - initial
    wins    = tlog.loc[tlog["dollar_pnl"] > 0, "dollar_pnl"]
    losses  = tlog.loc[tlog["dollar_pnl"] < 0, "dollar_pnl"]
    pf      = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf
    sharpe  = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    monthly = tlog.groupby(tlog["exit_date"].dt.to_period("M"))["dollar_pnl"].sum()
    monthly.index = monthly.index.to_timestamp()
    signs   = (tlog["dollar_pnl"] > 0).astype(int).values
    cw = cl = mcw = mcl = 0
    for s in signs:
        if s:  cw += 1; cl = 0
        else:  cl += 1; cw = 0
        mcw = max(mcw, cw); mcl = max(mcl, cl)
    days_in = tlog.apply(lambda r: (r.exit_date - r.entry_date).days, axis=1).sum()
    total_d = (equity.index[-1] - equity.index[0]).days
    new_hi  = equity[equity >= equity.cummax()]
    gaps    = new_hi.index.to_series().diff().dt.days.dropna()
    return dict(
        net_pnl=net_pnl, net_r=net_pnl / r_unit,
        max_dd=max_dd, max_dd_usd=abs(max_dd * initial),
        pf=pf, win_rate=(tlog["dollar_pnl"] > 0).mean(),
        n_trades=len(tlog), sharpe=sharpe,
        ret_maxdd=(net_pnl / initial) / abs(max_dd) if max_dd != 0 else 0,
        avg_trade=tlog["dollar_pnl"].mean(), avg_r=tlog["r"].mean(),
        avg_win=wins.mean() if not wins.empty else 0,
        avg_loss=losses.mean() if not losses.empty else 0,
        best=tlog["dollar_pnl"].max(), worst=tlog["dollar_pnl"].min(),
        gross_win=wins.sum(), gross_loss=losses.sum(),
        mcw=mcw, mcl=mcl, monthly=monthly,
        prof_months=(monthly > 0).sum(), total_months=len(monthly),
        best_month=monthly.max() if not monthly.empty else 0,
        worst_month=monthly.min() if not monthly.empty else 0,
        pct_in=days_in / total_d * 100 if total_d > 0 else 0,
        longest_flat=int(gaps.max()) if not gaps.empty else 0,
        longs=len(tlog), shorts=0,
        period=(f"{equity.index[0].strftime('%d/%m/%Y')} – "
                f"{equity.index[-1].strftime('%d/%m/%Y')}"),
        days_traded=tlog["exit_date"].nunique(),
    )
