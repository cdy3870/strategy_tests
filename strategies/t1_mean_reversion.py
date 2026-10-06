"""T1: Short-term mean reversion — RSI(N) < 20 or z-score(N) < -1.5, filtered by 200-MA."""
import numpy as np
import pandas as pd

LABEL   = "T1 Mean Rev"
ENABLED = True


def sidebar_controls(st) -> dict:
    n        = st.slider("Lookback N (RSI / z-score)", 2, 6, 3, key="t1_n")
    hold     = st.slider("Hold (days)", 2, 10, 4, key="t1_hold")
    ma       = st.checkbox("200-day MA filter", value=True, key="t1_ma")
    dyn_size = st.checkbox("Dynamic position sizing", value=True, key="t1_dyn")
    return {"n": n, "hold": hold, "use_ma": ma, "dynamic_size": dyn_size}


def _size(rsi_val: float, z_val: float) -> float:
    """Fixed tiers — not a tunable parameter."""
    if rsi_val < 10 or z_val < -2.5:
        return 2.0
    if rsi_val < 15 or z_val < -2.0:
        return 1.5
    return 1.0


def run(eem: pd.Series, spy: pd.Series, params: dict, cost_bps: float) -> pd.DataFrame:
    n, hold, use_ma  = params["n"], params["hold"], params["use_ma"]
    dynamic_size     = params.get("dynamic_size", False)

    d   = eem.diff()
    g   = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l   = (-d).clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    rsi = 100 - 100 / (1 + g / l.replace(0, np.nan))

    r = eem.pct_change()
    z = (r - r.rolling(n).mean()) / r.rolling(n).std().replace(0, np.nan)

    entry = (rsi < 20) | (z < -1.5)
    if use_ma:
        entry = entry & (eem > eem.rolling(200).mean())

    cost = cost_bps / 10_000
    trades, blocked = [], -1
    for i in range(len(eem) - hold - 1):
        if i <= blocked or not entry.iloc[i]:
            continue
        ei, xi  = i + 1, min(i + 1 + hold, len(eem) - 1)
        sz       = _size(rsi.iloc[i], z.iloc[i]) if dynamic_size else 1.0
        trades.append({"strat": LABEL, "entry_date": eem.index[ei],
                       "exit_date": eem.index[xi], "size": sz,
                       "net_ret": eem.iloc[xi] / eem.iloc[ei] - 1 - 2 * cost})
        blocked = xi
    return pd.DataFrame(trades)
