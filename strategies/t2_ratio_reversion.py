"""T2: EEM/SPY ratio z-score reversion — buy EEM when ratio stretched vs SPY."""
import numpy as np
import pandas as pd

LABEL   = "T2 Ratio Rev"
ENABLED = True


def sidebar_controls(st) -> dict:
    w    = st.slider("Z-score window (W)", 5, 30, 20, key="t2_w")
    hold = st.slider("Hold (days)", 2, 15, 5, key="t2_hold")
    ma   = st.checkbox("200-day MA filter", value=False, key="t2_ma")
    return {"w": w, "hold": hold, "use_ma": ma}


def run(eem: pd.Series, spy: pd.Series, params: dict, cost_bps: float) -> pd.DataFrame:
    w, hold, use_ma = params["w"], params["hold"], params["use_ma"]
    ratio = np.log(eem / spy)
    z     = (ratio - ratio.rolling(w).mean()) / ratio.rolling(w).std().replace(0, np.nan)

    entry = z < -1.5
    if use_ma:
        entry = entry & (eem > eem.rolling(200).mean())

    cost = cost_bps / 10_000
    trades, blocked = [], -1
    for i in range(len(eem) - hold - 1):
        if i <= blocked or not entry.iloc[i]:
            continue
        ei, xi = i + 1, min(i + 1 + hold, len(eem) - 1)
        trades.append({"strat": LABEL, "entry_date": eem.index[ei],
                       "exit_date": eem.index[xi],
                       "net_ret": eem.iloc[xi] / eem.iloc[ei] - 1 - 2 * cost})
        blocked = xi
    return pd.DataFrame(trades)
