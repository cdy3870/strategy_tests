"""
dashboard.py — Interactive trading dashboard (Streamlit + Plotly)
Run: streamlit run dashboard.py

To add a strategy: drop a new file in strategies/ following the interface:
  LABEL: str
  ENABLED: bool
  sidebar_controls(st) -> dict   # renders controls, returns params
  run(eem, spy, params, cost_bps) -> pd.DataFrame  # returns trades
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
import streamlit as st

import strategies
from engine  import build_equity, purged_walk_forward_splits, fold_stats
from metrics import compute_stats, permutation_pvalue
from charts  import equity_chart, monthly_chart, scatter_chart, annual_chart, kpi_card, km_row, section_header

# ─────────────────────────────────────────────────────────────────────────────
# Strategy discovery
# ─────────────────────────────────────────────────────────────────────────────

def load_strategies() -> list:
    mods = []
    for info in sorted(pkgutil.iter_modules(strategies.__path__), key=lambda x: x.name):
        mod = importlib.import_module(f"strategies.{info.name}")
        if hasattr(mod, "run") and hasattr(mod, "sidebar_controls"):
            mods.append(mod)
    return mods


STRATEGY_MODS = load_strategies()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="EEM Strategy Dashboard",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container { padding-top: 1rem; }
  .metric-card {
    background: #1e222d; border: 1px solid #2a2e39;
    border-radius: 6px; padding: 12px 16px; margin-bottom: 4px;
  }
  .metric-label { font-size: 11px; color: #787b86; margin-bottom: 4px; }
  .metric-value { font-size: 20px; font-weight: 700; }
  .metric-sub   { font-size: 11px; color: #787b86; margin-top: 2px; }

  /* ── Section-header tooltips ─────────────────────────────────────── */
  .tt-wrap {
    position: relative; display: inline-block;
    cursor: help; color: #787b86; font-size: 13px;
    margin-left: 6px; vertical-align: middle; font-weight: 400;
  }
  .tt-wrap .tt {
    visibility: hidden; opacity: 0;
    background: #2a2e39; color: #d1d4dc;
    font-size: 12px; font-weight: 400; line-height: 1.55;
    border: 1px solid #363a46; border-radius: 6px;
    padding: 10px 14px;
    position: absolute; left: 22px; top: -6px; width: 300px;
    z-index: 9999; pointer-events: none; white-space: normal;
    transition: opacity 0.15s;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45);
  }
  .tt-wrap:hover .tt { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading price data…")
def load_data(start: str, end: str) -> tuple[pd.Series, pd.Series]:
    raw   = yf.download(["EEM", "SPY"], start=start, end=end,
                        auto_adjust=True, progress=False)
    close = raw["Close"]
    eem, spy = close["EEM"].dropna(), close["SPY"].dropna()
    return eem.align(spy, join="inner")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("### Date range")
    start_date = st.date_input("Start", value=pd.Timestamp("2005-01-01"))
    end_date   = st.date_input("End",   value=pd.Timestamp("2024-12-31"))
    st.markdown("### Capital")
    initial  = float(st.number_input("Starting capital ($)", value=10_000, step=1_000))
    r_unit   = float(st.number_input("Risk per trade 1R ($)", value=250, step=25))
    cost_bps = st.slider("Cost (bps, one-way)", 0, 20, 5)
    st.markdown("### Walk-forward")
    use_wf       = st.checkbox("Purged walk-forward", value=True)
    n_splits     = st.slider("Folds", 2, 5, 3, key="wf_n",
                             disabled=not use_wf)
    test_years   = st.slider("Test window (years)", 1, 3, 2, key="wf_test",
                             disabled=not use_wf)
    embargo_days = st.slider("Embargo gap (days)", 0, 30, 10, key="wf_embargo",
                             disabled=not use_wf)
    st.markdown("---")

    st.markdown("### Strategies")
    all_params = {}
    for mod in STRATEGY_MODS:
        enabled_default = getattr(mod, "ENABLED", True)
        on = st.checkbox(mod.LABEL, value=enabled_default,
                         key=f"{mod.LABEL}_on")
        with st.expander("Parameters", expanded=False):
            params = mod.sidebar_controls(st)
        params["on"] = on
        all_params[mod.LABEL] = params

# ─────────────────────────────────────────────────────────────────────────────
# Run strategies
# ─────────────────────────────────────────────────────────────────────────────

# st.markdown("# Performance Dashboard")
st.markdown(f"Risk **${r_unit:.0f}** per trade (1R) &nbsp;·&nbsp; "
            f"{start_date.strftime('%d/%m/%Y')} – {end_date.strftime('%d/%m/%Y')}")

eem, spy = load_data(str(start_date), str(end_date))

strat_dfs = []
for mod in STRATEGY_MODS:
    params = all_params[mod.LABEL]
    if not params.get("on", True):
        continue
    df = mod.run(eem, spy, params, cost_bps)
    if not df.empty:
        strat_dfs.append(df)

if not strat_dfs:
    st.warning("No strategies enabled or no trades generated.")
    st.stop()

equity, tlog = build_equity(strat_dfs, eem.index, initial, r_unit)
stats = compute_stats(equity, tlog, initial, r_unit)

if not stats:
    st.warning("No trades in selected period.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────────────────────────────────────

spy_bah_pnl = (spy.iloc[-1] / spy.iloc[0] * initial) - initial

c1, c2, c3, c4, c5, c6 = st.columns(6)
net_col = "green" if stats["net_pnl"] >= 0 else "red"
c1.markdown(kpi_card("Net Profit",
    f"{'+'if stats['net_pnl']>=0 else ''}${abs(stats['net_pnl']):,.0f}",
    net_col, f"{stats['net_r']:+.1f}R"), unsafe_allow_html=True)
spy_col = "green" if spy_bah_pnl >= 0 else "red"
c2.markdown(kpi_card("SPY B&H Profit",
    f"{'+'if spy_bah_pnl>=0 else ''}${abs(spy_bah_pnl):,.0f}",
    spy_col, f"vs strategy: {'+'if stats['net_pnl']-spy_bah_pnl>=0 else ''}${stats['net_pnl']-spy_bah_pnl:,.0f}"),
    unsafe_allow_html=True)
c3.markdown(kpi_card("Max Drawdown",
    f"(${stats['max_dd_usd']:,.0f})", "red",
    f"{abs(stats['max_dd']*100):.1f}R"), unsafe_allow_html=True)
c4.markdown(kpi_card("Profit Factor", f"{stats['pf']:.2f}", "white",
    "All trades"), unsafe_allow_html=True)
wr_col = "green" if stats["win_rate"] >= 0.5 else "red"
c5.markdown(kpi_card("Percent Profitable", f"{stats['win_rate']*100:.1f}%",
    wr_col, f"{stats['n_trades']} trades"), unsafe_allow_html=True)
c6.markdown(kpi_card("Profitable Months",
    f"{stats['prof_months']}/{stats['total_months']}", "green",
    f"worst {stats['worst_month']/r_unit:+.1f}R"), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Equity curve
# ─────────────────────────────────────────────────────────────────────────────

folds = (purged_walk_forward_splits(eem.index, n_splits, test_years, embargo_days)
         if use_wf else [])

show_bah = st.checkbox("Show SPY buy-and-hold", value=False)
st.plotly_chart(
    equity_chart(equity, initial, tlog,
                 spy if show_bah else None,
                 folds or None),
    use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Purged walk-forward fold results
# ─────────────────────────────────────────────────────────────────────────────

if use_wf and folds:
    st.markdown(section_header(
        "Purged Walk-forward Results",
        "Each row is an out-of-sample (OOS) window the strategy never saw during "
        "training. Look for: consistent Sharpe ≥ 0.3 across all folds, win rate "
        "similar to in-sample, and no single fold carrying all the P&amp;L. "
        "3/3 folds passing is far stronger evidence than 1/3."
    ), unsafe_allow_html=True)
    fold_rows = [fold_stats(tlog, f, equity, r_unit) for f in folds]

    # summary table
    rows_display = []
    for fr in fold_rows:
        verdict = ("PASS" if pd.notna(fr["sharpe"]) and fr["sharpe"] >= 0.3
                   and fr["n_trades"] >= 10 else "FAIL")
        rows_display.append({
            "Fold":      fr["fold"],
            "OOS start": pd.Timestamp(fr["test_start"]).strftime("%Y-%m-%d"),
            "OOS end":   pd.Timestamp(fr["test_end"]).strftime("%Y-%m-%d"),
            "Trades":    fr["n_trades"],
            "Sharpe":    f"{fr['sharpe']:.3f}" if pd.notna(fr["sharpe"]) else "—",
            "Win %":     f"{fr['win_rate']*100:.1f}%" if pd.notna(fr.get("win_rate")) else "—",
            "Net P&L":   f"${fr['net_pnl']:,.0f}" if pd.notna(fr["net_pnl"]) else "—",
            "P-factor":  f"{fr['pf']:.2f}" if pd.notna(fr["pf"]) else "—",
            "Verdict":   verdict,
        })

    tbl = pd.DataFrame(rows_display)

    def _color_verdict(val):
        color = "#26a69a" if val == "PASS" else "#ef5350"
        return f"color: {color}; font-weight: 700"

    st.dataframe(
        tbl.style.applymap(_color_verdict, subset=["Verdict"]),
        use_container_width=True, hide_index=True,
    )

    pass_count = sum(1 for r in rows_display if r["Verdict"] == "PASS")
    summary_color = "#26a69a" if pass_count == len(fold_rows) else (
                    "#f39c12" if pass_count > 0 else "#ef5350")
    st.markdown(
        f'<div style="font-size:15px;font-weight:700;color:{summary_color};'
        f'margin:6px 0 12px">'
        f'{pass_count} / {len(fold_rows)} folds PASS</div>',
        unsafe_allow_html=True)

    # ── Permutation significance test ─────────────────────────────────────────
    st.markdown(section_header(
        "Permutation significance test",
        "1,000 random traders entered EEM at random dates in the same OOS window "
        "and held for the strategy's average hold period. "
        "p ≤ 0.05 means only 5% of random strategies matched your edge — statistically significant. "
        "p > 0.10 means the OOS result is consistent with luck.",
        tag="h4"
    ), unsafe_allow_html=True)
    oos_mask = tlog["entry_date"] >= pd.Timestamp(folds[0]["test_start"])
    oos_tlog = tlog[oos_mask]
    with st.spinner("Running 1 000 permutations…"):
        perm = permutation_pvalue(oos_tlog, eem)

    pc1, pc2, pc3, pc4 = st.columns(4)
    p = perm["pvalue"]
    p_color = "#26a69a" if p <= 0.05 else ("#f39c12" if p <= 0.10 else "#ef5350")
    p_label = "p ≤ 0.05  ✓ significant" if p <= 0.05 else (
              "p ≤ 0.10  borderline"     if p <= 0.10 else
              "p > 0.10  not significant")
    from charts import kpi_card as _kc
    pc1.markdown(_kc("p-value",        f"{p:.3f}", p_color, p_label),
                 unsafe_allow_html=True)
    pc2.markdown(_kc("Actual Sharpe",  f"{perm['actual_sharpe']:.3f}", "white",
                     "per-trade (unannualized)"), unsafe_allow_html=True)
    pc3.markdown(_kc("Null mean",      f"{perm['null_mean']:.3f}", "white",
                     "avg of 1 000 shuffles"), unsafe_allow_html=True)
    pc4.markdown(_kc("Null 95th pct",  f"{perm['null_p95']:.3f}", "white",
                     "threshold at α=0.05"), unsafe_allow_html=True)
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Key metrics (horizontal)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(section_header(
    "Key Metrics",
    "Full-period aggregate stats across all enabled strategies. "
    "These are in-sample figures — use the Walk-forward section above "
    "to assess out-of-sample validity."
), unsafe_allow_html=True)
km1, km2, km3, km4 = st.columns(4)

with km1:
    st.markdown(section_header(
        "Overview", "Trade count and time in market. Low % in market (&lt;20%) is healthy "
        "for mean reversion — it means the strategy is selective. Watch Longest Flat: "
        "prolonged gaps without new highs erode confidence and invite premature abandonment.",
        tag="bold"
    ), unsafe_allow_html=True)
    km_row("Total Trades",      str(stats["n_trades"]))
    km_row("Long / Short",      f"{stats['longs']} / {stats['shorts']}")
    km_row("Win Rate",          f"{stats['win_rate']*100:.1f}%",
           "#26a69a" if stats["win_rate"] >= 0.5 else "#ef5350")
    km_row("% in Market",       f"{stats['pct_in']:.1f}%")
    km_row("Longest Flat",      f"{stats['longest_flat']} days")

with km2:
    st.markdown(section_header(
        "P&amp;L", "Gross win vs. gross loss reveals where the edge lives. "
        "If avg win ≈ avg loss but win rate &gt;50%, the edge is frequency. "
        "If avg win &gt;&gt; avg loss at ~50% win rate, the edge is magnitude. "
        "Both are valid strategies, but fragile in different ways.",
        tag="bold"
    ), unsafe_allow_html=True)
    km_row("Net Profit",        f"${stats['net_pnl']:,.0f}",
           "#26a69a" if stats["net_pnl"] >= 0 else "#ef5350")
    km_row("Gross Profit",      f"${stats['gross_win']:,.0f}",  "#26a69a")
    km_row("Gross Loss",        f"(${abs(stats['gross_loss']):,.0f})", "#ef5350")
    km_row("Avg Trade",         f"${stats['avg_trade']:,.0f} ({stats['avg_r']:+.2f}R)")
    km_row("Avg Win / Loss",
           f"${stats['avg_win']:,.0f} / (${abs(stats['avg_loss']):,.0f})")
    km_row("Best / Worst",      f"${stats['best']:,.0f} / (${abs(stats['worst']):,.0f})")

with km3:
    st.markdown(section_header(
        "Ratios", "Sharpe ≥ 0.3 is the minimum bar for a credible edge (annualized, daily equity). "
        "Profit factor &gt;1.5 is solid. Return/Max DD shows dollars earned per dollar of peak drawdown — "
        "higher is better and more survivable psychologically.",
        tag="bold"
    ), unsafe_allow_html=True)
    km_row("Profit Factor",     f"{stats['pf']:.2f}")
    km_row("Sharpe (ann.)",     f"{stats['sharpe']:.2f}")
    km_row("Return / Max DD",   f"{stats['ret_maxdd']:.2f}")
    km_row("Max Consec. Wins",  str(stats["mcw"]))
    km_row("Max Consec. Losses",str(stats["mcl"]))

with km4:
    st.markdown(section_header(
        "Period", "Profitable months &gt;60% indicates consistent performance "
        "rather than a few large outlier months. "
        "Check best vs. worst month: if one bad month wipes out several good ones, "
        "the strategy has fragile tail risk.",
        tag="bold"
    ), unsafe_allow_html=True)
    km_row("Trading Period",    stats["period"])
    km_row("Days w/ Trade",     str(stats["days_traded"]))
    km_row("Profitable Months", f"{stats['prof_months']} / {stats['total_months']}")
    km_row("Best Month",        f"${stats['best_month']:,.0f}")
    km_row("Worst Month",       f"${stats['worst_month']:,.0f}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Chart panel
# ─────────────────────────────────────────────────────────────────────────────

col_l, col_r = st.columns([3, 1])
with col_l:
    view_desc = {
        "P&L": ("P&amp;L by Period",
                "Monthly/annual bars show seasonality and regime sensitivity. "
                "Look for consistent green bars across years, no single year "
                "dominating the total, and no catastrophic drawdown year. "
                "Switching to Year view reveals whether the strategy failed a "
                "specific regime (e.g. bear market) or degraded gradually."),
        "R-multiples": ("Trade R-multiples",
                        "Each dot is one trade's outcome in units of 1R (your risk per trade). "
                        "A positive cluster above zero = consistent edge. "
                        "Wide scatter with a few large outliers above zero = lucky outliers, "
                        "not reproducible skill. Aim for a tight positive distribution."),
    }
    chart_view_tmp = st.session_state.get("chart_view", "P&L")
    title, desc = view_desc.get(chart_view_tmp, view_desc["P&L"])
    st.markdown(section_header(title, desc), unsafe_allow_html=True)
with col_r:
    chart_view = st.selectbox("View", ["P&L", "R-multiples"], index=0, key="chart_view")

if chart_view == "R-multiples":
    st.plotly_chart(scatter_chart(tlog), use_container_width=True)
else:
    grp = st.radio("", ["Month", "Year"], horizontal=True, key="pnl_grp",
                   label_visibility="collapsed")
    if grp == "Month":
        st.plotly_chart(monthly_chart(stats["monthly"], r_unit), use_container_width=True)
    else:
        st.plotly_chart(annual_chart(tlog, initial), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# By strategy
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(section_header(
    "By Strategy",
    "Isolates each strategy's contribution. If combined results look good but one "
    "sub-strategy is dragging, consider disabling it. Ideally each strategy should "
    "independently clear a Sharpe ≥ 0.3 threshold — a good combined result built on "
    "one strong + one weak strategy is fragile."
), unsafe_allow_html=True)
strat_names = tlog["strat"].unique().tolist() if not tlog.empty else []

strat_rows = []
for sname in strat_names:
    sub = tlog[tlog["strat"] == sname]
    wins   = sub[sub["dollar_pnl"] > 0]["dollar_pnl"]
    losses = sub[sub["dollar_pnl"] < 0]["dollar_pnl"]
    pf     = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")
    rets   = sub["net_ret"]
    sharpe = (rets.mean() / rets.std() * (252 / max(1, (sub["exit_date"] - sub["entry_date"]).dt.days.mean())) ** 0.5
              if rets.std() > 0 else 0.0)
    net    = sub["dollar_pnl"].sum()
    strat_rows.append({
        "Strategy": sname,
        "Trades":   len(sub),
        "Win %":    f"{(sub['dollar_pnl'] > 0).mean() * 100:.0f}%",
        "Sharpe":   f"{sharpe:.2f}",
        "P-factor": f"{pf:.2f}" if pf != float('inf') else "∞",
        "Total R":  f"{sub['r'].sum():+.1f}",
        "Net $":    f"${net:,.0f}",
        "_net":     net,
    })

# combined row
strat_rows.append({
    "Strategy": "— Combined —",
    "Trades":   stats["n_trades"],
    "Win %":    f"{stats['win_rate'] * 100:.0f}%",
    "Sharpe":   f"{stats['sharpe']:.2f}",
    "P-factor": f"{stats['pf']:.2f}",
    "Total R":  f"{stats['net_r']:+.1f}",
    "Net $":    f"${stats['net_pnl']:,.0f}",
    "_net":     stats["net_pnl"],
})

strat_df = pd.DataFrame(strat_rows).drop(columns=["_net"])

def _color_net(val):
    if val.startswith("-") or val.startswith("($"):
        return "color: #ef5350"
    if val.startswith("$") or val.startswith("+"):
        return "color: #26a69a"
    return ""

st.dataframe(
    strat_df.style.applymap(_color_net, subset=["Net $"]),
    use_container_width=True, hide_index=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Trade log
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📋 Full trade log"):
    display = tlog.copy()
    display["entry_date"] = display["entry_date"].dt.strftime("%Y-%m-%d")
    display["exit_date"]  = display["exit_date"].dt.strftime("%Y-%m-%d")
    display["net_ret"]    = (display["net_ret"] * 100).round(3).astype(str) + "%"
    display["dollar_pnl"] = display["dollar_pnl"].round(2)
    display["r"]          = display["r"].round(3)
    st.dataframe(display, use_container_width=True)
