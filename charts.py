"""charts.py — Plotly chart builders and HTML UI helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Shared dark-theme layout defaults
_L = dict(
    paper_bgcolor="#131722", plot_bgcolor="#131722",
    font=dict(color="#d1d4dc", size=11),
    margin=dict(l=10, r=10, t=35, b=10),
    xaxis=dict(gridcolor="#2a2e39", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#2a2e39", showgrid=True, zeroline=False),
)


def equity_chart(equity: pd.Series, initial: float, tlog: pd.DataFrame,
                 benchmark: pd.Series | None = None,
                 folds: list[dict] | None = None) -> go.Figure:
    pnl = equity - initial
    dd  = equity - equity.cummax()
    fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28],
                        shared_xaxes=True, vertical_spacing=0.03)

    fig.add_trace(go.Scatter(x=equity.index, y=pnl, name="Strategy",
                             line=dict(color="#26a69a", width=1.5),
                             fill="tozeroy", fillcolor="rgba(38,166,154,0.1)"),
                  row=1, col=1)

    if folds:
        for f in folds:
            ts = pd.Timestamp(f["test_start"])
            te = pd.Timestamp(f["test_end"])
            # shade OOS window on both panels
            for row in [1, 2]:
                fig.add_vrect(x0=ts, x1=te,
                              fillcolor="rgba(243,156,18,0.07)",
                              line_width=0, row=row, col=1)
            # fold boundary line
            for row in [1, 2]:
                fig.add_vline(x=ts, line_color="#f39c12", line_dash="dash",
                              line_width=1, row=row, col=1)
            fig.add_annotation(x=ts, y=1.04, xref="x", yref="paper",
                               text=f"F{f['fold']}",
                               showarrow=False,
                               font=dict(color="#f39c12", size=9),
                               xanchor="left")

    if benchmark is not None:
        bah = (benchmark / benchmark.iloc[0] * initial - initial).reindex(
            equity.index, method="ffill")
        fig.add_trace(go.Scatter(x=bah.index, y=bah, name="SPY B&H",
                                 line=dict(color="#5c85d6", width=1.2, dash="dot"),
                                 opacity=0.8), row=1, col=1)

    if not tlog.empty:
        for mask, clr, sym, label in [
            (tlog["dollar_pnl"] > 0, "#26a69a", "triangle-up",   "Win"),
            (tlog["dollar_pnl"] < 0, "#ef5350", "triangle-down", "Loss"),
        ]:
            sub = tlog[mask]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["exit_date"], y=[pnl.asof(d) for d in sub["exit_date"]],
                mode="markers", name=label,
                marker=dict(symbol=sym, color=clr, size=7, opacity=0.8),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Exit: %{x|%Y-%m-%d}<br>"
                    "P&L: $%{customdata[1]:,.0f}<br>R: %{customdata[2]:.2f}R<extra></extra>"
                ),
                customdata=list(zip(sub["strat"], sub["dollar_pnl"], sub["r"])),
            ), row=1, col=1)

    fig.add_trace(go.Scatter(x=equity.index, y=dd, name="Drawdown",
                             line=dict(color="#ef5350", width=0.8),
                             fill="tozeroy", fillcolor="rgba(239,83,80,0.25)"),
                  row=2, col=1)
    fig.update_layout(**_L, height=420, hovermode="x unified",
                      title=dict(text="Equity curve and drawdown",
                                 font=dict(size=12), x=0),
                      legend=dict(bgcolor="#1e222d", bordercolor="#2a2e39",
                                  orientation="h", y=1.06, x=0))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", gridcolor="#2a2e39")
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    return fig


def monthly_chart(monthly: pd.Series, r_unit: float) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=monthly.index.strftime("%b %Y"), y=monthly.values,
        marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in monthly.values],
        text=[f"{v/r_unit:+.1f}R" for v in monthly.values],
        textposition="outside", textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#2a2e39", line_width=1)
    fig.update_layout(**_L, height=290,
                      title=dict(text="Monthly P&L", font=dict(size=12), x=0),
                      yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    return fig


def scatter_chart(tlog: pd.DataFrame) -> go.Figure:
    if tlog.empty:
        return go.Figure()
    palette = ["#26a69a", "#f39c12", "#5c6bc0", "#ec407a", "#ab47bc"]
    strats  = tlog["strat"].unique()
    colors  = {s: palette[i % len(palette)] for i, s in enumerate(strats)}
    fig = go.Figure(go.Scatter(
        x=tlog["entry_date"], y=tlog["r"], mode="markers",
        marker=dict(color=[colors[s] for s in tlog["strat"]],
                    size=8, opacity=0.75, line=dict(width=0.5, color="#131722")),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>Entry: %{x|%Y-%m-%d}<br>"
            "R: %{y:.2f}<br>$%{customdata[1]:,.0f}<extra></extra>"
        ),
        customdata=list(zip(tlog["strat"], tlog["dollar_pnl"])),
    ))
    fig.add_hline(y=0, line_color="#2a2e39", line_width=1)
    fig.update_layout(**_L, height=290,
                      title=dict(text="Trade R-multiples", font=dict(size=12), x=0),
                      yaxis_title="R")
    return fig


def annual_chart(tlog: pd.DataFrame, initial: float) -> go.Figure:
    if tlog.empty:
        return go.Figure()
    by_year = tlog.groupby(tlog["exit_date"].dt.year).agg(
        net_pnl=("dollar_pnl", "sum"),
        trades=("dollar_pnl", "count"),
        wins=("dollar_pnl", lambda x: (x > 0).sum()),
        r=("r", "sum"),
    ).reset_index()
    by_year["win_pct"]    = by_year["wins"] / by_year["trades"] * 100
    by_year["pct_return"] = by_year["net_pnl"] / initial * 100

    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in by_year["net_pnl"]]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                        shared_xaxes=True, vertical_spacing=0.06)

    fig.add_trace(go.Bar(
        x=by_year["exit_date"].astype(str), y=by_year["net_pnl"],
        marker_color=colors,
        text=[f"{r:+.1f}R<br>{t} trades"
              for r, t in zip(by_year["r"], by_year["trades"])],
        textposition="outside", textfont=dict(size=8),
        hovertemplate=(
            "<b>%{x}</b><br>P&L: $%{y:,.0f}<br>"
            "R: %{customdata[0]:+.1f}<br>Trades: %{customdata[1]}<br>"
            "Win %: %{customdata[2]:.0f}%<extra></extra>"
        ),
        customdata=list(zip(by_year["r"], by_year["trades"], by_year["win_pct"])),
        name="Annual P&L",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=by_year["exit_date"].astype(str), y=by_year["win_pct"],
        marker_color=["#26a69a" if v >= 50 else "#ef5350" for v in by_year["win_pct"]],
        name="Win %",
        hovertemplate="<b>%{x}</b><br>Win %%: %{y:.1f}%%<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=50, line_color="#787b86", line_dash="dot", line_width=1, row=2, col=1)

    fig.update_layout(**_L, height=380,
                      title=dict(text="Annual P&L breakdown", font=dict(size=12), x=0),
                      legend=dict(bgcolor="#1e222d", bordercolor="#2a2e39",
                                  orientation="h", y=1.05, x=0),
                      bargap=0.25)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1, gridcolor="#2a2e39")
    fig.update_yaxes(ticksuffix="%", title_text="Win %", row=2, col=1,
                     gridcolor="#2a2e39", range=[0, 100])
    fig.add_hline(y=0, line_color="#2a2e39", line_width=1, row=1, col=1)
    return fig


def section_header(title: str, description: str, tag: str = "h3") -> str:
    """Section title with an ⓘ icon that reveals a tooltip to the right on hover."""
    tip = (f'<span class="tt-wrap">ⓘ'
           f'<span class="tt">{description}</span></span>')
    if tag == "bold":
        return f"<p style='font-weight:600;margin-bottom:4px'>{title} {tip}</p>"
    return f"<{tag} style='margin-bottom:4px'>{title} {tip}</{tag}>"


def kpi_card(label: str, value: str, color: str, sub: str = "") -> str:
    clr = {"green": "#26a69a", "red": "#ef5350", "white": "#d1d4dc"}.get(color, color)
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{clr}">{value}</div>'
            f'<div class="metric-sub">{sub}</div></div>')


def km_row(label: str, value: str, color: str = "#d1d4dc") -> None:
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:3px 0;border-bottom:1px solid #2a2e39">'
        f'<span style="color:#787b86;font-size:12px">{label}</span>'
        f'<span style="color:{color};font-size:12px;font-weight:600">{value}</span>'
        f'</div>', unsafe_allow_html=True)
