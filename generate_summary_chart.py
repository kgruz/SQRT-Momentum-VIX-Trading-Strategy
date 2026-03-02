"""
Generate a single summary figure showing corrected backtest results.
Panels:
  1. OOS equity curves (Strategy vs QQQ)
  2. Strategy drawdown
  3. Yearly CAGR comparison (Strategy vs QQQ)
  4. Key metrics table
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Import corrected strategy outputs ──────────────────────────────
from Momentum_VIX_TS import (
    portfolio_ret_bt,
    vix,
    vix_threshold,
)
from Tests import (
    load_qqq_for_strategy,
    annualized_return,
    annualized_vol,
    sharpe_ratio,
    max_drawdown,
    sortino_ratio,
    t_test_excess_returns,
)

# ── Align strategy and QQQ ─────────────────────────────────────────
strat = pd.Series(portfolio_ret_bt).dropna()
qqq = load_qqq_for_strategy(strat)
if isinstance(qqq, pd.DataFrame):
    qqq = qqq.iloc[:, 0]
idx = strat.index.intersection(qqq.index)
strat = strat.loc[idx]
qqq = qqq.loc[idx]

eq_s = (1 + strat).cumprod()
eq_q = (1 + qqq).cumprod()

# ── Build figure ───────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.suptitle(
    "Corrected Backtest Results — Strategy vs QQQ (OOS 2008-2023)",
    fontsize=16, fontweight="bold", y=0.98,
)

gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                       left=0.06, right=0.97, top=0.92, bottom=0.06)

# ── Panel 1: Equity Curves ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(eq_s.index, eq_s.values, linewidth=1.5, label="Strategy (corrected)", color="#2166ac")
ax1.plot(eq_q.index, eq_q.values, linewidth=1.5, label="QQQ Buy & Hold", color="#b2182b")
ax1.set_title("Cumulative Equity (starting at $1)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Growth of $1")
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(eq_s.index[0], eq_s.index[-1])

# ── Panel 2: Strategy Drawdown ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
dd_s = eq_s / eq_s.cummax() - 1.0
dd_q = eq_q / eq_q.cummax() - 1.0
ax2.fill_between(dd_s.index, dd_s.values, 0, color="#2166ac", alpha=0.35, label="Strategy DD")
ax2.fill_between(dd_q.index, dd_q.values, 0, color="#b2182b", alpha=0.20, label="QQQ DD")
ax2.set_title("Drawdown Comparison", fontsize=12, fontweight="bold")
ax2.set_ylabel("Drawdown")
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(eq_s.index[0], eq_s.index[-1])

# ── Panel 3: Yearly CAGR Bars ──────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
years = sorted(set(d.year for d in strat.index))
cagr_s_list, cagr_q_list, yr_labels = [], [], []
for y in years:
    mask = strat.index.year == y
    rs = strat[mask]
    rq = qqq[mask]
    if len(rs) < 50:
        continue
    cagr_s_list.append(annualized_return(rs) * 100)
    cagr_q_list.append(annualized_return(rq) * 100)
    yr_labels.append(str(y))

x = np.arange(len(yr_labels))
w = 0.35
bars_s = ax3.bar(x - w/2, cagr_s_list, w, label="Strategy", color="#2166ac", edgecolor="white")
bars_q = ax3.bar(x + w/2, cagr_q_list, w, label="QQQ", color="#b2182b", edgecolor="white")
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(yr_labels, rotation=45, ha="right", fontsize=9)
ax3.set_ylabel("CAGR (%)")
ax3.set_title("Year-by-Year CAGR: Strategy vs QQQ", fontsize=12, fontweight="bold")
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis="y")

# ── Panel 4: Key Metrics Table ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")

excess = (strat - qqq).dropna()
t_ex, p_ex = t_test_excess_returns(strat, qqq)

metrics = [
    ["", "Strategy", "QQQ"],
    ["CAGR", f"{annualized_return(strat):.2%}", f"{annualized_return(qqq):.2%}"],
    ["Annualized Vol", f"{annualized_vol(strat):.2%}", f"{annualized_vol(qqq):.2%}"],
    ["Sharpe Ratio", f"{sharpe_ratio(strat):.3f}", f"{sharpe_ratio(qqq):.3f}"],
    ["Sortino Ratio", f"{sortino_ratio(strat):.3f}", f"{sortino_ratio(qqq):.3f}"],
    ["Max Drawdown", f"{max_drawdown(eq_s):.2%}", f"{max_drawdown(eq_q):.2%}"],
    ["", "", ""],
    ["Excess Return (ann.)", f"{excess.mean()*252:.2%}", "—"],
    ["Info Ratio vs QQQ", f"{sharpe_ratio(excess):.3f}", "—"],
    ["t-stat (excess > 0)", f"{t_ex:.3f}", "—"],
    ["p-value (one-sided)", f"{p_ex:.4f}", "—"],
    ["", "", ""],
    ["Verdict", "DOES NOT", "outperform QQQ"],
]

table = ax4.table(
    cellText=metrics,
    cellLoc="center",
    loc="center",
    bbox=[0.05, 0.0, 0.90, 1.0],
)
table.auto_set_font_size(False)
table.set_fontsize(11)

# Style header row
for j in range(3):
    table[0, j].set_facecolor("#333333")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Style verdict row
for j in range(3):
    table[len(metrics)-1, j].set_facecolor("#fee0d2")
    table[len(metrics)-1, j].set_text_props(fontweight="bold", color="#b2182b")

# Style separator rows
for row_idx in [6, 11]:
    for j in range(3):
        table[row_idx, j].set_facecolor("#f7f7f7")
        table[row_idx, j].set_edgecolor("#f7f7f7")

# Alternate row shading
for i in range(1, len(metrics)):
    if i in [6, 11, len(metrics)-1]:
        continue
    color = "#f0f0f0" if i % 2 == 0 else "white"
    for j in range(3):
        table[i, j].set_facecolor(color)

for key, cell in table.get_celld().items():
    cell.set_linewidth(0.5)

ax4.set_title("Key Metrics Summary", fontsize=12, fontweight="bold", pad=15)

# ── Save ───────────────────────────────────────────────────────────
out_path = "corrected_backtest_summary.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved summary chart to {out_path}")
plt.close(fig)
