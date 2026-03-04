"""
Student Strategy on Corrected Data
====================================
Runs the students' strategy *design choices* (VIX MA crossover, SH inverse ETF,
no TS momentum filter, drift simulation) on top of our corrected data pipeline
(historical S&P 500 membership, removed-stock prices, 20 bps TC).

This isolates what their strategy actually produces vs what the data errors gave them.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

# ── Re-use our corrected data loading ─────────────────────────────────
# (historical membership, removed-stock prices, no min_obs filter)

PRICES_CSV = "sp500_adjclose_2005_2023.csv"
SECTOR_CSV = "constituents.csv"
HISTORICAL_COMPONENTS_CSV = "sp500_historical_components.csv"
REMOVED_PRICES_CSV = "sp500_removed_prices.csv"

start_data = "2005-01-01"
end_data   = "2023-12-31"
vix_start_data = "1996-01-01"

# ── Student strategy parameters ───────────────────────────────────────
treasury_ticker  = "BIL"
inverse_ticker   = "SH"
benchmark_ticker = "QQQ"

transaction_cost_bps = 20       # CORRECTED (students had 5)
tc = transaction_cost_bps / 10_000

top_n_candidates  = [5, 10, 15, 20]
lookback_candidates = [6, 9, 12]
VIX_MA_WINDOW = 90
VIX_MA_FACTOR = 1.15
skip_months   = 1

TARGET_VOL = 0.15
VOL_WINDOW = 21
MAX_LEVER  = 1.0
MIN_LEVER  = 1.0

WFO_WINDOWS = [
    ("2005-01-01", "2007-12-31", "2008-01-01", "2010-12-31"),
    ("2008-01-01", "2010-12-31", "2011-01-01", "2013-12-31"),
    ("2011-01-01", "2013-12-31", "2014-01-01", "2016-12-31"),
    ("2014-01-01", "2016-12-31", "2017-01-01", "2019-12-31"),
    ("2017-01-01", "2019-12-31", "2020-01-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
]

# ── Sector map ────────────────────────────────────────────────────────

def load_sector_map(path=SECTOR_CSV):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    symbol_col = next((c for c in df.columns if c.lower() in ["symbol", "ticker"]), None)
    sector_col = next((c for c in df.columns if "sector" in c.lower()), None)
    if symbol_col is None or sector_col is None:
        return {}
    syms = df[symbol_col].astype(str).str.strip().str.replace(".", "-", regex=False)
    secs = df[sector_col].astype(str).str.strip()
    return dict(zip(syms, secs))

sector_map = load_sector_map()
USE_SECTOR_NEUTRALITY = len(sector_map) > 0
print(f"Sector map: {len(sector_map)} tickers.")

# ── Historical S&P 500 membership (our fix) ───────────────────────────

def load_historical_components(path=HISTORICAL_COMPONENTS_CSV):
    import bisect
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Warning: '{path}' not found -- membership filter disabled.")
        return [], []
    entries = []
    for _, row in df.iterrows():
        dt = pd.to_datetime(row["date"])
        tickers = set(t.strip().replace(".", "-") for t in str(row["tickers"]).split(","))
        entries.append((dt, tickers))
    entries.sort(key=lambda x: x[0])
    dates_list = [e[0] for e in entries]
    print(f"Historical membership: {len(entries)} snapshots ({entries[0][0].date()} to {entries[-1][0].date()}).")
    return entries, dates_list

_hist_entries, _hist_dates = load_historical_components()

def eligible_tickers(as_of_date, available_columns):
    import bisect
    if not _hist_entries:
        return list(available_columns)
    idx = bisect.bisect_right(_hist_dates, as_of_date) - 1
    if idx < 0:
        return []
    members = _hist_entries[idx][1]
    return [t for t in available_columns if t in members]

# ── Load prices (with removed-stock supplement) ───────────────────────

prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True).sort_index()
prices = prices[(prices.index >= pd.to_datetime(start_data)) &
                (prices.index <= pd.to_datetime(end_data))]

try:
    removed_prices = pd.read_csv(REMOVED_PRICES_CSV, index_col=0, parse_dates=True).sort_index()
    if hasattr(removed_prices.index, "tz") and removed_prices.index.tz is not None:
        removed_prices.index = removed_prices.index.tz_localize(None)
    new_cols = [c for c in removed_prices.columns if c not in prices.columns]
    if new_cols:
        prices = prices.join(removed_prices[new_cols], how="left")
        print(f"Merged {len(new_cols)} removed-stock price series.")
except FileNotFoundError:
    print("Warning: no removed-stock prices found.")

print(f"Price panel: {prices.shape}")

# ── Download VIX, BIL, SH, QQQ (VIX from 1996 for MA) ───────────────

print("Downloading VIX, BIL, SH, QQQ ...")
extra_data = yf.download(
    ["^VIX", treasury_ticker, inverse_ticker, benchmark_ticker],
    start=vix_start_data,
    end=end_data,
    auto_adjust=False,
    progress=False,
)

extra_adj = extra_data["Adj Close"]
extra_adj.index = pd.to_datetime(extra_adj.index).tz_localize(None)

vix_full = extra_adj["^VIX"].dropna()

common_index = prices.index.intersection(extra_adj.index)
prices    = prices.loc[common_index]
vix       = extra_adj["^VIX"].loc[common_index]
treasury  = extra_adj[treasury_ticker].loc[common_index]
inverse   = extra_adj[inverse_ticker].loc[common_index]
benchmark = extra_adj[benchmark_ticker].loc[common_index]

print(f"Aligned trading days: {len(common_index)}  |  Price matrix: {prices.shape}")

# ── Monthly momentum (NO TS filter — student design choice) ──────────

monthly_prices = prices.resample("ME").last()
# ts_mom_12 still computed but NOT used in weight selection (student's choice)
ts_mom_12 = monthly_prices.shift(1) / monthly_prices.shift(13) - 1

# ── Student's get_target_weights (cross-sectional only, no TS filter) ─

def get_target_weights(dt, lookback_months, top_n, momentum_df):
    if dt not in momentum_df.index:
        return {}
    mom_row = momentum_df.loc[dt]

    # Our corrected membership filter (not fallback-to-all)
    eligible = eligible_tickers(dt, mom_row.index)
    mom_row = mom_row[eligible]

    combined = mom_row.dropna()
    if combined.empty:
        return {}

    sorted_mom = combined.sort_values(ascending=False)

    if USE_SECTOR_NEUTRALITY:
        sec_buckets = {}
        for name in sorted_mom.index:
            sec = sector_map.get(name, "Unknown")
            sec_buckets.setdefault(sec, []).append(name)
        n_sectors = len(sec_buckets)
        if n_sectors == 0:
            return {}
        per_sector = max(1, int(np.ceil(top_n / n_sectors)))
        nominees = []
        for names in sec_buckets.values():
            nominees.extend(names[:per_sector])
        top_names = (
            sorted_mom
            .loc[sorted_mom.index.intersection(nominees)]
            .index[:top_n]
        )
        if len(top_names) == 0:
            return {}
        final_buckets = {}
        for name in top_names:
            sec = sector_map.get(name, "Unknown")
            final_buckets.setdefault(sec, []).append(name)
        w_sec = 1.0 / len(final_buckets)
        result = {}
        for sec, names in final_buckets.items():
            w_each = w_sec / len(names)
            for name in names:
                result[name] = w_each
        return result
    else:
        top_names = sorted_mom.index[:top_n]
        w = 1.0 / len(top_names)
        return {name: w for name in top_names}

# ── Daily returns ─────────────────────────────────────────────────────

stock_rets     = prices.pct_change(fill_method=None).fillna(0.0)
treasury_rets  = treasury.pct_change(fill_method=None).fillna(0.0)
inverse_rets   = inverse.pct_change(fill_method=None).fillna(0.0)
benchmark_rets = benchmark.pct_change(fill_method=None).fillna(0.0)
realized_vol   = benchmark_rets.rolling(VOL_WINDOW).std() * np.sqrt(252)

# ── Helpers ───────────────────────────────────────────────────────────

def compute_cagr(eq):
    if len(eq) == 0:
        return np.nan
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    return np.nan if n_years <= 0 else (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1

def compute_sharpe(rets, ppyr=252):
    if len(rets) < 2:
        return np.nan
    mu = rets.mean() * ppyr
    sigma = rets.std() * np.sqrt(ppyr)
    return np.nan if sigma == 0 else mu / sigma

def max_drawdown(eq):
    if len(eq) == 0:
        return np.nan
    return (eq / eq.cummax() - 1.0).min()

def _nan_stats(lookback_months, top_n):
    return dict(
        cagr_strategy=np.nan, cagr_benchmark=np.nan,
        sharpe_strategy=np.nan, sharpe_benchmark=np.nan,
        mdd_strategy=np.nan, mdd_benchmark=np.nan,
        avg_turnover=np.nan,
        lookback_months=lookback_months, top_n=top_n,
        portfolio_ret_bt=pd.Series(dtype=float),
        benchmark_ret_bt=pd.Series(dtype=float),
        portfolio_eq_bt=pd.Series(dtype=float),
        benchmark_eq_bt=pd.Series(dtype=float),
    )

# ── Student's backtest (VIX MA crossover, SH risk-off, drift sim) ────

def run_backtest(lookback_months, top_n, start_date, end_date):
    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)

    # Student's VIX MA crossover regime (uses vix_full for longer history)
    vix_ma   = vix_full.rolling(VIX_MA_WINDOW, min_periods=20).mean()
    risk_off = (vix_full > VIX_MA_FACTOR * vix_ma).reindex(prices.index).fillna(False)

    window_idx = prices.index[(prices.index >= start_dt) & (prices.index <= end_dt)]
    if len(window_idx) == 0:
        return _nan_stats(lookback_months, top_n)

    momentum = (
        monthly_prices.shift(skip_months) /
        monthly_prices.shift(skip_months + lookback_months) - 1
    )

    # Student's rebalance logic: month-ends + forced day-1
    month_ends = pd.date_range(start=window_idx[0], end=window_idx[-1], freq="ME")
    rebalance_dates = {window_idx[0]}
    for me in month_ends:
        candidates = window_idx[window_idx <= me]
        if len(candidates):
            rebalance_dates.add(candidates[-1])

    # Dollar-value drift simulation (student's fix)
    all_tickers    = prices.columns.tolist()
    n_assets       = len(all_tickers)
    ticker_idx_map = {t: i for i, t in enumerate(all_tickers)}

    nav          = 1.0
    held_dollars = np.zeros(n_assets)
    cash_dollars = 0.0

    port_rets_list  = []
    bench_rets_list = []
    turnover_list   = []
    dates_list      = []

    stock_rets_np  = stock_rets.values
    stock_rets_pos = {d: i for i, d in enumerate(stock_rets.index)}

    for date in window_idx:
        ro_today = bool(risk_off.get(date, False))

        if date in rebalance_dates:
            avail_months = monthly_prices.index[monthly_prices.index <= date]
            turnover_today = 0.0
            if len(avail_months) > 0:
                last_month = avail_months[-1]
                tgt = get_target_weights(last_month, lookback_months, top_n, momentum)

                target_w = np.zeros(n_assets)
                for t, w in tgt.items():
                    if t in ticker_idx_map:
                        target_w[ticker_idx_map[t]] = w

                if ro_today:
                    new_stock_d = np.zeros(n_assets)
                    new_cash_d  = nav
                else:
                    new_stock_d = target_w * nav
                    new_cash_d  = 0.0

                old_alloc = np.append(held_dollars, cash_dollars)
                new_alloc = np.append(new_stock_d, new_cash_d)
                turnover_today = float(np.abs(new_alloc - old_alloc).sum() / 2.0 / max(nav, 1e-12))

                held_dollars = new_stock_d
                cash_dollars = new_cash_d
        else:
            turnover_today = 0.0

        di = stock_rets_pos.get(date)
        sr = np.nan_to_num(stock_rets_np[di] if di is not None else np.zeros(n_assets))
        treas_r = float(treasury_rets.get(date, 0.0))
        sh_r    = float(inverse_rets.get(date, 0.0))

        # Student's choice: SH during risk-off, treasury during risk-on cash
        cash_r = sh_r if ro_today else treas_r

        stock_pnl = float(held_dollars @ sr)
        cash_pnl  = cash_dollars * cash_r
        tc_cost   = turnover_today * tc * max(nav, 1e-12)

        port_ret_pct = (stock_pnl + cash_pnl - tc_cost) / max(nav, 1e-12)

        held_dollars = held_dollars * (1.0 + sr)
        cash_dollars = cash_dollars * (1.0 + cash_r)
        nav          = max(nav + stock_pnl + cash_pnl - tc_cost, 1e-12)

        bench_r = float(benchmark_rets.get(date, np.nan))

        port_rets_list.append(port_ret_pct)
        bench_rets_list.append(bench_r)
        turnover_list.append(turnover_today)
        dates_list.append(date)

    port_ret_w  = pd.Series(port_rets_list, index=dates_list)
    bench_ret_w = pd.Series(bench_rets_list, index=dates_list).dropna()
    port_ret_w  = port_ret_w.reindex(bench_ret_w.index).fillna(0.0)
    turn_w      = pd.Series(turnover_list, index=dates_list).reindex(bench_ret_w.index).fillna(0.0)

    port_eq_w  = (1 + port_ret_w).cumprod()
    bench_eq_w = (1 + bench_ret_w).cumprod()

    if len(port_eq_w) == 0:
        return _nan_stats(lookback_months, top_n)

    return dict(
        cagr_strategy=compute_cagr(port_eq_w),
        cagr_benchmark=compute_cagr(bench_eq_w),
        sharpe_strategy=compute_sharpe(port_ret_w),
        sharpe_benchmark=compute_sharpe(bench_ret_w),
        mdd_strategy=max_drawdown(port_eq_w),
        mdd_benchmark=max_drawdown(bench_eq_w),
        avg_turnover=float(turn_w.mean()),
        lookback_months=lookback_months,
        top_n=top_n,
        portfolio_ret_bt=port_ret_w,
        benchmark_ret_bt=bench_ret_w,
        portfolio_eq_bt=port_eq_w,
        benchmark_eq_bt=bench_eq_w,
    )


# ══════════════════════════════════════════════════════════════════════
# WALK-FORWARD OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════

all_oos_portfolio = []
all_oos_benchmark = []
wfo_summary       = []
chosen_params     = []

print("\n=== WALK-FORWARD OPTIMIZATION (Student Strategy, Clean Data) ===\n")

for (train_start, train_end, test_start, test_end) in WFO_WINDOWS:
    print(f"TRAIN {train_start}-{train_end}  |  TEST {test_start}-{test_end}")

    best_stats = None
    for L in lookback_candidates:
        for n in top_n_candidates:
            s  = run_backtest(L, n, train_start, train_end)
            sh = s["sharpe_strategy"]
            if not np.isnan(sh) and (best_stats is None or sh > best_stats["sharpe_strategy"]):
                best_stats = s

    if best_stats is None:
        print("  -> Skipped.\n")
        continue

    best_L = best_stats["lookback_months"]
    best_n = best_stats["top_n"]
    chosen_params.append((best_L, best_n))

    print(
        f"  TRAIN best: L={best_L}, n={best_n} "
        f"| Sharpe={best_stats['sharpe_strategy']:.2f}, "
        f"CAGR={best_stats['cagr_strategy']:.2%}"
    )

    oos = run_backtest(best_L, best_n, test_start, test_end)
    print(
        f"  OOS: Sharpe={oos['sharpe_strategy']:.2f}, "
        f"CAGR={oos['cagr_strategy']:.2%}, "
        f"MaxDD={oos['mdd_strategy']:.2%}\n"
    )

    all_oos_portfolio.append(oos["portfolio_ret_bt"])
    all_oos_benchmark.append(oos["benchmark_ret_bt"])
    wfo_summary.append(dict(
        train_start=train_start, train_end=train_end,
        test_start=test_start, test_end=test_end,
        train_L=best_L, train_n=best_n,
        train_sharpe=best_stats["sharpe_strategy"],
        train_cagr=best_stats["cagr_strategy"],
        test_sharpe=oos["sharpe_strategy"],
        test_cagr=oos["cagr_strategy"],
        test_mdd=oos["mdd_strategy"],
    ))


# ══════════════════════════════════════════════════════════════════════
# STITCH OOS
# ══════════════════════════════════════════════════════════════════════

if not all_oos_portfolio:
    raise RuntimeError("No OOS segments produced.")

portfolio_ret_bt = pd.concat(all_oos_portfolio).sort_index()
benchmark_ret_bt = pd.concat(all_oos_benchmark).sort_index()

portfolio_ret_bt = portfolio_ret_bt[~portfolio_ret_bt.index.duplicated(keep="first")]
benchmark_ret_bt = benchmark_ret_bt[~benchmark_ret_bt.index.duplicated(keep="first")]

common_idx       = portfolio_ret_bt.index.intersection(benchmark_ret_bt.index)
portfolio_ret_bt = portfolio_ret_bt.reindex(common_idx).fillna(0.0)
benchmark_ret_bt = benchmark_ret_bt.reindex(common_idx).fillna(0.0)

portfolio_eq_oos = (1 + portfolio_ret_bt).cumprod()
benchmark_eq_oos = (1 + benchmark_ret_bt).cumprod()

# Also load our corrected strategy for 3-way comparison
try:
    from Momentum_VIX_TS import (
        portfolio_ret_bt as corr_ret,
        portfolio_eq_oos as corr_eq,
    )
    corr_ret = corr_ret.reindex(common_idx)
    corr_eq_aligned = (1 + corr_ret.fillna(0.0)).cumprod()
    HAS_CORRECTED = True
except Exception:
    HAS_CORRECTED = False


# ══════════════════════════════════════════════════════════════════════
# PRINT RESULTS
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FULL OOS RESULTS (2008-2023) — Student Strategy on Clean Data")
print("=" * 70)
print(f"Student Strategy CAGR:   {compute_cagr(portfolio_eq_oos):.2%}")
print(f"QQQ CAGR:                {compute_cagr(benchmark_eq_oos):.2%}")
print(f"Student Strategy Sharpe: {compute_sharpe(portfolio_ret_bt):.2f}")
print(f"QQQ Sharpe:              {compute_sharpe(benchmark_ret_bt):.2f}")
print(f"Student Strategy MaxDD:  {max_drawdown(portfolio_eq_oos):.2%}")
print(f"QQQ MaxDD:               {max_drawdown(benchmark_eq_oos):.2%}")
print(f"OOS days:                {len(portfolio_ret_bt)}")

if HAS_CORRECTED:
    print(f"\nCorrected Original CAGR: {compute_cagr(corr_eq_aligned):.2%}")
    print(f"Corrected Original Sharpe: {compute_sharpe(corr_ret.dropna()):.2f}")

excess = portfolio_ret_bt - benchmark_ret_bt
ann_excess = excess.mean() * 252
excess_sharpe = compute_sharpe(excess)
from scipy import stats as sp_stats
t_stat, p_val_two = sp_stats.ttest_1samp(excess.dropna(), 0)
p_val_one = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2

print(f"\nExcess return vs QQQ (ann.): {ann_excess:.2%}")
print(f"Info ratio vs QQQ:           {excess_sharpe:.3f}")
print(f"t-stat (excess > 0):         {t_stat:.3f}")
print(f"p-value (one-sided):         {p_val_one:.4f}")

verdict = "OUTPERFORMS" if (p_val_one < 0.05 and t_stat > 0) else "DOES NOT outperform"
print(f"\nVerdict: Student strategy {verdict} QQQ at 5% significance.")

print("\n=== WFO SUMMARY ===")
for row in wfo_summary:
    print(
        f"{row['train_start']}-{row['train_end']} "
        f"(L={row['train_L']}, n={row['train_n']}): "
        f"train Sharpe={row['train_sharpe']:.2f} | "
        f"OOS {row['test_start']}-{row['test_end']} "
        f"Sharpe={row['test_sharpe']:.2f}, CAGR={row['test_cagr']:.2%}, "
        f"MaxDD={row['test_mdd']:.2%}"
    )

if chosen_params:
    Ls = [p[0] for p in chosen_params]
    ns = [p[1] for p in chosen_params]
    print(f"\nParameter stability: L={Ls}, n={ns}")


# ══════════════════════════════════════════════════════════════════════
# COMPARISON CHART
# ══════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 13))
fig.suptitle(
    "Student Strategy on Corrected Data vs QQQ (OOS 2008-2023)",
    fontsize=16, fontweight="bold", y=0.98,
)
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                       left=0.06, right=0.97, top=0.92, bottom=0.06)

# Panel 1: Equity curves
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(portfolio_eq_oos.index, portfolio_eq_oos.values,
         linewidth=1.5, label="Student Strategy (clean data)", color="#2166ac")
ax1.plot(benchmark_eq_oos.index, benchmark_eq_oos.values,
         linewidth=1.5, label="QQQ Buy & Hold", color="#b2182b")
if HAS_CORRECTED:
    ax1.plot(corr_eq_aligned.index, corr_eq_aligned.values,
             linewidth=1.2, label="Corrected Original", color="#4daf4a", linestyle="--")
ax1.set_title("Cumulative Equity (starting at $1)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Growth of $1")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Drawdowns
ax2 = fig.add_subplot(gs[0, 1])
dd_s = portfolio_eq_oos / portfolio_eq_oos.cummax() - 1.0
dd_q = benchmark_eq_oos / benchmark_eq_oos.cummax() - 1.0
ax2.fill_between(dd_s.index, dd_s.values, 0, color="#2166ac", alpha=0.35, label="Student Strat DD")
ax2.fill_between(dd_q.index, dd_q.values, 0, color="#b2182b", alpha=0.20, label="QQQ DD")
ax2.set_title("Drawdown Comparison", fontsize=12, fontweight="bold")
ax2.set_ylabel("Drawdown")
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(True, alpha=0.3)

# Panel 3: Yearly CAGR bars
ax3 = fig.add_subplot(gs[1, 0])
from Tests import annualized_return
years = sorted(set(d.year for d in portfolio_ret_bt.index))
cagr_s_list, cagr_q_list, yr_labels = [], [], []
for y in years:
    mask = portfolio_ret_bt.index.year == y
    rs = portfolio_ret_bt[mask]
    rq = benchmark_ret_bt[mask]
    if len(rs) < 50:
        continue
    cagr_s_list.append(annualized_return(rs) * 100)
    cagr_q_list.append(annualized_return(rq) * 100)
    yr_labels.append(str(y))

x = np.arange(len(yr_labels))
w = 0.35
ax3.bar(x - w/2, cagr_s_list, w, label="Student Strat", color="#2166ac", edgecolor="white")
ax3.bar(x + w/2, cagr_q_list, w, label="QQQ", color="#b2182b", edgecolor="white")
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(yr_labels, rotation=45, ha="right", fontsize=9)
ax3.set_ylabel("CAGR (%)")
ax3.set_title("Year-by-Year CAGR: Student Strategy vs QQQ", fontsize=12, fontweight="bold")
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis="y")

# Panel 4: Key metrics table
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")

from Tests import annualized_vol, sharpe_ratio, sortino_ratio

strat_s = portfolio_ret_bt
qqq_s   = benchmark_ret_bt
eq_s    = portfolio_eq_oos
eq_q    = benchmark_eq_oos

metrics = [
    ["", "Student Strat", "QQQ"],
    ["CAGR", f"{compute_cagr(eq_s):.2%}", f"{compute_cagr(eq_q):.2%}"],
    ["Annualized Vol", f"{annualized_vol(strat_s):.2%}", f"{annualized_vol(qqq_s):.2%}"],
    ["Sharpe Ratio", f"{compute_sharpe(strat_s):.3f}", f"{compute_sharpe(qqq_s):.3f}"],
    ["Sortino Ratio", f"{sortino_ratio(strat_s):.3f}", f"{sortino_ratio(qqq_s):.3f}"],
    ["Max Drawdown", f"{max_drawdown(eq_s):.2%}", f"{max_drawdown(eq_q):.2%}"],
    ["", "", ""],
    ["Excess Return (ann.)", f"{ann_excess:.2%}", ""],
    ["Info Ratio vs QQQ", f"{excess_sharpe:.3f}", ""],
    ["t-stat (excess > 0)", f"{t_stat:.3f}", ""],
    ["p-value (one-sided)", f"{p_val_one:.4f}", ""],
    ["", "", ""],
    ["Verdict", verdict.split()[0], verdict.split()[-1] if len(verdict.split()) > 1 else ""],
]

table = ax4.table(
    cellText=metrics,
    cellLoc="center",
    loc="center",
    bbox=[0.05, 0.0, 0.90, 1.0],
)
table.auto_set_font_size(False)
table.set_fontsize(11)

for j in range(3):
    table[0, j].set_facecolor("#333333")
    table[0, j].set_text_props(color="white", fontweight="bold")

does_not = "DOES NOT" in verdict
verdict_color = "#fee0d2" if does_not else "#d4edda"
verdict_text_color = "#b2182b" if does_not else "#155724"
for j in range(3):
    table[len(metrics)-1, j].set_facecolor(verdict_color)
    table[len(metrics)-1, j].set_text_props(fontweight="bold", color=verdict_text_color)

for row_idx in [6, 11]:
    for j in range(3):
        table[row_idx, j].set_facecolor("#f7f7f7")
        table[row_idx, j].set_edgecolor("#f7f7f7")

for i in range(1, len(metrics)):
    if i in [6, 11, len(metrics)-1]:
        continue
    color = "#f0f0f0" if i % 2 == 0 else "white"
    for j in range(3):
        table[i, j].set_facecolor(color)

for key, cell in table.get_celld().items():
    cell.set_linewidth(0.5)

ax4.set_title("Key Metrics Summary", fontsize=12, fontweight="bold", pad=15)

out_path = "student_strategy_clean_data.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved chart to {out_path}")
plt.close(fig)
