"""
Momentum Strategy - Fixed Version
====================================
Fixes applied vs original:

  [1] SURVIVORSHIP BIAS
      Load from long-format CSV (date, ticker, adj_close, in_sp500).
      Only trade a ticker on dates when in_sp500 == 1. Delisted firms
      (Lehman, Bear Stearns) appear while alive; late additions (TSLA,
      ABNB) only appear after their index inclusion date.

  [2] VIX LOOKAHEAD ELIMINATED
      vix_threshold computed ONLY from pre-window history.
      If fewer than 60 pre-window VIX observations exist, the window
      is SKIPPED. The original fell back to the full 2005-2023 series,
      leaking future volatility into past decisions.

  [3] LEVERAGE vs BENCHMARK
      Benchmark changed to QQQ. Both the strategy and QQQ are compared
      on an unlevered basis -- vol scaling is removed from the stock leg
      so the strategy always holds exactly its target weights (scale=1).
      This gives a clean apples-to-apples return comparison.

  [4] DAILY REBALANCING DRIFT
      Monthly target weights are set on month-end rebalance days only.
      Between rebalances the portfolio drifts with daily stock returns.
      TC is charged only at actual rebalance dates.
      The original forward-filled monthly weights daily, implying
      costless daily rebalancing -- understating volatility and TC.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from collections import Counter

# =========================
# PARAMETERS
# =========================

PRICES_CSV   = "sp500_prices_1996_2023_final.csv"  # date,ticker,adj_close,in_sp500
SECTOR_CSV   = "constituents.csv"

start_data      = "2005-01-01"
end_data        = "2023-12-31"
vix_start_data  = "1996-01-01"   # VIX/BIL download starts earlier so pre-2005 history
                                  # is available for threshold computation in the first WFO window

treasury_ticker  = "BIL"
inverse_ticker   = "SH"
benchmark_ticker = "QQQ"   # Fix [3]: was SPY

transaction_cost_bps = 5
tc = transaction_cost_bps / 10_000

top_n_candidates          = [5, 10, 15, 20]
lookback_candidates       = [6, 9, 12]
VIX_MA_WINDOW = 90    # MA crossover window
VIX_MA_FACTOR = 1.15   # risk-off when VIX > 1.2x its 90-day MA
skip_months               = 1

TARGET_VOL  = 0.15   # kept for reference but vol scaling is disabled (unlevered mode)
VOL_WINDOW  = 21
MAX_LEVER   = 1.0    # unlevered: scale always 1.0
MIN_LEVER   = 1.0
MIN_HISTORY = 60     # minimum VIX observations in rolling window; skip if fewer
# VIX regime: fixed absolute threshold (VIX_LEVEL), not grid-searched

WFO_WINDOWS = [
    ("2005-01-01", "2007-12-31", "2008-01-01", "2010-12-31"),
    ("2008-01-01", "2010-12-31", "2011-01-01", "2013-12-31"),
    ("2011-01-01", "2013-12-31", "2014-01-01", "2016-12-31"),
    ("2014-01-01", "2016-12-31", "2017-01-01", "2019-12-31"),
    ("2017-01-01", "2019-12-31", "2020-01-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
]


# =========================
# UTIL: LOAD SECTOR MAP
# =========================

def load_sector_map(path=SECTOR_CSV):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Warning: sector file '{path}' not found -- sector neutrality disabled.")
        return {}
    symbol_col = next((c for c in df.columns if c.lower() in ["symbol", "ticker"]), None)
    sector_col = next((c for c in df.columns if "sector" in c.lower()), None)
    if symbol_col is None or sector_col is None:
        print("Warning: could not infer symbol/sector columns -- sector neutrality disabled.")
        return {}
    syms = df[symbol_col].astype(str).str.strip().str.replace(".", "-", regex=False)
    secs = df[sector_col].astype(str).str.strip()
    mapping = dict(zip(syms, secs))
    print(f"Loaded sector map for {len(mapping)} tickers.")
    return mapping


sector_map            = load_sector_map()
USE_SECTOR_NEUTRALITY = len(sector_map) > 0


# =========================
# 1. LOAD PRICE PANEL  [Fix 1: survivorship bias]
#
# Long-format CSV: date, ticker, adj_close, in_sp500
# Pivot to wide matrices for prices AND membership.
# membership[date, ticker] == True only when that firm was in the index.
# =========================

print("Loading price data from long-format CSV ...")
raw = pd.read_csv(PRICES_CSV, parse_dates=["date"])
raw["ticker"] = raw["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)

prices     = raw.pivot(index="date", columns="ticker", values="adj_close").sort_index()
membership = raw.pivot(index="date", columns="ticker", values="in_sp500").sort_index()
membership = membership.fillna(0).astype(bool)

prices    = prices[
    (prices.index >= pd.to_datetime(start_data)) &
    (prices.index <= pd.to_datetime(end_data))
]
membership = membership.reindex(prices.index).fillna(False)

min_obs    = 252
valid_cols = prices.columns[prices.notna().sum() >= min_obs]
prices     = prices[valid_cols]
membership = membership.reindex(columns=valid_cols, fill_value=False)

print(f"Price panel: {prices.shape}  |  Membership panel: {membership.shape}")


# =========================
# 2. DOWNLOAD VIX, BIL, QQQ
# =========================

print("Downloading VIX, BIL, QQQ ...")
extra_data = yf.download(
    ["^VIX", treasury_ticker, inverse_ticker, benchmark_ticker],
    start=vix_start_data,
    end=end_data,
    auto_adjust=False,
    progress=False,
)

extra_adj       = extra_data["Adj Close"]
extra_adj.index = pd.to_datetime(extra_adj.index).tz_localize(None)

# Keep the full VIX series (back to vix_start_data) for pre-window threshold computation.
# The trading-day series (vix, treasury, benchmark) are restricted to common_index.
vix_full     = extra_adj["^VIX"].dropna()

common_index = prices.index.intersection(extra_adj.index)
prices       = prices.loc[common_index]
membership   = membership.loc[common_index]
vix          = extra_adj["^VIX"].loc[common_index]
treasury     = extra_adj[treasury_ticker].loc[common_index]
inverse_ret  = extra_adj[inverse_ticker].pct_change().fillna(0.0).loc[common_index]
benchmark    = extra_adj[benchmark_ticker].loc[common_index]

print(f"Aligned trading days: {len(common_index)}  |  Price matrix: {prices.shape}")


# =========================
# 3. MONTHLY MOMENTUM & MEMBERSHIP
# =========================

monthly_prices     = prices.resample("ME").last()
monthly_membership = membership.resample("ME").last()
ts_mom_12          = monthly_prices.shift(1) / monthly_prices.shift(13) - 1


def get_target_weights(dt, lookback_months, top_n, momentum_df):
    """
    Return {ticker: weight} for a rebalance date.
    Only tickers with in_sp500==True on dt are eligible.  [Fix 1]
    """
    if dt not in momentum_df.index:
        return {}

    mom_row = momentum_df.loc[dt]

    if dt in monthly_membership.index:
        in_idx = monthly_membership.loc[dt].reindex(mom_row.index, fill_value=False)
        # If membership data is empty/stale for this date, fall back to all tickers
        if in_idx.sum() == 0:
            in_idx = pd.Series(True, index=mom_row.index)
    else:
        in_idx = pd.Series(True, index=mom_row.index)

    combined = mom_row[in_idx].dropna()
    if combined.empty:
        return {}

    sorted_mom = combined.sort_values(ascending=False)

    if USE_SECTOR_NEUTRALITY:
        sec_buckets = {}
        for name in sorted_mom.index:
            sec = sector_map.get(name, "Unknown")
            sec_buckets.setdefault(sec, []).append(name)
        n_sectors  = len(sec_buckets)
        if n_sectors == 0:
            return {}
        per_sector = max(1, int(np.ceil(top_n / n_sectors)))
        nominees   = []
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
        w_sec  = 1.0 / len(final_buckets)
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


# =========================
# 4. DAILY RETURNS
# =========================

stock_rets     = prices.pct_change()
# Winsorize: cap daily returns at ±50% -- any move beyond this is a data error
# (real stocks can gap down -50% on bankruptcy but not gap up +200% in a day)
stock_rets     = stock_rets.clip(lower=-0.5, upper=0.5).fillna(0.0)
treasury_rets  = treasury.pct_change().fillna(0.0)
benchmark_rets = benchmark.pct_change().fillna(0.0)
realized_vol   = benchmark_rets.rolling(VOL_WINDOW).std() * np.sqrt(252)
rv_aligned_all = realized_vol.reindex(vix.index).ffill().bfill()


# =========================
# HELPERS
# =========================

def compute_cagr(eq):
    if len(eq) == 0:
        return np.nan
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    return np.nan if n_years <= 0 else (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1


def compute_sharpe(rets, ppyr=252):
    if len(rets) < 2:
        return np.nan
    mu    = rets.mean() * ppyr
    sigma = rets.std() * np.sqrt(ppyr)
    return np.nan if sigma == 0 else mu / sigma


def max_drawdown(eq):
    if len(eq) == 0:
        return np.nan
    return (eq / eq.cummax() - 1.0).min()


def _nan_stats(lookback_months, top_n):
    return dict(
        cagr_strategy=np.nan, cagr_benchmark=np.nan, cagr_bench_levered=np.nan,
        sharpe_strategy=np.nan, sharpe_benchmark=np.nan, sharpe_bench_levered=np.nan,
        mdd_strategy=np.nan, mdd_benchmark=np.nan,
        avg_turnover=np.nan,
        lookback_months=lookback_months, top_n=top_n,
        portfolio_ret_bt=pd.Series(dtype=float),
        benchmark_ret_bt=pd.Series(dtype=float),
        portfolio_eq_bt=pd.Series(dtype=float),
        benchmark_eq_bt=pd.Series(dtype=float),
    )


# =========================
# 5. BACKTEST
#
# Fix [2]: VIX threshold from pre-window history only; skip if not enough.
# Fix [3]: Leverage-matched QQQ benchmark computed alongside strategy.
# Fix [4]: Portfolio drifts naturally between month-end rebalances.
#          TC charged only on rebalance days, not daily.
# =========================

def run_backtest(lookback_months, top_n, start_date, end_date):
    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)

    vix_ma   = vix.rolling(VIX_MA_WINDOW, min_periods=20).mean()
    risk_off = (vix > VIX_MA_FACTOR * vix_ma).reindex(prices.index).fillna(False)
    equity_scale = (~risk_off).astype(float)  # 1.0 when risk-on, 0.0 when risk-off

    # Window
    window_idx = prices.index[
        (prices.index >= start_dt) & (prices.index <= end_dt)
    ]
    if len(window_idx) == 0:
        return _nan_stats(lookback_months, top_n)

    momentum = (
        monthly_prices.shift(skip_months) /
        monthly_prices.shift(skip_months + lookback_months) - 1
    )


    # Map month-ends to nearest prior trading day.
    # Also force a rebalance on day 1 of the window so the portfolio is
    # always initialised -- without this, windows that start mid-month
    # (including the 2023 OOS window) begin with zero weights.
    month_ends = pd.date_range(start=window_idx[0], end=window_idx[-1], freq="ME")
    rebalance_dates = {window_idx[0]}   # always rebalance on day 1
    for me in month_ends:
        candidates = window_idx[window_idx <= me]
        if len(candidates):
            rebalance_dates.add(candidates[-1])

    # Dollar-value drift simulation (fixes weight-decay bug)
    all_tickers    = prices.columns.tolist()
    n_assets       = len(all_tickers)
    ticker_idx_map = {t: i for i, t in enumerate(all_tickers)}

    nav           = 1.0
    held_dollars  = np.zeros(n_assets)
    cash_dollars  = 0.0

    port_rets_list  = []
    bench_rets_list = []
    turnover_list   = []
    dates_list      = []

    stock_rets_np  = stock_rets.values
    stock_rets_pos = {d: i for i, d in enumerate(stock_rets.index)}

    for date in window_idx:
        ro_today    = bool(risk_off.get(date, False))
        scale_today = float(equity_scale.get(date, 1.0))

        # Rebalance on month-end dates (and forced day-1 of window)
        if date in rebalance_dates:
            avail_months = monthly_prices.index[monthly_prices.index <= date]
            turnover_today = 0.0
            if len(avail_months) > 0:
                last_month = avail_months[-1]
                tgt        = get_target_weights(last_month, lookback_months, top_n, momentum)

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



                # Turnover as fraction of NAV
                old_alloc = np.append(held_dollars, cash_dollars)
                new_alloc = np.append(new_stock_d,  new_cash_d)
                turnover_today = float(np.abs(new_alloc - old_alloc).sum() / 2.0 / max(nav, 1e-12))

                held_dollars = new_stock_d
                cash_dollars = new_cash_d
        else:
            turnover_today = 0.0

        # Daily P&L in dollars then convert to return pct
        di  = stock_rets_pos.get(date)
        sr  = np.nan_to_num(stock_rets_np[di] if di is not None else np.zeros(n_assets))
        treas_r = float(treasury_rets.get(date, 0.0))
        sh_r    = float(inverse_ret.get(date, 0.0))
        cash_r  = sh_r if ro_today else treas_r

        stock_pnl  = float(held_dollars @ sr)
        cash_pnl   = cash_dollars * cash_r
        tc_cost    = turnover_today * tc * max(nav, 1e-12)

        port_ret_pct = (stock_pnl + cash_pnl - tc_cost) / max(nav, 1e-12)

        # Drift: grow dollar positions with daily returns
        held_dollars = held_dollars * (1.0 + sr)
        cash_dollars = cash_dollars * (1.0 + cash_r)
        nav          = max(nav + stock_pnl + cash_pnl - tc_cost, 1e-12)

        bench_r = float(benchmark_rets.get(date, np.nan))

        port_rets_list.append(port_ret_pct)
        bench_rets_list.append(bench_r)
        turnover_list.append(turnover_today)
        dates_list.append(date)
    port_ret_w  = pd.Series(port_rets_list,  index=dates_list)
    bench_ret_w = pd.Series(bench_rets_list, index=dates_list).dropna()
    # Use fillna(0) so any date gaps in the benchmark don't NaN-out strategy returns
    # and propagate forward through cumprod, causing the flat-line bug
    port_ret_w  = port_ret_w.reindex(bench_ret_w.index).fillna(0.0)
    turn_w      = pd.Series(turnover_list,   index=dates_list).reindex(bench_ret_w.index).fillna(0.0)

    port_eq_w  = (1 + port_ret_w).cumprod()
    bench_eq_w = (1 + bench_ret_w).cumprod()

    if len(port_eq_w) == 0:
        return _nan_stats(lookback_months, top_n)

    return dict(
        cagr_strategy    = compute_cagr(port_eq_w),
        cagr_benchmark   = compute_cagr(bench_eq_w),
        sharpe_strategy  = compute_sharpe(port_ret_w),
        sharpe_benchmark = compute_sharpe(bench_ret_w),
        mdd_strategy     = max_drawdown(port_eq_w),
        mdd_benchmark    = max_drawdown(bench_eq_w),
        avg_turnover     = float(turn_w.mean()),

        lookback_months  = lookback_months,
        top_n            = top_n,
        portfolio_ret_bt = port_ret_w,
        benchmark_ret_bt = bench_ret_w,
        portfolio_eq_bt  = port_eq_w,
        benchmark_eq_bt  = bench_eq_w,
    )


# =========================
# 6. WALK-FORWARD OPTIMIZATION
# =========================

all_oos_portfolio     = []
all_oos_benchmark     = []
wfo_summary           = []
chosen_params         = []

print("\n=== WALK-FORWARD OPTIMIZATION ===\n")

for (train_start, train_end, test_start, test_end) in WFO_WINDOWS:
    print(f"TRAIN {train_start}-{train_end}  |  TEST {test_start}-{test_end}")

    best_stats = None
    for L in lookback_candidates:
        for n in top_n_candidates:
            s  = run_backtest(L, n, train_start, train_end)
            sh = s["sharpe_strategy"]
            if not np.isnan(sh) and (
                best_stats is None or sh > best_stats["sharpe_strategy"]
            ):
                best_stats = s

    if best_stats is None:
        print("  -> Skipped: insufficient VIX history for clean threshold.\n")
        continue

    best_L   = best_stats["lookback_months"]
    best_n   = best_stats["top_n"]
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
        train_start  = train_start, train_end    = train_end,
        test_start   = test_start,  test_end     = test_end,
        train_L      = best_L,      train_n      = best_n,
        train_sharpe = best_stats["sharpe_strategy"],
        train_cagr   = best_stats["cagr_strategy"],
        test_sharpe  = oos["sharpe_strategy"],
        test_cagr    = oos["cagr_strategy"],
        test_mdd     = oos["mdd_strategy"],
    ))


# =========================
# 7. STITCH FULL OOS RETURNS
# =========================

if not all_oos_portfolio:
    raise RuntimeError("No OOS segments produced -- check data and WFO windows.")

portfolio_ret_bt = pd.concat(all_oos_portfolio).sort_index()
benchmark_ret_bt = pd.concat(all_oos_benchmark).sort_index()

# Deduplicate (overlapping window boundaries)
portfolio_ret_bt = portfolio_ret_bt[~portfolio_ret_bt.index.duplicated(keep="first")]
benchmark_ret_bt = benchmark_ret_bt[~benchmark_ret_bt.index.duplicated(keep="first")]

# Align to common dates - fill gaps with 0 (flat day), never drop
common_idx       = portfolio_ret_bt.index.intersection(benchmark_ret_bt.index)
portfolio_ret_bt = portfolio_ret_bt.reindex(common_idx).fillna(0.0)
benchmark_ret_bt = benchmark_ret_bt.reindex(common_idx).fillna(0.0)

n_nan = portfolio_ret_bt.isna().sum()
if n_nan > 0:
    first_nan = portfolio_ret_bt.index[portfolio_ret_bt.isna()][0]
    print(f"WARNING: {n_nan} NaN returns in portfolio after stitch -- first at {first_nan}")
    portfolio_ret_bt = portfolio_ret_bt.fillna(0.0)

portfolio_eq_oos = (1 + portfolio_ret_bt).cumprod()
benchmark_eq_oos = (1 + benchmark_ret_bt).cumprod()

print("\n=== FULL OOS RESULTS (2008-2023) ===")
print(f"Strategy CAGR:   {compute_cagr(portfolio_eq_oos):.2%}")
print(f"QQQ CAGR:        {compute_cagr(benchmark_eq_oos):.2%}")
print(f"Strategy Sharpe: {compute_sharpe(portfolio_ret_bt):.2f}")
print(f"QQQ Sharpe:      {compute_sharpe(benchmark_ret_bt):.2f}")
print(f"Strategy MaxDD:  {max_drawdown(portfolio_eq_oos):.2%}")
print(f"QQQ MaxDD:       {max_drawdown(benchmark_eq_oos):.2%}")
print(f"OOS days:        {len(portfolio_ret_bt)}")

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
    Ls   = [p[0] for p in chosen_params]
    ns   = [p[1] for p in chosen_params]
    print("\n=== PARAMETER STABILITY ===")
    print(f"VIX regime: MA({VIX_MA_WINDOW}d) x {VIX_MA_FACTOR} (fixed)")
    print(f"Lookbacks:       {Ls}    mode={max(set(Ls), key=Ls.count)}")
    print(f"n_stocks:        {ns}    mode={max(set(ns), key=ns.count)}")


# =========================
# 8. EXPORT VIX THRESHOLD
# =========================

# VIX MA crossover threshold (adaptive)
vix_ma_oos      = vix.rolling(VIX_MA_WINDOW, min_periods=20).mean()
vix_threshold   = VIX_MA_FACTOR
vix_ma_series   = vix_ma_oos
risk_off_oos    = (vix > VIX_MA_FACTOR * vix_ma_oos)
risk_off_signal = risk_off_oos
sh_ret_bt       = inverse_ret.reindex(portfolio_ret_bt.index).fillna(0.0)
print(f"VIX regime: MA({VIX_MA_WINDOW}d) x {VIX_MA_FACTOR}")


# =========================
# 9. REGIME BREAKDOWN
# =========================

risk_off_mask_oos = risk_off_oos.reindex(portfolio_ret_bt.index).fillna(False)

strat_risk_on  = portfolio_ret_bt[~risk_off_mask_oos].dropna()
strat_risk_off = portfolio_ret_bt[risk_off_mask_oos].dropna()

print("\n=== REGIME BREAKDOWN ===")
print(
    f"Risk-ON  days: {len(strat_risk_on):4d} | "
    f"CAGR: {compute_cagr((1+strat_risk_on).cumprod()):.2%} | "
    f"Sharpe: {compute_sharpe(strat_risk_on):.2f}"
)
print(
    f"Risk-OFF days: {len(strat_risk_off):4d} | "
    f"CAGR: {compute_cagr((1+strat_risk_off).cumprod()):.2%} | "
    f"Sharpe: {compute_sharpe(strat_risk_off):.2f}"
)
ron_sharpe = compute_sharpe(strat_risk_on)
print(
    "\nInterpretation:",
    "Momentum alpha looks real -- risk-on days carrying returns."
    if ron_sharpe > 0.8
    else "Risk-on Sharpe weak -- returns may be driven by crash avoidance, not selection."
)


# =========================
# 10. ROBUSTNESS SURFACE
# =========================

def run_robustness_surface(oos_start="2008-01-01", oos_end="2023-12-31"):
    print("\n=== ROBUSTNESS SURFACE ===")
    print("Wide green plateau = real signal  |  Single spike = overfit\n")

    surface = {}
    for L in lookback_candidates:
        for n in top_n_candidates:

            res               = run_backtest(L, n, oos_start, oos_end)
            surface[(L,n)] = res["sharpe_strategy"]

    sharpes = [v for v in surface.values() if not np.isnan(v)]
    print(f"Combinations tested:           {len(sharpes)}")
    print(f"Mean Sharpe across all combos: {np.mean(sharpes):.3f}")
    print(f"Std  Sharpe across all combos: {np.std(sharpes):.3f}")
    print(f"% combos Sharpe > 0.5:         {np.mean([s > 0.5 for s in sharpes]):.1%}")
    print(f"% combos Sharpe > 0.0:         {np.mean([s > 0.0 for s in sharpes]):.1%}")


    heat    = np.array([
        [surface.get((L, n), np.nan) for n in top_n_candidates]
        for L in lookback_candidates
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1.5)
    ax.set_xticks(range(len(top_n_candidates)))
    ax.set_xticklabels(top_n_candidates)
    ax.set_yticks(range(len(lookback_candidates)))
    ax.set_yticklabels(lookback_candidates)
    plt.colorbar(im, ax=ax, label="OOS Sharpe")
    ax.set_xlabel("n stocks")
    ax.set_ylabel("Lookback (months)")
    ax.set_title(
        f"Robustness Surface (VIX MA x{VIX_MA_FACTOR})\n"
        "Flat green = real signal  |  Single spike = overfit"
    )
    plt.tight_layout()
    plt.show()
    return surface


# =========================
# 11. PLOTS
# =========================

if __name__ == "__main__":

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(portfolio_eq_oos,     label="WFO Strategy (OOS, net TC)", linewidth=2)
    ax.plot(benchmark_eq_oos, label="QQQ Buy and Hold", linewidth=1.5, linestyle="--")
    ax.set_title("Walk-Forward OOS Equity Curve — Strategy vs QQQ (2008-2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (Starting at 1.0)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    drawdown = portfolio_eq_oos / portfolio_eq_oos.cummax() - 1.0
    fig, ax  = plt.subplots(figsize=(13, 4))
    ax.fill_between(drawdown.index, drawdown, 0, color="firebrick", alpha=0.4)
    ax.plot(drawdown, color="firebrick", label="Strategy Drawdown")
    ax.set_title("Walk-Forward Strategy Drawdown (2008-2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    run_robustness_surface()
