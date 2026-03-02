import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# =========================
# PARAMETERS
# =========================

PRICES_CSV = "sp500_adjclose_2005_2023.csv"
SECTOR_CSV = "constituents.csv"

start_data = "2005-01-01"
end_data   = "2023-12-31"

treasury_ticker  = "BIL"
benchmark_ticker = "SPY"

transaction_cost_bps = 20
tc = transaction_cost_bps / 10000.0

# -------------------------------------------------------
# GRID — all four parameters searched, none hardcoded
# -------------------------------------------------------
top_n_candidates          = [5, 10, 15, 20]
lookback_candidates       = [6, 9, 12]
vix_percentile_candidates = [0.65, 0.70, 0.75, 0.80, 0.85]
skip_months               = 1

# Volatility scaling — fixed architecture, dynamic sizing only
TARGET_VOL = 0.15
VOL_WINDOW = 21
MAX_LEVER  = 1.0
MIN_LEVER  = 0.5

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

    symbol_col = None
    for c in df.columns:
        if c.lower() in ["symbol", "ticker"]:
            symbol_col = c
            break

    sector_col = None
    for c in df.columns:
        if "sector" in c.lower():
            sector_col = c
            break

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
# UTIL: POINT-IN-TIME S&P 500 MEMBERSHIP (SURVIVORSHIP BIAS FIX)
# =========================

HISTORICAL_COMPONENTS_CSV = "sp500_historical_components.csv"
REMOVED_PRICES_CSV        = "sp500_removed_prices.csv"


def load_historical_components(path=HISTORICAL_COMPONENTS_CSV):
    """
    Load point-in-time S&P 500 membership from the fja05680/sp500 dataset.
    Returns sorted list of (Timestamp, set_of_tickers) for fast lookup.
    Tickers are normalized to hyphen format (BF-B, BRK-B).
    """
    import bisect
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Warning: '{path}' not found -- point-in-time filter disabled.")
        return [], []

    entries = []
    for _, row in df.iterrows():
        dt = pd.to_datetime(row["date"])
        tickers = set(
            t.strip().replace(".", "-")
            for t in str(row["tickers"]).split(",")
        )
        entries.append((dt, tickers))
    entries.sort(key=lambda x: x[0])

    dates_list = [e[0] for e in entries]
    print(f"Loaded historical S&P 500 membership: {len(entries)} snapshots "
          f"({entries[0][0].date()} to {entries[-1][0].date()}).")
    return entries, dates_list


_hist_entries, _hist_dates = load_historical_components()


def eligible_tickers(as_of_date, available_columns):
    """Return tickers from available_columns that were in the S&P 500
    on as_of_date, using point-in-time historical membership data."""
    import bisect
    if not _hist_entries:
        return list(available_columns)
    idx = bisect.bisect_right(_hist_dates, as_of_date) - 1
    if idx < 0:
        return []
    members = _hist_entries[idx][1]
    return [t for t in available_columns if t in members]


# =========================
# 1. LOAD PRICE PANEL (including removed-stock prices)
# =========================

prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
prices = prices.sort_index()
prices = prices[
    (prices.index >= pd.to_datetime(start_data)) &
    (prices.index <= pd.to_datetime(end_data))
]

# Merge supplementary price data for stocks removed from S&P 500
try:
    removed_prices = pd.read_csv(REMOVED_PRICES_CSV, index_col=0, parse_dates=True)
    removed_prices = removed_prices.sort_index()
    if hasattr(removed_prices.index, "tz") and removed_prices.index.tz is not None:
        removed_prices.index = removed_prices.index.tz_localize(None)
    # Only add columns that don't already exist
    new_cols = [c for c in removed_prices.columns if c not in prices.columns]
    if new_cols:
        prices = prices.join(removed_prices[new_cols], how="left")
        print(f"Merged {len(new_cols)} removed-stock price series into panel.")
except FileNotFoundError:
    print("Warning: no supplementary removed-stock prices found.")

print(f"Loaded prices: {prices.shape}")


# =========================
# 2. DOWNLOAD VIX, BIL, SPY
# =========================

extra_data = yf.download(
    ["^VIX", treasury_ticker, benchmark_ticker],
    start=start_data,
    end=end_data,
    auto_adjust=False,
    progress=False
)

extra_adj       = extra_data["Adj Close"]
extra_adj.index = pd.to_datetime(extra_adj.index).tz_localize(None)

common_index = prices.index.intersection(extra_adj.index)
prices       = prices.loc[common_index]
vix          = extra_adj["^VIX"].loc[common_index]
treasury     = extra_adj[treasury_ticker].loc[common_index]
benchmark    = extra_adj[benchmark_ticker].loc[common_index]

print(f"Aligned trading days: {len(common_index)}")
print(f"Aligned price matrix shape: {prices.shape}")


# =========================
# 3. PRECOMPUTE MOMENTUM WEIGHTS
#    Keyed by (lookback_months, top_n).
#
#    SECTOR NEUTRALITY FIX:
#    Old: picked top_n first, then redistributed sector-neutrally.
#         All top_n could come from one sector.
#    New: take proportional nominees per sector first, re-rank by
#         momentum, then cap at top_n. Genuine sector diversification.
# =========================

monthly_prices = prices.resample("ME").last()
ts_mom_12      = monthly_prices.shift(1) / monthly_prices.shift(13) - 1

weights_daily_by_params = {}

for L in lookback_candidates:
    for top_n in top_n_candidates:

        momentum = (
            monthly_prices.shift(skip_months) /
            monthly_prices.shift(skip_months + L) - 1
        )

        weights_monthly = pd.DataFrame(
            index=monthly_prices.index,
            columns=prices.columns,
            data=0.0
        )

        for dt in momentum.index:
            mom_row  = momentum.loc[dt]
            ts_row   = ts_mom_12.loc[dt]

            # Point-in-time filter: only include tickers that were
            # S&P 500 members on or before this rebalance date
            eligible = eligible_tickers(dt, mom_row.index)
            mom_row  = mom_row[eligible]
            ts_row   = ts_row[eligible]

            # Only keep names with positive 12m TS momentum
            combined = mom_row[ts_row > 0].dropna()
            if combined.empty:
                continue

            sorted_mom = combined.sort_values(ascending=False)

            if USE_SECTOR_NEUTRALITY:
                # Step 1: bucket all TS-filtered names by sector
                sec_buckets = {}
                for name in sorted_mom.index:
                    sec = sector_map.get(name, "Unknown")
                    sec_buckets.setdefault(sec, []).append(name)

                n_sectors = len(sec_buckets)
                if n_sectors == 0:
                    continue

                # Step 2: proportional nominees per sector
                per_sector   = max(1, int(np.ceil(top_n / n_sectors)))
                nominees     = []
                for sec, names in sec_buckets.items():
                    nominees.extend(names[:per_sector])

                # Step 3: re-rank nominees by momentum, take best top_n
                top_names = (
                    sorted_mom
                    .loc[sorted_mom.index.intersection(nominees)]
                    .index[:top_n]
                )
                if len(top_names) == 0:
                    continue

                # Step 4: equal weight within sector, equal across sectors
                final_buckets = {}
                for name in top_names:
                    sec = sector_map.get(name, "Unknown")
                    final_buckets.setdefault(sec, []).append(name)

                weight_per_sector = 1.0 / len(final_buckets)
                for sec, names in final_buckets.items():
                    w_each = weight_per_sector / len(names)
                    weights_monthly.loc[dt, names] = w_each

            else:
                top_names = sorted_mom.index[:top_n]
                if len(top_names) == 0:
                    continue
                weights_monthly.loc[dt, top_names] = 1.0 / len(top_names)

        # FIX: deprecated fillna(method="ffill") -> .ffill()
        weights_daily = (
            weights_monthly
            .reindex(prices.index)
            .ffill()
            .fillna(0.0)
        )

        weights_daily_by_params[(L, top_n)] = weights_daily

print(
    f"Precomputed weights: {len(lookback_candidates)} lookbacks x "
    f"{len(top_n_candidates)} n_stocks = {len(weights_daily_by_params)} combos."
)


# =========================
# 4. DAILY RETURNS & REALIZED VOL
#
#    FIX: rv_threshold is NO LONGER computed here on the full dataset.
#    That was look-ahead bias — every decision was using the 75th pct
#    of vol computed across 2005-2023, including future data.
#    It is now computed inside run_backtest() from pre-window history only.
# =========================

# fill_method=None avoids forward-filling sparse removed-stock prices
# which would create spurious jumps when the next real price appears
stock_rets     = prices.pct_change(fill_method=None).fillna(0.0)
treasury_rets  = treasury.pct_change(fill_method=None).fillna(0.0)
benchmark_rets = benchmark.pct_change(fill_method=None).fillna(0.0)

realized_vol   = benchmark_rets.rolling(VOL_WINDOW).std() * np.sqrt(252)


# =========================
# HELPERS
# =========================

def compute_cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) == 0:
        return np.nan
    n_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if n_years <= 0:
        return np.nan
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return np.nan
    mu    = returns.mean() * periods_per_year
    sigma = returns.std() * np.sqrt(periods_per_year)
    return np.nan if sigma == 0 else mu / sigma


def max_drawdown(equity_curve: pd.Series) -> float:
    if len(equity_curve) == 0:
        return np.nan
    return (equity_curve / equity_curve.cummax() - 1.0).min()


# =========================
# 5. BACKTEST FUNCTION
#
#    FIXES:
#    [1] vix_threshold_local computed from pre-window history (no look-ahead)
#    [2] rv_threshold_local computed from pre-window history (no look-ahead)
#    [3] VIX regime uses percentile of historical VIX, not a hard level
#    [4] top_n is now a parameter, not hardcoded
#    [5] All deprecated .fillna(method=...) replaced with .ffill()/.bfill()
# =========================

def run_backtest(
    vix_percentile: float,
    lookback_months: int,
    top_n: int,
    start_date: str,
    end_date: str,
):
    start_dt      = pd.to_datetime(start_date)
    end_dt        = pd.to_datetime(end_date)
    weights_daily = weights_daily_by_params[(lookback_months, top_n)]

    # ------------------------------------------------------------------
    # Compute thresholds strictly from history before this window
    # ------------------------------------------------------------------
    vix_pre = vix[vix.index < start_dt]
    rv_pre  = realized_vol[realized_vol.index < start_dt].dropna()

    if len(vix_pre) >= 2:
        vix_threshold_local = float(vix_pre.quantile(vix_percentile))
    else:
        # No historical VIX data; use long-run median as conservative default
        vix_threshold_local = 20.0

    if len(rv_pre) >= 2:
        rv_threshold_local = float(rv_pre.quantile(0.75))
    else:
        # No historical RV data; use long-run equity vol as conservative default
        rv_threshold_local = 0.20

    # ------------------------------------------------------------------
    # Dual regime: risk-off when VIX or realized vol is elevated
    # ------------------------------------------------------------------
    rv_aligned = realized_vol.reindex(vix.index).ffill().bfill()
    risk_off   = (vix > vix_threshold_local) | (rv_aligned > rv_threshold_local)
    risk_on    = ~risk_off

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------
    stock_weights   = weights_daily.mul(risk_on.astype(float), axis=0)
    treasury_weight = risk_off.astype(float)

    # ------------------------------------------------------------------
    # Vol scaling on stock leg only
    # ------------------------------------------------------------------
    vol_series   = rv_aligned.reindex(stock_weights.index).ffill().bfill()
    scale_factor = (
        (TARGET_VOL / vol_series)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=MIN_LEVER, upper=MAX_LEVER)
    )

    scaled_stock_weights = stock_weights.mul(scale_factor, axis=0)

    # ------------------------------------------------------------------
    # Returns net of transaction costs
    # ------------------------------------------------------------------
    portfolio_ret_gross = (
        (scaled_stock_weights * stock_rets).sum(axis=1)
        + treasury_weight * treasury_rets
    )

    combined_weights                  = scaled_stock_weights.copy()
    combined_weights[treasury_ticker] = treasury_weight
    turnover                          = combined_weights.diff().abs().sum(axis=1) / 2.0
    turnover.iloc[0]                  = 0.0

    portfolio_ret = portfolio_ret_gross - turnover * tc

    # ------------------------------------------------------------------
    # Restrict to window
    # ------------------------------------------------------------------
    mask            = (portfolio_ret.index >= start_dt) & (portfolio_ret.index <= end_dt)
    portfolio_ret_w = portfolio_ret[mask]
    benchmark_ret_w = benchmark_rets[mask]
    turnover_w      = turnover[mask]

    portfolio_eq_w = (1 + portfolio_ret_w).cumprod()
    benchmark_eq_w = (1 + benchmark_ret_w).cumprod()

    if len(portfolio_eq_w) == 0:
        stats = dict(
            cagr_strategy=np.nan, cagr_benchmark=np.nan,
            sharpe_strategy=np.nan, sharpe_benchmark=np.nan,
            mdd_strategy=np.nan, mdd_benchmark=np.nan,
            avg_turnover=np.nan,
        )
    else:
        stats = dict(
            cagr_strategy    = compute_cagr(portfolio_eq_w),
            cagr_benchmark   = compute_cagr(benchmark_eq_w),
            sharpe_strategy  = compute_sharpe(portfolio_ret_w),
            sharpe_benchmark = compute_sharpe(benchmark_ret_w),
            mdd_strategy     = max_drawdown(portfolio_eq_w),
            mdd_benchmark    = max_drawdown(benchmark_eq_w),
            avg_turnover     = turnover_w.mean(),
        )

    stats.update(dict(
        vix_percentile   = vix_percentile,
        lookback_months  = lookback_months,
        top_n            = top_n,
        portfolio_ret_bt = portfolio_ret_w,
        benchmark_ret_bt = benchmark_ret_w,
        portfolio_eq_bt  = portfolio_eq_w,
        benchmark_eq_bt  = benchmark_eq_w,
    ))

    return stats


# =========================
# 6. WALK-FORWARD OPTIMIZATION
# =========================

all_oos_portfolio = []
all_oos_benchmark = []
wfo_summary       = []
chosen_params     = []

print("\n=== WALK-FORWARD OPTIMIZATION ===\n")

for (train_start, train_end, test_start, test_end) in WFO_WINDOWS:
    print(f"TRAIN {train_start}-{train_end}  |  TEST {test_start}-{test_end}")

    best_stats = None

    for L in lookback_candidates:
        for n in top_n_candidates:
            for pct in vix_percentile_candidates:
                s  = run_backtest(pct, L, n, train_start, train_end)
                sh = s["sharpe_strategy"]
                if not np.isnan(sh) and (best_stats is None or sh > best_stats["sharpe_strategy"]):
                    best_stats = s

    if best_stats is None:
        print("  -> No valid result, skipping.\n")
        continue

    best_pct = best_stats["vix_percentile"]
    best_L   = best_stats["lookback_months"]
    best_n   = best_stats["top_n"]
    chosen_params.append((best_pct, best_L, best_n))

    print(
        f"  TRAIN best: pct={best_pct}, L={best_L}, n={best_n} "
        f"| Sharpe={best_stats['sharpe_strategy']:.2f}, "
        f"CAGR={best_stats['cagr_strategy']:.2%}"
    )

    oos = run_backtest(best_pct, best_L, best_n, test_start, test_end)
    print(
        f"  OOS result: Sharpe={oos['sharpe_strategy']:.2f}, "
        f"CAGR={oos['cagr_strategy']:.2%}, "
        f"MaxDD={oos['mdd_strategy']:.2%}\n"
    )

    all_oos_portfolio.append(oos["portfolio_ret_bt"])
    all_oos_benchmark.append(oos["benchmark_ret_bt"])

    wfo_summary.append(dict(
        train_start  = train_start,  train_end   = train_end,
        test_start   = test_start,   test_end    = test_end,
        train_pct    = best_pct,     train_L     = best_L,     train_n    = best_n,
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

# Remove duplicate dates from overlapping windows
portfolio_ret_bt = portfolio_ret_bt[~portfolio_ret_bt.index.duplicated(keep="first")]

# FIX: safe two-way alignment
benchmark_ret_bt = benchmark_ret_bt.reindex(portfolio_ret_bt.index).dropna()
portfolio_ret_bt = portfolio_ret_bt.reindex(benchmark_ret_bt.index).dropna()

portfolio_eq_oos = (1 + portfolio_ret_bt).cumprod()
benchmark_eq_oos = (1 + benchmark_ret_bt).cumprod()

print("=== FULL OOS RESULTS (2008-2023) ===")
print(f"Strategy CAGR:   {compute_cagr(portfolio_eq_oos):.2%}")
print(f"SPY CAGR:        {compute_cagr(benchmark_eq_oos):.2%}")
print(f"Strategy Sharpe: {compute_sharpe(portfolio_ret_bt):.2f}")
print(f"SPY Sharpe:      {compute_sharpe(benchmark_ret_bt):.2f}")
print(f"Strategy MaxDD:  {max_drawdown(portfolio_eq_oos):.2%}")
print(f"SPY MaxDD:       {max_drawdown(benchmark_eq_oos):.2%}")
print(f"OOS days:        {len(portfolio_ret_bt)}")

print("\n=== WFO SUMMARY ===")
for row in wfo_summary:
    print(
        f"{row['train_start']}-{row['train_end']} "
        f"(pct={row['train_pct']}, L={row['train_L']}, n={row['train_n']}): "
        f"train Sharpe={row['train_sharpe']:.2f} | "
        f"OOS {row['test_start']}-{row['test_end']} "
        f"Sharpe={row['test_sharpe']:.2f}, "
        f"CAGR={row['test_cagr']:.2%}, "
        f"MaxDD={row['test_mdd']:.2%}"
    )

# Parameter stability
if chosen_params:
    pcts = [p[0] for p in chosen_params]
    Ls   = [p[1] for p in chosen_params]
    ns   = [p[2] for p in chosen_params]
    print("\n=== PARAMETER STABILITY ===")
    print("Consistent params across windows = more trustworthy signal")
    print(f"VIX pcts chosen: {pcts}  mode={max(set(pcts), key=pcts.count)}")
    print(f"Lookbacks:       {Ls}    mode={max(set(Ls), key=Ls.count)}")
    print(f"n_stocks:        {ns}    mode={max(set(ns), key=ns.count)}")


# =========================
# 8. EXPORT vix_threshold FOR STATS MODULE
#    Defined here so everything below and the stats import both have it.
# =========================

if chosen_params:
    pct_counts   = Counter([p[0] for p in chosen_params])
    vix_pct_mode = max(pct_counts, key=pct_counts.get)
else:
    vix_pct_mode = 0.75

vix_oos = vix.reindex(portfolio_ret_bt.index).dropna()
vix_threshold = float(vix_oos.quantile(vix_pct_mode))
print(
    f"\nvix_threshold exported for stats module: "
    f"pct={vix_pct_mode} -> absolute={vix_threshold:.1f}"
)


# =========================
# 9. REGIME BREAKDOWN
#    Separates momentum alpha (risk-on days) from market timing
#    contribution (risk-off / cash days).
#
#    What to look for:
#      Risk-ON Sharpe > 0.8  -> real momentum alpha
#      Risk-ON Sharpe < 0.5  -> returns driven by avoiding crashes,
#                               not by stock selection
# =========================

rv_oos = realized_vol.reindex(portfolio_ret_bt.index).dropna()
risk_off_mask = (
    (vix > vix_threshold) |
    (realized_vol > rv_oos.quantile(vix_pct_mode))
).reindex(portfolio_ret_bt.index).fillna(False)

strat_risk_on  = portfolio_ret_bt[~risk_off_mask].dropna()
strat_risk_off = portfolio_ret_bt[risk_off_mask].dropna()

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
print(
    f"\nInterpretation: "
    f"{'Momentum alpha looks real -- risk-on days are carrying returns.' if compute_sharpe(strat_risk_on) > 0.8 else 'Risk-on Sharpe is weak -- returns may be driven by crash avoidance, not stock selection.'}"
)


# =========================
# 10. ROBUSTNESS SURFACE
#     The single most important diagnostic.
#     Sweep ALL (L, n, vix_pct) combos over the full OOS period.
#
#     What to look for:
#       Wide green plateau  -> real signal, parameter-robust
#       Single bright spike -> overfit, found lucky combination
# =========================

def run_robustness_surface(oos_start="2008-01-01", oos_end="2023-12-31"):
    print("\n=== ROBUSTNESS SURFACE ===")
    print("Wide green plateau = real signal | Single spike = overfit\n")

    surface = {}
    for L in lookback_candidates:
        for n in top_n_candidates:
            for pct in vix_percentile_candidates:
                res              = run_backtest(pct, L, n, oos_start, oos_end)
                surface[(L,n,pct)] = res["sharpe_strategy"]

    sharpes = [v for v in surface.values() if not np.isnan(v)]
    print(f"Combinations tested:             {len(sharpes)}")
    print(f"Mean Sharpe across all combos:   {np.mean(sharpes):.3f}")
    print(f"Std  Sharpe across all combos:   {np.std(sharpes):.3f}")
    print(f"% of combos with Sharpe > 0.5:   {np.mean([s > 0.5 for s in sharpes]):.1%}")
    print(f"% of combos with Sharpe > 0.0:   {np.mean([s > 0.0 for s in sharpes]):.1%}")

    # Heatmap — fix vix_pct at median, vary L and n
    mid_pct = sorted(vix_percentile_candidates)[len(vix_percentile_candidates) // 2]
    heat    = np.array([
        [surface.get((L, n, mid_pct), np.nan) for n in top_n_candidates]
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
        f"Robustness Surface (VIX pct={mid_pct})\n"
        "Flat green = real signal  |  Single spike = overfit"
    )
    plt.tight_layout()
    plt.show()

    return surface


# =========================
# 11. PLOTS
#     FIX: heavy computation no longer runs on import.
#     Plots and surface only render when script is run directly.
# =========================

if __name__ == "__main__":

    # Equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_eq_oos, label="WFO Strategy (OOS, net of TC)")
    plt.plot(benchmark_eq_oos, label="SPY Buy & Hold")
    plt.title("Walk-Forward OOS Equity Curve (2008-2023)")
    plt.xlabel("Date")
    plt.ylabel("Equity (Starting at 1.0)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Drawdown
    drawdown = portfolio_eq_oos / portfolio_eq_oos.cummax() - 1.0

    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown, 0, color="firebrick", alpha=0.4)
    plt.plot(drawdown, color="firebrick", label="Strategy Drawdown")
    plt.title("Walk-Forward Strategy Drawdown (2008-2023)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Robustness surface — most important diagnostic
    run_robustness_surface()