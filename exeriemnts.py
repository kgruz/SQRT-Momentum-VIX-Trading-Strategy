"""
Experiment runner -- tests signal variations against baseline.
Run from your project directory:
    python3 run_experiments.py

Baseline: equal-weight, skip_months=1, VIX MA 1.2x, BIL risk-off
Target to beat: CAGR=15.13%, Sharpe=0.98, MaxDD=-41.96%
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ── config (match your main strategy) ──────────────────────────────────────
PRICES_CSV    = "sp500_prices_1996_2023_final.csv"
start_data    = "2005-01-01"
end_data      = "2023-12-31"
vix_start     = "1996-01-01"
VIX_MA_WINDOW = 90
tc            = 5 / 10_000
WFO_WINDOWS   = [
    ("2005-01-01","2007-12-31","2008-01-01","2010-12-31"),
    ("2008-01-01","2010-12-31","2011-01-01","2013-12-31"),
    ("2011-01-01","2013-12-31","2014-01-01","2016-12-31"),
    ("2014-01-01","2016-12-31","2017-01-01","2019-12-31"),
    ("2017-01-01","2019-12-31","2020-01-01","2022-12-31"),
    ("2020-01-01","2022-12-31","2023-01-01","2023-12-31"),
]
top_n_candidates    = [5, 10, 15, 20]
lookback_candidates = [6, 9, 12]

# ── load data once ──────────────────────────────────────────────────────────
print("Loading data...")
raw        = pd.read_csv(PRICES_CSV, parse_dates=["date"])
raw["ticker"] = raw["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
prices     = raw.pivot(index="date", columns="ticker", values="adj_close").sort_index()
membership = raw.pivot(index="date", columns="ticker", values="in_sp500").sort_index().fillna(0).astype(bool)
prices     = prices[(prices.index >= pd.to_datetime(start_data)) & (prices.index <= pd.to_datetime(end_data))]
membership = membership.reindex(prices.index).fillna(False)
valid_cols = prices.columns[prices.notna().sum() >= 252]
prices     = prices[valid_cols]
membership = membership.reindex(columns=valid_cols, fill_value=False)

print("Downloading VIX/BIL/QQQ/SH...")
extra = yf.download(["^VIX","BIL","QQQ","SH"], start=vix_start, end=end_data,
                    auto_adjust=False, progress=False)["Adj Close"]
extra.index = pd.to_datetime(extra.index).tz_localize(None)
common      = prices.index.intersection(extra.index)
prices      = prices.loc[common]
membership  = membership.loc[common]
vix         = extra["^VIX"].loc[common]
bil         = extra["BIL"].pct_change().fillna(0).loc[common]
sh          = extra["SH"].pct_change().fillna(0).loc[common]
qqq         = extra["QQQ"].loc[common]
print(f"Ready. {len(common)} trading days, {prices.shape[1]} tickers\n")

monthly_prices     = prices.resample("ME").last()
monthly_membership = membership.resample("ME").last()
stock_rets         = prices.pct_change().clip(-0.5, 0.5).fillna(0)

# precompute vol panel for vol-scaling experiments
rolling_vol  = stock_rets.rolling(63, min_periods=21).std() * np.sqrt(252)
monthly_vol  = rolling_vol.resample("ME").last()

# ── helpers ─────────────────────────────────────────────────────────────────
def compute_cagr(eq):
    n_years = len(eq) / 252
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan

def compute_sharpe(r):
    return (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan

def max_drawdown(eq):
    roll_max = eq.cummax()
    return ((eq - roll_max) / roll_max).min()

def get_weights(dt, lookback, top_n, skip, weighting, momentum_df):
    if dt not in momentum_df.index:
        return {}
    mom_row = momentum_df.loc[dt]
    in_idx  = monthly_membership.loc[dt].reindex(mom_row.index, fill_value=False) \
              if dt in monthly_membership.index else pd.Series(True, index=mom_row.index)
    if in_idx.sum() == 0:
        in_idx = pd.Series(True, index=mom_row.index)
    combined = mom_row[in_idx].dropna()
    if combined.empty:
        return {}
    top = combined.nlargest(top_n).index
    n   = len(top)
    if weighting == "equal":
        return {t: 1/n for t in top}
    elif weighting == "linear_rank":
        ranks = np.arange(1, n+1, dtype=float)
        w = (n+1-ranks); w /= w.sum()
        return dict(zip(top, w))
    elif weighting == "inv_vol":
        if dt in monthly_vol.index:
            v = monthly_vol.loc[dt].reindex(top).clip(lower=0.05).fillna(0.20)
            w = (1/v); w /= w.sum()
            return dict(zip(top, w.values))
        return {t: 1/n for t in top}
    return {t: 1/n for t in top}

def run_backtest(lookback, skip, top_n, start_date, end_date,
                 vix_factor, risk_off_asset, weighting):
    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date)
    momentum = (monthly_prices.shift(skip) /
                monthly_prices.shift(skip + lookback) - 1)
    vix_ma   = vix.rolling(VIX_MA_WINDOW, min_periods=20).mean()
    risk_off = (vix > vix_factor * vix_ma).reindex(prices.index).fillna(False)

    window_idx = prices.index[(prices.index >= start_dt) & (prices.index <= end_dt)]
    if len(window_idx) == 0:
        return None

    month_ends = pd.date_range(start=start_dt, end=end_dt, freq="ME")
    rebal_dates = set(
        prices.index[prices.index.searchsorted(m, side="right") - 1]
        for m in month_ends if m >= start_dt
    ) | {window_idx[0]}

    tickers    = prices.columns.tolist()
    ticker_map = {t: i for i, t in enumerate(tickers)}
    n_assets   = len(tickers)
    sr_np      = stock_rets.values
    sr_pos     = {d: i for i, d in enumerate(stock_rets.index)}

    nav = 1.0
    held_dollars = np.zeros(n_assets)
    cash_dollars = 1.0
    rets, bench_rets = [], []

    for date in window_idx:
        ro = bool(risk_off.get(date, False))
        turnover = 0.0
        if date in rebal_dates:
            avail = monthly_prices.index[monthly_prices.index <= date]
            if len(avail):
                tgt = get_weights(avail[-1], lookback, top_n, skip, weighting, momentum)
                tw  = np.zeros(n_assets)
                for t, w in tgt.items():
                    if t in ticker_map:
                        tw[ticker_map[t]] = w
                new_s = tw * nav if not ro else np.zeros(n_assets)
                new_c = 0.0   if not ro else nav
                old_a = np.append(held_dollars, cash_dollars)
                new_a = np.append(new_s, new_c)
                turnover = float(np.abs(new_a - old_a).sum() / 2 / max(nav, 1e-12))
                held_dollars, cash_dollars = new_s, new_c

        di  = sr_pos.get(date)
        sr  = np.nan_to_num(sr_np[di] if di is not None else np.zeros(n_assets))
        if risk_off_asset == "BIL":
            cash_r = float(bil.get(date, 0.0))
        elif risk_off_asset == "SH":
            cash_r = float(sh.get(date, 0.0))
        else:  # BIL always (even risk-on)
            cash_r = float(bil.get(date, 0.0))

        treas_r = float(bil.get(date, 0.0))
        actual_cash_r = cash_r if ro else treas_r

        stock_pnl = float(held_dollars @ sr)
        cash_pnl  = cash_dollars * actual_cash_r
        tc_cost   = turnover * tc * max(nav, 1e-12)
        port_r    = (stock_pnl + cash_pnl - tc_cost) / max(nav, 1e-12)

        held_dollars = held_dollars * (1 + sr)
        cash_dollars = cash_dollars * (1 + actual_cash_r)
        nav = max(nav + stock_pnl + cash_pnl - tc_cost, 1e-12)

        rets.append(port_r)
        bench_rets.append(qqq.pct_change().get(date, np.nan))

    if not rets:
        return None
    pr = pd.Series(rets, index=window_idx).fillna(0)
    br = pd.Series(bench_rets, index=window_idx).dropna()
    pr = pr.reindex(br.index).fillna(0)
    eq = (1 + pr).cumprod()
    return dict(sharpe=compute_sharpe(pr), cagr=compute_cagr(eq),
                mdd=max_drawdown(eq), port_rets=pr)

def run_wfo(cfg):
    all_rets = []
    for train_s, train_e, test_s, test_e in WFO_WINDOWS:
        best, best_sh = None, -np.inf
        for L in lookback_candidates:
            for n in top_n_candidates:
                s = run_backtest(L, cfg["skip"], n, train_s, train_e,
                                 cfg["vix_factor"], cfg["risk_off"], cfg["weighting"])
                if s and not np.isnan(s["sharpe"]) and s["sharpe"] > best_sh:
                    best_sh, best = s["sharpe"], (L, n)
        if best is None:
            continue
        oos = run_backtest(best[0], cfg["skip"], best[1], test_s, test_e,
                           cfg["vix_factor"], cfg["risk_off"], cfg["weighting"])
        if oos:
            all_rets.append(oos["port_rets"])

    if not all_rets:
        return None
    combined = pd.concat(all_rets).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")].fillna(0)
    eq = (1 + combined).cumprod()
    return dict(cagr=compute_cagr(eq), sharpe=compute_sharpe(combined),
                mdd=max_drawdown(eq))

# ── experiments ─────────────────────────────────────────────────────────────
experiments = [
    # name, config
    ("BASELINE (equal, skip=1, BIL, vix=1.2)",
     dict(skip=1, vix_factor=1.2, risk_off="BIL", weighting="equal")),

    ("skip=2",
     dict(skip=2, vix_factor=1.2, risk_off="BIL", weighting="equal")),

    ("skip=3",
     dict(skip=3, vix_factor=1.2, risk_off="BIL", weighting="equal")),

    ("vix_factor=1.1",
     dict(skip=1, vix_factor=1.1, risk_off="BIL", weighting="equal")),

    ("vix_factor=1.15",
     dict(skip=1, vix_factor=1.15, risk_off="BIL", weighting="equal")),

    ("vix_factor=1.25",
     dict(skip=1, vix_factor=1.25, risk_off="BIL", weighting="equal")),

    ("vix_factor=1.3",
     dict(skip=1, vix_factor=1.3, risk_off="BIL", weighting="equal")),

    ("SH risk-off",
     dict(skip=1, vix_factor=1.2, risk_off="SH", weighting="equal")),

    ("SH + vix=1.15",
     dict(skip=1, vix_factor=1.15, risk_off="SH", weighting="equal")),

    ("SH + vix=1.25",
     dict(skip=1, vix_factor=1.25, risk_off="SH", weighting="equal")),

    ("SH + skip=2",
     dict(skip=2, vix_factor=1.2, risk_off="SH", weighting="equal")),

    ("linear_rank weighting",
     dict(skip=1, vix_factor=1.2, risk_off="BIL", weighting="linear_rank")),

    ("inv_vol weighting",
     dict(skip=1, vix_factor=1.2, risk_off="BIL", weighting="inv_vol")),

    ("linear_rank + SH",
     dict(skip=1, vix_factor=1.2, risk_off="SH", weighting="linear_rank")),

    ("inv_vol + SH",
     dict(skip=1, vix_factor=1.2, risk_off="SH", weighting="inv_vol")),

    ("linear_rank + SH + vix=1.15",
     dict(skip=1, vix_factor=1.15, risk_off="SH", weighting="linear_rank")),

    ("inv_vol + SH + vix=1.15",
     dict(skip=1, vix_factor=1.15, risk_off="SH", weighting="inv_vol")),
]

print(f"{'Experiment':<45} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>9}  BEAT?")
print("-" * 80)

baseline_cagr, baseline_sharpe = 0.1513, 0.98
results = []

for name, cfg in experiments:
    r = run_wfo(cfg)
    if r is None:
        print(f"{name:<45} {'ERROR':>8}")
        continue
    beat = "✓ BEAT" if r["cagr"] > baseline_cagr and r["sharpe"] > baseline_sharpe else \
           "~ CAGR" if r["cagr"] > baseline_cagr else \
           "~ SHP"  if r["sharpe"] > baseline_sharpe else ""
    print(f"{name:<45} {r['cagr']:>7.2%} {r['sharpe']:>8.2f} {r['mdd']:>8.2%}  {beat}")
    results.append((name, r))

print("\n=== TOP 3 BY SHARPE ===")
top = sorted(results, key=lambda x: x[1]["sharpe"], reverse=True)[:3]
for name, r in top:
    print(f"  {name}: CAGR={r['cagr']:.2%} Sharpe={r['sharpe']:.2f} MaxDD={r['mdd']:.2%}")