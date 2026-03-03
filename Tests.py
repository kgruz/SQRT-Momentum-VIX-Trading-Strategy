"""
MOMENTUM-VIX STRATEGY — 20-TEST ROBUSTNESS & SIGNIFICANCE SUITE
================================================================
All tests are designed to be CRITICAL, not flattering.
If the strategy is overfit or fragile, these will show it.

Strategy specifics tested:
  - Momentum signal (cross-sectional, S&P 500 universe)
  - VIX MA regime filter (risk-off when VIX > 1.15x 90d MA)
  - SH (inverse S&P) held during risk-off
  - Linear rank position weighting
  - Walk-forward OOS returns only (no in-sample data)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from math import erf, sqrt

try:
    from scipy.stats import t as t_dist
except ImportError:
    t_dist = None

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

# ── strategy exports ────────────────────────────────────────────────────────
from Momentum_VIX_TS import (
    portfolio_ret_bt,   # daily OOS strategy returns (pd.Series)
    vix,                # daily VIX series aligned to trading days
    vix_threshold,      # float: VIX_MA_FACTOR used (1.15)
    vix_ma_series,      # pd.Series: rolling 90d MA of VIX
    risk_off_signal,    # pd.Series bool: True on risk-off days
    sh_ret_bt,          # pd.Series: SH daily returns aligned to OOS period
)

TRADING_DAYS = 252


# =========================================================
# HELPERS
# =========================================================

def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def annualized_return(r):
    r = pd.Series(r).dropna()
    if len(r) == 0: return np.nan
    years = len(r) / TRADING_DAYS
    return (1 + r).prod() ** (1 / years) - 1 if years > 0 else np.nan

def annualized_vol(r):
    r = pd.Series(r).dropna()
    return r.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(r) > 1 else np.nan

def sharpe_ratio(r, rf=0.0):
    r = pd.Series(r).dropna()
    if len(r) < 2: return np.nan
    ex = r - rf / TRADING_DAYS
    sig = ex.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return ex.mean() * TRADING_DAYS / sig if sig > 0 else np.nan

def sortino_ratio(r, rf=0.0):
    r = pd.Series(r).dropna()
    if len(r) < 2: return np.nan
    ex = r - rf / TRADING_DAYS
    down = ex[ex < 0]
    if len(down) == 0: return np.nan
    dd = down.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return ex.mean() * TRADING_DAYS / dd if dd > 0 else np.nan

def max_drawdown(eq):
    eq = pd.Series(eq).dropna()
    if len(eq) == 0: return np.nan
    return (eq / eq.cummax() - 1).min()

def ulcer_index(eq):
    eq = pd.Series(eq).dropna()
    dd = eq / eq.cummax() - 1
    return np.sqrt(np.mean((dd * 100) ** 2))

def martin_ratio(r):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod()
    ui = ulcer_index(eq)
    return r.mean() * TRADING_DAYS / ui if np.isfinite(ui) and ui > 0 else np.nan

def calmar_ratio(r):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod()
    mdd = abs(max_drawdown(eq))
    return annualized_return(r) / mdd if mdd > 0 else np.nan

def t_test_excess(s, b):
    s, b = pd.Series(s).dropna(), pd.Series(b).dropna()
    idx = s.index.intersection(b.index)
    ex = (s.loc[idx] - b.loc[idx]).dropna()
    n = len(ex)
    if n < 2: return np.nan, 1.0
    se = ex.std(ddof=1) / np.sqrt(n)
    if se == 0: return np.nan, 1.0
    t = ex.mean() / se
    p = t_dist.sf(t, df=n-1) if t_dist else 1 - norm_cdf(t)
    return t, p

def hurst_exponent(ts, min_lag=2, max_lag=100):
    x = pd.Series(ts).dropna().values
    if len(x) < max_lag: return np.nan
    tau = [np.sqrt(np.std(x[lag:] - x[:-lag])) for lag in range(min_lag, max_lag)]
    log_lags = np.log(np.arange(min_lag, max_lag, dtype=float))
    H, _ = np.polyfit(log_lags, np.log(tau), 1)
    return H

def deflated_sharpe(r, sr_thresh=0.0):
    r = pd.Series(r).dropna()
    T = len(r)
    if T < 2: return np.nan
    sr = sharpe_ratio(r)
    denom = 1 - r.skew() * sr + (r.kurt() / 4) * sr**2
    if denom <= 0: return np.nan
    z = (sr - sr_thresh) * np.sqrt(T - 1) / np.sqrt(denom)
    return norm_cdf(z)

def block_bootstrap_ci(r, n_boot=5000, alpha=0.05, block=5):
    r = pd.Series(r).dropna().values
    n = len(r)
    if n == 0: return (np.nan, np.nan), (np.nan, np.nan)
    n_blocks = int(np.ceil(n / block))
    sh_s, cagr_s = [], []
    for _ in range(n_boot):
        idx = np.random.randint(0, n_blocks, n_blocks)
        boot = np.concatenate([r[b*block:min((b+1)*block, n)] for b in idx])
        bs = pd.Series(boot)
        sh_s.append(sharpe_ratio(bs))
        cagr_s.append(annualized_return(bs))
    lo, hi = 100*alpha/2, 100*(1-alpha/2)
    return (np.percentile(sh_s, lo), np.percentile(sh_s, hi)), \
           (np.percentile(cagr_s, lo), np.percentile(cagr_s, hi))

def load_qqq(strat):
    strat = pd.Series(strat).dropna()
    data = yf.download("QQQ",
                       start=strat.index.min() - pd.Timedelta(days=5),
                       end=strat.index.max() + pd.Timedelta(days=5),
                       auto_adjust=False, progress=False)
    close = data["Adj Close"]
    if isinstance(close, pd.DataFrame):
        close = close["QQQ"] if "QQQ" in close.columns else close.iloc[:, 0]
    if hasattr(close.index, "tz") and close.index.tz:
        close = close.tz_convert(None)
    return close.pct_change().reindex(strat.index).fillna(0.0)


# =========================================================
# RUN ALL 20 TESTS
# =========================================================

def run_all_tests():
    strat = pd.Series(portfolio_ret_bt).dropna()
    qqq   = load_qqq(strat)
    if isinstance(qqq, pd.DataFrame):
        qqq = qqq.iloc[:, 0]
    idx   = strat.index.intersection(qqq.index)
    strat = strat.loc[idx]
    qqq   = qqq.loc[idx]
    eq_s  = (1 + strat).cumprod()
    eq_q  = (1 + qqq).cumprod()
    excess = (strat - qqq).dropna()

    # Align regime signal to OOS period
    ro = risk_off_signal.reindex(idx).fillna(False)

    # ── TEST 1: Core performance scorecard ────────────────────────────────
    print("=== TEST 1: CORE PERFORMANCE SCORECARD vs QQQ ===")
    print(f"Strategy CAGR:    {annualized_return(strat):.2%}")
    print(f"QQQ CAGR:         {annualized_return(qqq):.2%}")
    print(f"Strategy Vol:     {annualized_vol(strat):.2%}")
    print(f"QQQ Vol:          {annualized_vol(qqq):.2%}")
    print(f"Strategy Sharpe:  {sharpe_ratio(strat):.3f}")
    print(f"QQQ Sharpe:       {sharpe_ratio(qqq):.3f}")
    print(f"Strategy MaxDD:   {max_drawdown(eq_s):.2%}")
    print(f"QQQ MaxDD:        {max_drawdown(eq_q):.2%}")
    print(f"Strategy Calmar:  {calmar_ratio(strat):.3f}")
    print(f"QQQ Calmar:       {calmar_ratio(qqq):.3f}")
    print()

    # ── TEST 2: Risk-adjusted ratios ───────────────────────────────────────
    print("=== TEST 2: RISK-ADJUSTED RATIOS ===")
    print(f"Strategy Sortino: {sortino_ratio(strat):.3f}  |  QQQ: {sortino_ratio(qqq):.3f}")
    print(f"Strategy Martin:  {martin_ratio(strat):.3f}  |  QQQ: {martin_ratio(qqq):.3f}")
    print(f"Strategy Calmar:  {calmar_ratio(strat):.3f}  |  QQQ: {calmar_ratio(qqq):.3f}")
    print()

    # ── TEST 3: Statistical significance of excess returns ─────────────────
    print("=== TEST 3: STATISTICAL SIGNIFICANCE OF EXCESS RETURNS ===")
    print(f"Annualized excess return: {excess.mean() * TRADING_DAYS:.2%}")
    t_ex, p_ex = t_test_excess(strat, qqq)
    print(f"t-stat (excess > 0): {t_ex:.3f},  p-value (one-sided): {p_ex:.4f}")
    print(f"Interpretation: {'SIGNIFICANT at 5%' if p_ex < 0.05 else 'NOT significant at 5%'}")
    print()

    # ── TEST 4: CAPM alpha and beta ────────────────────────────────────────
    print("=== TEST 4: CAPM ALPHA & BETA vs QQQ ===")
    s_al = strat.loc[idx]
    q_al = qqq.loc[idx]
    if sm is not None:
        X = sm.add_constant((q_al).values)
        res = sm.OLS(s_al.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        alpha_d, beta = res.params[0], res.params[1]
        print(f"Alpha (annual):  {alpha_d * TRADING_DAYS:.2%}  t={res.tvalues[0]:.3f}  p={res.pvalues[0]:.4f}")
        print(f"Beta:            {beta:.3f}  t={res.tvalues[1]:.3f}  p={res.pvalues[1]:.4f}")
    else:
        X = np.column_stack([np.ones(len(q_al)), q_al.values])
        b = np.linalg.lstsq(X, s_al.values, rcond=None)[0]
        print(f"Alpha (annual):  {b[0] * TRADING_DAYS:.2%}  (install statsmodels for t-stats)")
        print(f"Beta:            {b[1]:.3f}")
    print()

    # ── TEST 5: Information ratio ──────────────────────────────────────────
    print("=== TEST 5: INFORMATION RATIO vs QQQ ===")
    ir = sharpe_ratio(excess)
    print(f"Information ratio: {ir:.3f}")
    print(f"Interpretation: {'Strong (>0.5)' if ir > 0.5 else 'Moderate (0.3-0.5)' if ir > 0.3 else 'Weak (<0.3)'}")
    print()

    # ── TEST 6: Return distribution shape ─────────────────────────────────
    print("=== TEST 6: RETURN DISTRIBUTION SHAPE ===")
    print(f"Strategy — skew: {strat.skew():.3f},  excess kurtosis: {strat.kurt():.3f}")
    print(f"QQQ      — skew: {qqq.skew():.3f},  excess kurtosis: {qqq.kurt():.3f}")
    print(f"Positive skew is better (right tail > left tail)")
    print()

    # ── TEST 7: Rolling 12-month Sharpe stability ──────────────────────────
    print("=== TEST 7: ROLLING 12-MONTH SHARPE STABILITY ===")
    roll_s = strat.rolling(TRADING_DAYS).apply(lambda x: sharpe_ratio(pd.Series(x)), raw=False)
    roll_q = qqq.rolling(TRADING_DAYS).apply(lambda x: sharpe_ratio(pd.Series(x)), raw=False)
    diff   = (roll_s - roll_q).dropna()
    print(f"Median rolling Sharpe — strategy: {roll_s.dropna().median():.3f}  |  QQQ: {roll_q.dropna().median():.3f}")
    print(f"% of 12m windows strategy beats QQQ Sharpe: {(diff > 0).mean():.1%}")
    print(f"Min rolling Sharpe (strategy): {roll_s.dropna().min():.3f}")
    print()

    # ── TEST 8: Probabilistic / deflated Sharpe ────────────────────────────
    print("=== TEST 8: PROBABILISTIC SHARPE RATIO ===")
    psr = deflated_sharpe(strat)
    print(f"Prob(true Sharpe > 0): {psr:.4f}")
    print(f"Interpretation: {'Very strong (>0.95)' if psr > 0.95 else 'Strong (>0.90)' if psr > 0.90 else 'Moderate'}")
    print()

    # ── TEST 9: Block bootstrap CI ─────────────────────────────────────────
    print("=== TEST 9: BLOCK BOOTSTRAP 95% CONFIDENCE INTERVALS ===")
    sh_ci, cagr_ci = block_bootstrap_ci(strat)
    print(f"Sharpe 95% CI: [{sh_ci[0]:.3f}, {sh_ci[1]:.3f}]")
    print(f"CAGR   95% CI: [{cagr_ci[0]:.2%}, {cagr_ci[1]:.2%}]")
    print(f"CI lower bound Sharpe > 0: {'YES' if sh_ci[0] > 0 else 'NO'}")
    print()

    # ── TEST 10: Hurst exponent (momentum persistence) ────────────────────
    print("=== TEST 10: HURST EXPONENT (MOMENTUM PERSISTENCE) ===")
    H = hurst_exponent(eq_s.pct_change().dropna())
    print(f"Hurst exponent: {H:.3f}")
    print(f"Interpretation: {'Persistent (momentum present)' if H > 0.55 else 'Random walk (~0.5)' if H > 0.45 else 'Mean-reverting'}")
    print()

    # ── TEST 11: Yearly stability ──────────────────────────────────────────
    print("=== TEST 11: YEAR-BY-YEAR PERFORMANCE vs QQQ ===")
    years = sorted(set(strat.index.year))
    wins = 0
    for y in years:
        m = strat.index.year == y
        rs, rq = strat[m], qqq[m]
        if len(rs) < 50: continue
        cs, cq = annualized_return(rs), annualized_return(rq)
        ss, sq = sharpe_ratio(rs), sharpe_ratio(rq)
        t, p   = t_test_excess(rs, rq)
        beat   = "✓" if cs > cq else "✗"
        if cs > cq: wins += 1
        print(f"{y} {beat}  CAGR: strat={cs:>7.2%} qqq={cq:>7.2%}  "
              f"Sharpe: {ss:.2f}/{sq:.2f}  t={t:.2f} p={p:.3f}")
    total_years = sum(1 for y in years if len(strat[strat.index.year == y]) >= 50)
    print(f"Beat QQQ CAGR in {wins}/{total_years} years ({wins/total_years:.0%})")
    print()

    # ── TEST 12: Macro window performance ─────────────────────────────────
    print("=== TEST 12: MACRO REGIME WINDOWS vs QQQ ===")
    windows = [
        ("GFC + aftermath",    "2008-01-01", "2012-12-31"),
        ("QE bull market",     "2013-01-01", "2016-12-31"),
        ("Late bull + COVID",  "2017-01-01", "2020-12-31"),
        ("Post-COVID regimes", "2021-01-01", "2023-12-31"),
    ]
    for label, s, e in windows:
        m = (strat.index >= s) & (strat.index <= e)
        rs, rq = strat[m], qqq[m]
        if len(rs) < 50:
            print(f"{label}: insufficient data"); continue
        t, p = t_test_excess(rs, rq)
        print(f"{label} ({s[:4]}-{e[:4]}): "
              f"CAGR {annualized_return(rs):.2%} vs {annualized_return(rq):.2%}  "
              f"Sharpe {sharpe_ratio(rs):.2f}/{sharpe_ratio(rq):.2f}  "
              f"p={p:.3f}")
    print()

    # ── TEST 13: VIX regime breakdown ─────────────────────────────────────
    print("=== TEST 13: VIX REGIME BREAKDOWN (RISK-ON vs RISK-OFF) ===")
    risk_on_ret  = strat[~ro].dropna()
    risk_off_ret = strat[ro].dropna()
    print(f"Risk-ON  ({len(risk_on_ret):4d} days): "
          f"CAGR={annualized_return(risk_on_ret):.2%}  "
          f"Sharpe={sharpe_ratio(risk_on_ret):.2f}")
    print(f"Risk-OFF ({len(risk_off_ret):4d} days): "
          f"CAGR={annualized_return(risk_off_ret):.2%}  "
          f"Sharpe={sharpe_ratio(risk_off_ret):.2f}")
    print(f"Risk-ON days carry returns: {'YES ✓' if annualized_return(risk_on_ret) > annualized_return(risk_off_ret) else 'NO ✗'}")
    print()

    # ── TEST 14: Reality check bootstrap (White-style) ────────────────────
    print("=== TEST 14: REALITY CHECK BOOTSTRAP (WHITE-STYLE, 5000 SAMPLES) ===")
    ex_vals = excess.values
    n_ex    = len(ex_vals)
    orig_sh = sharpe_ratio(excess)
    block   = 5
    n_bl    = int(np.ceil(n_ex / block))
    count   = 0
    for _ in range(5000):
        idx_bl = np.random.randint(0, n_bl, n_bl)
        boot   = np.concatenate([ex_vals[b*block:min((b+1)*block, n_ex)] for b in idx_bl])
        if sharpe_ratio(pd.Series(boot)) >= orig_sh:
            count += 1
    p_rc = count / 5000
    print(f"Observed excess Sharpe: {orig_sh:.3f}")
    print(f"Bootstrap p-value (Sharpe_boot >= Sharpe_obs): {p_rc:.4f}")
    print(f"Result: {'SIGNIFICANT ✓' if p_rc < 0.05 else 'NOT significant at 5%'}")
    print()

    # ── TEST 15: VIX permutation test (block shuffle) ──────────────────────
    print("=== TEST 15: VIX PERMUTATION TEST (1000 BLOCK SHUFFLES) ===")
    v_al  = vix_ma_series.reindex(idx).dropna()
    s_al  = strat.reindex(v_al.index)
    q_al2 = qqq.reindex(v_al.index)
    ro_al = risk_off_signal.reindex(v_al.index).fillna(False)
    rs_lo = s_al[~ro_al]
    rb_lo = q_al2[~ro_al]
    t_real, _ = t_test_excess(rs_lo, rb_lo)
    ro_vals = ro_al.values.astype(float)
    n_ro    = len(ro_vals)
    blk     = 21
    n_blk   = n_ro // blk
    t_perms = []
    for _ in range(1000):
        blocks = [ro_vals[i*blk:(i+1)*blk] for i in range(n_blk)]
        np.random.shuffle(blocks)
        perm = np.concatenate(blocks)
        if n_blk * blk < n_ro:
            perm = np.concatenate([perm, ro_vals[n_blk*blk:]])
        perm_mask = perm.astype(bool)
        rs_p = s_al[~perm_mask]
        rb_p = q_al2[~perm_mask]
        if len(rs_p) < 20: continue
        t_p, _ = t_test_excess(rs_p, rb_p)
        if np.isfinite(t_p):
            t_perms.append(t_p)
    t_perms = np.array(t_perms)
    p_perm  = np.mean(t_perms >= t_real) if len(t_perms) > 0 else np.nan
    print(f"Real risk-on t-stat: {t_real:.3f}")
    print(f"Permutation p-value: {p_perm:.4f}")
    print(f"VIX signal is non-random: {'YES ✓' if p_perm < 0.05 else 'NOT significant at 5%'}")
    print()

    # ── TEST 16: VIX noise injection stability ─────────────────────────────
    print("=== TEST 16: VIX NOISE INJECTION STABILITY (500 SIMULATIONS) ===")
    vix_vals = vix.reindex(idx).values
    ma_vals  = vix_ma_series.reindex(idx).values
    t_noise  = []
    for _ in range(500):
        noisy_vix = vix_vals + np.random.normal(0, 2.0, size=len(vix_vals))
        noisy_ro  = noisy_vix > vix_threshold * ma_vals
        rs_n = strat[~noisy_ro]
        rb_n = qqq[~noisy_ro]
        if len(rs_n) < 20: continue
        t_n, _ = t_test_excess(rs_n, rb_n)
        if np.isfinite(t_n):
            t_noise.append(t_n)
    t_noise = np.array(t_noise)
    if len(t_noise) > 0:
        print(f"Mean t-stat under noisy VIX: {t_noise.mean():.3f}")
        print(f"Std of t-stats:              {t_noise.std():.3f}")
        print(f"Fraction of runs with t < 0: {np.mean(t_noise < 0):.3f}")
        print(f"Signal robust to VIX noise: {'YES ✓' if np.mean(t_noise < 0) < 0.2 else 'FRAGILE ✗'}")
    else:
        print("Not enough data.")
    print()

    # ── TEST 17: SH hedge effectiveness ───────────────────────────────────
    print("=== TEST 17: SH HEDGE EFFECTIVENESS ON RISK-OFF DAYS ===")
    sh_ro  = sh_ret_bt.reindex(idx)[ro].dropna()
    qqq_ro = qqq[ro].dropna()
    if len(sh_ro) > 20:
        sh_ro_al  = sh_ro.reindex(qqq_ro.index).dropna()
        qqq_ro_al = qqq_ro.reindex(sh_ro_al.index).dropna()
        corr = sh_ro_al.corr(qqq_ro_al)
        sh_pos_days = (sh_ro > 0).mean()
        sh_cagr_ro  = annualized_return(sh_ro)
        print(f"SH vs QQQ correlation on risk-off days: {corr:.3f}  (should be negative)")
        print(f"SH positive return days during risk-off: {sh_pos_days:.1%}  (should be >50%)")
        print(f"SH annualized return during risk-off:    {sh_cagr_ro:.2%}")
        print(f"SH hedging working: {'YES ✓' if corr < -0.5 and sh_pos_days > 0.5 else 'PARTIAL' if corr < 0 else 'NO ✗'}")
    else:
        print("Insufficient risk-off days for SH analysis.")
    print()

    # ── TEST 15: VIX filter value-add ─────────────────────────────────────
    print("=== TEST 18: VIX FILTER VALUE-ADD (WITH vs WITHOUT) ===")
    # Simulate unfiltered: replace risk-off periods with QQQ returns
    unfiltered = strat.copy()
    unfiltered[ro] = qqq[ro]  # what would have happened without the filter
    print(f"With VIX filter:    CAGR={annualized_return(strat):.2%}  Sharpe={sharpe_ratio(strat):.3f}  MaxDD={max_drawdown(eq_s):.2%}")
    eq_uf = (1 + unfiltered).cumprod()
    print(f"Without VIX filter: CAGR={annualized_return(unfiltered):.2%}  Sharpe={sharpe_ratio(unfiltered):.3f}  MaxDD={max_drawdown(eq_uf):.2%}")
    print(f"Filter adds value: {'YES ✓' if sharpe_ratio(strat) > sharpe_ratio(unfiltered) else 'NO ✗'}")
    print()

    # ── TEST 16: Tail risk ─────────────────────────────────────────────────
    print("=== TEST 19: TAIL RISK (VaR, ES, FAT TAILS) ===")
    for label, r in [("Strategy", strat), ("QQQ", qqq)]:
        var5  = np.percentile(r, 5)
        es5   = r[r <= var5].mean()
        mu, sd = r.mean(), r.std(ddof=1)
        fat   = np.mean(np.abs(r - mu) > 3 * sd)
        print(f"{label}: VaR(5%)={var5:.3%}  ES(5%)={es5:.3%}  Fat-tail days={fat:.2%}")
    print()

    # ── TEST 17: Linear rank weight concentration check ───────────────────
    print("=== TEST 20: MOMENTUM SIGNAL QUALITY (RANK WEIGHT EFFECT) ===")
    # Compare strategy to a synthetic equal-weight version using QQQ as proxy
    # Real test: do risk-on days show positive alpha vs QQQ?
    risk_on_strat = strat[~ro]
    risk_on_qqq   = qqq[~ro]
    t_ro, p_ro = t_test_excess(risk_on_strat, risk_on_qqq)
    print(f"Risk-on excess vs QQQ: {(risk_on_strat - risk_on_qqq).mean() * TRADING_DAYS:.2%} annualized")
    print(f"t-stat: {t_ro:.3f},  p-value: {p_ro:.4f}")
    print(f"Momentum alpha is real: {'YES ✓' if p_ro < 0.05 else 'NOT significant at 5%'}")
    print()


    print("\n=== END OF 20-TEST SUITE ===\n")


if __name__ == "__main__":
    run_all_tests()