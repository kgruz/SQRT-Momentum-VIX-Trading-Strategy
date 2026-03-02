"""
QQQ-BASED ROBUSTNESS & SIGNIFICANCE TEST SUITE

All tests are designed to be CRITICAL, not flattering.
If the strategy is overfit or fragile, these will show it.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from math import erf, sqrt

# Try to use proper t-distribution & HAC if available
try:
    from scipy.stats import t as t_dist
except ImportError:
    t_dist = None

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

# Import strategy outputs from your main backtest file
from Momentum_VIX_TS import (
    portfolio_ret_bt,   # daily OOS strategy returns
    vix,                # daily VIX series
    vix_threshold,      # regime threshold exported from strategy script
)

TRADING_DAYS = 252


# =========================================================
# 0. BASIC HELPERS
# =========================================================

def norm_cdf(x: float) -> float:
    """Standard normal CDF using erf (used if SciPy missing)."""
    return 0.5 * (1 + erf(x / sqrt(2)))


def annualized_return(returns: pd.Series) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        return np.nan
    cum = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return cum ** (1 / years) - 1 if years > 0 else np.nan


def annualized_vol(returns: pd.Series) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        return np.nan
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sharpe vs constant rf (annualized)."""
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        return np.nan
    excess = returns - rf / TRADING_DAYS
    mu = excess.mean() * TRADING_DAYS
    sig = excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return np.nan if sig == 0 else mu / sig


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sortino ratio."""
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        return np.nan
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    mu = excess.mean() * TRADING_DAYS
    dd = downside.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return np.nan if dd == 0 else mu / dd


def max_drawdown(equity_curve: pd.Series) -> float:
    equity_curve = pd.Series(equity_curve).dropna()
    if len(equity_curve) == 0:
        return np.nan
    dd = equity_curve / equity_curve.cummax() - 1.0
    return dd.min()


def ulcer_index(equity_curve: pd.Series) -> float:
    """Ulcer index: root mean square of percentage drawdowns."""
    equity_curve = pd.Series(equity_curve).dropna()
    if len(equity_curve) == 0:
        return np.nan
    dd = equity_curve / equity_curve.cummax() - 1.0
    return np.sqrt(np.mean((dd * 100.0) ** 2))


def martin_ratio(returns: pd.Series) -> float:
    """Martin ratio = excess return / ulcer index."""
    returns = pd.Series(returns).dropna()
    eq = (1 + returns).cumprod()
    ui = ulcer_index(eq)
    if not np.isfinite(ui) or ui == 0:
        return np.nan
    mu = returns.mean() * TRADING_DAYS
    return mu / ui


def t_test_mean(series: pd.Series):
    """
    One-sample t-test on mean(series).
    H0: mean <= 0, H1: mean > 0
    Returns t_stat, p_one_sided
    """
    x = pd.Series(series).dropna().values
    n = len(x)
    if n < 2:
        return np.nan, 1.0
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return np.nan, 1.0
    se = s / np.sqrt(n)
    t_stat = m / se
    if t_dist is not None:
        p = t_dist.sf(t_stat, df=n - 1)
    else:
        p = 1 - norm_cdf(t_stat)
    return t_stat, p


def t_test_excess_returns(strategy: pd.Series, benchmark: pd.Series):
    """
    One-sample t-test on daily excess returns (strategy - benchmark).
    H0: mean(excess) <= 0, H1: mean(excess) > 0
    Returns t_stat, p_one_sided.
    """
    s = pd.Series(strategy).dropna()
    b = pd.Series(benchmark).dropna()
    idx = s.index.intersection(b.index)
    excess = (s.loc[idx] - b.loc[idx]).dropna()

    n = len(excess)
    if n < 2:
        return np.nan, 1.0

    m = excess.mean()
    sd = excess.std(ddof=1)
    if sd == 0:
        return np.nan, 1.0

    se = sd / np.sqrt(n)
    t_stat = m / se
    if t_dist is not None:
        p = t_dist.sf(t_stat, df=n - 1)
    else:
        p = 1 - norm_cdf(t_stat)
    return t_stat, p


# =========================================================
# 1. LOAD QQQ AND ALIGN
# =========================================================

def load_qqq_for_strategy(strat: pd.Series) -> pd.Series:
    """
    Download QQQ Adj Close and compute daily returns aligned to strategy.
    Always returns a 1D Series of daily returns.
    """
    strat = pd.Series(strat).dropna()
    start = strat.index.min() - pd.Timedelta(days=5)
    end = strat.index.max() + pd.Timedelta(days=5)

    data = yf.download("QQQ", start=start, end=end, auto_adjust=False, progress=False)

    close = data["Adj Close"]
    if isinstance(close, pd.DataFrame):
        if "QQQ" in close.columns:
            close = close["QQQ"]
        else:
            close = close.iloc[:, 0]

    if hasattr(close.index, "tz") and close.index.tz is not None:
        close = close.tz_convert(None)
    close.index = pd.to_datetime(close.index)

    q_ret = close.pct_change()
    q_ret = q_ret.reindex(strat.index).fillna(0.0)
    return q_ret


# =========================================================
# 2. HURST EXPONENT
# =========================================================

def hurst_exponent(ts: pd.Series, min_lag=2, max_lag=100) -> float:
    """
    Simple R/S-style Hurst exponent.
    H > 0.5: persistence (momentum)
    H < 0.5: mean reversion
    ~0.5 : random walk
    """
    x = pd.Series(ts).dropna().values
    if len(x) < max_lag:
        return np.nan

    lags = range(min_lag, max_lag)
    tau = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        if len(diff) == 0:
            continue
        tau.append(np.sqrt(np.std(diff)))
    if len(tau) < 2:
        return np.nan

    log_lags = np.log(list(lags)[:len(tau)])
    log_tau = np.log(tau)
    H, _ = np.polyfit(log_lags, log_tau, 1)
    return H


# =========================================================
# 3. DEFLATED / PROBABILISTIC SHARPE
# =========================================================

def deflated_sharpe_ratio(
    returns: pd.Series,
    sr_threshold: float = 0.0,
    n_strats: int = 1,
):
    """
    Approximate Probabilistic Sharpe Ratio (Bailey & Lopez de Prado-style).
    Uses correct ex-kurtosis term: ex_kurt / 4.
    """
    r = pd.Series(returns).dropna()
    T = len(r)
    if T < 2:
        return np.nan

    sr = sharpe_ratio(r)
    skew = r.skew()
    ex_kurt = r.kurt()  # excess kurtosis

    if n_strats > 1:
        penalty = 0.5 * np.log(n_strats) / np.sqrt(max(T - 1, 1))
    else:
        penalty = 0.0

    sr_star = sr_threshold + penalty

    denom = 1 - skew * sr + (ex_kurt / 4.0) * (sr ** 2)
    if denom <= 0:
        return np.nan

    z = (sr - sr_star) * np.sqrt(T - 1) / np.sqrt(denom)
    psr = norm_cdf(z)
    return psr


# =========================================================
# 4. BLOCK BOOTSTRAP CI
# =========================================================

def bootstrap_sharpe_cagr_ci(
    returns: pd.Series,
    n_boot: int = 5000,
    alpha: float = 0.05,
    block_size: int = 5,
):
    """
    Block bootstrap CIs for Sharpe and CAGR.
    """
    returns = pd.Series(returns).dropna()
    r = returns.values
    n = len(r)
    if n == 0:
        return (np.nan, np.nan), (np.nan, np.nan), np.array([]), np.array([])

    n_blocks = int(np.ceil(n / block_size))
    sh_samples = []
    cagr_samples = []

    for _ in range(n_boot):
        idx_blocks = np.random.randint(0, n_blocks, size=n_blocks)
        boot = []
        for b in idx_blocks:
            start = b * block_size
            end = min(start + block_size, n)
            boot.extend(r[start:end])
        boot_series = pd.Series(boot)
        sh_samples.append(sharpe_ratio(boot_series))
        cagr_samples.append(annualized_return(boot_series))

    sh_samples = np.array(sh_samples)
    cagr_samples = np.array(cagr_samples)

    sh_ci = (
        np.percentile(sh_samples, 100 * alpha / 2),
        np.percentile(sh_samples, 100 * (1 - alpha / 2)),
    )
    cagr_ci = (
        np.percentile(cagr_samples, 100 * alpha / 2),
        np.percentile(cagr_samples, 100 * (1 - alpha / 2)),
    )
    return sh_ci, cagr_ci, sh_samples, cagr_samples


# =========================================================
# 5. REALITY CHECK-STYLE BOOTSTRAP
# =========================================================

def reality_check_bootstrap(
    strategy: pd.Series,
    benchmark: pd.Series,
    n_boot: int = 5000,
    block_size: int = 5,
):
    """
    Simple White-style reality check on excess Sharpe of THIS strategy.
    Not multi-strategy – that would need the whole parameter grid matrix.
    """
    s = pd.Series(strategy).dropna()
    b = pd.Series(benchmark).dropna()
    idx = s.index.intersection(b.index)
    excess = (s.loc[idx] - b.loc[idx]).dropna()

    if len(excess) == 0:
        return np.nan, np.nan

    orig_sharpe = sharpe_ratio(excess)

    r = excess.values
    n = len(r)
    n_blocks = int(np.ceil(n / block_size))
    count = 0

    for _ in range(n_boot):
        idx_blocks = np.random.randint(0, n_blocks, size=n_blocks)
        boot = []
        for b_idx in idx_blocks:
            start = b_idx * block_size
            end = min(start + block_size, n)
            boot.extend(r[start:end])
        boot_series = pd.Series(boot)
        boot_sharpe = sharpe_ratio(boot_series)
        if boot_sharpe >= orig_sharpe:
            count += 1

    p_value = count / n_boot
    return orig_sharpe, p_value


# =========================================================
# 6. YEARLY STABILITY REPORT (vs QQQ)
# =========================================================

def yearly_stability_report(strategy: pd.Series, benchmark: pd.Series):
    """
    Per-calendar-year CAGR, Sharpe, and t-test on excess vs QQQ.
    """
    s = pd.Series(strategy).dropna()
    b = pd.Series(benchmark).dropna()
    idx = s.index.intersection(b.index)
    s = s.loc[idx]
    b = b.loc[idx]

    years = sorted(set(d.year for d in s.index))
    results = []

    for y in years:
        mask = s.index.year == y
        r_s = s[mask]
        r_b = b[mask]
        if len(r_s) < 50:
            continue

        cagr_s = annualized_return(r_s)
        cagr_b = annualized_return(r_b)
        sh_s = sharpe_ratio(r_s)
        sh_b = sharpe_ratio(r_b)
        t_stat, p_val = t_test_excess_returns(r_s, r_b)

        results.append({
            "year": y,
            "cagr_strategy": cagr_s,
            "cagr_qqq": cagr_b,
            "sharpe_strategy": sh_s,
            "sharpe_qqq": sh_b,
            "t_stat_excess": t_stat,
            "p_val_excess": p_val,
        })

    return results


# =========================================================
# 7. SUBSAMPLE MACRO WINDOWS (vs QQQ)
# =========================================================

def subsample_windows_report(strategy: pd.Series, qqq: pd.Series):
    """
    Performance in major macro windows vs QQQ:
      - 2008-2012 (GFC + aftermath)
      - 2013-2016 (QE bull)
      - 2017-2020 (late bull + COVID shock)
      - 2021-2023 (post-COVID regimes)
    """
    windows = [
        ("2008-01-01", "2012-12-31"),
        ("2013-01-01", "2016-12-31"),
        ("2017-01-01", "2020-12-31"),
        ("2021-01-01", "2023-12-31"),
    ]

    results = []
    for start, end in windows:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        mask = (strategy.index >= start_dt) & (strategy.index <= end_dt)
        s = strategy[mask]
        q = qqq[mask]
        if len(s) < 50:
            continue

        cagr_s = annualized_return(s)
        cagr_q = annualized_return(q)
        sh_s = sharpe_ratio(s)
        sh_q = sharpe_ratio(q)
        t_stat, p_val = t_test_excess_returns(s, q)

        results.append({
            "window": f"{start} to {end}",
            "cagr_strategy": cagr_s,
            "cagr_qqq": cagr_q,
            "sharpe_strategy": sh_s,
            "sharpe_qqq": sh_q,
            "t_stat_excess": t_stat,
            "p_val_excess": p_val,
        })

    return results


# =========================================================
# 8. CAPM REGRESSION vs QQQ (HAC if possible)
# =========================================================

def capm_alpha_beta(strategy: pd.Series, qqq: pd.Series, rf_annual: float = 0.0):
    """
    CAPM regression: r_s - rf = alpha + beta*(r_q - rf) + eps
    Uses HAC (Newey-West) if statsmodels is available.
    Returns dict of alpha, beta, t-stats, p-values.
    """
    s = pd.Series(strategy).dropna()
    q = pd.Series(qqq).dropna()
    idx = s.index.intersection(q.index)
    s = s.loc[idx]
    q = q.loc[idx]

    rf_daily = rf_annual / TRADING_DAYS
    y = s - rf_daily
    X = q - rf_daily

    if sm is None:
        X_mat = np.column_stack([np.ones(len(X)), X.values])
        beta_hat = np.linalg.lstsq(X_mat, y.values, rcond=None)[0]
        alpha, beta = beta_hat[0], beta_hat[1]
        return {
            "alpha_daily": alpha,
            "alpha_annual": alpha * TRADING_DAYS,
            "beta": beta,
            "alpha_t": np.nan,
            "beta_t": np.nan,
            "alpha_p": np.nan,
            "beta_p": np.nan,
        }

    X_design = sm.add_constant(X.values)
    model = sm.OLS(y.values, X_design)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    alpha = res.params[0]
    beta = res.params[1]
    alpha_t = res.tvalues[0]
    beta_t = res.tvalues[1]
    alpha_p = res.pvalues[0]
    beta_p = res.pvalues[1]

    return {
        "alpha_daily": alpha,
        "alpha_annual": alpha * TRADING_DAYS,
        "beta": beta,
        "alpha_t": alpha_t,
        "beta_t": beta_t,
        "alpha_p": alpha_p,
        "beta_p": beta_p,
    }


# =========================================================
# 9. VIX REGIME ANALYSIS & TESTS
# =========================================================

def vix_regime_analysis(strategy: pd.Series, qqq: pd.Series,
                        vix_series: pd.Series, threshold: float):
    """
    Performance conditional on VIX regimes:
        low-vol: VIX <= threshold
        high-vol: VIX > threshold
    """
    s = pd.Series(strategy).dropna()
    q = pd.Series(qqq).dropna()
    v = pd.Series(vix_series).dropna()
    idx = s.index.intersection(q.index).intersection(v.index)
    s = s.loc[idx]
    q = q.loc[idx]
    v = v.loc[idx]

    low_mask = v <= threshold
    high_mask = ~low_mask

    def metrics(mask):
        rs = s[mask]
        rq = q[mask]
        if len(rs) < 20:
            return None
        out = {}
        out["n_days"] = len(rs)
        out["cagr_strategy"] = annualized_return(rs)
        out["cagr_qqq"] = annualized_return(rq)
        out["sharpe_strategy"] = sharpe_ratio(rs)
        out["sharpe_qqq"] = sharpe_ratio(rq)
        excess = rs - rq
        out["mean_excess_daily"] = excess.mean()
        out["mean_excess_annual"] = excess.mean() * TRADING_DAYS
        t_stat, p_val = t_test_excess_returns(rs, rq)
        out["t_stat_excess"] = t_stat
        out["p_t_excess"] = p_val
        return out

    return {
        "low_vol": metrics(low_mask),
        "high_vol": metrics(high_mask),
    }


def vix_permutation_test_block(
    strategy: pd.Series,
    benchmark: pd.Series,
    vix_series: pd.Series,
    threshold: float,
    n_perm: int = 1000,
    block_size: int = 21,
):
    """
    VIX permutation test using BLOCK shuffling to preserve local autocorrelation.
    - Compute real low-vol t-stat on excess returns
    - Block-shuffle VIX many times
    - Recompute low-vol t-stat each time
    - p-value = P(t_perm >= t_real)

    FIX: v is aligned to (s,b) BEFORE taking v.values, so the boolean mask
    has the correct length for s.
    """
    s = pd.Series(strategy).dropna()
    b = pd.Series(benchmark).dropna()
    v = pd.Series(vix_series).dropna()

    # Align all three
    idx = s.index.intersection(b.index).intersection(v.index)
    s = s.loc[idx]
    b = b.loc[idx]
    v = v.loc[idx]
    v_vals = v.values  # same length as s and b now

    # Real low-vol mask and t-stat
    low_mask_real = v_vals <= threshold
    rs_low = s[low_mask_real]
    rb_low = b[low_mask_real]
    t_real, _ = t_test_excess_returns(rs_low, rb_low)
    if not np.isfinite(t_real):
        return np.nan, np.nan, np.array([])

    n = len(v_vals)
    n_blocks = n // block_size
    if n_blocks == 0:
        return t_real, np.nan, np.array([])

    t_samples = []
    for _ in range(n_perm):
        blocks = [
            v_vals[i * block_size:(i + 1) * block_size]
            for i in range(n_blocks)
        ]
        np.random.shuffle(blocks)
        perm_v = np.concatenate(blocks)
        if n_blocks * block_size < n:
            perm_v = np.concatenate([perm_v, v_vals[n_blocks * block_size:]])

        perm_mask = perm_v <= threshold
        rs_perm = s[perm_mask]
        rb_perm = b[perm_mask]
        if len(rs_perm) < 20:
            continue
        t_perm, _ = t_test_excess_returns(rs_perm, rb_perm)
        if np.isfinite(t_perm):
            t_samples.append(t_perm)

    t_samples = np.array(t_samples)
    if len(t_samples) == 0:
        return t_real, np.nan, np.array([])

    p_val = np.mean(t_samples >= t_real)
    return t_real, p_val, t_samples


def vix_noise_injection_test(
    strategy: pd.Series,
    benchmark: pd.Series,
    vix_series: pd.Series,
    threshold: float,
    noise_std: float = 2.0,
    n_sims: int = 500,
):
    """
    Add Gaussian noise to VIX and recompute low-vol t-stat each time.
    Tests how fragile the regime edge is to mis-measurement of VIX.
    """
    s = pd.Series(strategy).dropna()
    b = pd.Series(benchmark).dropna()
    v = pd.Series(vix_series).dropna()

    idx = s.index.intersection(b.index).intersection(v.index)
    s = s.loc[idx]
    b = b.loc[idx]
    v_vals = v.loc[idx].values

    t_stats = []
    for _ in range(n_sims):
        noisy_vix = v_vals + np.random.normal(0, noise_std, size=len(v_vals))
        mask = noisy_vix <= threshold
        rs = s[mask]
        rb = b[mask]
        if len(rs) < 20:
            continue
        t_sim, _ = t_test_excess_returns(rs, rb)
        if np.isfinite(t_sim):
            t_stats.append(t_sim)

    return np.array(t_stats)


# =========================================================
# 10. TAIL RISK & AUTOCORRELATION STRUCTURE
# =========================================================

def tail_risk_metrics(returns: pd.Series):
    """5% daily VaR and ES + fraction of >3σ days."""
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return {
            "VaR_5": np.nan,
            "ES_5": np.nan,
            "frac_3sigma": np.nan,
        }
    var_5 = np.percentile(r, 5)
    es_5 = r[r <= var_5].mean()
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0:
        frac_3 = np.nan
    else:
        frac_3 = np.mean(np.abs(r - mu) > 3 * sd)
    return {
        "VaR_5": var_5,
        "ES_5": es_5,
        "frac_3sigma": frac_3,
    }


def autocorr_and_lbq(returns: pd.Series, lags=10):
    """
    Simple autocorrelation stats. If statsmodels is present, use Ljung-Box.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 2:
        return {
            "autocorr_1": np.nan,
            "lbq_pvalue": np.nan,
        }

    autocorr_1 = r.autocorr(lag=1)

    if sm is None:
        lbq_p = np.nan
    else:
        lbq = sm.stats.acorr_ljungbox(r, lags=[lags], return_df=True)
        lbq_p = lbq["lb_pvalue"].iloc[0]

    return {
        "autocorr_1": autocorr_1,
        "lbq_pvalue": lbq_p,
    }


# =========================================================
# 11. RANDOM SUBSAMPLE ROBUSTNESS
# =========================================================

def subsample_robustness(
    strategy: pd.Series,
    benchmark: pd.Series,
    n_sims: int = 1000,
    frac: float = 0.7,
    block_size: int = 5,
):
    """
    Randomly subsample ~70% of days (block resampling) many times and
    recompute mean excess + Sharpe each time. If the edge is real,
    most subsamples should still show positive excess and Sharpe.
    """
    s = pd.Series(strategy).dropna()
    b = pd.Series(benchmark).dropna()
    idx = s.index.intersection(b.index)
    s = s.loc[idx]
    b = b.loc[idx]
    excess = (s - b).dropna()
    r = excess.values
    n = len(r)
    if n < 2:
        return np.array([]), np.array([])

    n_blocks_total = n // block_size
    n_blocks_keep = int(np.round(n_blocks_total * frac))
    if n_blocks_keep < 1:
        return np.array([]), np.array([])

    mean_samples = []
    sharpe_samples = []

    for _ in range(n_sims):
        block_ids = np.random.choice(n_blocks_total, size=n_blocks_keep, replace=False)
        subs = []
        for bid in block_ids:
            start = bid * block_size
            end = min(start + block_size, n)
            subs.extend(r[start:end])
        subs_series = pd.Series(subs)
        mean_samples.append(subs_series.mean())
        sharpe_samples.append(sharpe_ratio(subs_series))

    return np.array(mean_samples), np.array(sharpe_samples)


# =========================================================
# 12. MASTER RUNNER – 20 TESTS
# =========================================================

def run_all_tests():
    # Strategy OOS returns
    strat = pd.Series(portfolio_ret_bt).dropna()

    # QQQ returns
    qqq = load_qqq_for_strategy(strat)

    # Safety: if somehow still DataFrame, take first column
    if isinstance(qqq, pd.DataFrame):
        qqq = qqq.iloc[:, 0]

    # Align
    idx = strat.index.intersection(qqq.index)
    strat = strat.loc[idx]
    qqq = qqq.loc[idx]

    print("=== TEST 1: BASIC PERFORMANCE vs QQQ ===")
    eq_s = (1 + strat).cumprod()
    eq_q = (1 + qqq).cumprod()
    print(f"Strategy CAGR:   {annualized_return(strat):.2%}")
    print(f"QQQ CAGR:        {annualized_return(qqq):.2%}")
    print(f"Strategy Vol:    {annualized_vol(strat):.2%}")
    print(f"QQQ Vol:         {annualized_vol(qqq):.2%}")
    print(f"Strategy Sharpe: {sharpe_ratio(strat):.3f}")
    print(f"QQQ Sharpe:      {sharpe_ratio(qqq):.3f}")
    print(f"Strategy MaxDD:  {max_drawdown(eq_s):.2%}")
    print(f"QQQ MaxDD:       {max_drawdown(eq_q):.2%}")
    print()

    print("=== TEST 2: DISTRIBUTIONAL STATS ===")
    print(f"Strategy skew:   {strat.skew():.3f}, kurtosis (excess): {strat.kurt():.3f}")
    print(f"QQQ skew:        {qqq.skew():.3f}, kurtosis (excess): {qqq.kurt():.3f}")
    print()

    print("=== TEST 3: EXCESS RETURN STATS vs QQQ ===")
    excess = (strat - qqq).dropna()
    print(f"Mean daily excess:   {excess.mean():.6f}")
    print(f"Annualized excess:   {excess.mean() * TRADING_DAYS:.2%}")
    print(f"Std of daily excess: {excess.std(ddof=1):.6f}")
    t_ex, p_ex = t_test_excess_returns(strat, qqq)
    print(f"t-stat (mean excess > 0): {t_ex:.3f}, p_one_sided = {p_ex:.4f}")
    print()

    print("=== TEST 4: CAPM ALPHA/BETA vs QQQ (HAC if available) ===")
    capm = capm_alpha_beta(strat, qqq)
    print(f"Alpha (daily):   {capm['alpha_daily']:.6f}")
    print(f"Alpha (annual):  {capm['alpha_annual']:.2%}")
    print(f"Beta:            {capm['beta']:.3f}")
    print(f"Alpha t-stat:    {capm['alpha_t']:.3f}, p = {capm['alpha_p']:.4f}")
    print(f"Beta t-stat:     {capm['beta_t']:.3f}, p = {capm['beta_p']:.4f}")
    print()

    print("=== TEST 5: INFORMATION RATIO vs QQQ ===")
    ir = sharpe_ratio(excess)  # Sharpe on excess returns = info ratio
    print(f"Information ratio (vs QQQ): {ir:.3f}")
    print()

    print("=== TEST 6: SORTINO & MARTIN RATIOS ===")
    print(f"Strategy Sortino: {sortino_ratio(strat):.3f}")
    print(f"QQQ Sortino:      {sortino_ratio(qqq):.3f}")
    print(f"Strategy Martin:  {martin_ratio(strat):.3f}")
    print(f"QQQ Martin:       {martin_ratio(qqq):.3f}")
    print()

    print("=== TEST 7: ROLLING 12-MONTH SHARPE vs QQQ (summary) ===")
    window = TRADING_DAYS
    roll_sh_s = strat.rolling(window).apply(lambda x: sharpe_ratio(pd.Series(x)), raw=False)
    roll_sh_q = qqq.rolling(window).apply(lambda x: sharpe_ratio(pd.Series(x)), raw=False)
    diff_roll = (roll_sh_s - roll_sh_q).dropna()
    print(f"Median rolling Sharpe (strategy): {roll_sh_s.dropna().median():.3f}")
    print(f"Median rolling Sharpe (QQQ):      {roll_sh_q.dropna().median():.3f}")
    print(f"Median rolling Sharpe diff:       {diff_roll.median():.3f}")
    print(f"% of windows where strategy Sharpe > QQQ: {(diff_roll > 0).mean():.1%}")
    print()

    print("=== TEST 8: HURST EXPONENT OF EQUITY-CURVE RETURNS ===")
    equity = eq_s
    eq_ret = equity.pct_change().dropna()
    H = hurst_exponent(eq_ret)
    print(f"Hurst exponent: {H:.3f}")
    print()

    print("=== TEST 9: BLOCK BOOTSTRAP CI FOR SHARPE & CAGR ===")
    sh_ci, cagr_ci, sh_samples, cagr_samples = bootstrap_sharpe_cagr_ci(strat)
    print(f"Sharpe 95% CI: [{sh_ci[0]:.3f}, {sh_ci[1]:.3f}]")
    print(f"CAGR   95% CI: [{cagr_ci[0]:.2%}, {cagr_ci[1]:.2%}]")
    print()

    print("=== TEST 10: DEFLATED / PROBABILISTIC SHARPE (PSR) ===")
    psr = deflated_sharpe_ratio(strat, sr_threshold=0.0, n_strats=1)
    print(f"Prob(Sharpe > 0): {psr:.3f}")
    print()

    print("=== TEST 11: REALITY CHECK-STYLE BOOTSTRAP (EXCESS) ===")
    orig_sh, p_rc = reality_check_bootstrap(strat, qqq)
    print(f"Sharpe(excess): {orig_sh:.3f}")
    print(f"Bootstrap p (Sharpe_boot >= Sharpe_orig): {p_rc:.4f}")
    print()

    print("=== TEST 12: YEARLY STABILITY vs QQQ ===")
    yearly = yearly_stability_report(strat, qqq)
    for row in yearly:
        print(
            f"{row['year']}: "
            f"CAGR_strat={row['cagr_strategy']:.2%}, "
            f"CAGR_QQQ={row['cagr_qqq']:.2%}, "
            f"Sharpe_strat={row['sharpe_strategy']:.2f}, "
            f"Sharpe_QQQ={row['sharpe_qqq']:.2f}, "
            f"t_excess={row['t_stat_excess']:.3f}, "
            f"p_excess={row['p_val_excess']:.4f}"
        )
    print()

    print("=== TEST 13: SUBSAMPLE MACRO WINDOWS vs QQQ ===")
    subs = subsample_windows_report(strat, qqq)
    for row in subs:
        print(
            f"{row['window']}: "
            f"CAGR_strat={row['cagr_strategy']:.2%}, "
            f"CAGR_QQQ={row['cagr_qqq']:.2%}, "
            f"Sharpe_strat={row['sharpe_strategy']:.2f}, "
            f"Sharpe_QQQ={row['sharpe_qqq']:.2f}, "
            f"t_excess={row['t_stat_excess']:.3f}, "
            f"p_excess={row['p_val_excess']:.4f}"
        )
    print()

    print("=== TEST 14: VIX REGIME ANALYSIS (LOW vs HIGH VOL) ===")
    regimes = vix_regime_analysis(strat, qqq, vix, vix_threshold)
    for regime, vals in regimes.items():
        print(f"\n--- {regime.upper()} ---")
        if vals is None:
            print("No data in this regime.")
            continue
        for k, v in vals.items():
            if isinstance(v, float):
                if "cagr" in k:
                    print(f"{k}: {v:.2%}")
                elif "mean_excess" in k or "t_stat" in k or "p_t" in k:
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
            else:
                print(f"{k}: {v}")
    print()

    print("=== TEST 15: VIX PERMUTATION TEST (BLOCK SHUFFLE) ===")
    t_real, p_perm, t_perm_samples = vix_permutation_test_block(strat, qqq, vix, vix_threshold)
    print(f"Real low-vol t-stat: {t_real:.3f}")
    print(f"Permutation p-value (t_perm >= t_real): {p_perm:.4f}")
    print()

    print("=== TEST 16: VIX NOISE INJECTION STABILITY ===")
    t_noise = vix_noise_injection_test(strat, qqq, vix, vix_threshold)
    if len(t_noise) > 0:
        print(f"Mean simulated t-stat: {t_noise.mean():.3f}")
        print(f"Std of simulated t-stats: {t_noise.std():.3f}")
        frac_fail = np.mean(t_noise < 0)
        print(f"Fraction of noisy runs with t < 0: {frac_fail:.3f}")
    else:
        print("Not enough data for noise test.")
    print()

    print("=== TEST 17: TAIL RISK COMPARISON vs QQQ ===")
    tail_s = tail_risk_metrics(strat)
    tail_q = tail_risk_metrics(qqq)
    print("Strategy:", tail_s)
    print("QQQ:     ", tail_q)
    print()

    print("=== TEST 18: AUTOCORRELATION & LJUNG-BOX ===")
    ac_s = autocorr_and_lbq(strat)
    ac_q = autocorr_and_lbq(qqq)
    print(f"Strategy: autocorr_1={ac_s['autocorr_1']:.3f}, Ljung-Box p={ac_s['lbq_pvalue']:.4f}")
    print(f"QQQ:      autocorr_1={ac_q['autocorr_1']:.3f}, Ljung-Box p={ac_q['lbq_pvalue']:.4f}")
    print()

    print("=== TEST 19: RANDOM SUBSAMPLE ROBUSTNESS (70% DAYS) ===")
    mean_sub, sharpe_sub = subsample_robustness(strat, qqq)
    if len(mean_sub) > 0:
        print(f"Mean of mean_excess over subsamples: {mean_sub.mean():.6f}")
        print(f"Fraction of subsamples with mean_excess > 0: {(mean_sub > 0).mean():.3f}")
        print(f"Mean Sharpe(excess) over subsamples: {sharpe_sub.mean():.3f}")
        print(f"Fraction of subsamples with Sharpe > 0: {(sharpe_sub > 0).mean():.3f}")
    else:
        print("Not enough data for subsample robustness test.")
    print()

    print("=== TEST 20: POST-2017 \"GO LIVE\" PERFORMANCE vs QQQ ===")
    start_live = pd.to_datetime("2017-01-01")
    mask_live = strat.index >= start_live
    s_live = strat[mask_live]
    q_live = qqq[mask_live]
    if len(s_live) < 50:
        print("Not enough post-2017 data for live-style evaluation.")
    else:
        eq_s_live = (1 + s_live).cumprod()
        eq_q_live = (1 + q_live).cumprod()
        print(f"Strategy CAGR (2017+):   {annualized_return(s_live):.2%}")
        print(f"QQQ CAGR (2017+):        {annualized_return(q_live):.2%}")
        print(f"Strategy Sharpe (2017+): {sharpe_ratio(s_live):.3f}")
        print(f"QQQ Sharpe (2017+):      {sharpe_ratio(q_live):.3f}")
        print(f"Strategy MaxDD (2017+):  {max_drawdown(eq_s_live):.2%}")
        print(f"QQQ MaxDD (2017+):       {max_drawdown(eq_q_live):.2%}")
    print("\n=== END OF 20-TEST QQQ SUITE ===\n")


if __name__ == "__main__":
    run_all_tests()