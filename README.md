# SQRT-Momentum-VIX-Trading-Strategy
# SQRT Momentum + VIX Regime Trading Strategy

A systematic equity momentum strategy with VIX-based regime filtering, volatility scaling, and walk-forward optimization — backtested on S&P 500 constituents from 2005–2023.

---

## Overview

This strategy combines **cross-sectional momentum** (buying recent outperformers) with a **dual VIX + realized-volatility regime filter** (stepping aside into T-Bills during high-volatility environments) and **dynamic volatility scaling** to target a stable annualized risk level.

The backtest avoids look-ahead bias by using a full **walk-forward optimization (WFO)** framework — all parameters are selected on in-sample data and evaluated on unseen out-of-sample periods. All reported results cover the **stitched out-of-sample period from 2008 to 2023**.

---

## Repository Structure

```
├── Momentum_VIX_TS.py              # Main strategy + WFO backtest engine
├── Momentum_VIX_TS_Stats.py        # 20-test robustness & significance suite (vs QQQ)
├── sp500_adjclose_2005_2023.csv    # Daily adjusted close prices (S&P 500 constituents)
├── constituents.csv                # S&P 500 ticker → GICS sector mapping
└── .gitignore
```

> **Note:** `sp500_adjclose_2005_2023.csv` and `constituents.csv` are not included in the repo due to file size. See [Data](#data) below for how to source them.

---

## Strategy Logic

### 1. Signal Construction
- **Cross-sectional momentum**: Rank S&P 500 stocks by their returns over the past `L` months (skipping the most recent month to avoid short-term reversal).
- **Time-series momentum filter**: Only consider stocks with positive 12-month momentum — a trend-confirmation gate that reduces whipsawing into reversing names.
- **Sector neutrality**: Stocks are nominated proportionally from each GICS sector before final re-ranking by momentum, preventing concentration in a single sector.

### 2. Dual Regime Filter (Risk-On / Risk-Off)
The portfolio shifts entirely to **T-Bills (BIL)** when either condition is met:
- VIX exceeds its historical `p`-th percentile (tuned via WFO), **or**
- 21-day realized volatility of SPY exceeds its 75th historical percentile

Both thresholds are computed strictly from **pre-window history** — no look-ahead bias.

### 3. Volatility Scaling
The equity leg is scaled daily to target **15% annualized volatility**, with leverage bounded between **0.5× and 2.0×**.

### 4. Transaction Costs
All trades are charged **5 basis points** one-way, applied to the daily turnover of the full combined weight vector (equity + T-Bill).

---

## Parameters

| Parameter | Search Grid | Description |
|---|---|---|
| `top_n` | 5, 10, 15, 20 | Number of stocks held |
| `lookback_months` | 6, 9, 12 | Momentum lookback window |
| `vix_percentile` | 0.65, 0.70, 0.75, 0.80, 0.85 | VIX risk-off threshold percentile |
| `skip_months` | 1 (fixed) | Months skipped before lookback (reversal avoidance) |

All three free parameters are selected independently per WFO window by maximizing **in-sample Sharpe ratio**.

---

## Walk-Forward Optimization Windows

| Training Period | Out-of-Sample Period |
|---|---|
| 2005–2007 | 2008–2010 |
| 2008–2010 | 2011–2013 |
| 2011–2013 | 2014–2016 |
| 2014–2016 | 2017–2019 |
| 2017–2019 | 2020–2022 |
| 2020–2022 | 2023 |

---

## Data

The strategy requires two CSV files that are not committed to this repo:

| File | Description | How to obtain |
|---|---|---|
| `sp500_adjclose_2005_2023.csv` | Daily adjusted close prices. Index = dates, columns = tickers. | Download via `yfinance`, Tiingo, or similar for S&P 500 constituents |
| `constituents.csv` | Must have a `Symbol` (or `Ticker`) column and a `Sector` column | Available from [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies) or similar |

VIX (`^VIX`), BIL, and SPY are downloaded automatically at runtime via `yfinance`.

> **Survivorship bias warning:** Using a static constituent list overstates historical performance. For research use, point-in-time index membership data (e.g. from Compustat or a data vendor) is recommended.

---

## Installation

```bash
pip install pandas numpy yfinance matplotlib scipy statsmodels
```

`scipy` and `statsmodels` are optional but recommended — they enable the proper t-distribution, HAC-robust (Newey-West) CAPM regression, and Ljung-Box tests in the stats suite.

---

## Usage

### Run the backtest

```bash
python Momentum_VIX_TS.py
```

This will:
1. Load price and sector data, download VIX/BIL/SPY via `yfinance`
2. Precompute momentum weights for all 60 parameter combinations
3. Run walk-forward optimization across 6 windows
4. Print full OOS performance vs SPY and parameter stability analysis
5. Display equity curve, drawdown chart, and robustness surface heatmap

### Run the robustness & significance suite

```bash
python Momentum_VIX_TS_Stats.py
```

This imports the OOS returns from `Momentum_VIX_TS.py` and runs 20 statistical tests comparing the strategy against **QQQ** as a high-bar benchmark.

---

## 20-Test Robustness Suite (`Momentum_VIX_TS_Stats.py`)

All tests are designed to be **critical, not flattering**.

| # | Test | What it checks |
|---|---|---|
| 1 | Basic performance vs QQQ | CAGR, Vol, Sharpe, MaxDD |
| 2 | Distributional stats | Skewness, excess kurtosis |
| 3 | Excess return t-test vs QQQ | Is mean daily alpha > 0? |
| 4 | CAPM alpha/beta (HAC) | Newey-West robust alpha and market exposure |
| 5 | Information ratio | Sharpe of excess returns vs QQQ |
| 6 | Sortino & Martin ratios | Downside-risk-adjusted performance |
| 7 | Rolling 12-month Sharpe | % of windows outperforming QQQ |
| 8 | Hurst exponent | Return persistence (H > 0.5 = momentum) |
| 9 | Block bootstrap CI | 95% confidence intervals for Sharpe and CAGR |
| 10 | Probabilistic Sharpe (PSR) | P(true Sharpe > 0) after skew/kurtosis adjustment |
| 11 | Reality check bootstrap | Is excess Sharpe vs QQQ statistically significant? |
| 12 | Yearly stability | Per-year CAGR, Sharpe, and t-test vs QQQ |
| 13 | Macro subsample windows | GFC, QE bull, COVID, post-COVID regime performance |
| 14 | VIX regime analysis | Low-vol vs high-vol conditional performance |
| 15 | VIX permutation test | Block-shuffle VIX — does the regime edge survive? |
| 16 | VIX noise injection | Perturb VIX by ±2pts — how fragile is the threshold? |
| 17 | Tail risk (VaR / ES) | 5% VaR, Expected Shortfall, fat-tail frequency |
| 18 | Autocorrelation & Ljung-Box | Return serial dependence structure |
| 19 | Random 70% subsample | Edge consistency across random subsets of days |
| 20 | Post-2017 "go-live" | Performance in the most recent unseen period |

---

## Key Design Choices & Bias Mitigations

| Risk | Mitigation |
|---|---|
| Look-ahead bias in VIX threshold | Computed from pre-window history only, inside `run_backtest()` |
| Look-ahead bias in realized vol threshold | Same — computed from pre-window SPY vol history |
| Sector concentration | Proportional sector nomination before final momentum re-ranking |
| Parameter overfitting | Full 3-dimensional grid search evaluated strictly OOS via WFO |
| Transaction cost understatement | 5 bps applied to daily turnover of the full weight vector |

---

## Robustness Diagnostics

`run_robustness_surface()` sweeps all (lookback × n_stocks × VIX percentile) combinations across the full OOS period and generates a Sharpe heatmap.

- **Wide green plateau** → real signal, parameter-robust
- **Single bright spike** → likely overfit

The **Regime Breakdown** section separates momentum alpha from market-timing contribution:
- Risk-ON Sharpe > 0.8 → genuine stock-selection alpha
- Risk-ON Sharpe < 0.5 → returns driven primarily by crash avoidance

---

## Limitations

- **Survivorship bias** — static constituent list inflates performance; point-in-time membership data is needed for production use.
- **Capacity** — high-turnover momentum strategies face liquidity constraints at scale; 5 bps TC may understate real-world costs at large AUM.
- **Regime dependency** — the strategy holds cash during high-vol periods that may recover quickly, causing underperformance in sharp V-shaped recoveries (e.g. March–April 2020).
- **Parameter instability** — if chosen parameters vary widely across WFO windows, the signal may not generalize reliably out-of-sample.

---

## License

MIT
