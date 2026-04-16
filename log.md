# Project Change Log

## 2026-04-16

- Archived old driver scripts to archive/: main.py, main2_heston.py, main3.py, main4.py.
- Created a new unified main.py with two modes:
  - rolling (rolling volatility target)
  - instantaneous (absolute log-return target)

### Errors Found

- DA_utility.py: kalman_DA and particle_filter_DA returned inside the loop (early return bug).
- DA_utility_heston.py: return indentation mismatch in kalman_heston_DA and particle_filter_heston_DA.
- Main scripts expected 3 outputs from kalman/particle methods after utility signature simplification.
- Evaluation risk: posterior estimates and unshifted rolling baselines could introduce look-ahead bias.

### Fixes Applied

- DA_utility.py:
  - moved returns outside loops for kalman_DA and particle_filter_DA.
  - kept minimal outputs for these methods: (sigma_prior, sigma_est).
- DA_utility_heston.py:
  - fixed return indentation.
  - kept minimal outputs for kalman_heston_DA and particle_filter_heston_DA: (sigma_prior, sigma_est).
- main.py:
  - updated unpacking for kalman/particle methods to match new signatures.
  - switched scoring to prior-only DA signals for fairness.
  - shifted rolling volatility baselines by 1 step in instantaneous mode.
  - added burn-in masking for all metric calculations.
  - removed in-sample warmup scoring for GARCH by scoring only post-warmup forecasts.
  - added per-method evaluation audit printout (total, burn_in_excluded, nan_excluded, scored).

### Incremental Update

- Error: `probabilities do not sum to 1` in particle-filter resampling during instantaneous runs.
- Fix: stabilized PF weights in DA_utility.py and DA_utility_heston.py using log-weights, max-shift, finite-sum check, and uniform fallback.
- Why: guarantees valid probability vectors for `np.random.choice` and prevents underflow-induced failures.
- Validation: `python main.py --mode instantaneous --symbols BTC ETH --burn-in 200 --output inst_results.csv` completed successfully.

### Incremental Update

- Request: add minimal plotting to compare actual rolling volatility vs Kalman prediction.
- Fix: added `--plot` and `--plot-dir` in main.py, plus a small `save_plot(...)` helper.
- Output: saves PNG files like `plots/BTC_rolling_vol_100_kalman_prior.png`.
- Validation: `python main.py --mode rolling --symbols BTC --burn-in 200 --output rolling_results_test.csv --plot --plot-dir plots` completed and saved plot.

### Incremental Update

- Request: reduce overestimation risk by making rolling evaluation lag-aware and tighten instantaneous target definition.
- Fix (rolling): added `--rolling-window` and `--rolling-lag`; DA now assimilates lagged rolling observations and is scored against current rolling target.
- Fix (instantaneous): added `--inst-scale` and defaulted to `sqrt(pi/2)` to score against a sigma-consistent proxy.
- Bug found: lagged rolling setup initially caused all-NaN Kalman scores due initial rolling NaN propagation.
- Bug fix: filled initial rolling/lagged NaNs with 0.0 in main.py before DA calls.
- Validation:
  - `python main.py --mode rolling --symbols BTC ETH --burn-in 200 --rolling-window 20 --rolling-lag 20 --output rolling_results_lagged.csv --plot --plot-dir plots`
  - `python main.py --mode instantaneous --symbols BTC ETH --burn-in 200 --inst-scale 1.2533141373 --output inst_results_scaled.csv`

### Incremental Update

- Request: each run should save one plot per symbol with actual vs all methods in the same figure.
- Fix: replaced single-line plot helper with a comparison plot helper in main.py.
- Plot style: all series now use `alpha=0.5` for readability.
- Output examples:
  - `plots/BTC_rolling_vol_20_lag20_comparison.png`
  - `plots/ETH_rolling_vol_20_lag20_comparison.png`
  - `plots/BTC_inst_sigma_proxy_comparison.png`
  - `plots/ETH_inst_sigma_proxy_comparison.png`

### Incremental Update

- Request: update README with new results and explain why old rolling results looked too perfect.
- Fix: refreshed README performance section with:
  - current instantaneous scaled-proxy results,
  - legacy rolling (`rolling_vol_100`) results,
  - current lag-aware rolling (`rolling_vol_20_lag20`) results,
  - explicit explanation of legacy over-optimism (smooth target + online tracking alignment).
- Also updated project structure notes to reflect archived legacy main scripts and the new unified `main.py`.

### Incremental Update

- Request: README should be concise with 2-line approach summaries and a clear methodology section.
- Fix: rewrote README with:
  - 2-line summaries for instantaneous and rolling approaches,
  - explicit state-space equations,
  - Kalman and Particle Filter equations,
  - no-look-ahead protocol notes,
  - legacy vs current rolling results explanation.

### Incremental Update

- Request: add HMM-based version for rolling volatility.
- Fix: created `hmm.py` with Gaussian HMM regime model for lag-aware rolling volatility.
- Includes: train split fitting, online no-look-ahead HMM prior prediction, GARCH baselines, audit metrics, and optional comparison plots (`--plot`).

### Incremental Update

- Request: apply the "plot last 1000 points" change in `main.py` (not only `hmm.py`).
- Fix: added `--plot-points` (default `1000`) to `main.py`; rolling and instantaneous comparison plots now render only the most recent `N` points.
- Validation: `python main.py --mode rolling --symbols BTC --burn-in 200 --rolling-window 20 --rolling-lag 20 --output rolling_results_lagged.csv --plot --plot-dir plots --plot-points 1000` completed and saved plot.

### Incremental Update

- Request: reorder README so rolling appears before instantaneous in approach, methodology, and results.
- Fix: reordered sections accordingly and updated run command order.
- Added summary note: instantaneous volatility attempt (log-return proxy based) is currently unreliable under extreme crypto fluctuations.

### Incremental Update

- Request: rewrite `research_report.tex` based on current methodology/results and keep it under 2 pages.
- Fix: replaced long draft with a concise updated report emphasizing lag-aware rolling methodology first, then instantaneous proxy attempt.
- Included: strict evaluation protocol changes, legacy-vs-current rolling interpretation, BTC/ETH snapshot table, and short future-work section.
