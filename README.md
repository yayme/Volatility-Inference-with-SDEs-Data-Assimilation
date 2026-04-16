# Volatility Inference with SDEs and Bayesian Filtering

## Summary Conclusion
Volatility in crypto markets is difficult to predict because of rapid sentiment shifts, liquidity shocks, leverage cascades, and fast regime changes in market microstructure.
This exploratory project studies online volatility estimation using a classic SDE state model with Kalman/Particle filtering under lag-aware evaluation.

Current experiments use mid-price based returns for simplicity and tractability.
Future work should incorporate sentiment/news signals, order-book and flow features, and multidimensional state-space models for improved robustness.

## Approach Summary

### Pipeline 1: Instantaneous Volatility (Heston-lite)
Infers latent volatility from price returns using a mean-reverting stochastic volatility state and Bayesian updates.
Evaluated against a scaled instantaneous proxy: `|log-return| * sqrt(pi/2)`.

### Pipeline 2: Rolling Volatility (Lagged)
Tracks smoothed volatility with a mean reverting state model and filtering under strict lag separation.
DA methods observe lagged rolling volatility, while scoring is done on the current rolling target.

## Methodology

### 1) State Dynamics (common form)
We model latent volatility as a 1D state:

$$
\sigma_{t+1} = \sigma_t + \kappa(\theta - \sigma_t)\Delta t + \xi\,\eta_t,
\quad \eta_t \sim \mathcal{N}(0, \Delta t)
$$

State variable:

$$
x_t \equiv \sigma_t
$$

### 2) Observation Models

Instantaneous proxy target:

$$
y_t^{(inst)} = \left|\log\frac{S_t}{S_{t-1}}\right|\sqrt{\frac{\pi}{2}}
$$

Rolling target (window `w`) with lag-aware DA observation (`L`):

$$
y_t^{(roll)} = \operatorname{std}(r_{t-w+1:t}),
\quad z_t^{(DA)} = y_{t-L}^{(roll)}
$$

### 3) Kalman Filter (scalar)
Prediction:

$$
\hat x^-_t = \hat x_{t-1} + \kappa(\theta - \hat x_{t-1})\Delta t,
\quad P^-_t = P_{t-1} + Q
$$

Update:

$$
K_t = \frac{P^-_t}{P^-_t + R},
\quad \hat x_t = \hat x^-_t + K_t(y_t - \hat x^-_t),
\quad P_t = (1-K_t)P^-_t
$$

### 4) Particle Filter (bootstrap)
Propagation for particles \(x_t^{(i)}\):

$$
x_t^{(i)} = x_{t-1}^{(i)} + \kappa(\theta - x_{t-1}^{(i)})\Delta t + \xi\eta_t^{(i)}
$$

Weights from Gaussian observation likelihood:

$$
\tilde w_t^{(i)} \propto \exp\left(-\frac{1}{2}\left(\frac{y_t - x_t^{(i)}}{R}\right)^2\right),
\quad w_t^{(i)}=\frac{\tilde w_t^{(i)}}{\sum_j \tilde w_t^{(j)}}
$$

Posterior estimate:

$$
\hat x_t = \frac{1}{N}\sum_{i=1}^N x_t^{(i)}
$$

## Evaluation Protocol
- DA metrics are computed on prior predictions (`*_prior`) rather than posterior estimates.
- Burn-in is excluded from scoring.
- Rolling mode uses explicit lag between DA input and scored target.
- GARCH is scored only on post-warmup forecasts.

## Why Earlier Rolling Results Looked Too Perfect
- Legacy target `rolling_vol_100` is highly smooth and strongly autocorrelated.
- Online DA tracking on a closely aligned observation stream can inflate tracking R2.
- This is not direct index look-ahead, but it overstates strict forecast realism.

## Current BTC/ETH Results Snapshot

### Instantaneous (scaled proxy)
Source: `inst_results_scaled.csv`

| Symbol | Kalman Heston R2 | Particle Heston R2 | Rolling(20) R2 |
|--------|-------------------|--------------------|----------------|
| BTC    | -0.202            | -0.024             | 0.100          |
| ETH    | -0.208            | -0.017             | 0.126          |

### Rolling (legacy, no explicit lag)
Source: `rolling_results.csv`

| Symbol | Kalman DA R2 | Particle Filter R2 |
|--------|---------------|--------------------|
| BTC    | 0.9919        | 0.8383             |
| ETH    | 0.9932        | 0.9181             |

### Rolling (current, lag-aware: window=20, lag=20)
Source: `rolling_results_lagged.csv`

| Symbol | Kalman DA R2 | Particle Filter R2 | GARCH(2,2) R2 |
|--------|---------------|--------------------|---------------|
| BTC    | 0.1416        | 0.1044             | 0.3295        |
| ETH    | 0.2375        | 0.2387             | 0.1708        |

## Run

```bash
python main.py --mode instantaneous --symbols BTC ETH --burn-in 200 --inst-scale 1.2533141373 --output inst_results_scaled.csv --plot --plot-dir plots
python main.py --mode rolling --symbols BTC ETH --burn-in 200 --rolling-window 20 --rolling-lag 20 --output rolling_results_lagged.csv --plot --plot-dir plots
```
