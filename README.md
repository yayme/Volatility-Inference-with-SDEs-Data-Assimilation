
# Volatility Inference with SDEs and Bayesian Filtering

This project implements two volatility esti### Key Findings:
- **Pipeline 1 (Instantaneous)**: Kalman Heston achieves near-perfect performance (R² ≈ 0.9998) for real-time volatility inference
- **Pipeline 2 (Rolling)**: Both Kalman and Particle Filter DA methods achieve R² > 0.99 for smoothed volatility estimation
- **Bayesian filtering methods** substantially outperform traditional GARCH models in both pipelines
- **Particle Filter** shows robust performance across different volatility proxies and time scales
- **Traditional methods** (Naive DA, GARCH) provide limited explanatory power compared to SDE-based approachesn pipelines for cryptocurrency data, combining stochastic differential equations (SDEs) with Bayesian filtering techniques to produce real-time and smoothed volatility estimates.

## Methodology Overview

### Pipeline 1: Instantaneous Volatility Inference
- **Approach**: Heston-lite SDE model with Euler–Maruyama discretization + Bayesian filtering  
- **Model**:  
```
dS_t = μ S_t dt + σ_t S_t dW^S_t
dσ_t = κ(θ - σ_t) dt + ξ dW^σ_t
```
- **Ground Truth**: Absolute log returns `|log(S_t / S_{t-1})|` as a proxy for instantaneous volatility  
- **Methods**: Naive Data Assimilation, Generic DA, Kalman Filter, Particle Filter  
- **Output**: Real-time volatility estimates capturing rapid market dynamics  

### Pipeline 2: Rolling Volatility Estimation
- **Approach**: Ornstein–Uhlenbeck (OU) mean-reverting SDE model + Bayesian filtering  
- **Model**: Mean-reverting volatility dynamics with long-term equilibrium:
```
dσ_t = κ(θ - σ_t) dt + ξ dW_t
```
  where `κ` = mean reversion speed, `θ` = long-term volatility mean, `ξ` = volatility of volatility
- **Ground Truth**: Rolling standard deviation computed over 100-period windows  
- **Methods**: Kalman Filter, Particle Filter with Bayesian state estimation for smoothed volatility tracking  
- **Discretization**: Euler-Maruyama scheme with adaptive time step `dt = 1/1440` (minute-level resolution)  
- **Output**: Smoothed volatility estimates capturing mean-reverting behavior, ideal for risk management and forecasting  

## Key Features
- Combines SDE-based modeling with Bayesian filtering techniques
- Real-time and smoothed volatility estimation
- More adaptive and accurate than traditional GARCH models
- Euler–Maruyama discretization for SDE numerical integration
- Systematic validation against GARCH benchmarks across multiple cryptocurrencies

## Quick Start

### Pipeline 1: Instantaneous Volatility (Heston-lite)
```python
from DA_utility_heston import kalman_heston_DA, particle_filter_heston_DA
import pandas as pd

# Load cryptocurrency data
df = pd.read_csv('BTC_spot_full.csv')
prices = df['bam_close']

# Kalman Filter with Heston-lite dynamics
sigma_model, sigma_est = kalman_heston_DA(
    prices, kappa=2.0, theta=None, xi=0.3, dt=1/1440, R=1e-4, Q=0.01
)

# Particle Filter with Heston-lite dynamics
sigma_model_pf, sigma_est_pf = particle_filter_heston_DA(
    prices, kappa=2.0, xi=0.3, N_particles=150, R=0.001
)
```

### Pipeline 2: Rolling Volatility (Mean-Reverting)
```python
from DA_utility import kalman_DA, particle_filter_DA, compute_rolling_volatility
import pandas as pd

# Load data and compute rolling volatility
df = pd.read_csv('BTC_spot_full.csv')
returns = df['bam_close'].pct_change().fillna(0)
sigma_obs = compute_rolling_volatility(returns, window=100)

# Kalman Filter for rolling volatility
sigma_model_kf, sigma_est_kf = kalman_DA(
    sigma_obs, kappa=2.0, xi=0.3, dt=1/1440, R=1e-4
)

# Particle Filter for rolling volatility  
sigma_model_pf, sigma_est_pf = particle_filter_DA(
    sigma_obs, kappa=2.0, xi=0.3, N_particles=200, R=0.001
)
```

## Performance Results

### Pipeline 1: Instantaneous Volatility Inference (Heston-lite + Bayesian Filtering)
Performance metrics using absolute log returns as ground truth for instantaneous volatility:

| Symbol | Naive Heston MSE | Naive Heston R² | Generic Heston MSE | Generic Heston R² | Kalman Heston MSE | Kalman Heston R² | Particle Filter MSE | Particle Filter R² | Rolling Vol (20) R² |
|--------|------------------|-----------------|--------------------|--------------------|-------------------|------------------|---------------------|-------------------|---------------------|
| BNB    | 0.001833         | -33.23          | 0.000672           | -11.55             | 8.30e-09         | 0.9998           | 2.06e-05           | 0.615             | 0.173               |
| BTC    | 0.001437         | -43.87          | 0.001303           | -39.68             | 5.33e-09         | 0.9998           | 1.05e-05           | 0.673             | 0.088               |
| TRX    | 0.001737         | -28.41          | 0.001125           | -18.05             | 8.76e-09         | 0.9999           | 2.67e-05           | 0.547             | 0.169               |
| XRP    | 0.001888         | -21.50          | 0.000947           | -10.28             | 1.27e-08         | 0.9998           | 4.36e-05           | 0.481             | 0.151               |
| ETH    | 0.001481         | -30.01          | 0.000845           | -16.69             | 8.01e-09         | 0.9998           | 1.74e-05           | 0.635             | 0.115               |

### Pipeline 2: Rolling Volatility Estimation (Mean-reverting + Bayesian Filtering)
Performance metrics using rolling standard deviation (100 periods) as ground truth:

The following table shows Mean Squared Error (MSE) and R² scores for **Pipeline 2** (rolling volatility estimation) across five cryptocurrencies:

| Symbol | Naive DA MSE | Kalman DA MSE | Particle Filter MSE | GARCH(1,1) MSE | GARCH(2,2) MSE | Naive DA R² | Kalman DA R² | Particle Filter R² | GARCH(1,1) R² | GARCH(2,2) R² |
|--------|--------------|---------------|---------------------|----------------|----------------|-------------|--------------|-------------------|---------------|---------------|
| BNB    | 0.005269     | 7.11e-08      | 4.98e-08           | 1.04e-05       | 1.03e-05       | -224.24     | 0.997        | 0.998             | 0.556         | 0.561         |
| BTC    | 0.004721     | 4.87e-08      | 5.62e-08           | 7.54e-06       | 7.34e-06       | -416.09     | 0.996        | 0.995             | 0.334         | 0.352         |
| TRX    | 0.005798     | 1.20e-07      | 8.47e-08           | 1.16e-05       | 1.11e-05       | -191.43     | 0.996        | 0.997             | 0.615         | 0.631         |
| XRP    | 0.004816     | 1.64e-07      | 4.13e-08           | 2.30e-05       | 2.25e-05       | -122.68     | 0.996        | 0.999             | 0.408         | 0.423         |
| ETH    | 0.004662     | 6.00e-08      | 4.67e-08           | 7.10e-06       | 7.11e-06       | -281.46     | 0.996        | 0.997             | 0.570         | 0.569         |

### Key Findings:
- **Bayesian filtering methods (Kalman Filter & Particle Filter)** achieve R² > 0.99, significantly outperforming traditional approaches
- **Particle Filter** demonstrates superior performance with lowest MSE across most cryptocurrencies  
- **GARCH models** provide moderate performance with R² between 0.33-0.63
- **SDE-based data assimilation** captures volatility dynamics more effectively than classical econometric models

## Project Structure and File Descriptions

### Core Pipeline Scripts
1. **main.py** - Original experimental framework for SDE simulations and data assimilation

2. **main2.py** - **Pipeline 2**: Rolling volatility estimation with comprehensive visualization
   - Mean-reverting SDE model + Bayesian filtering
   - Rolling standard deviation as volatility proxy
   - Detailed plots comparing DA estimates vs GARCH models

3. **main2_heston.py** - **Pipeline 1**: Instantaneous volatility visualization  
   - Heston-lite SDE model + Bayesian filtering
   - Absolute log returns as instantaneous volatility proxy
   - Real-time volatility dynamics analysis

4. **main3.py** - Automated batch processing for Pipeline 2 performance evaluation
   - MSE/R² calculations across all cryptocurrencies
   - CSV output for statistical analysis

5. **main4.py** - **Pipeline 1**: Instantaneous volatility inference engine
   - Heston-based data assimilation with comprehensive performance metrics
   - Generates `instantaneous_volatility_results.csv` (72MB, 264K+ observations)

### Core Data Assimilation Modules  
6. **DA_utility.py** - **Pipeline 2** implementation
   - Mean-reverting model with generic predictor framework
   - `kalman_DA()`, `particle_filter_DA()`: Bayesian filtering methods
   - Rolling volatility computation and combination strategies

7. **DA_utility_heston.py** - **Pipeline 1** implementation  
   - Heston-lite SDE dynamics: dσ = κ(θ - σ)dt + ξdW
   - `kalman_heston_DA()`, `particle_filter_heston_DA()`: Specialized Heston filtering
   - Direct price-to-volatility estimation with stochastic volatility models

### Supporting Modules
8. **simulator.py** - SDE simulation functions for model validation
9. **utility.py** - Data preprocessing and basic visualization utilities

### Data and Configuration
10. **Cryptocurrency Data**: Hourly spot prices (6+ years, 52K+ observations per symbol)
    - `BTC_spot_full.csv`, `ETH_spot_full.csv`, `BNB_spot_full.csv`, `XRP_spot_full.csv`, `TRX_spot_full.csv`

11. **Results Files**:
    - `mse_results.csv` - Pipeline 2 performance metrics  
    - `instantaneous_volatility_results.csv` - Pipeline 1 comprehensive results
    - `instantaneous_volatility_performance.csv` - Pipeline 1 performance analysis

