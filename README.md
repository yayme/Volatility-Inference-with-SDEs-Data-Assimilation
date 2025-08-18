
# Volatility-Inference-with-SDEs-Data-Assimilation

This project implements two distinct volatility estimation pipelines combining stochastic differential equations (SDEs) with Bayesian filtering techniques for cryptocurrency data analysis.

## Methodology Overview

### Pipeline 1: Instantaneous Volatility Inference
**Approach**: Heston-lite SDE model + Bayesian filtering with absolute log returns as instantaneous volatility proxy
- **Model**: dσ = κ(θ - σ)dt + ξdW (Heston-lite dynamics)
- **Ground Truth**: |log returns| as proxy for instantaneous volatility
- **Methods**: Naive DA, Generic DA, Kalman Filter, Particle Filter
- **Output**: Real-time volatility estimates capturing instantaneous market dynamics

### Pipeline 2: Rolling Volatility Estimation  
**Approach**: Mean-reverting SDE model + Bayesian filtering with rolling standard deviation as volatility proxy
- **Model**: Generic mean-reverting process (non-Heston)
- **Ground Truth**: Rolling window standard deviation (20/50/100 periods)
- **Methods**: Kalman Filter, Particle Filter with mean-reverting dynamics
- **Output**: Smoothed volatility estimates suitable for risk management

Both pipelines leverage advanced Bayesian filtering to combine prior SDE model predictions with observed data, achieving superior performance compared to traditional GARCH models.

## Performance Results

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

## Usage

### Pipeline 1: Instantaneous Volatility (Heston-lite + Bayesian Filtering)
```bash
python main4.py          # Batch processing all cryptocurrencies
python main2_heston.py   # Detailed visualization for single asset
```

### Pipeline 2: Rolling Volatility (Mean-reverting + Bayesian Filtering)  
```bash
python main3.py          # Batch performance evaluation
python main2.py          # Comprehensive visualization and analysis
```

## Technical Innovation

This framework advances financial volatility modeling by:
1. **Dual-proxy approach**: Instantaneous (|log returns|) vs smoothed (rolling std) volatility
2. **SDE-Bayesian integration**: Principled combination of stochastic model priors with observed data
3. **Real-time inference**: Particle filtering enables non-linear, non-Gaussian volatility dynamics
4. **Comparative validation**: Systematic evaluation against GARCH benchmarks across multiple assets

The results demonstrate that SDE-based Bayesian filtering substantially outperforms traditional econometric approaches for cryptocurrency volatility estimation.
     - `simulate_garch()`: GARCH(1,1) volatility simulation
     - `simulate_heston_lite()`: Heston-lite stochastic volatility model

6. **utility.py**
   - **General utility functions** for:
     - Data collection and processing
     - Basic plotting and visualization
     - Volatility calculation helpers

### Data and Configuration
7. **data_collection.py**
   - Script for collecting and preparing spot price data from raw sources

8. **universal_config.json**
   - Configuration file for symbols and other universal settings

9. **requirements.txt**
   - Python dependencies: pandas, numpy, matplotlib, scikit-learn, arch, etc.

### Data Files
10. **Cryptocurrency Data CSV Files:**
    - `BTC_spot_full.csv` - Bitcoin hourly spot price data
    - `ETH_spot_full.csv` - Ethereum hourly spot price data  
    - `BNB_spot_full.csv` - Binance Coin hourly spot price data
    - `XRP_spot_full.csv` - Ripple hourly spot price data
    - `TRX_spot_full.csv` - Tron hourly spot price data

11. **mse_results.csv**
    - Generated results file containing MSE and R² scores for all methods and symbols

### Development and Experimentation
12. **notebooks/**
    - Directory containing Jupyter notebooks for development and analysis:
    - `demo_experiment.ipynb` - Demonstration experiments and prototyping

13. **src/** (Alternative organized structure)
    - `config.py` - Configuration management
    - `pipeline.py` - Processing pipeline
    - `filters/` - Filter implementations (EnKF, Particle Filter)
    - `models/` - Model implementations (crypto fetcher, SDE simulator)
    - `utils/` - Utility functions (metrics, plotting)

## Usage

### Quick Analysis (Recommended)
```bash
python main3.py
```
Runs MSE/R² analysis on all cryptocurrencies and saves results to CSV.

### Detailed Analysis with Plots
```bash
python main2.py
```
Generates comprehensive plots and analysis for a single cryptocurrency (XRP by default).

### Individual Experiments
```bash
python main.py
```
Original experimental script for custom analysis.

## Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `pandas`, `numpy` - Data processing
- `matplotlib` - Visualization  
- `scikit-learn` - Machine learning metrics
- `arch` - GARCH modeling
- Custom modules for data assimilation methods
