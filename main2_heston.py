import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DA_utility_heston import (
    compute_returns,
    compute_rolling_volatility,
    naive_heston_DA,
    kalman_heston_DA,
    particle_filter_heston_DA,
    heston_predictor,
    generic_heston_DA,
    naive_combiner
)
from arch import arch_model
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('BTC_spot_full.csv')
prices = df['bam_close']
log_returns = compute_returns(prices)
df['log_return'] = log_returns
df['abs_log_return'] = np.abs(log_returns)

# Use absolute log returns as observed instantaneous volatility
df['sigma_obs'] = df['abs_log_return']
# Also compute rolling volatility for comparison
df['rolling_vol_100'] = compute_rolling_volatility(log_returns, window=100)
df=df[1000:1100]

# Update prices to match the subsetted dataframe
prices = df['bam_close']

# Subset data for faster plotting (optional)
# df = df[:1000]
# prices = prices[:1000]

print("Running Heston-based Data Assimilation methods...")

# --- Heston-based Data Assimilation (DA) methods ---
# 1. Naive Heston DA
print("  Running Naive Heston DA...")
sigma_model_naive, sigma_est_naive = naive_heston_DA(prices, alpha=0.3)
df['heston_naive_model'] = sigma_model_naive
df['heston_naive_est'] = sigma_est_naive

# 2. Generic Heston DA 
print("  Running Generic Heston DA...")
sigma_model_generic, sigma_est_generic = generic_heston_DA(
    prices, 
    predictor=lambda s, t: heston_predictor(s, t), 
    combiner=lambda pred, obs: naive_combiner(pred, obs, alpha=0.2)
)
df['heston_generic_model'] = sigma_model_generic
df['heston_generic_est'] = sigma_est_generic

# 3. Kalman Heston DA
print("  Running Kalman Heston DA...")
sigma_model_kalman, sigma_est_kalman = kalman_heston_DA(
    prices, R=1e-4, Q=0.01
)
df['heston_kalman_model'] = sigma_model_kalman
df['heston_kalman_est'] = sigma_est_kalman

# 4. Particle Filter Heston DA
print("  Running Particle Filter Heston DA...")
sigma_model_pf, sigma_est_pf = particle_filter_heston_DA(
    prices, N_particles=150, R=0.001
)
df['heston_pf_model'] = sigma_model_pf
df['heston_pf_est'] = sigma_est_pf

# GARCH models for comparison
print("  Running GARCH models...")
# Use log_returns from the subsetted dataframe
log_returns_subset = df['log_return']
returns_pct = log_returns_subset * 100
garch = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='normal')
res = garch.fit(disp='off')
sigma_garch = res.conditional_volatility / 100
df['sigma_garch'] = sigma_garch.values

# GARCH(2,2)
garch22 = arch_model(returns_pct, vol='Garch', p=2, q=2, dist='normal')
res22 = garch22.fit(disp='off')
sigma_garch22 = res22.conditional_volatility / 100
df['sigma_garch22'] = sigma_garch22.values

print("Creating plots...")

# --- PLOTS ---

# 1) Observed Instantaneous Volatility vs Heston DA estimates
plt.figure(figsize=(14, 6))
plt.plot(df['sigma_obs'], label='Instantaneous Volatility (|log returns|)', color='red', alpha=0.8)
plt.plot(df['heston_naive_est'], label='Naive Heston DA', color='blue', alpha=0.5)
plt.plot(df['heston_generic_est'], label='Generic Heston DA', color='cyan', alpha=0.5)
plt.plot(df['heston_kalman_est'], label='Kalman Heston DA', color='green', alpha=0.5)
plt.plot(df['heston_pf_est'], label='Particle Filter Heston DA', color='purple', alpha=0.5)
plt.plot(df['rolling_vol_100'], label='Rolling Vol (100)', color='orange', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.title('BTC Instantaneous Volatility: Observed vs Heston DA Estimates')
plt.legend()
plt.tight_layout()
plt.show()

# 2) Individual comparison: Kalman Heston
plt.figure(figsize=(10,4))
plt.plot(df['sigma_obs'], label='Instantaneous Volatility', color='red', alpha=0.8)
plt.plot(df['heston_kalman_est'], label='Kalman Heston Estimate', color='green', alpha=0.5)
plt.title('Instantaneous Volatility vs Kalman Heston Estimate')
plt.legend()
plt.tight_layout()
plt.show()

# 3) Individual comparison: Particle Filter Heston
plt.figure(figsize=(10,4))
plt.plot(df['sigma_obs'], label='Instantaneous Volatility', color='red', alpha=0.8)
plt.plot(df['heston_pf_est'], label='Particle Filter Heston Estimate', color='purple', alpha=0.5)
plt.title('Instantaneous Volatility vs Particle Filter Heston Estimate')
plt.legend()
plt.tight_layout()
plt.show()

# 4) Closeup: All Heston DA estimates (zoomed in)
plt.figure(figsize=(12,5))
start, end = 100, 600  # adjust as needed for your data
plt.plot(df['sigma_obs'].iloc[start:end], label='Instantaneous Volatility', color='red', alpha=0.8)
plt.plot(df['heston_kalman_est'].iloc[start:end], label='Kalman Heston', color='green', alpha=0.5)
plt.plot(df['heston_pf_est'].iloc[start:end], label='Particle Filter Heston', color='purple', alpha=0.5)
plt.plot(df['heston_generic_est'].iloc[start:end], label='Generic Heston', color='cyan', alpha=0.5)
plt.title('Heston DA Volatility Estimates (Closeup View)')
plt.legend()
plt.tight_layout()
plt.show()

# 5) Kalman Filter: Model Prior vs DA Estimate
plt.figure(figsize=(14, 6))
plt.plot(df['heston_kalman_model'], label='Kalman Heston Model Prior', linestyle='--', color='lightblue', alpha=0.5)
plt.plot(df['heston_kalman_est'], label='Kalman Heston DA', color='green', alpha=0.5)
plt.plot(df['sigma_obs'], label='Instantaneous Volatility', color='red', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.title('Kalman Heston Filter: Model Prior vs DA Estimate')
plt.legend()
plt.tight_layout()
plt.show()

# 6) Particle Filter: Model Prior vs DA Estimate
plt.figure(figsize=(14, 6))
plt.plot(df['heston_pf_model'], label='Particle Filter Heston Model Prior', linestyle='--', color='plum', alpha=0.5)
plt.plot(df['heston_pf_est'], label='Particle Filter Heston DA', color='purple', alpha=0.5)
plt.plot(df['sigma_obs'], label='Instantaneous Volatility', color='red', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.title('Particle Filter Heston: Model Prior vs DA Estimate')
plt.legend()
plt.tight_layout()
plt.show()

# 7) Heston DA Estimates vs GARCH vs Rolling Volatility
plt.figure(figsize=(14, 6))
plt.plot(df['heston_kalman_est'], label='Kalman Heston DA', color='green', alpha=0.5)
plt.plot(df['heston_pf_est'], label='Particle Filter Heston DA', color='purple', alpha=0.5)
plt.plot(df['sigma_garch'], label='GARCH(1,1)', color='orange', alpha=0.5)
plt.plot(df['sigma_garch22'], label='GARCH(2,2)', color='brown', alpha=0.5)
plt.plot(df['rolling_vol_100'], label='Rolling Vol (100)', color='gray', alpha=0.5)
plt.plot(df['sigma_obs'], label='Instantaneous Volatility', color='red', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.title('Heston DA Estimates vs GARCH vs Rolling Volatility')
plt.legend()
plt.tight_layout()
plt.show()

# 8) Model comparison: All Heston model priors
plt.figure(figsize=(14, 6))
plt.plot(df['heston_naive_model'], label='Naive Heston Model', linestyle='--', color='lightcyan', alpha=0.6)
plt.plot(df['heston_generic_model'], label='Generic Heston Model', linestyle='--', color='lightblue', alpha=0.6)
plt.plot(df['heston_kalman_model'], label='Kalman Heston Model', linestyle='--', color='lightgreen', alpha=0.6)
plt.plot(df['heston_pf_model'], label='Particle Filter Heston Model', linestyle='--', color='plum', alpha=0.6)
plt.plot(df['sigma_obs'], label='Instantaneous Volatility', color='red', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.title('Comparison of All Heston Model Priors')
plt.legend()
plt.tight_layout()
plt.show()

# --- MSE CALCULATION ---
print("\nCalculating MSE performance metrics...")

# Use valid data points for MSE calculation
m1 = (~np.isnan(df['sigma_obs'])) & (~np.isnan(df['heston_naive_est']))
mse_naive = mean_squared_error(df['sigma_obs'][m1], df['heston_naive_est'][m1])

m2 = (~np.isnan(df['sigma_obs'])) & (~np.isnan(df['heston_generic_est']))
mse_generic = mean_squared_error(df['sigma_obs'][m2], df['heston_generic_est'][m2])

m3 = (~np.isnan(df['sigma_obs'])) & (~np.isnan(df['heston_kalman_est']))
mse_kalman = mean_squared_error(df['sigma_obs'][m3], df['heston_kalman_est'][m3])

m4 = (~np.isnan(df['sigma_obs'])) & (~np.isnan(df['heston_pf_est']))
mse_pf = mean_squared_error(df['sigma_obs'][m4], df['heston_pf_est'][m4])

m5 = (~np.isnan(df['sigma_obs'])) & (~np.isnan(df['sigma_garch']))
mse_garch = mean_squared_error(df['sigma_obs'][m5], df['sigma_garch'][m5])

m6 = (~np.isnan(df['sigma_obs'])) & (~np.isnan(df['sigma_garch22']))
mse_garch22 = mean_squared_error(df['sigma_obs'][m6], df['sigma_garch22'][m6])

m7 = (~np.isnan(df['sigma_obs'])) & (~np.isnan(df['rolling_vol_100']))
mse_rolling = mean_squared_error(df['sigma_obs'][m7], df['rolling_vol_100'][m7])

print("MSE vs Instantaneous Volatility (|log returns|):")
print(f"Naive Heston DA:    {mse_naive:.10f}")
print(f"Generic Heston DA:  {mse_generic:.10f}")
print(f"Kalman Heston DA:   {mse_kalman:.10f}")
print(f"Particle Filter:    {mse_pf:.10f}")
print(f"GARCH(1,1):         {mse_garch:.10f}")
print(f"GARCH(2,2):         {mse_garch22:.10f}")
print(f"Rolling Vol (100):  {mse_rolling:.10f}")

print("\nPlots generated successfully!")
print("Note: This analysis uses absolute log returns as instantaneous volatility ground truth.")
print("Heston-based methods directly model volatility evolution using SDE dynamics.")
