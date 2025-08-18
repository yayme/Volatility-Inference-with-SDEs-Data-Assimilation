import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DA_utility import (
    compute_rolling_volatility,
    naive_DA,
    heston_predictor,
    generic_DA,
    naive_combiner,
    kalman_DA,
    particle_filter_DA
)
from arch import arch_model
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('XRP_spot_full.csv')
df['return'] = df['bam_close'].pct_change().fillna(0)
df['sigma_obs'] = compute_rolling_volatility(df['return'], window=100)
# df = df[:1000]
df=df[:1000]
# --- Data Assimilation (DA) methods ---
# Naive DA
sigma_model_naive, sigma_est_naive = generic_DA(
    df['sigma_obs'], predictor=heston_predictor, combiner=naive_combiner
)
df['sigma_model_naive'] = sigma_model_naive
df['sigma_est_naive'] = sigma_est_naive

# Kalman Filter DA
sigma_model_kf, sigma_est_kf = kalman_DA(
    df['sigma_obs'], kappa=2.0, xi=0.3, dt=1/1440, R=1e-4
)
df['sigma_model_kf'] = sigma_model_kf
df['sigma_est_kf'] = sigma_est_kf

# Particle Filter DA
sigma_model_pf, sigma_est_pf = particle_filter_DA(
    df['sigma_obs'], kappa=2.0, xi=0.3, dt=1/1440, N_particles=200, R=0.001
)
df['sigma_model_pf'] = sigma_model_pf
df['sigma_est_pf'] = sigma_est_pf

# GARCH(1,1)
returns = df['return'] * 100
garch = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
res = garch.fit(disp='off')
sigma_garch = res.conditional_volatility / 100
df['sigma_garch'] = sigma_garch.values

# GARCH(2,2)
garch22 = arch_model(returns, vol='Garch', p=2, q=2, dist='normal')
res22 = garch22.fit(disp='off')
sigma_garch22 = res22.conditional_volatility / 100
df['sigma_garch22'] = sigma_garch22.values

# --- PLOTS ---

# 1) Observed vs DA estimates
plt.figure(figsize=(14, 6))
plt.plot(df['sigma_obs'], label='Observed Volatility', color='red', alpha=0.8)
plt.plot(df['sigma_est_naive'], label='Naive DA', color='blue', alpha=0.5)
plt.plot(df['sigma_est_kf'], label='Kalman DA', color='green', alpha=0.5)
plt.plot(df['sigma_est_pf'], label='Particle Filter DA', color='purple', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.title('XRP Spot Volatility: Observed vs DA Estimates')
plt.legend()

plt.figure(figsize=(10,4))
plt.plot(df['sigma_obs'], label='Observed Volatility', color='red', alpha=0.8)
plt.plot(df['sigma_est_kf'], label='Kalman Estimate', color='green', alpha=0.5)
plt.title('Observed Volatility vs Kalman Estimate')
plt.legend()
plt.tight_layout()

plt.show()

# Closeup: All DA estimates (zoomed in)
plt.figure(figsize=(12,5))
start, end = 100, 600  # adjust as needed for your data
plt.plot(df['sigma_obs'].iloc[start:end], label='Observed Volatility', color='red', alpha=0.8)
plt.plot(df['sigma_est_kf'].iloc[start:end], label='Kalman Estimate', color='green', alpha=0.5)
plt.plot(df['sigma_est_pf'].iloc[start:end], label='Particle Filter', color='purple', alpha=0.5)
plt.title('Volatility Estimates (Closeup)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(df['sigma_obs'], label='Observed Volatility', color='red', alpha=0.8)
plt.plot(df['sigma_est_pf'], label='Particle Filter Estimate', color='purple', alpha=0.5)
plt.title('Observed Volatility vs Particle Filter Estimate')
plt.legend()
plt.tight_layout()
plt.show()

# 2) Kalman Filter: Model Prior vs DA Estimate
plt.figure(figsize=(14, 6))
plt.plot(df['sigma_model_kf'], label='Kalman Model Prior', linestyle='--', color='lightblue', alpha=0.5)
plt.plot(df['sigma_est_kf'], label='Kalman DA', color='green', alpha=0.5)
plt.plot(df['sigma_obs'], label='Observed Volatility', color='red', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.title('Kalman Filter: Model Prior vs DA Estimate')
plt.legend()
plt.show()

# 3) Particle Filter: Model Prior vs DA Estimate
plt.figure(figsize=(14, 6))
plt.plot(df['sigma_model_pf'], label='Particle Filter Model Prior', linestyle='--', color='plum', alpha=0.5)
plt.plot(df['sigma_est_pf'], label='Particle Filter DA', color='purple', alpha=0.5)
plt.plot(df['sigma_obs'], label='Observed Volatility', color='red', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.title('Particle Filter: Model Prior vs DA Estimate')
plt.legend()
plt.show()

# 4) DA Estimates vs GARCH
plt.figure(figsize=(14, 6))
plt.plot(df['sigma_est_kf'], label='Kalman DA', color='green', alpha=0.5)
plt.plot(df['sigma_est_pf'], label='Particle Filter DA', color='purple', alpha=0.5)
plt.plot(df['sigma_garch'], label='GARCH(1,1)', color='orange', alpha=0.5)
plt.plot(df['sigma_garch22'], label='GARCH(2,2)', color='brown', alpha=0.5)
plt.plot(df['sigma_obs'], label='Observed Volatility', color='red', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.title('DA Estimates vs GARCH')
plt.legend()
plt.show()



