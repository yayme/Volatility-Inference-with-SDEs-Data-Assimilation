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

df = pd.read_csv('BTC_spot_full.csv')
df['return'] = df['bam_close'].pct_change().fillna(0)
df['sigma_obs'] = compute_rolling_volatility(df['return'], window=100)
df = df[:1000]

sigma_model_naive, sigma_prior_naive, sigma_est_naive = generic_DA(
    df['sigma_obs'], predictor=heston_predictor, combiner=naive_combiner
)
df['sigma_model_naive'] = sigma_model_naive
df['sigma_est_naive'] = sigma_est_naive

sigma_model_kf, sigma_prior_kf, sigma_est_kf = kalman_DA(
    df['sigma_obs'], kappa=2.0, xi=0.3, dt=1/1440, R=1e-4
)
df['sigma_model_kf'] = sigma_model_kf
df['sigma_est_kf'] = sigma_est_kf

sigma_model_pf, sigma_prior_pf, sigma_est_pf = particle_filter_DA(
    df['sigma_obs'], kappa=2.0, xi=0.3, dt=1/1440, N_particles=200, R=0.001
)
df['sigma_model_pf'] = sigma_model_pf
df['sigma_est_pf'] = sigma_est_pf

returns = df['return'] * 100
garch = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
res = garch.fit(last_obs=100, disp='off')
forecasts = res.forecast(horizon=1, start=100)
sigma_garch = np.sqrt(forecasts.variance['h.1'].values) / 100
sigma_garch[:100] = res.conditional_volatility[:100] / 100
df['sigma_garch'] = sigma_garch

df[['bam_close', 'sigma_obs', 'sigma_est_naive', 'sigma_est_kf', 'sigma_est_pf']].plot(
    subplots=True, figsize=(12, 10), alpha=0.7, title="BTC Spot: DA Comparison"
)
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df['sigma_obs'], label='Observed Volatility', color='blue', alpha=0.5)
plt.plot(df['sigma_model_kf'], label='KF Model Prior', linestyle='--', color='green', alpha=0.3)
plt.plot(df['sigma_est_kf'], label='Kalman DA', color='green', alpha=0.7)
plt.plot(df['sigma_model_pf'], label='PF Model Prior', linestyle='--', color='red', alpha=0.3)
plt.plot(df['sigma_est_pf'], label='Particle Filter DA', color='red', alpha=0.7)
plt.plot(df['sigma_garch'], label='GARCH(1,1)', color='purple', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.title('BTC Spot Volatility Estimates Comparison')
plt.xlim(0, 1000)
# plt.ylim(-0.01, 0.03)
plt.legend()
plt.show()

mse_naive = mean_squared_error(df['sigma_obs'], sigma_prior_naive)
mse_kf = mean_squared_error(df['sigma_obs'], sigma_prior_kf)
mse_pf = mean_squared_error(df['sigma_obs'], sigma_prior_pf)
mse_garch = mean_squared_error(df['sigma_obs'], df['sigma_garch'])

print("MSE vs sigma_obs")
print(f"Naive DA:       {mse_naive:.6f}")
print(f"Kalman DA:      {mse_kf:.6f}")
print(f"Particle Filter:{mse_pf:.6f}")
print(f"GARCH(1,1):     {mse_garch:.6f}")
