
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DA_utility import compute_rolling_volatility, naive_DA, heston_predictor, generic_DA, naive_combiner, kalman_DA, particle_filter_DA



df = pd.read_csv('BTC_spot_full.csv')
df['return'] = df['bam_close'].pct_change().fillna(0)
df['sigma_obs'] = compute_rolling_volatility(df['return'], window=100)
df = df[10000:]
sigma_model_naive, sigma_est_naive = generic_DA(df['sigma_obs'], predictor=heston_predictor, combiner=naive_combiner)
df['sigma_est_naive'] = sigma_est_naive
sigma_model_kf, sigma_est_kf = kalman_DA(df['sigma_obs'])
df['sigma_est_kf'] = sigma_est_kf
sigma_model_pf, sigma_est_pf = particle_filter_DA(df['sigma_obs'])
df['sigma_est_pf'] = sigma_est_pf
df[['bam_close', 'sigma_obs', 'sigma_est_naive', 'sigma_est_kf', 'sigma_est_pf']].plot(subplots=True, figsize=(12,10))
plt.show()
from arch import arch_model

# Fit GARCH(1,1) on the returns
returns = df['return']*100  # scale to percent for stability
garch = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
res = garch.fit(disp='off')

# In-sample conditional volatility
sigma_garch = res.conditional_volatility / 100  # scale back to original
df['sigma_garch'] = sigma_garch.values

plt.figure(figsize=(14,6))

plt.plot(df['sigma_obs'], label='Observed Volatility', color='blue', alpha=0.5)
# plt.plot(df['sigma_est_naive'], label='Naive DA', color='orange')
plt.plot(df['sigma_est_kf'], label='Kalman DA', color='green', alpha=0.5)
plt.plot(df['sigma_est_pf'], label='Particle Filter DA', color='red', alpha=0.5)
plt.plot(df['sigma_garch'], label='GARCH(1,1)', color='purple' ,alpha =0.5)
plt.xlabel('Time')
plt.ylabel('Value / Volatility')
plt.title('BTC Spot Volatility Estimates Comparison')
plt.xlim(10000,20000)
plt.ylim(-0.01,0.03)
plt.legend()
plt.show()