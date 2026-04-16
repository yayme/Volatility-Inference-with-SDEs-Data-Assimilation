import pandas as pd
import numpy as np

def load_crypto_data(filepath, limit=1000, window=100):
    df = pd.read_csv(filepath)
    if limit is not None:
        df = df.iloc[:limit].copy()
        
    prices = df['bam_close']
    log_returns = np.log(prices / prices.shift(1)).fillna(0)
    
    # Target 1: Instantaneous Volatility
    target_inst = np.abs(log_returns)
    
    # Target 2: Rolling Volatility 
    target_roll = log_returns.rolling(window=window, min_periods=1).std()
    
    df['log_return'] = log_returns
    df['target_inst'] = target_inst
    df['target_roll'] = target_roll
    
    return df

def kalman_filter_sde(observations, sde='OU', kappa=2.0, theta=None, xi=0.3, dt=1/1440, R=1e-4):
    """
    Minimialist Kalman Filter for Volatility Tracking.
    sde: 'OU' (Ornstein-Uhlenbeck) or 'CIR' (Heston-style sqrt process)
    """
    if theta is None:
        theta = np.nanmean(observations[:100])
        
    priors, posteriors = [theta], [theta]
    P = 1.0 # Initial uncertainty
    
    for t in range(1, len(observations)):
        prev_est = posteriors[-1]
        
        # 1. Predict Step
        pred_mean = prev_est + kappa * (theta - prev_est) * dt
        pred_mean = max(pred_mean, 1e-6)
        
        process_var = (xi**2 * dt) if sde == 'OU' else (xi**2 * prev_est * dt)
        P_pred = P + process_var
        priors.append(pred_mean)
        
        # 2. Update Step
        obs = observations.iloc[t]
        if np.isnan(obs):
            posteriors.append(pred_mean)
            P = P_pred
            continue
            
        K = P_pred / (P_pred + R)
        new_est = pred_mean + K * (obs - pred_mean)
        P = (1 - K) * P_pred
        posteriors.append(max(new_est, 1e-6))
        
    return np.array(priors), np.array(posteriors)

def particle_filter_sde(observations, sde='OU', kappa=2.0, theta=None, xi=0.3, dt=1/1440, R=1e-4, N_particles=200):
    """
    Minimialist Particle Filter for Volatility Tracking.
    """
    if theta is None:
        theta = np.nanmean(observations[:100])
        
    particles = np.full(N_particles, theta)
    priors, posteriors = [theta], [theta]
    
    for t in range(1, len(observations)):
        # 1. Propagation
        dW = np.random.normal(0, np.sqrt(dt), size=N_particles)
        if sde == 'OU':
            particles = particles + kappa*(theta - particles)*dt + xi*dW
        else:
            particles = particles + kappa*(theta - particles)*dt + xi*np.sqrt(np.maximum(particles, 1e-6))*dW
            
        particles = np.maximum(particles, 1e-6)
        priors.append(np.mean(particles))
        
        # 2. Weighting & Resampling
        obs = observations.iloc[t]
        if np.isnan(obs):
            posteriors.append(np.mean(particles))
            continue
            
        weights = np.exp(-0.5 * ((obs - particles) / R)**2)
        weights_sum = np.sum(weights)
        
        if weights_sum == 0 or np.isnan(weights_sum):
            weights = np.ones(N_particles) / N_particles
        else:
            weights = weights / weights_sum
            
        indices = np.random.choice(np.arange(N_particles), size=N_particles, p=weights)
        particles = particles[indices]
        posteriors.append(np.mean(particles))
        
    return np.array(priors), np.array(posteriors)

def run_out_of_sample_garch(returns, burn_in=100):
    """
    Minimialist GARCH(1,1) strictly evaluated out-of-sample.
    """
    from arch import arch_model
    # Rescale returns to avoid optimizer warnings
    scale_factor = 100
    rescaled_returns = returns * scale_factor
    
    garch = arch_model(rescaled_returns, vol='Garch', p=1, q=1, dist='normal')
    # Fit strictly on burn-in phase to eliminate look-ahead
    res = garch.fit(last_obs=burn_in, disp='off')
    
    # Predict all future variances recursively based ONLY on past data
    forecasts = res.forecast(horizon=1, start=burn_in)
    
    n = len(returns)
    out_of_sample_priors = np.full(n, np.nan)
    
    # ALIGNMENT BUG FIX:
    # `forecasts.variance` at index `t` represents the forecast for `t+1` using data up to `t`.
    # To compare the 1-step forecast against observation `t+1`, we shift the forecast array forward by 1 index.
    garch_variances = forecasts.variance['h.1'].values[:-1]
    out_of_sample_priors[burn_in+1:] = np.sqrt(garch_variances) / scale_factor
    
    return out_of_sample_priors
