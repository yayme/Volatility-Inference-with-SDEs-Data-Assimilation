import numpy as np
import pandas as pd

def compute_returns(S):
    """Compute log returns from close prices."""
    return np.log(S).diff().fillna(0)

def compute_rolling_volatility(returns, window=100):
    """Classic rolling standard deviation (for benchmarking)."""
    return returns.rolling(window=window, min_periods=1).std()

# --- Naive Heston DA ---
def naive_heston_DA(S, kappa=2.0, theta=None, xi=0.3, dt=1/1440, alpha=0.5):
    returns = compute_returns(S)
    if theta is None:
        theta = returns.std()
    sigma_model = [theta]
    sigma_est = [theta]
    for t in range(1, len(returns)):
        dW = np.random.normal(0, np.sqrt(dt))
        sigma_pred = sigma_model[-1] + kappa*(theta - sigma_model[-1])*dt + xi*dW
        sigma_pred = max(sigma_pred, 1e-6)
        sigma_model.append(sigma_pred)
        sigma_est.append(alpha*sigma_pred + (1-alpha)*abs(returns.iloc[t]))
    return sigma_model, sigma_est

# --- Heston predictor ---
def heston_predictor(sigma_prev, theta, kappa=2.0, xi=0.3, dt=1/1440):
    dW = np.random.normal(0, np.sqrt(dt))
    sigma_pred = sigma_prev + kappa*(theta - sigma_prev)*dt + xi*dW
    return max(sigma_pred, 1e-6)

# --- Generic Heston DA ---
def generic_heston_DA(S, predictor, combiner, theta=None):
    returns = compute_returns(S)
    if theta is None:
        theta = returns.std()
    sigma_model = [theta]
    sigma_est = [theta]
    for t in range(1, len(returns)):
        sigma_pred = predictor(sigma_model[-1], theta)
        sigma_model.append(sigma_pred)
        sigma_est.append(combiner(sigma_pred, abs(returns.iloc[t])))
    return sigma_model, sigma_est

def naive_combiner(sigma_pred, sigma_obs_t, alpha=0.5):
    return max(alpha*sigma_pred + (1-alpha)*sigma_obs_t, 1e-6)

# --- Kalman Heston DA ---
def kalman_heston_DA(S, theta=None, kappa=2.0, xi=0.3, dt=1/1440, R=1e-4, Q=None):
    returns = compute_returns(S)
    if theta is None:
        theta = returns.std()
    if Q is None:
        Q = xi**2 * dt

    sigma_est = [theta]
    sigma_model = [theta]
    P = 1.0

    for t in range(1, len(returns)):
        # Model prediction
        dW = np.random.normal(0, np.sqrt(dt))
        sigma_pred_model = sigma_model[-1] + kappa*(theta - sigma_model[-1])*dt + xi*dW
        sigma_pred_model = max(sigma_pred_model, 1e-6)
        sigma_model.append(sigma_pred_model)

        # Kalman prediction
        sigma_pred = sigma_est[-1] + kappa*(theta - sigma_est[-1])*dt
        P_pred = P + Q

        # Observation update
        obs = abs(returns.iloc[t])
        K = P_pred / (P_pred + R)
        sigma_new = sigma_pred + K * (obs - sigma_pred)
        P = (1 - K) * P_pred

        sigma_est.append(max(sigma_new, 1e-6))

    return sigma_model, sigma_est

# --- Particle Filter Heston DA ---
def particle_filter_heston_DA(S, theta=None, kappa=2.0, xi=0.3, dt=1/1440,
                              N_particles=100, R=0.001):
    returns = compute_returns(S)
    if theta is None:
        theta = returns.std()

    particles = np.full(N_particles, theta)
    sigma_est = [theta]
    sigma_model = [theta]

    for t in range(1, len(returns)):
        # Propagation
        dW = np.random.normal(0, np.sqrt(dt), size=N_particles)
        particles = particles + kappa*(theta - particles)*dt + xi*dW
        particles = np.clip(particles, 1e-6, None)

        # Model-only trajectory
        dW_model = np.random.normal(0, np.sqrt(dt))
        sigma_pred_model = sigma_model[-1] + kappa*(theta - sigma_model[-1])*dt + xi*dW_model
        sigma_model.append(max(sigma_pred_model, 1e-6))

        # Weighting
        obs = abs(returns.iloc[t])
        weights = np.exp(-0.5*((obs - particles)/R)**2)
        weights = weights + 1e-12  # Add small constant to prevent zeros
        weights = weights / np.sum(weights)
        
        # Check for NaN weights and fix if necessary
        if np.any(np.isnan(weights)) or np.sum(weights) == 0:
            weights = np.ones(N_particles) / N_particles  # Uniform weights as fallback
        else:
            weights = weights / np.sum(weights)  # Ensure normalization

        # Resampling
        indices = np.random.choice(np.arange(N_particles), size=N_particles, p=weights)
        particles = particles[indices]

        # Posterior mean
        sigma_new = np.mean(particles)
        sigma_est.append(sigma_new)

    return sigma_model, sigma_est
