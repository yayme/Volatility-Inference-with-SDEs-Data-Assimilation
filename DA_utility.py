import numpy as np

def compute_rolling_volatility(returns, window=100):
    return returns.rolling(window=window, min_periods=1).std()

def naive_DA(sigma_obs, kappa=2.0, theta=None, xi=0.3, dt=1/1440, alpha=0.5):
    if theta is None:
        theta = sigma_obs.mean()
    sigma_model = [theta]
    sigma_est = [theta]
    for t in range(1, len(sigma_obs)):
        dW = np.random.normal(0, np.sqrt(dt))
        sigma_pred = sigma_model[-1] + kappa*(theta - sigma_model[-1])*dt + xi*dW
        sigma_pred = max(sigma_pred, 1e-6)
        sigma_model.append(sigma_pred)
        sigma_new = alpha*sigma_pred + (1-alpha)*sigma_obs.iloc[t]
        sigma_est.append(sigma_new)
    return sigma_model, sigma_est

def heston_predictor(sigma_prev, theta, kappa=2.0, xi=0.3, dt=1/1440):
    dW = np.random.normal(0, np.sqrt(dt))
    sigma_pred = sigma_prev + kappa*(theta - sigma_prev)*dt + xi*dW
    return max(sigma_pred, 1e-6)

def generic_DA(sigma_obs, predictor, combiner, theta=None):
    if theta is None:
        theta = sigma_obs.mean()
    sigma_model = [theta]
    sigma_est = [theta]
    for t in range(1, len(sigma_obs)):
        sigma_pred = predictor(sigma_model[-1], theta)
        sigma_model.append(sigma_pred)
        sigma_new = combiner(sigma_pred, sigma_obs.iloc[t])
        sigma_est.append(sigma_new)
    return sigma_model, sigma_est

def naive_combiner(sigma_pred, sigma_obs_t, alpha=0.5):
    return alpha*sigma_pred + (1-alpha)*sigma_obs_t

def kalman_DA(sigma_obs, theta=None, kappa=2.0, xi=0.3, dt=1/1440, R=1e-4, Q=1e-5):
    if theta is None:
        theta = sigma_obs.mean()
    sigma_est = [theta]
    P = 1.0
    for t in range(1, len(sigma_obs)):
        sigma_pred = sigma_est[-1] + kappa*(theta - sigma_est[-1])*dt
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        sigma_new = sigma_pred + K * (sigma_obs.iloc[t] - sigma_pred)
        P = (1 - K) * P_pred
        sigma_est.append(sigma_new)
    sigma_model = [theta] + [theta + kappa*(theta - theta)*dt for _ in range(1, len(sigma_obs))]
    return sigma_model, sigma_est

def particle_filter_DA(sigma_obs, theta=None, kappa=2.0, xi=0.3, dt=1/1440, N_particles=100, alpha=0.5):
    if theta is None:
        theta = sigma_obs.mean()
    particles = np.full(N_particles, theta)
    sigma_est = [theta]
    for t in range(1, len(sigma_obs)):
        dW = np.random.normal(0, np.sqrt(dt), size=N_particles)
        particles = particles + kappa*(theta - particles)*dt + xi*dW
        particles = np.clip(particles, 1e-6, None)
        obs = sigma_obs.iloc[t]
        weights = np.exp(-0.5*((obs - particles)/0.001)**2)
        weights /= np.sum(weights)
        indices = np.random.choice(np.arange(N_particles), size=N_particles, p=weights)
        particles = particles[indices]
        sigma_new = np.mean(particles)
        sigma_est.append(sigma_new)
    sigma_model = [theta] + [theta + kappa*(theta - theta)*dt for _ in range(1, len(sigma_obs))]
    return sigma_model, sigma_est
