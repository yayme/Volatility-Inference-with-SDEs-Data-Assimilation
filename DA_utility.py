import numpy as np

def compute_rolling_volatility(returns, window=100):
    return returns.rolling(window=window, min_periods=1).std()

def naive_DA(sigma_obs, kappa=2.0, theta=None, xi=0.3, dt=1/1440, alpha=0.5):
    if theta is None:
        theta = sigma_obs.iloc[:min(100, len(sigma_obs))].mean()
    sigma_model = [theta]
    sigma_prior = [theta]
    sigma_est = [theta]
    for t in range(1, len(sigma_obs)):
        dW = np.random.normal(0, np.sqrt(dt))
        sigma_pred = sigma_model[-1] + kappa*(theta - sigma_model[-1])*dt + xi*dW
        sigma_pred = max(sigma_pred, 1e-6)
        sigma_model.append(sigma_pred)
        sigma_prior.append(sigma_pred)
        sigma_est.append(alpha*sigma_pred + (1-alpha)*sigma_obs.iloc[t])
    return sigma_model, sigma_prior, sigma_est

def heston_predictor(sigma_prev, theta, kappa=2.0, xi=0.3, dt=1/1440):
    dW = np.random.normal(0, np.sqrt(dt))
    sigma_pred = sigma_prev + kappa*(theta - sigma_prev)*dt + xi*dW
    return max(sigma_pred, 1e-6)

def generic_DA(sigma_obs, predictor, combiner, theta=None):
    if theta is None:
        theta = sigma_obs.iloc[:min(100, len(sigma_obs))].mean()
    sigma_model = [theta]
    sigma_prior = [theta]
    sigma_est = [theta]
    for t in range(1, len(sigma_obs)):
        sigma_pred = predictor(sigma_model[-1], theta)
        sigma_model.append(sigma_pred)
        sigma_prior.append(sigma_pred)
        sigma_est.append(combiner(sigma_pred, sigma_obs.iloc[t]))
    return sigma_model, sigma_prior, sigma_est

def naive_combiner(sigma_pred, sigma_obs_t, alpha=0.5):
    return max(alpha*sigma_pred + (1-alpha)*sigma_obs_t, 1e-6)


def kalman_DA(sigma_obs, theta=None, kappa=2.0, xi=0.3, dt=1/1440, R=1e-4, Q=None):
    if theta is None:
        theta = sigma_obs.iloc[:min(100, len(sigma_obs))].mean()
    if Q is None:
        Q = xi**2 * dt

    sigma_prior = [theta]
    sigma_est = [theta]
    P = 1.0

    for t in range(1, len(sigma_obs)):
        sigma_pred = sigma_est[-1] + kappa*(theta - sigma_est[-1])*dt
        sigma_prior.append(max(sigma_pred, 1e-6))
        P_pred = P + Q

        K = P_pred / (P_pred + R)
        sigma_new = sigma_pred + K * (sigma_obs.iloc[t] - sigma_pred)
        P = (1 - K) * P_pred

        sigma_est.append(max(sigma_new, 1e-6))

    return sigma_prior, sigma_est


def particle_filter_DA(
    sigma_obs, theta=None, kappa=2.0, xi=0.3, dt=1/1440,
    N_particles=100, R=0.001
):
    if theta is None:
        theta = sigma_obs.iloc[:min(100, len(sigma_obs))].mean()

    particles = np.full(N_particles, theta)
    sigma_prior = [theta]
    sigma_est = [theta]

    for t in range(1, len(sigma_obs)):
        dW = np.random.normal(0, np.sqrt(dt), size=N_particles)
        particles = particles + kappa*(theta - particles)*dt + xi*dW
        particles = np.clip(particles, 1e-6, None)
        
        sigma_prior.append(np.mean(particles))

        obs = sigma_obs.iloc[t]
        log_w = -0.5 * ((obs - particles) / R) ** 2
        log_w = log_w - np.max(log_w)
        weights = np.exp(log_w)
        weight_sum = np.sum(weights)
        if (not np.isfinite(weight_sum)) or weight_sum <= 0:
            weights = np.full(N_particles, 1.0 / N_particles)
        else:
            weights = weights / weight_sum
            weights = weights / np.sum(weights)

        indices = np.random.choice(np.arange(N_particles), size=N_particles, p=weights)
        particles = particles[indices]

        sigma_est.append(np.mean(particles))

    return sigma_prior, sigma_est

