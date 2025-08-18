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
        sigma_est.append(alpha*sigma_pred + (1-alpha)*sigma_obs.iloc[t])
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
        sigma_est.append(combiner(sigma_pred, sigma_obs.iloc[t]))
    return sigma_model, sigma_est

def naive_combiner(sigma_pred, sigma_obs_t, alpha=0.5):
    return max(alpha*sigma_pred + (1-alpha)*sigma_obs_t, 1e-6)


def kalman_DA(sigma_obs, theta=None, kappa=2.0, xi=0.3, dt=1/1440, R=1e-4, Q=None):
    """Kalman Filter based Data Assimilation for Heston-lite vol model."""
    if theta is None:
        theta = sigma_obs.mean()
    if Q is None:
        Q = xi**2 * dt  # natural choice: process noise ~ variance of diffusion

    sigma_est = [theta]
    sigma_model = [theta]
    P = 1.0  # initial uncertainty

    for t in range(1, len(sigma_obs)):
        # ---- Model prediction (OU drift + noise) ----
        dW = np.random.normal(0, np.sqrt(dt))
        sigma_pred_model = sigma_model[-1] + kappa*(theta - sigma_model[-1])*dt + xi*dW
        sigma_pred_model = max(sigma_pred_model, 1e-6)
        sigma_model.append(sigma_pred_model)

        # ---- Kalman prediction (mean reversion only, uncertainty grows) ----
        sigma_pred = sigma_est[-1] + kappa*(theta - sigma_est[-1])*dt
        P_pred = P + Q

        # ---- Update with observation ----
        K = P_pred / (P_pred + R)
        sigma_new = sigma_pred + K * (sigma_obs.iloc[t] - sigma_pred)
        P = (1 - K) * P_pred

        sigma_est.append(max(sigma_new, 1e-6))

    return sigma_model, sigma_est


def particle_filter_DA(
    sigma_obs, theta=None, kappa=2.0, xi=0.3, dt=1/1440,
    N_particles=100, R=0.001
):
    """Particle Filter Data Assimilation for Heston-lite vol model."""
    if theta is None:
        theta = sigma_obs.mean()

    # initialize particles
    particles = np.full(N_particles, theta)
    sigma_est = [theta]

    # generate a model-only reference path
    sigma_model = [theta]

    for t in range(1, len(sigma_obs)):
        # ---- Propagation step (Heston-lite dynamics) ----
        dW = np.random.normal(0, np.sqrt(dt), size=N_particles)
        particles = particles + kappa*(theta - particles)*dt + xi*dW
        particles = np.clip(particles, 1e-6, None)

        # propagate a "model-only" trajectory (one sample)
        dW_model = np.random.normal(0, np.sqrt(dt))
        sigma_pred_model = sigma_model[-1] + kappa*(theta - sigma_model[-1])*dt + xi*dW_model
        sigma_model.append(max(sigma_pred_model, 1e-6))

        # ---- Weighting step ----
        obs = sigma_obs.iloc[t]
        weights = np.exp(-0.5 * ((obs - particles) / R) ** 2)
        weights /= np.sum(weights) + 1e-12  # avoid NaN
        weights = weights / np.sum(weights)  # ensure weights sum to exactly 1

        # ---- Resampling step ----
        indices = np.random.choice(np.arange(N_particles), size=N_particles, p=weights)
        particles = particles[indices]

        # ---- Estimate posterior mean ----
        sigma_new = np.mean(particles)
        sigma_est.append(sigma_new)

    return sigma_model, sigma_est

