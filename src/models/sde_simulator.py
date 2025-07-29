import numpy as np

def simulate_gbm(S0, mu, sigma, dt, steps):
    """Simulate a Geometric Brownian Motion path."""
    S = np.zeros(steps)
    S[0] = S0
    for t in range(1, steps):
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn())
    return S 