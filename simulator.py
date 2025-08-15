import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def simulate_brownian_motion(T=1, dt=0.001):
    N = int(T/dt)
    t = np.linspace(0, T, N)
    dw = np.random.randn(N) * np.sqrt(dt)
    w = np.cumsum(dw)
    return t, w

def simulate_garch(n=1000, omega=0.001, alpha=0.05, beta=0.94, seed=42):
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    sigma = np.sqrt(sigma2[0])
    np.random.seed(seed)
    for t in range(1, n):
        z = np.random.normal()
        returns[t] = sigma * z
        sigma2[t] = omega + alpha * (returns[t] ** 2) + beta * sigma2[t - 1]
        sigma = np.sqrt(sigma2[t])
    return returns, sigma2

def simulate_heston_lite(T=1, dt=1/250, mu=0.05, kappa=2.0, theta=0.04, xi=0.3, S0=100, sigma0=0.2):
    N = int(T/dt)
    time = np.linspace(0, T, N)
    S = np.zeros(N)
    sigma = np.zeros(N)
    S[0] = S0
    sigma[0] = sigma0
    for t in range(1, N):
        dW_S = np.random.normal(0, np.sqrt(dt))
        dW_sigma = np.random.normal(0, np.sqrt(dt))
        sigma[t] = np.abs(sigma[t-1] + kappa * (theta - sigma[t-1]) * dt + xi * dW_sigma)
        S[t] = S[t-1] + mu * S[t-1] * dt + sigma[t] * S[t-1] * dW_S
    return time, S, sigma
def simulate_gbm(S0, mu, sigma, dt, steps):
    """Simulate a Geometric Brownian Motion path."""
    S = np.zeros(steps)
    S[0] = S0
    for t in range(1, steps):
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn())
    return S 