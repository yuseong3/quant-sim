"""European option pricing: Black-Scholes closed form vs Monte Carlo.

The point of this module is to show *why* Monte Carlo works and how fast it
converges: the standard error shrinks like 1/sqrt(N), and antithetic variates
reduce the constant in front of that rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def black_scholes(S: float, K: float, T: float, r: float, sigma: float,
                  kind: str = "call") -> float:
    """Closed-form Black-Scholes price of a European option."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
        return intrinsic
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    if kind == "put":
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    raise ValueError("kind must be 'call' or 'put'")


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def monte_carlo(S: float, K: float, T: float, r: float, sigma: float,
                n_paths: int, kind: str = "call", antithetic: bool = False,
                seed: int | None = None) -> MCResult:
    """Price a European option by simulating terminal prices under GBM.

    Under the risk-neutral measure, S_T = S * exp((r - sigma^2/2) T + sigma sqrt(T) Z).
    The price is the discounted average payoff. With antithetic=True each draw Z
    is paired with -Z; the pair average is used as one sample, which lowers the
    variance because the two payoffs are negatively correlated.
    """
    rng = np.random.default_rng(seed)
    payoff = (lambda s: np.maximum(s - K, 0.0)) if kind == "call" else (lambda s: np.maximum(K - s, 0.0))
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    disc = np.exp(-r * T)

    if antithetic:
        m = n_paths // 2
        z = rng.standard_normal(m)
        samples = 0.5 * (payoff(S * np.exp(drift + vol * z)) + payoff(S * np.exp(drift - vol * z)))
    else:
        m = n_paths
        z = rng.standard_normal(m)
        samples = payoff(S * np.exp(drift + vol * z))

    samples = disc * samples
    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(m)), n_paths)
