"""Bet sizing with the Kelly criterion.

A repeated bet wins b units per unit staked with probability p and loses the
stake otherwise. Betting a fraction f of wealth each time, the long-run growth
rate of log wealth is

    g(f) = p * ln(1 + b f) + (1 - p) * ln(1 - f).

g is maximised at the Kelly fraction f* = p - (1 - p) / b. Betting more than
f* raises risk *and* lowers growth; at 2 f* the growth rate is roughly zero.

The practical twist studied here: p is never known, only estimated. Because g
falls off faster on the over-betting side, a fraction of Kelly (e.g. half)
is more robust to estimation error than full Kelly.
"""
from __future__ import annotations

import numpy as np


def kelly_fraction(p: float, b: float = 1.0) -> float:
    """Growth-optimal fraction of wealth to stake. Negative means: do not bet."""
    return p - (1.0 - p) / b


def growth_rate(f, p: float, b: float = 1.0):
    """Expected log growth per bet for fraction(s) f (0 <= f < 1)."""
    f = np.asarray(f, dtype=float)
    return p * np.log1p(b * f) + (1.0 - p) * np.log1p(-f)


def simulate_wealth(f: float, p: float, b: float = 1.0, n_bets: int = 1000,
                    n_paths: int = 10000, seed: int | None = 0) -> dict:
    """Simulate log-wealth paths starting from wealth 1.

    Returns final log wealth, the realised growth per bet, and the maximum
    drawdown of each path (largest fall from a running peak, as a fraction).
    """
    rng = np.random.default_rng(seed)
    wins = rng.random((n_paths, n_bets)) < p
    step = np.where(wins, np.log1p(b * f), np.log1p(-f))
    logw = np.cumsum(step, axis=1)
    peak = np.maximum.accumulate(np.maximum(logw, 0.0), axis=1)
    max_dd = 1.0 - np.exp((logw - peak).min(axis=1))
    return {"final_log_wealth": logw[:, -1],
            "growth_per_bet": logw[:, -1] / n_bets,
            "max_drawdown": max_dd,
            "paths": logw}


def estimation_error_growth(multiplier: float, p_true: float, b: float = 1.0,
                            n_obs: int = 200, n_trials: int = 20000,
                            seed: int | None = 0) -> float:
    """Expected true growth when p must be estimated from n_obs past bets.

    The trader estimates p_hat from n_obs Bernoulli observations, then stakes
    multiplier * f*(p_hat), clipped to [0, 0.99]. The returned value is the
    average *true* growth rate of that choice across many possible samples.
    """
    rng = np.random.default_rng(seed)
    p_hat = rng.binomial(n_obs, p_true, size=n_trials) / n_obs
    f = np.clip(multiplier * kelly_fraction(p_hat, b), 0.0, 0.99)
    return float(growth_rate(f, p_true, b).mean())
