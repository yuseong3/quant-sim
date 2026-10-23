"""Adverse selection: the Glosten & Milgrom (1985) market maker.

Why do bid-ask spreads exist? Avellaneda-Stoikov (market_making.py) explains
one reason: inventory risk. Glosten-Milgrom explains the other: some traders
know more than the market maker. Every buy is a little evidence that the asset
is worth more, every sell that it is worth less, so a market maker who quotes
the same price to everyone loses to informed traders. The spread is what the
uninformed pay to cover those losses.

Model
-----
- The asset is worth V_high or V_low; the market maker's prior is P(V_high) = pi.
- Each period one trader arrives. With probability mu they are informed (buy if
  V = V_high, sell if V = V_low); otherwise they buy or sell with probability 1/2.
- A competitive, risk-neutral market maker sets
      ask = E[V | next trade is a buy],   bid = E[V | next trade is a sell]
  so it breaks even on average, and updates pi by Bayes' rule after each trade.

Closed forms used in the tests
------------------------------
- With pi = 1/2 the first spread is exactly  mu * (V_high - V_low).
- A naive maker that never learns and quotes prior_mean +/- s/2 earns
      s/2 - mu * (V_high - V_low)/2   per trade,
  so it loses money once mu > s / (V_high - V_low).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GMParams:
    v_low: float = 90.0
    v_high: float = 110.0
    prior_high: float = 0.5
    mu: float = 0.3          # share of informed traders
    n_trades: int = 300


def quotes(pi, p: GMParams):
    """Bayesian bid and ask given belief pi = P(V_high).

    Returns (bid, ask, posterior_if_sell, posterior_if_buy).
    """
    pi = np.asarray(pi, dtype=float)
    buy_if_high = p.mu + (1.0 - p.mu) / 2.0
    buy_if_low = (1.0 - p.mu) / 2.0
    post_buy = pi * buy_if_high / (pi * buy_if_high + (1.0 - pi) * buy_if_low)
    post_sell = pi * (1.0 - buy_if_high) / (pi * (1.0 - buy_if_high) + (1.0 - pi) * (1.0 - buy_if_low))
    width = p.v_high - p.v_low
    return p.v_low + width * post_sell, p.v_low + width * post_buy, post_sell, post_buy


@dataclass
class GMResult:
    high: np.ndarray            # True if V = V_high, per path
    belief: np.ndarray          # (n_paths, n_trades + 1) market maker's P(V_high)
    spread: np.ndarray          # (n_paths, n_trades) quoted spread
    mm_pnl: np.ndarray          # per path, Bayesian market maker
    informed_pnl: np.ndarray    # per path
    noise_pnl: np.ndarray       # per path
    naive_pnl: np.ndarray       # per path, fixed-quote maker facing the same order flow


def simulate(p: GMParams, n_paths: int = 5000, naive_spread: float = 4.0,
             seed: int | None = 0) -> GMResult:
    """Simulate many independent markets, vectorised over paths.

    The naive maker sees exactly the same traders (common random numbers) but
    always quotes prior_mean +/- naive_spread / 2 and never updates.
    """
    rng = np.random.default_rng(seed)
    high = rng.random(n_paths) < p.prior_high
    value = np.where(high, p.v_high, p.v_low)
    pi = np.full(n_paths, p.prior_high)
    belief = np.empty((n_paths, p.n_trades + 1)); belief[:, 0] = pi
    spread = np.empty((n_paths, p.n_trades))
    mm = np.zeros(n_paths); inf = np.zeros(n_paths); noise = np.zeros(n_paths); naive = np.zeros(n_paths)
    mean0 = p.v_low + (p.v_high - p.v_low) * p.prior_high
    n_bid, n_ask = mean0 - naive_spread / 2.0, mean0 + naive_spread / 2.0

    for t in range(p.n_trades):
        bid, ask, post_sell, post_buy = quotes(pi, p)
        spread[:, t] = ask - bid
        informed = rng.random(n_paths) < p.mu
        noise_buy = rng.random(n_paths) < 0.5
        buy = np.where(informed, high, noise_buy)

        trader_gain = np.where(buy, value - ask, bid - value)   # trader buys at ask / sells at bid
        mm -= trader_gain
        inf += np.where(informed, trader_gain, 0.0)
        noise += np.where(informed, 0.0, trader_gain)
        naive -= np.where(buy, value - n_ask, n_bid - value)

        pi = np.where(buy, post_buy, post_sell)
        belief[:, t + 1] = pi

    return GMResult(high, belief, spread, mm, inf, noise, naive)


def trades_to_discover(res: GMResult, p: GMParams, tol: float = 0.05) -> np.ndarray:
    """Number of trades until the market maker's expected value first comes within
    tol * (V_high - V_low) of the true value. Paths that never get there return n_trades."""
    exp_v = p.v_low + (p.v_high - p.v_low) * res.belief
    true_v = np.where(res.high, p.v_high, p.v_low)[:, None]
    close = np.abs(exp_v - true_v) <= tol * (p.v_high - p.v_low) + 1e-12
    return np.where(close.any(axis=1), close.argmax(axis=1), p.n_trades)


def expected_discovery_time(mu: float, tol: float = 0.05, prior_high: float = 0.5) -> float:
    """Exact expected number of trades to reach the tolerance, starting from prior 1/2.

    Each buy multiplies the odds P(V_high)/P(V_low) by (1+mu)/(1-mu) and each sell
    divides by it, so the log-odds is a simple random walk with step
    L = ln((1+mu)/(1-mu)). Given the true value, a step toward the truth happens with
    probability (1+mu)/2, so the walk drifts by mu steps per trade. Reaching
    belief 1 - tol needs k = ceil(ln((1-tol)/tol) / L) net steps, and a simple random
    walk with drift mu hits level k after k / mu trades on average.
    """
    if prior_high != 0.5:
        raise NotImplementedError("closed form written for a symmetric prior")
    step = np.log((1.0 + mu) / (1.0 - mu))
    k = np.ceil(np.log((1.0 - tol) / tol) / step - 1e-12)
    return float(k / mu)
