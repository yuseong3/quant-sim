"""Market-making simulation based on Avellaneda & Stoikov (2008),
"High-frequency trading in a limit order book", Quantitative Finance 8(3).

A market maker posts a bid and an ask around the mid price and earns the spread
when both sides fill. The risk is inventory: if only one side keeps filling, the
maker accumulates a position exposed to price moves.

Model
-----
- Mid price:            dS = sigma dW  (arithmetic Brownian motion)
- Order arrival rate:   lambda(delta) = A * exp(-k * delta), where delta is the
                        distance of a quote from the mid. Quoting further away
                        fills less often.
- Inventory strategy:   quotes are centred on a *reservation price*
                            r = S - q * gamma * sigma^2 * (T - t)
                        so a long maker (q > 0) shades both quotes down to sell
                        more and buy less. The optimal total spread is
                            gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k).
- Symmetric strategy:   same total spread, but always centred on the mid,
                        ignoring inventory. This is the benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MMParams:
    s0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    gamma: float = 0.1      # risk aversion
    k: float = 1.5          # order-book depth decay
    A: float = 140.0        # base arrival intensity

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))


@dataclass
class MMResult:
    pnl: np.ndarray            # final mark-to-market PnL per path
    inventory: np.ndarray      # final inventory per path
    spread: float              # average quoted total spread
    sample_path: dict          # one path for plotting

    def summary(self) -> dict:
        return {
            "mean_pnl": float(self.pnl.mean()),
            "std_pnl": float(self.pnl.std(ddof=1)),
            "sharpe_like": float(self.pnl.mean() / self.pnl.std(ddof=1)),
            "mean_abs_inventory": float(np.abs(self.inventory).mean()),
            "std_inventory": float(self.inventory.std(ddof=1)),
            "avg_spread": self.spread,
        }


def simulate(p: MMParams, strategy: str = "inventory", n_paths: int = 1000,
             seed: int | None = 0) -> MMResult:
    """Simulate n_paths independent trading days, vectorised over paths.

    strategy: "inventory" (Avellaneda-Stoikov) or "symmetric" (benchmark).
    Using the same seed for both strategies gives them identical price paths and
    identical random numbers for fills, so differences come from quoting alone
    (common random numbers).
    """
    if strategy not in ("inventory", "symmetric"):
        raise ValueError("strategy must be 'inventory' or 'symmetric'")
    rng = np.random.default_rng(seed)
    n = p.n_steps
    S = np.full(n_paths, p.s0)
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    spreads = []
    path = {"t": [], "mid": [], "bid": [], "ask": [], "q": []}

    const = (2.0 / p.gamma) * np.log(1.0 + p.gamma / p.k)
    for i in range(n):
        tau = p.T - i * p.dt
        total_spread = p.gamma * p.sigma**2 * tau + const
        half = total_spread / 2.0
        centre = S - q * p.gamma * p.sigma**2 * tau if strategy == "inventory" else S
        bid, ask = centre - half, centre + half

        # fill probabilities over one step: lambda * dt (small enough to stay < 1)
        d_bid = S - bid
        d_ask = ask - S
        prob_bid = np.clip(p.A * np.exp(-p.k * d_bid) * p.dt, 0.0, 1.0)
        prob_ask = np.clip(p.A * np.exp(-p.k * d_ask) * p.dt, 0.0, 1.0)
        u_bid = rng.random(n_paths)
        u_ask = rng.random(n_paths)
        buy = u_bid < prob_bid     # our bid is hit: we buy one unit
        sell = u_ask < prob_ask    # our ask is lifted: we sell one unit

        q += buy.astype(float) - sell.astype(float)
        cash += sell * ask - buy * bid
        spreads.append(total_spread)

        path["t"].append(i * p.dt)
        path["mid"].append(S[0])
        path["bid"].append(bid[0])
        path["ask"].append(ask[0])
        path["q"].append(q[0])

        S = S + p.sigma * np.sqrt(p.dt) * rng.standard_normal(n_paths)

    pnl = cash + q * S  # mark remaining inventory to the final mid
    return MMResult(pnl=pnl, inventory=q, spread=float(np.mean(spreads)),
                    sample_path={k: np.array(v) for k, v in path.items()})
