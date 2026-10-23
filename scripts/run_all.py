"""Run all four experiments, save figures to docs/img and numbers to docs/results.json.

Usage:  python scripts/run_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantsim import (MMParams, black_scholes, monte_carlo, simulate,  # noqa: E402
                      kelly_fraction, growth_rate, simulate_wealth, estimation_error_growth)
from quantsim import adverse_selection as gm  # noqa: E402

IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2563eb", "#ea580c", "#6b7280"


def option_experiment() -> dict:
    S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.03, 0.2
    exact = black_scholes(S, K, T, r, sigma)
    Ns = np.unique(np.logspace(2, 6, 17).astype(int))
    plain, anti = [], []
    for n in Ns:
        plain.append(monte_carlo(S, K, T, r, sigma, int(n), seed=1))
        anti.append(monte_carlo(S, K, T, r, sigma, int(n), antithetic=True, seed=1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for res, c, lab in [(plain, BLUE, "Plain MC"), (anti, ORANGE, "Antithetic MC")]:
        p = np.array([x.price for x in res]); e = np.array([x.std_error for x in res])
        ax[0].plot(Ns, p, color=c, label=lab, marker="o", ms=3)
        ax[0].fill_between(Ns, p - 1.96 * e, p + 1.96 * e, color=c, alpha=0.15)
        ax[1].loglog(Ns, e, color=c, label=lab, marker="o", ms=3)
    ax[0].axhline(exact, color=GREY, ls="--", label=f"Black-Scholes = {exact:.4f}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("Number of paths N"); ax[0].set_ylabel("Price")
    ax[0].set_title("MC estimate with 95% CI"); ax[0].legend(frameon=False)
    ref = plain[0].std_error * np.sqrt(Ns[0] / Ns)
    ax[1].loglog(Ns, ref, color=GREY, ls=":", label=r"$\propto 1/\sqrt{N}$")
    ax[1].set_xlabel("Number of paths N"); ax[1].set_ylabel("Standard error")
    ax[1].set_title("Error shrinks like 1/sqrt(N)"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(IMG / "option_convergence.png"); plt.close(fig)

    big_p = monte_carlo(S, K, T, r, sigma, 1_000_000, seed=7)
    big_a = monte_carlo(S, K, T, r, sigma, 1_000_000, antithetic=True, seed=7)
    vr = (big_p.std_error / big_a.std_error) ** 2
    slope = float(np.polyfit(np.log(Ns), np.log([x.std_error for x in plain]), 1)[0])
    return {"params": {"S": S, "K": K, "T": T, "r": r, "sigma": sigma},
            "black_scholes": exact,
            "mc_1m": {"price": big_p.price, "se": big_p.std_error},
            "antithetic_1m": {"price": big_a.price, "se": big_a.std_error},
            "variance_reduction_factor": vr,
            "log_log_slope": slope}


def mm_experiment() -> dict:
    p = MMParams()
    out = {}
    res = {s: simulate(p, s, n_paths=2000, seed=42) for s in ("inventory", "symmetric")}
    for s, r_ in res.items():
        out[s] = r_.summary()

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.linspace(min(r_.pnl.min() for r_ in res.values()),
                       max(r_.pnl.max() for r_ in res.values()), 60)
    ax[0].hist(res["symmetric"].pnl, bins=bins, color=GREY, alpha=0.6, label="Symmetric")
    ax[0].hist(res["inventory"].pnl, bins=bins, color=BLUE, alpha=0.6, label="Inventory-aware (A-S)")
    ax[0].set_xlabel("Final PnL"); ax[0].set_ylabel("Paths"); ax[0].set_title("PnL distribution (2,000 days)")
    ax[0].legend(frameon=False)
    sp = res["inventory"].sample_path
    ax[1].plot(sp["t"], sp["mid"], color=GREY, lw=1, label="Mid")
    ax[1].plot(sp["t"], sp["bid"], color=BLUE, lw=0.8, label="Bid")
    ax[1].plot(sp["t"], sp["ask"], color=ORANGE, lw=0.8, label="Ask")
    ax[1].set_xlabel("Time"); ax[1].set_ylabel("Price"); ax[1].set_title("One simulated day (A-S quotes)")
    ax2 = ax[1].twinx(); ax2.step(sp["t"], sp["q"], color="#16a34a", lw=0.8, alpha=0.7)
    ax2.set_ylabel("Inventory", color="#16a34a"); ax2.spines["top"].set_visible(False)
    lo, hi = ax[1].get_ylim()
    ax[1].set_ylim(lo, hi + 0.25 * (hi - lo))
    ax[1].legend(frameon=False, loc="upper left", ncol=3)
    fig.tight_layout(); fig.savefig(IMG / "market_making.png"); plt.close(fig)

    # sensitivity: risk aversion gamma trades mean PnL against PnL risk
    gammas = [0.01, 0.05, 0.1, 0.5, 1.0]
    sens = []
    for g in gammas:
        s = simulate(MMParams(gamma=g), "inventory", n_paths=1000, seed=3).summary()
        sens.append({"gamma": g, "mean_pnl": s["mean_pnl"], "std_pnl": s["std_pnl"]})
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot([x["std_pnl"] for x in sens], [x["mean_pnl"] for x in sens], color=BLUE, marker="o")
    for x in sens:
        ax.annotate(f"γ={x['gamma']}", (x["std_pnl"], x["mean_pnl"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Std of PnL (risk)"); ax.set_ylabel("Mean PnL (return)")
    ax.set_title("Risk aversion: return vs risk")
    fig.tight_layout(); fig.savefig(IMG / "gamma_tradeoff.png"); plt.close(fig)
    out["gamma_sensitivity"] = sens
    out["params"] = p.__dict__
    return out


def kelly_experiment() -> dict:
    p, b, n_bets = 0.55, 1.0, 1000
    fstar = kelly_fraction(p, b)
    mults = [0.25, 0.5, 1.0, 1.5, 2.0]
    rows = []
    sims = {}
    for m in mults:
        sim = simulate_wealth(m * fstar, p, b, n_bets=n_bets, n_paths=5000, seed=21)
        sims[m] = sim
        rows.append({"multiple_of_kelly": m, "fraction": m * fstar,
                     "theory_growth": float(growth_rate(m * fstar, p, b)),
                     "sim_growth": float(sim["growth_per_bet"].mean()),
                     "median_final_wealth": float(np.exp(np.median(sim["final_log_wealth"]))),
                     "prob_loss": float((sim["final_log_wealth"] < 0).mean()),
                     "prob_drawdown_50": float((sim["max_drawdown"] > 0.5).mean())})

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    fs = np.linspace(0, 2.4 * fstar, 200)
    ax[0].plot(fs / fstar, growth_rate(fs, p, b) * 1000, color=GREY, label="Theory g(f)")
    ax[0].scatter([r["multiple_of_kelly"] for r in rows], [r["sim_growth"] * 1000 for r in rows],
                  color=BLUE, zorder=3, label="Simulation (5,000 paths)")
    ax[0].axhline(0, color=GREY, lw=0.6); ax[0].axvline(1, color=ORANGE, ls="--", lw=0.8, label="Kelly f*")
    ax[0].set_xlabel("Bet size (multiple of Kelly)"); ax[0].set_ylabel("Log growth per 1,000 bets")
    ax[0].set_title("Over-betting lowers growth"); ax[0].legend(frameon=False)
    t = np.arange(1, n_bets + 1)
    for m, c in [(0.5, BLUE), (1.0, ORANGE), (2.0, GREY)]:
        q = np.percentile(sims[m]["paths"], [10, 50, 90], axis=0)
        ax[1].plot(t, q[1], color=c, label=f"{m}× Kelly (median)")
        ax[1].fill_between(t, q[0], q[2], color=c, alpha=0.12)
    ax[1].set_xlabel("Bet number"); ax[1].set_ylabel("Log wealth")
    ax[1].set_title("Wealth paths: median and 10–90% band"); ax[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(IMG / "kelly_growth.png"); plt.close(fig)

    grid = np.round(np.arange(0.1, 2.01, 0.1), 2)
    est = [estimation_error_growth(float(m), p, b, n_obs=200, seed=5) for m in grid]
    known = [float(growth_rate(m * fstar, p, b)) for m in grid]
    best = float(grid[int(np.argmax(est))])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(grid, np.array(known) * 1000, color=GREY, ls="--", label="p known exactly")
    ax.plot(grid, np.array(est) * 1000, color=BLUE, marker="o", ms=3, label="p estimated from 200 bets")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xlabel("Multiple of estimated Kelly"); ax.set_ylabel("True log growth per 1,000 bets")
    ax.set_title("Estimation error favours fractional Kelly"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(IMG / "kelly_estimation.png"); plt.close(fig)

    # known continuous-time law: P(wealth ever falls to half) = 0.5^(2/c - 1) at c x Kelly
    ruin = []
    for c in (0.5, 1.0):
        paths = simulate_wealth(c * fstar, p, b, n_bets=5000, n_paths=4000, seed=2)["paths"]
        ruin.append({"multiple": c, "sim": float((paths.min(axis=1) <= np.log(0.5)).mean()),
                     "theory": 0.5 ** (2 / c - 1)})
    sd_f = 2 * np.sqrt(p * (1 - p) / 200)            # f_hat = 2 p_hat - 1 for an even-money bet
    m_star = fstar**2 / (fstar**2 + sd_f**2)
    return {"p": p, "b": b, "kelly_fraction": fstar, "n_bets": n_bets, "table": rows,
            "ruin_half": ruin, "m_star_theory": float(m_star),
            "estimation": {"n_obs": 200, "best_multiplier": best,
                           "growth_full_kelly": est[list(grid).index(1.0)],
                           "growth_half_kelly": est[list(grid).index(0.5)],
                           "growth_known_full": known[list(grid).index(1.0)]}}


def gm_experiment() -> dict:
    out = {}
    # 1) one market and the average spread over time
    p = gm.GMParams(mu=0.3, n_trades=60)
    res = gm.simulate(p, n_paths=4000, seed=8)
    i = int(np.argmax(res.high))  # a path where the asset is worth V_high
    bid, ask, _, _ = gm.quotes(res.belief[i, :-1], p)
    ev = p.v_low + (p.v_high - p.v_low) * res.belief[i]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    t = np.arange(p.n_trades)
    ax[0].fill_between(t, bid, ask, color=BLUE, alpha=0.15, step="post", label="Bid-ask band")
    ax[0].step(t, ev[:-1], color=BLUE, where="post", lw=1.2, label="Market maker's E[V]")
    ax[0].axhline(p.v_high, color=ORANGE, ls="--", lw=1, label="True value")
    ax[0].set_xlabel("Trade number"); ax[0].set_ylabel("Price")
    ax[0].set_title("Prices learn the truth from order flow (μ = 0.3)"); ax[0].legend(frameon=False, loc="lower right")
    for mu, c in [(0.1, GREY), (0.3, BLUE), (0.5, ORANGE)]:
        q = gm.GMParams(mu=mu, n_trades=60)
        r_ = gm.simulate(q, n_paths=4000, seed=9)
        ax[1].plot(np.arange(q.n_trades), r_.spread.mean(axis=0), color=c, label=f"μ = {mu}")
    ax[1].set_xlabel("Trade number"); ax[1].set_ylabel("Average spread")
    ax[1].set_title("Spread starts at μ·(V_H − V_L), then shrinks"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(IMG / "gm_quotes.png"); plt.close(fig)

    # 2) PnL decomposition at mu = 0.3
    p = gm.GMParams(mu=0.3, n_trades=300)
    res = gm.simulate(p, n_paths=20000, seed=1)
    n = res.mm_pnl.size
    out["pnl_per_market_mu_0_3"] = {
        "market_maker": float(res.mm_pnl.mean()), "market_maker_se": float(res.mm_pnl.std(ddof=1) / np.sqrt(n)),
        "informed": float(res.informed_pnl.mean()), "uninformed": float(res.noise_pnl.mean())}
    out["first_spread_mu_0_3"] = float(res.spread[0, 0])

    # 3) discovery speed and the naive maker
    mus = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    disc, naive = [], []
    for mu in mus:
        q = gm.GMParams(mu=mu, n_trades=4000)
        r_ = gm.simulate(q, n_paths=3000, seed=2)
        d = gm.trades_to_discover(r_, q)
        disc.append({"mu": mu, "sim_mean": float(d.mean()), "sim_se": float(d.std(ddof=1) / np.sqrt(d.size)),
                     "exact": gm.expected_discovery_time(mu), "unfinished": float((d >= q.n_trades).mean())})
        q2 = gm.GMParams(mu=mu, n_trades=200)
        r2 = gm.simulate(q2, n_paths=4000, seed=4, naive_spread=4.0)
        naive.append({"mu": mu, "naive_per_trade": float(r2.naive_pnl.mean() / q2.n_trades),
                      "theory": 2.0 - mu * 10.0,
                      "bayesian_per_trade": float(r2.mm_pnl.mean() / q2.n_trades)})
    out["discovery"] = disc
    out["naive_vs_bayesian"] = naive

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    mgrid = np.linspace(0.04, 0.55, 200)
    ax[0].loglog(mgrid, [gm.expected_discovery_time(m) for m in mgrid], color=GREY, lw=1, label="Exact formula k/μ")
    ax[0].loglog(mus, [d["sim_mean"] for d in disc], "o", color=BLUE, label="Simulation (3,000 markets)")
    ax[0].loglog(mgrid, np.log(19) / 2 / mgrid**2, color=ORANGE, ls=":", lw=1,
                 label=r"small-μ limit $\ln 19 / (2\mu^2)$")
    ax[0].set_xticks(mus); ax[0].set_xticklabels([str(m) for m in mus]); ax[0].minorticks_off()
    ax[0].set_xlabel("Share of informed traders μ"); ax[0].set_ylabel("Trades until price within 5% of truth")
    ax[0].set_title("Halving informed flow quadruples discovery time"); ax[0].legend(frameon=False)
    ax[1].plot(mus, [x["theory"] for x in naive], color=GREY, lw=1, label="Theory: s/2 − μΔ/2")
    ax[1].plot(mus, [x["naive_per_trade"] for x in naive], "o", color=ORANGE, label="Naive fixed quotes (sim)")
    ax[1].plot(mus, [x["bayesian_per_trade"] for x in naive], "s", color=BLUE, ms=4, label="Bayesian maker (sim)")
    ax[1].axhline(0, color=GREY, lw=0.6); ax[1].axvline(0.2, color=GREY, ls="--", lw=0.6)
    ax[1].set_xlabel("Share of informed traders μ"); ax[1].set_ylabel("Market maker PnL per trade")
    ax[1].set_title("Ignoring information loses money past μ = s/Δ"); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(IMG / "gm_discovery.png"); plt.close(fig)
    return out


if __name__ == "__main__":
    results = {"options": option_experiment(), "market_making": mm_experiment(),
               "kelly": kelly_experiment(), "adverse_selection": gm_experiment()}
    (ROOT / "docs" / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
