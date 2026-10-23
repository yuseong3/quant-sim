"""Run with:  python -m unittest discover tests"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantsim import MMParams, black_scholes, monte_carlo, simulate  # noqa: E402


class TestBlackScholes(unittest.TestCase):
    def test_known_value(self):
        # Textbook example (Hull): S=42, K=40, T=0.5, r=0.1, sigma=0.2 -> call ~ 4.76
        self.assertAlmostEqual(black_scholes(42, 40, 0.5, 0.1, 0.2), 4.76, places=2)

    def test_put_call_parity(self):
        S, K, T, r, s = 100, 95, 0.75, 0.02, 0.3
        lhs = black_scholes(S, K, T, r, s, "call") - black_scholes(S, K, T, r, s, "put")
        self.assertAlmostEqual(lhs, S - K * math.exp(-r * T), places=10)


class TestMonteCarlo(unittest.TestCase):
    def test_mc_within_ci(self):
        exact = black_scholes(100, 105, 1, 0.03, 0.2)
        res = monte_carlo(100, 105, 1, 0.03, 0.2, 400_000, seed=11)
        # 4 standard errors: fails by chance with probability < 0.01%
        self.assertLess(abs(res.price - exact), 4 * res.std_error)

    def test_antithetic_reduces_error(self):
        a = monte_carlo(100, 100, 1, 0.0, 0.2, 200_000, seed=5)
        b = monte_carlo(100, 100, 1, 0.0, 0.2, 200_000, antithetic=True, seed=5)
        self.assertLess(b.std_error, a.std_error)


class TestMarketMaking(unittest.TestCase):
    def test_inventory_strategy_lowers_risk(self):
        p = MMParams()
        inv = simulate(p, "inventory", n_paths=500, seed=1).summary()
        sym = simulate(p, "symmetric", n_paths=500, seed=1).summary()
        self.assertLess(inv["std_pnl"], sym["std_pnl"])
        self.assertLess(inv["std_inventory"], sym["std_inventory"])

    def test_deterministic_with_seed(self):
        p = MMParams()
        a = simulate(p, "inventory", n_paths=50, seed=9).pnl
        b = simulate(p, "inventory", n_paths=50, seed=9).pnl
        self.assertTrue((a == b).all())

    def test_bad_strategy(self):
        with self.assertRaises(ValueError):
            simulate(MMParams(), "nope", n_paths=1)


class TestKelly(unittest.TestCase):
    def test_fraction_and_maximum(self):
        from quantsim import kelly_fraction, growth_rate
        import numpy as np
        self.assertAlmostEqual(kelly_fraction(0.55, 1.0), 0.10)
        fs = np.linspace(0.001, 0.5, 5000)
        self.assertAlmostEqual(fs[np.argmax(growth_rate(fs, 0.55))], 0.10, places=3)
        self.assertLess(growth_rate(0.2, 0.55), 0.0)  # about 2x Kelly: growth turns negative

    def test_simulation_matches_theory(self):
        from quantsim import growth_rate, simulate_wealth
        sim = simulate_wealth(0.1, 0.55, n_bets=500, n_paths=4000, seed=1)["growth_per_bet"].mean()
        self.assertAlmostEqual(sim, float(growth_rate(0.1, 0.55)), delta=3e-4)

    def test_half_kelly_ruin_probability(self):
        # continuous-time result: P(wealth ever hits x) = x^(2/c - 1) for c x Kelly
        import numpy as np
        from quantsim import simulate_wealth
        paths = simulate_wealth(0.05, 0.55, n_bets=5000, n_paths=3000, seed=2)["paths"]
        prob = (paths.min(axis=1) <= np.log(0.5)).mean()
        self.assertAlmostEqual(prob, 0.125, delta=0.03)

    def test_estimation_error_favours_fraction(self):
        from quantsim import estimation_error_growth
        self.assertGreater(estimation_error_growth(0.5, 0.55, seed=3),
                           estimation_error_growth(1.0, 0.55, seed=3))


class TestAdverseSelection(unittest.TestCase):
    def test_first_spread_closed_form(self):
        from quantsim.adverse_selection import GMParams, quotes
        for mu in (0.1, 0.3, 0.7):
            bid, ask, _, _ = quotes(0.5, GMParams(mu=mu))
            self.assertAlmostEqual(float(ask - bid), mu * 20.0, places=10)

    def test_zero_profit_and_transfer(self):
        import numpy as np
        from quantsim.adverse_selection import GMParams, simulate
        r = simulate(GMParams(mu=0.3, n_trades=100), n_paths=20000, seed=3)
        se = r.mm_pnl.std(ddof=1) / np.sqrt(r.mm_pnl.size)
        self.assertLess(abs(r.mm_pnl.mean()), 4 * se)          # competitive maker breaks even
        total = r.mm_pnl + r.informed_pnl + r.noise_pnl
        self.assertTrue(np.allclose(total, 0.0))                  # zero-sum, path by path
        self.assertGreater(r.informed_pnl.mean(), 0.0)
        self.assertLess(r.noise_pnl.mean(), 0.0)

    def test_naive_maker_break_even(self):
        from quantsim.adverse_selection import GMParams, simulate
        for mu in (0.1, 0.3):
            r = simulate(GMParams(mu=mu, n_trades=200), n_paths=4000, seed=4, naive_spread=4.0)
            self.assertAlmostEqual(r.naive_pnl.mean() / 200, 2.0 - mu * 10.0, delta=0.05)

    def test_discovery_time_exact(self):
        from quantsim.adverse_selection import GMParams, simulate, trades_to_discover, expected_discovery_time
        for mu in (0.2, 0.4):
            q = GMParams(mu=mu, n_trades=2000)
            d = trades_to_discover(simulate(q, n_paths=3000, seed=5), q)
            se = d.std(ddof=1) / d.size ** 0.5
            self.assertLess(abs(d.mean() - expected_discovery_time(mu)), 4 * se)


if __name__ == "__main__":
    unittest.main()
