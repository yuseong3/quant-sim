"""quant-sim: small, readable simulations of option pricing, market making, bet sizing and adverse selection."""
from .options import black_scholes, monte_carlo, MCResult
from .market_making import MMParams, MMResult, simulate
from .kelly import kelly_fraction, growth_rate, simulate_wealth, estimation_error_growth
from . import adverse_selection

__all__ = ["black_scholes", "monte_carlo", "MCResult", "MMParams", "MMResult", "simulate",
           "kelly_fraction", "growth_rate", "simulate_wealth", "estimation_error_growth", "adverse_selection"]
