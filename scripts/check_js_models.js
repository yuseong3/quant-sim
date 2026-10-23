// Cross-check the browser models (site/models.js) against the Python results.
// Usage: node scripts/check_js_models.js   (run scripts/run_all.py first)
const path = require('path');
const fs = require('fs');
const QS = require(path.join(__dirname, '..', 'site', 'models.js'));
const R = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'docs', 'results.json'), 'utf8'));

let failed = 0;
function check(name, js, py, tol) {
  const ok = Math.abs(js - py) <= tol;
  if (!ok) failed++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(46)} js=${js.toFixed(4)}  python=${py.toFixed(4)}  tol=${tol}`);
}
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };

// Options
const o = R.options.params;
check('Black-Scholes price', QS.blackScholes(o.S, o.K, o.T, o.r, o.sigma), R.options.black_scholes, 1e-5);
const mc = QS.monteCarlo(o.S, o.K, o.T, o.r, o.sigma, 1000000, 7);
const last = mc.plain[mc.plain.length - 1];
check('MC price, 1e6 paths (within 4 se of exact)', last.mean, R.options.black_scholes, 4 * last.se);
const vr = (last.se / mc.anti[mc.anti.length - 1].se) ** 2;
check('antithetic variance reduction factor', vr, R.options.variance_reduction_factor, 0.15);

// Market making
const inv = QS.marketMaking(0.1, true, 2000, 42), sym = QS.marketMaking(0.1, false, 2000, 42);
check('A-S inventory-aware mean PnL', mean(Array.from(inv.pnl)), R.market_making.inventory.mean_pnl, 1.0);
check('A-S inventory-aware std PnL', sd(Array.from(inv.pnl)), R.market_making.inventory.std_pnl, 0.6);
check('symmetric mean PnL', mean(Array.from(sym.pnl)), R.market_making.symmetric.mean_pnl, 1.5);
check('symmetric std PnL', sd(Array.from(sym.pnl)), R.market_making.symmetric.std_pnl, 1.0);

// Kelly
const K = R.kelly, row = (m) => K.table.find((r) => r.multiple_of_kelly === m);
check('Kelly fraction p=0.55', QS.kellyFraction(0.55), K.kelly_fraction, 1e-12);
check('growth at half Kelly (theory)', QS.growthRate(0.05, 0.55), row(0.5).theory_growth, 1e-12);
const kp = QS.kellyPaths(0.1, 0.55, 1000, 5000, 21, 10);
check('P(loss after 1000 bets), full Kelly', kp.probLoss, row(1.0).prob_loss, 0.015);
const k2 = QS.kellyPaths(0.2, 0.55, 1000, 5000, 22, 10);
check('P(loss after 1000 bets), 2x Kelly', k2.probLoss, row(2.0).prob_loss, 0.03);

// Adverse selection
const G = R.adverse_selection;
const q = QS.gmQuotes(0.5, 0.3);
check('first G-M spread, mu=0.3', q.ask - q.bid, G.first_spread_mu_0_3, 1e-9);
for (const d of G.discovery.filter((d) => [0.1, 0.3, 0.5].includes(d.mu))) {
  check(`discovery exact formula, mu=${d.mu}`, QS.gmExpectedDiscovery(d.mu), d.exact, 1e-9);
  check(`discovery simulated, mu=${d.mu}`, QS.gmDiscoverySim(d.mu, 3000, 11, 20000), d.exact, 4 * d.sim_se + 1e-9);
}
for (const d of G.naive_vs_bayesian.filter((d) => [0.1, 0.3].includes(d.mu))) {
  check(`naive maker PnL/trade, mu=${d.mu}`, QS.gmNaive(d.mu, 4, 4000, 200, 13), d.theory, 0.05);
}

console.log(failed ? `\n${failed} check(s) failed` : '\nall checks passed');
process.exit(failed ? 1 : 0);
