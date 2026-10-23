/* Browser versions of the four quant-sim models. Inlined into the page by
 * scripts/build_site.py and cross-checked against the Python results by
 * scripts/check_js_models.js. Same formulas as quantsim/*.py; the random
 * number generator differs, so simulated numbers agree statistically, not digit for digit. */
(function (root) {
  'use strict';

  // ---------- random numbers ----------
  function rng(seed) {
    let a = seed >>> 0, spare = null;
    function u() {                      // mulberry32
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    function n() {                      // Box-Muller, one value cached
      if (spare !== null) { const s = spare; spare = null; return s; }
      let x; do { x = u(); } while (x <= 1e-12);
      const r = Math.sqrt(-2 * Math.log(x)), th = 2 * Math.PI * u();
      spare = r * Math.sin(th);
      return r * Math.cos(th);
    }
    return { u, n };
  }

  // ---------- Monte Carlo option pricing ----------
  function erf(x) {                     // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const ncdf = (x) => 0.5 * (1 + erf(x / Math.SQRT2));

  function blackScholes(S, K, T, r, sigma) {
    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);
    return S * ncdf(d1) - K * Math.exp(-r * T) * ncdf(d2);
  }

  /* Running Monte Carlo estimate at log-spaced checkpoints.
   * Plain and antithetic estimators use the same normal draws. */
  function monteCarlo(S, K, T, r, sigma, nMax, seed) {
    const g = rng(seed), drift = (r - 0.5 * sigma * sigma) * T, vol = sigma * Math.sqrt(T), disc = Math.exp(-r * T);
    const pay = (z) => disc * Math.max(S * Math.exp(drift + vol * z) - K, 0);
    const checkpoints = [];
    for (let i = 0; i <= 40; i++) checkpoints.push(Math.round(100 * Math.pow(nMax / 100, i / 40)));
    const plain = [], anti = [];
    let ps = 0, pss = 0, pn = 0, as = 0, ass = 0, an = 0, c = 0;
    for (let path = 1; path <= nMax; path++) {
      const z = g.n(); const x = pay(z);
      ps += x; pss += x * x; pn++;
      if (path % 2 === 1) {                     // one antithetic pair per two paths
        const y = 0.5 * (x + pay(-z));
        as += y; ass += y * y; an++;
      }
      while (c < checkpoints.length && path === checkpoints[c]) {
        const pm = ps / pn, am = as / an;
        plain.push({ n: path, mean: pm, se: Math.sqrt(Math.max(pss / pn - pm * pm, 0) / pn) });
        anti.push({ n: path, mean: am, se: Math.sqrt(Math.max(ass / an - am * am, 0) / an) });
        c++;
      }
    }
    return { plain, anti };
  }

  // ---------- Avellaneda-Stoikov market making ----------
  const AS = { s0: 100, T: 1, sigma: 2, dt: 0.005, k: 1.5, A: 140 };

  /* Final PnL of nPaths simulated days. Inventory-aware and symmetric runs with the
   * same seed consume random numbers identically, so they see the same prices and order flow. */
  function marketMaking(gamma, inventoryAware, nPaths, seed) {
    const g = rng(seed), n = Math.round(AS.T / AS.dt), s2 = AS.sigma * AS.sigma;
    const cst = (2 / gamma) * Math.log(1 + gamma / AS.k), sq = Math.sqrt(AS.dt);
    const pnl = new Float64Array(nPaths), inv = new Float64Array(nPaths);
    for (let p = 0; p < nPaths; p++) {
      let S = AS.s0, q = 0, cash = 0;
      for (let i = 0; i < n; i++) {
        const tau = AS.T - i * AS.dt, half = (gamma * s2 * tau + cst) / 2;
        const centre = inventoryAware ? S - q * gamma * s2 * tau : S;
        const bid = centre - half, ask = centre + half;
        const pb = Math.min(1, AS.A * Math.exp(-AS.k * (S - bid)) * AS.dt);
        const pa = Math.min(1, AS.A * Math.exp(-AS.k * (ask - S)) * AS.dt);
        const buy = g.u() < pb, sell = g.u() < pa;
        if (buy) { q += 1; cash -= bid; }
        if (sell) { q -= 1; cash += ask; }
        S += AS.sigma * sq * g.n();
      }
      pnl[p] = cash + q * S; inv[p] = q;
    }
    return { pnl, inv };
  }

  // ---------- Kelly criterion ----------
  const kellyFraction = (p) => 2 * p - 1;     // even-money bet: p - (1 - p)
  const growthRate = (f, p) => p * Math.log1p(f) + (1 - p) * Math.log1p(-f);

  /* Log wealth of nPaths after every `every` bets; returns quantile bands per checkpoint. */
  function kellyPaths(f, p, nBets, nPaths, seed, every) {
    const g = rng(seed), up = Math.log1p(f), dn = Math.log1p(-f), m = Math.floor(nBets / every) + 1;
    const snaps = []; for (let j = 0; j < m; j++) snaps.push(new Float64Array(nPaths));
    for (let i = 0; i < nPaths; i++) {
      let w = 0;
      for (let b = 1; b <= nBets; b++) {
        w += g.u() < p ? up : dn;
        if (b % every === 0) snaps[b / every][i] = w;
      }
    }
    const q = (arr, x) => arr[Math.min(arr.length - 1, Math.floor(x * arr.length))];
    const bands = snaps.map((a, j) => {
      const s = Array.from(a).sort((x, y) => x - y);
      return { bet: j * every, p10: q(s, 0.1), p50: q(s, 0.5), p90: q(s, 0.9) };
    });
    const last = snaps[m - 1];
    let losses = 0; for (let i = 0; i < nPaths; i++) if (last[i] < 0) losses++;
    return { bands, probLoss: losses / nPaths, medianFinal: bands[m - 1].p50 };
  }

  // ---------- Glosten-Milgrom adverse selection ----------
  const GM = { vLow: 90, vHigh: 110 };

  function gmQuotes(pi, mu) {
    const bh = mu + (1 - mu) / 2, bl = (1 - mu) / 2, w = GM.vHigh - GM.vLow;
    const pBuy = pi * bh / (pi * bh + (1 - pi) * bl);
    const pSell = pi * (1 - bh) / (pi * (1 - bh) + (1 - pi) * (1 - bl));
    return { bid: GM.vLow + w * pSell, ask: GM.vLow + w * pBuy, pBuy, pSell };
  }

  /* One market: quotes and expected value before each trade, plus the trade direction. */
  function gmPath(mu, nTrades, seed, high) {
    const g = rng(seed); let pi = 0.5; const out = [];
    for (let t = 0; t < nTrades; t++) {
      const q = gmQuotes(pi, mu);
      const informed = g.u() < mu, noiseBuy = g.u() < 0.5;
      const buy = informed ? high : noiseBuy;
      out.push({ t, bid: q.bid, ask: q.ask, ev: GM.vLow + (GM.vHigh - GM.vLow) * pi, buy, informed });
      pi = buy ? q.pBuy : q.pSell;
    }
    return out;
  }

  function gmExpectedDiscovery(mu, tol) {
    tol = tol || 0.05;
    const step = Math.log((1 + mu) / (1 - mu));
    return Math.ceil(Math.log((1 - tol) / tol) / step - 1e-12) / mu;
  }

  /* Naive maker quoting 100 +/- s/2 forever, against the same kind of order flow. */
  function gmNaive(mu, s, markets, trades, seed) {
    const g = rng(seed); let tot = 0;
    for (let m = 0; m < markets; m++) {
      const high = g.u() < 0.5, V = high ? GM.vHigh : GM.vLow;
      for (let t = 0; t < trades; t++) {
        const informed = g.u() < mu, noiseBuy = g.u() < 0.5;
        const buy = informed ? high : noiseBuy;
        tot -= buy ? V - (100 + s / 2) : (100 - s / 2) - V;
      }
    }
    return tot / (markets * trades);
  }

  /* Mean trades until E[V] first comes within 5% of the range of the true value. */
  function gmDiscoverySim(mu, markets, seed, cap) {
    const g = rng(seed); let tot = 0;
    for (let m = 0; m < markets; m++) {
      const high = g.u() < 0.5; let pi = 0.5, t = 0;
      const done = () => (high ? pi >= 0.95 - 1e-12 : pi <= 0.05 + 1e-12);
      while (!done() && t < cap) {
        const q = gmQuotes(pi, mu);
        const informed = g.u() < mu, noiseBuy = g.u() < 0.5;
        pi = (informed ? high : noiseBuy) ? q.pBuy : q.pSell; t++;
      }
      tot += t;
    }
    return tot / markets;
  }

  const QS = { rng, blackScholes, monteCarlo, AS, marketMaking, kellyFraction, growthRate, kellyPaths,
    GM, gmQuotes, gmPath, gmExpectedDiscovery, gmNaive, gmDiscoverySim };
  if (typeof module !== 'undefined' && module.exports) module.exports = QS; else root.QS = QS;
})(this);
