"""Build the portfolio site from site/page.html.

Every number on the page is filled in from docs/results.json, so the site cannot
drift from the code. The browser models (site/models.js) are inlined.

Usage:
    python scripts/run_all.py                       # produce docs/results.json first
    python scripts/build_site.py --github <your-id> # writes docs/index.html for GitHub Pages

Outputs:
    docs/index.html   complete HTML document (GitHub Pages serves this)
    build/page.html   the same page without the document wrapper (for hosts that add one)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def num(v: float, d: int) -> str:
    return f"{v:.{d}f}".replace("-", "−")


def pct(v: float, d: int = 1) -> str:
    return f"{100 * v:.{d}f}%"


def values(r: dict, n_tests: int) -> dict[str, str]:
    o, mm, k, g = r["options"], r["market_making"], r["kelly"], r["adverse_selection"]
    inv, sym = mm["inventory"], mm["symmetric"]
    row = {x["multiple_of_kelly"]: x for x in k["table"]}
    disc = next(d for d in g["discovery"] if d["mu"] == 0.3)
    naive = next(d for d in g["naive_vs_bayesian"] if d["mu"] == 0.3)
    ruin = next(x for x in k["ruin_half"] if x["multiple"] == 0.5)
    return {
        "gm_first_theory": num(0.3 * 20, 3),
        "gm_first_sim": num(g["first_spread_mu_0_3"], 3),
        "gm_disc_exact": num(disc["exact"], 2),
        "gm_disc_sim": f'{num(disc["sim_mean"], 2)} ± {num(disc["sim_se"], 2)}',
        "gm_naive_theory": num(naive["theory"], 3),
        "gm_naive_sim": num(naive["naive_per_trade"], 3),
        "k_ruin_theory": pct(ruin["theory"]),
        "k_ruin_sim": pct(ruin["sim"]),
        "k_mstar_theory": f'{k["m_star_theory"]:.2f}×',
        "k_mstar_sim": f'{k["estimation"]["best_multiplier"]:.1f}×',
        "k_mstar_val": f'{k["m_star_theory"]:.2f}',
        "k_half_share": pct(row[0.5]["theory_growth"] / row[1.0]["theory_growth"], 0),
        "k_loss_half": pct(row[0.5]["prob_loss"]),
        "k_loss_full": pct(row[1.0]["prob_loss"]),
        "k_loss_double": pct(row[2.0]["prob_loss"]),
        "mc_slope": num(o["log_log_slope"], 2),
        "bs": num(o["black_scholes"], 4),
        "mc_price": num(o["mc_1m"]["price"], 4),
        "mc_ci": num(1.96 * o["mc_1m"]["se"], 4),
        "mc_vr": f'{o["variance_reduction_factor"]:.1f}×',
        "as_drop": pct(1 - inv["mean_pnl"] / sym["mean_pnl"], 0),
        "as_inv_mean": num(inv["mean_pnl"], 1), "as_sym_mean": num(sym["mean_pnl"], 1),
        "as_inv_std": num(inv["std_pnl"], 1), "as_sym_std": num(sym["std_pnl"], 1),
        "as_inv_ratio": num(inv["sharpe_like"], 1), "as_sym_ratio": num(sym["sharpe_like"], 1),
        "n_tests": str(n_tests),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--github", help="your GitHub user name, used for the source-code link")
    args = ap.parse_args()

    page = (ROOT / "site" / "page.html").read_text()
    models = (ROOT / "site" / "models.js").read_text()
    results = json.loads((ROOT / "docs" / "results.json").read_text())
    n_tests = len(re.findall(r"^\s+def test_", (ROOT / "tests" / "test_quantsim.py").read_text(), re.M))

    for key, val in values(results, n_tests).items():
        page = page.replace("{{" + key + "}}", val)
    if args.github:
        page = page.replace("{{GITHUB_URL}}", f"https://github.com/{args.github}/quant-sim")
    page = page.replace("<script>/*MODELS*/</script>", "<script>\n" + models + "</script>")

    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", page)) - {"GITHUB_URL"})
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")

    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "page.html").write_text(page)
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           '</head>\n<body>\n' + page + '\n</body>\n</html>\n')
    (ROOT / "docs" / "index.html").write_text(doc)
    print(f"wrote docs/index.html ({len(doc) // 1024} KB), {n_tests} tests counted,"
          f" github link {'set' if args.github else 'hidden until --github is given'}")


if __name__ == "__main__":
    main()
