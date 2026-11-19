# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Fit eps per patch and basis from LER vs rounds, and Lambda = eps(d=3) / eps(d=5); optional plot.

  python fit_ler.py                          # our stored hardware decodes (data/index.csv)
  python fit_ler.py --source hardware        # your re-decode, results/hardware_ler.csv
  python fit_ler.py --source pauli_sim       # your Pauli-only simulation, results/pauli_sim_ler.csv
  python fit_ler.py --source hardware --plot # also results/ler_vs_R_<source>.png (needs matplotlib)

Fit: LER(R) = (1 - b (1 - 2 eps)^R) / 2, unweighted least squares over R >= 5, b free (SPAM offset).
Lambda uses the mean eps of the three d = 3 patches (they are the three corners of the d = 5 patch).
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

import ccrepro

D5, D3 = "d5_L04", ["d3_TL", "d3_TR", "d3_BOT"]
NAMES = {"d5_L04": "d = 5", "d3_TL": "Patch A (d = 3)", "d3_TR": "Patch B (d = 3)", "d3_BOT": "Patch C (d = 3)"}


def load_curves(source):
    if source == "ours":
        rows = [dict(r, ler=r["our_ler"], shots_decoded=r["shots"]) for r in ccrepro.index()]
    else:
        with open(ccrepro.ROOT / f"results/{source}_ler.csv") as f:
            rows = list(csv.DictReader(f))
    curves = defaultdict(list)
    for r in rows:
        curves[r["patch"], r["basis"]].append((int(r["rounds"]), float(r["ler"]), int(r["shots_decoded"])))
    return {k: np.array(sorted(v)).T for k, v in curves.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ours", choices=["ours", "hardware", "pauli_sim"])
    ap.add_argument("--rmin", type=int, default=5)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    curves = load_curves(a.source)
    fits = {}
    print(f"source: {a.source}\n{'patch':18s}{'basis':>6s}{'eps':>10s}{'± err':>9s}{'b':>7s}  points")
    for (g, b), (R, ler, n) in sorted(curves.items()):
        try:
            fits[g, b] = ccrepro.fit_eps(R, ler, rmin=a.rmin)
        except (ValueError, RuntimeError, TypeError):
            continue                                  # too few points for a 2-parameter fit
        eps, err, bb = fits[g, b]
        print(f"{NAMES.get(g, g):18s}{b:>6s}{eps:10.5f}{err:9.5f}{bb:7.3f}  {int((R >= a.rmin).sum())}")
    for b in "XZ":
        if (D5, b) in fits and all((g, b) in fits for g in D3):
            e5, s5 = fits[D5, b][:2]
            e3 = np.mean([fits[g, b][0] for g in D3])
            s3 = np.sqrt(sum(fits[g, b][1] ** 2 for g in D3)) / 3
            lam = e3 / e5
            print(f"Lambda {b} = {lam:.3f} ± {lam * np.hypot(s3 / e3, s5 / e5):.3f}")

    if a.plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
        for i, b in enumerate("XZ"):
            for j, g in enumerate([D5, *D3]):
                ax = axs[i, j]
                if (g, b) not in curves:
                    continue
                R, ler, n = curves[g, b]
                ax.errorbar(R, ler, np.sqrt(ler * (1 - ler) / n), fmt="o", ms=4, capsize=2)
                if (g, b) in fits:
                    eps, _, bb = fits[g, b]
                    r = np.linspace(a.rmin, R.max(), 100)
                    ax.plot(r, ccrepro.decay(r, bb, eps), label=f"ε = {eps:.4f}")
                    ax.legend(loc="upper left")
                ax.set_title(f"{NAMES[g]}, {b} basis"); ax.grid(alpha=0.3)
                if i == 1: ax.set_xlabel("QEC rounds R")
                if j == 0: ax.set_ylabel("logical error rate")
        fig.suptitle(f"ibm_phoenix v3: LER vs rounds ({a.source}); fit over R ≥ {a.rmin}")
        fig.tight_layout()
        out = ccrepro.ROOT / f"results/ler_vs_R_{a.source}.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=120)
        print("wrote", out)


if __name__ == "__main__":
    main()
