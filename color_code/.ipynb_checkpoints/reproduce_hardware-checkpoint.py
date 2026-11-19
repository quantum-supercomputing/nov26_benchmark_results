# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Hardware: stored shots -> detectors -> Tesseract -> LER, checked against our stored results at every step.

  python reproduce_hardware.py --patch d3_TL --basis X             # one patch/basis, all rounds, every shot
  python reproduce_hardware.py --distance 3 --workers 8            # all d = 3 experiments
  python reproduce_hardware.py --patch d5_L04 --shots 200          # quick d = 5 check on the first 200 shots
  python reproduce_hardware.py --workers 16                        # everything (d = 5 is ~1 CPU-day, see README)

Writes results/hardware_ler.csv (one row per experiment; rows are replaced on re-run). Then: python fit_ler.py
"""
import argparse
import csv
import pathlib
import time

import numpy as np

import ccrepro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", help="d5_L04, d3_TL, d3_TR, d3_BOT (default all)")
    ap.add_argument("--basis", choices=["X", "Z"])
    ap.add_argument("--distance", type=int, choices=[3, 5])
    ap.add_argument("--rounds", type=int, nargs="*")
    ap.add_argument("--shots", type=int, default=None, help="decode only the first N shots (default all)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=pathlib.Path, default=ccrepro.ROOT / "results/hardware_ler.csv")
    a = ap.parse_args()

    rows = {}
    if a.out.exists():
        with open(a.out) as f:
            rows = {r["path"]: r for r in csv.DictReader(f)}
    for e in ccrepro.index():
        if (a.patch and e["patch"] != a.patch) or (a.basis and e["basis"] != a.basis) \
                or (a.distance and int(e["distance"]) != a.distance) or (a.rounds and int(e["rounds"]) not in a.rounds):
            continue
        ex = ccrepro.load(e["path"])
        ours = ex.our_results()
        n = a.shots or ex.shots

        # 1) Qiskit circuit: one classical bit per Stim measurement, same order.
        qc_clbits = ex.qiskit_circuit().num_clbits
        assert qc_clbits == ex.stim_circuit().num_measurements, (qc_clbits, ex.stim_circuit().num_measurements)

        # 2) Stored shots -> 3) detection events and observable flips.
        m = ex.measurements()[:n]
        dets, obs = ex.to_detectors(m)
        same_dets = bool((dets == ours["detectors"][:n]).all() and (obs == ours["observables"][:n]).all())

        # 4) Tesseract.
        t = time.time()
        fail = ccrepro.decode_parallel(ex, dets, obs, workers=a.workers)
        dt = time.time() - t
        same_fail = float((fail == ours["logical_failure"][:n]).mean())
        rows[e["path"]] = dict(path=e["path"], patch=e["patch"], distance=e["distance"], basis=e["basis"],
                               rounds=e["rounds"], shots_decoded=n, failures=int(fail.sum()), ler=fail.mean(),
                               our_ler_same_shots=ours["logical_failure"][:n].mean(), detectors_match=same_dets,
                               per_shot_agreement=same_fail, seconds=round(dt, 1))
        print(f"{e['path']:16s} {n:5d} shots  LER {fail.mean():.4f}  ours {ours['logical_failure'][:n].mean():.4f}  "
              f"detectors match: {same_dets}  per-shot agreement {same_fail:.4f}  ({1e3 * dt / n:.0f} ms/shot)",
              flush=True)

        a.out.parent.mkdir(exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values()))))
            w.writeheader(); w.writerows(sorted(rows.values(), key=lambda r: (r["patch"], r["basis"], int(r["rounds"]))))


if __name__ == "__main__":
    main()
