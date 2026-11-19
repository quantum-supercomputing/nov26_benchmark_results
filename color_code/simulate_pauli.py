# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pauli-only simulation: sample the stored Stim circuit, decode with the same DEM, report the LER.

  python simulate_pauli.py --patch d3_TL --basis X --shots 5000
  python simulate_pauli.py --distance 3 --workers 8

The Stim circuit is the scheduled hardware circuit with Pauli noise from IBM's calibration (gate, readout and
reset errors); it is also the decoder's prior. This is NOT the "Beyond Pauli Sim" curve of the figure, which
adds leakage and needs our simulator. Seeds default to crc32(path), so reruns are identical.

Writes results/pauli_sim_ler.csv. Then: python fit_ler.py --source pauli_sim
"""
import argparse
import csv
import pathlib
import time

import ccrepro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch")
    ap.add_argument("--basis", choices=["X", "Z"])
    ap.add_argument("--distance", type=int, choices=[3, 5])
    ap.add_argument("--rounds", type=int, nargs="*")
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=None, help="default: crc32 of the experiment path")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=pathlib.Path, default=ccrepro.ROOT / "results/pauli_sim_ler.csv")
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
        seed = ex.default_seed() if a.seed is None else a.seed
        dets, obs = ex.sample_pauli(a.shots, seed=seed)
        t = time.time()
        fail = ccrepro.decode_parallel(ex, dets, obs, workers=a.workers)
        dt = time.time() - t
        rows[e["path"]] = dict(path=e["path"], patch=e["patch"], distance=e["distance"], basis=e["basis"],
                               rounds=e["rounds"], shots_decoded=a.shots, failures=int(fail.sum()), ler=fail.mean(),
                               seed=seed, detector_fraction=dets.mean(), seconds=round(dt, 1))
        print(f"{e['path']:16s} {a.shots:6d} shots  Pauli-sim LER {fail.mean():.4f}  (hardware {float(e['our_ler']):.4f})"
              f"  detector fraction {dets.mean():.4f}", flush=True)

        a.out.parent.mkdir(exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values()))))
            w.writeheader(); w.writerows(sorted(rows.values(), key=lambda r: (r["patch"], r["basis"], int(r["rounds"]))))


if __name__ == "__main__":
    main()
