# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Shots -> detectors -> Tesseract -> LER for the color-code hardware data, with only numpy + stim + tesseract.

    import ccrepro
    ex = ccrepro.load("d3_TL_X/R10")
    m = ex.measurements()                      # (shots, num_measurements) bool, raw classified bits from IBM
    dets, obs = ex.to_detectors(m)             # stim measurement -> detector conversion
    fail = ccrepro.decode(ex.dem(), dets, obs) # per-shot logical failure (Tesseract)
    fail.mean(), ex.our_results()["logical_failure"].mean()

    dets, obs = ex.sample_pauli(20000, seed=1) # Pauli-only simulation of the same circuit
    qc = ex.qiskit_circuit()                   # the scheduled circuit that ran on ibm_phoenix (needs qiskit)
"""
from __future__ import annotations

import csv
import json
import pathlib
import zlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"

# Tesseract settings used for every decode we report.
TESSERACT = dict(det_beam=10, beam_climbing=True, no_revisit_dets=True, merge_errors=True)


def unpack(path, key):
    """Bool arrays are stored bit-packed along axis 1, with the true width in '<key>_width'."""
    z = np.load(path)
    return np.unpackbits(z[key], axis=1)[:, : int(z[f"{key}_width"])].astype(bool)


def index():
    """One dict per experiment, from data/index.csv."""
    with open(DATA / "index.csv") as f:
        return list(csv.DictReader(f))


class Experiment:
    def __init__(self, path):
        self.path = DATA / path
        self.name = str(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.rounds, self.shots, self.distance, self.basis = (self.meta[k] for k in ("rounds", "shots", "distance", "basis"))

    def __repr__(self):
        return f"<{self.meta['patch']} {self.basis} R={self.rounds}, {self.shots} shots>"

    def qiskit_circuit(self):
        """The transpiled, scheduled circuit sent to the device (qiskit.QuantumCircuit)."""
        from qiskit import qpy
        with open(self.path / "circuit.qpy", "rb") as f:
            return qpy.load(f)[0]

    def stim_circuit(self):
        """Plain-Stim circuit with Pauli noise from calibration; defines detectors and the observable."""
        import stim
        return stim.Circuit((self.path / "circuit.stim").read_text())

    def measurements(self):
        """Raw classified readout bits, (shots, num_measurements), in Qiskit clbit order = Stim measurement order."""
        return unpack(self.path / "measurements.npz", "measurements")

    def to_detectors(self, measurements):
        """(detectors, observables) from raw measurements, using the circuit's DETECTOR / OBSERVABLE_INCLUDE."""
        conv = self.stim_circuit().compile_m2d_converter()
        return conv.convert(measurements=measurements, separate_observables=True)

    def dem(self):
        """Detector error model the hardware was decoded with."""
        return self.stim_circuit().detector_error_model()

    def sample_pauli(self, shots, seed=None):
        """(detectors, observables) sampled from the Pauli-noise circuit."""
        return self.stim_circuit().compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)

    def default_seed(self):
        return zlib.crc32(self.name.encode()) & 0x7FFFFFFF

    def our_results(self):
        """Our stored detectors, observables and per-shot Tesseract failures, for comparison."""
        p = self.path / "our_results.npz"
        return {"detectors": unpack(p, "detectors"), "observables": unpack(p, "observables"),
                "logical_failure": np.load(p)["logical_failure"]}


def load(path):
    return Experiment(path)


def decoder(dem):
    from tesseract_decoder import tesseract
    return tesseract.TesseractConfig(dem=dem, **TESSERACT).compile_decoder()


def decode(dem, detectors, observables, dec=None):
    """Per-shot logical failure: Tesseract's predicted observable flip != the measured one."""
    dec = dec or decoder(dem)
    pred = np.asarray(dec.decode_batch(detectors), bool)
    return (pred != observables).any(axis=1)


# Parallel decode: each worker process builds the decoder for an experiment once and decodes shot chunks.
_DEC = {}


def _decode_chunk(args):
    name, detectors, observables = args
    if name not in _DEC:
        _DEC[name] = decoder(load(name).dem())
    return decode(None, detectors, observables, _DEC[name])


def decode_parallel(ex, detectors, observables, workers=1, chunk=200):
    if workers <= 1:
        return decode(ex.dem(), detectors, observables)
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(ex.name, detectors[i:i + chunk], observables[i:i + chunk]) for i in range(0, len(detectors), chunk)]
    with ProcessPoolExecutor(workers) as pool:
        return np.concatenate(list(pool.map(_decode_chunk, jobs)))


def decay(R, b, eps):
    """LER after R rounds: per-round logical flip probability eps, SPAM offset b."""
    return (1 - b * (1 - 2 * eps) ** np.asarray(R, float)) / 2


def fit_eps(R, ler, rmin=5, saturation=0.49):
    """Unweighted least-squares fit of decay() over R >= rmin, LER < saturation. Returns (eps, eps_err, b)."""
    from scipy.optimize import curve_fit
    R, ler = np.asarray(R, float), np.asarray(ler, float)
    m = (R >= rmin) & (ler < saturation)
    if m.sum() < 3:
        raise ValueError(f"need >= 3 rounds with R >= {rmin} to fit (b, eps), have {m.sum()}")
    (b, eps), cov = curve_fit(decay, R[m], ler[m], p0=(1.0, 0.01), bounds=([0, 0], [2, 0.5]))
    return float(eps), float(np.sqrt(cov[1, 1])), float(b)
