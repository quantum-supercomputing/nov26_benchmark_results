# Color-code memory on ibm_phoenix (v3, 2026-10-01): shots → LER reproducibility package

This package lets you check our hardware logical error rates (LERs) from the raw shots, using only public
tools (numpy, stim, tesseract-decoder; qiskit only to open the circuits). It has:

1. the circuits that ran on the device (Qiskit QPY);
2. the shots, as raw classified readout bits from IBM;
3. code that turns the shots into detection events (Stim);
4. code that decodes them with Tesseract and fits the error per round ε and Λ;
5. a Pauli-noise Stim circuit for each experiment, with code to sample it and decode the samples the same way.

**The experiment.** Memory experiments on the triangular 6.6.6 color code, in the X and Z bases, on IBM
ibm_phoenix (Nighthawk). There are four patches:

| Patch | Distance | Data qubits |
|---|---|---|
| `d5_L04` | d = 5 | 37 physical qubits in total |
| Patch A (`d3_TL`) | d = 3 | the d = 5 patch's top-left corner |
| Patch B (`d3_TR`) | d = 3 | its top-right corner |
| Patch C (`d3_BOT`) | d = 3 | its bottom corner |

Each patch was run at R = 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27 and 30 rounds, with 5000 shots per
circuit and one IBM job per circuit. That makes 96 experiments in total.

## Quick start

```bash
pip install -r requirements.txt
python fit_ler.py                                      # ε and Λ from our stored per-shot decodes
python reproduce_hardware.py --patch d3_TL --basis X   # re-do shots -> detectors -> decode for Patch A, X
python fit_ler.py --source hardware --plot             # fit your own decode; results/ler_vs_R_hardware.png
python simulate_pauli.py --distance 3                  # Pauli-only simulation of the same circuits
python fit_ler.py --source pauli_sim
```

`fit_ler.py` on the stored decodes prints the hardware ε of our figure (unweighted fit over R ≥ 5):

| ε | d = 5 | Patch A | Patch B | Patch C |
|---|---|---|---|---|
| X | 0.0114 | 0.0108 | 0.0132 | 0.0112 |
| Z | 0.0138 | 0.0095 | 0.0135 | 0.0119 |

Λ = mean ε of the three d = 3 corners ÷ ε of d = 5:
- 1.03 ± 0.03 (X) and 0.84 ± 0.03 (Z) for fits over R ≥ 5;
- 1.04 ± 0.03 and 0.86 ± 0.03 for fits over all rounds, R = 2–30 (`--rmin 2`). These are the headline numbers.

## Files

```
data/index.csv                one row per experiment: path, patch, basis, rounds, shots, IBM job id, our LER
data/<patch>_<basis>/R<rr>/
    circuit.qpy               the transpiled, scheduled circuit sent to ibm_phoenix (Qiskit QPY)
    measurements.npz          the shots: raw classified readout bits, (5000, num_measurements)
    circuit.stim              the same circuit in plain Stim with Pauli noise; defines DETECTORs + OBSERVABLE
    our_results.npz           our detection events, observable flips and per-shot Tesseract failures
    meta.json                 layout (physical qubits), transpiler settings, job id, decoder settings,
                              detector table (round, plaquette weight, ancilla pair), our LER
ccrepro.py                    the whole pipeline in ~150 lines (load, convert, decode, sample, fit)
reproduce_hardware.py         hardware: shots -> detectors -> Tesseract -> LER, checked against ours
simulate_pauli.py             Pauli-only sampling of circuit.stim, decoded the same way
fit_ler.py                    ε fits, Λ, plot
```

Bool arrays in the `.npz` files are bit-packed along axis 1. Unpack with
`np.unpackbits(z[k], axis=1)[:, :z[k + "_width"]].astype(bool)`, or use `ccrepro.unpack`.

## The pipeline, step by step

```python
import ccrepro
ex = ccrepro.load("d3_TL_X/R10")

qc = ex.qiskit_circuit()         # 1. the circuit that ran (120-qubit device register, 61 clbits)
m = ex.measurements()            # 2. shots: (5000, 61) bool, Qiskit clbit order
dets, obs = ex.to_detectors(m)   # 3. stim: circuit.stim.compile_m2d_converter().convert(...)
dem = ex.dem()                   #    circuit.stim.detector_error_model()
fail = ccrepro.decode(dem, dets, obs)   # 4. Tesseract (det_beam=10, beam_climbing, no_revisit_dets, merge_errors)
print(fail.mean())               # 0.0812, the same as ex.our_results()["logical_failure"].mean()

dets, obs = ex.sample_pauli(20000, seed=1)   # Pauli-only simulation of the same circuit
print(ccrepro.decode(dem, dets, obs).mean())
```

How the pieces fit together:
- **Measurement order.** Clbit *k* of the Qiskit circuit is measurement *k* of `circuit.stim`.
  `reproduce_hardware.py` checks that the counts agree.
- **Qubit labels.** Stim qubit indices are the physical qubit numbers of ibm_phoenix.
- **The Qiskit circuit** has already been through our pipeline:
  - transpiled to a fixed layout with no routing, and scheduled ALAP;
  - each ancilla measure + reset fused into IBM's `measure_reset_2` (calibration `auto_qec`);
  - XY4 dynamical decoupling on idle windows.

  `measure_reset_2` loads as an opaque instruction.
- **circuit.stim** is that same scheduled circuit, gate for gate. Each gate is followed by Pauli noise taken
  from IBM's calibration for that qubit or coupler:
  - depolarizing noise after one- and two-qubit gates;
  - X errors on reset and readout;
  - no idle noise.

  Its detector error model is the one we decoded the hardware with. We built it with our own tools; you only
  need Stim to use it.
- **Basis handling.** Stim's m2d converter applies the detector and observable definitions in the file,
  including the reference (noiseless) parities. Nothing basis-specific is needed on your side.

## What we checked

For all 96 experiments:
- `ex.to_detectors(ex.measurements())` reproduces our stored detection events and observable flips bit for bit.
- The DEM regenerated from `circuit.stim` gives the same per-shot Tesseract decisions as the DEM we used.
  The stored DEM text was rounded to 6 significant figures; spot-checked on d = 3 R = 10, d = 5 R = 5 and
  d = 5 R = 30.

We also re-ran this package's own code, from shots to LER:
- **Patch A, X basis:** all 12 round counts, every shot.
- **d = 5, Z basis:** R = 5 and R = 30, first 50 shots.

The per-shot Tesseract failures agree with ours exactly (agreement = 1.0), so the LER matches to every digit.
`reproduce_hardware.py` prints this agreement for every experiment.

## Decode cost

Tesseract, single core, Apple-silicon laptop:

| Experiment | Time per shot |
|---|---|
| d = 3, R = 10 | ~6 ms |
| d = 5, R = 5 | ~0.1 s |
| d = 5, R = 30 | ~3 s |

- **One d = 3 patch and basis** (12 round counts × 5000 shots): about 15 minutes.
- **All 72 d = 3 experiments:** about 2 CPU-hours.
- **All 24 d = 5 experiments** at 5000 shots: about 45 CPU-hours. Our own run took ~4.5 h on 10 cores.
  - `--workers N` splits the shots across N processes.
  - `--shots 200` gives a quick d = 5 check. Per-shot agreement on the first 200 shots is still exact; only
    the LER is noisier.

