# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Qiskit Serverless entrypoint for the CLOPS_v benchmark.

This runs the unmodified ``clops_v.py`` remotely, so the benchmark client sits
inside IBM Quantum's cloud rather than on the submitting machine. The
submitting machine can disconnect while it runs.

Arguments, passed to ``function.run(...)``:
    backend_name (required), quantum_volume (required), templates (100),
    parameter_updates (10), shots (100), pipelines (5), qubits (None),
    seed (None)

The saved result is the ``ClopsV.result()`` dictionary. On failure it is
``{"error": ..., "traceback": ...}`` instead.
"""

import traceback
from datetime import datetime

from qiskit_serverless import get_arguments, get_runtime_service, save_result

from clops_v import ClopsV


def _jsonable(value):
    """Make the result JSON serializable; save_result drops anything that isn't"""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, range)):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    if hasattr(value, "tolist"):  # numpy array
        return value.tolist()
    return str(value)


def main():
    args = get_arguments()
    try:
        bench = ClopsV(
            get_runtime_service(),
            args["backend_name"],
            int(args["quantum_volume"]),
            templates=int(args.get("templates", 100)),
            parameter_updates=int(args.get("parameter_updates", 10)),
            shots=int(args.get("shots", 100)),
            pipelines=int(args.get("pipelines", 5)),
            qubits=args.get("qubits"),
            seed=args.get("seed"),
        )
        result = bench.run().result()
    except Exception as exc:  # noqa: BLE001 - report any failure to the caller
        result = {
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc()[-3000:],
            "arguments": args,
        }
    result = _jsonable(result)
    if not save_result(result):
        print("save_result failed; result follows:", result)


main()
