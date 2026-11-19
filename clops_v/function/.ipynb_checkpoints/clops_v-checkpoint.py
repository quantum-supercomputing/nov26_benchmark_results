# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals

"""
CLOPS_v benchmark (standalone)

Implements the original Circuit Layer Operations Per Second benchmark of
Wack et al., "Scale, Quality, and Speed: three key attributes to measure the
performance of near-term quantum computers", arXiv:2110.14108 (2021). It is
referred to as CLOPS_v to distinguish it from the later utility scale CLOPS_h
benchmark. The two use different circuit families and their values must not be
compared on a common scale.

This file is self contained. It requires only numpy, qiskit (>= 2.0) and
qiskit-ibm-runtime with Executor support (>= 0.50); running against a local
fake backend additionally needs qiskit-aer.

Protocol (paper section III.C):

1. M parameterized Quantum Volume templates are generated, each with
   D = log2(QV) layers of a random qubit permutation followed by floor(D/2)
   fully parameterized SU(4) blocks. The templates are compiled once.
2. Time is started.
3. Round 0 parameters for every template are drawn from a PRNG.
4. For each round k > 0, the measured output of template m in round k-1 seeds
   the PRNG that draws template m's round k parameters, so round k of a
   template cannot run until round k-1 of that template has returned.
5. Every circuit runs with S shots on the same qubits and with the same
   system settings (gate lengths, rep delay, ...) used for quantum volume.
6. Otherwise, circuits may be executed in any order or combination.
7. Time is stopped when all results have been received.

CLOPS_v = M * K * S * D / time_taken

This implementation submits work through the Executor primitive inside a
Session. Each round of a group of templates ("pipeline") is a single
QuantumProgram with one circuit item per template, carrying that template's
parameter values for the round. With P pipelines the templates are split into
P groups, each running its own chain of K dependent jobs; the chains are
interleaved so one group's client side work overlaps another group's QPU time.

The time is taken from server reported job timestamps: from the time the first
job started running (not when it was queued) to the time the last job finished.
Using server timestamps for both ends keeps the measurement on one clock.
"""

import hashlib
import importlib.metadata
import inspect
import math
import os
import platform
import sys
import time
import warnings
from datetime import datetime
from typing import List, Optional, Sequence

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.transpiler import Target
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Paper operating point, an unqualified "CLOPS_v" label is reserved for it
CANONICAL_TEMPLATES = 100
CANONICAL_PARAMETER_UPDATES = 10
CANONICAL_SHOTS = 100

# Parameters per SU(4) block of the paper's Fig. 1 decomposition
PARAMS_PER_SU4 = 15


def parameterized_su4(theta: Sequence) -> QuantumCircuit:
    """Return the fully parameterized SU(4) block of the CLOPS paper, Fig. 1

    The block is a KAK style decomposition with three two qubit gates and 15
    free Z rotation angles: an rz-sx-rz-sx-rz frame on each qubit, three
    entangling gates interleaved with rotations, and a closing rz-sx-rz-sx-rz
    frame on each qubit. It is written with CX as the entangling gate and is
    lowered to the target's native two qubit gate by the transpiler, which
    preserves all 15 parameters.

    Args:
        theta: sequence of 15 parameters (or values)

    Returns:
        a 2 qubit QuantumCircuit
    """
    if len(theta) != PARAMS_PER_SU4:
        raise ValueError(
            "SU(4) block takes %d parameters, got %d" % (PARAMS_PER_SU4, len(theta))
        )

    qc = QuantumCircuit(2)

    for qubit, offset in ((0, 0), (1, 3)):
        qc.rz(theta[offset], qubit)
        qc.sx(qubit)
        qc.rz(theta[offset + 1], qubit)
        qc.sx(qubit)
        qc.rz(theta[offset + 2], qubit)

    qc.cx(1, 0)
    qc.rz(theta[6], 0)
    qc.sx(1)
    qc.rz(theta[7], 1)
    qc.cx(1, 0)
    qc.rz(theta[8], 0)
    qc.sx(1)
    qc.cx(1, 0)

    for qubit, offset in ((0, 9), (1, 12)):
        qc.rz(theta[offset], qubit)
        qc.sx(qubit)
        qc.rz(theta[offset + 1], qubit)
        qc.sx(qubit)
        qc.rz(theta[offset + 2], qubit)

    return qc


def num_template_parameters(num_qubits: int) -> int:
    """Number of free parameters in a square parameterized QV template"""
    return PARAMS_PER_SU4 * (num_qubits // 2) * num_qubits


def create_parameterized_qv(
    num_qubits: int, rng: np.random.Generator, name: str = "clops_v"
) -> QuantumCircuit:
    """Create a square parameterized Quantum Volume template

    Each of the `num_qubits` layers is a random permutation of the qubits
    followed by floor(num_qubits / 2) parameterized SU(4) blocks on the permuted
    pairs. The permutations are fixed when the template is created; only the
    SU(4) parameters vary between executions.

    Args:
        num_qubits: width and depth D of the template
        rng: random generator used for the permutations
        name: circuit name, also used as the parameter vector name

    Returns:
        an untranspiled QuantumCircuit with measurements on all qubits
    """
    if num_qubits < 2:
        raise ValueError("A QV template needs at least 2 qubits")

    pairs_per_layer = num_qubits // 2
    theta = ParameterVector(name, num_template_parameters(num_qubits))

    qc = QuantumCircuit(num_qubits, name=name)
    block = 0
    for _ in range(num_qubits):
        perm = rng.permutation(num_qubits)
        for pair in range(pairs_per_layer):
            params = theta[PARAMS_PER_SU4 * block : PARAMS_PER_SU4 * (block + 1)]
            qubits = [int(perm[2 * pair]), int(perm[2 * pair + 1])]
            qc.compose(parameterized_su4(params), qubits, inplace=True)
            block += 1

    qc.measure_all()
    return qc


def chain_target(backend, qubits: Sequence[int]) -> Target:
    """Return a copy of the backend target restricted to `qubits`

    Every instruction keeps its backend properties (durations, errors, custom
    instructions such as ``measure_2``), but only on qargs that lie entirely
    inside `qubits`. All physical qubits stay in the target, so the layout can
    be applied, while layout, routing and translation can only use the selected
    qubits and the edges between them. This keeps every template, including its
    SWAPs, on the chosen qubit set.

    The restriction is made on the target rather than by passing a reduced
    ``coupling_map`` and ``basis_gates`` to the pass manager: real backends
    report non standard instructions that ``basis_gates`` rejects, and loose
    constraints would discard the backend's durations and error rates.
    """
    source = backend.target
    keep = set(int(q) for q in qubits)
    target = Target(
        description=source.description,
        num_qubits=source.num_qubits,
        dt=source.dt,
        granularity=source.granularity,
        min_length=source.min_length,
        pulse_alignment=source.pulse_alignment,
        acquire_alignment=source.acquire_alignment,
        qubit_properties=source.qubit_properties,
        concurrent_measurements=source.concurrent_measurements,
    )
    for name in source.operation_names:
        operation = source.operation_from_name(name)
        props = source[name]
        if inspect.isclass(operation) or props is None or None in props:
            # globally defined instruction, e.g. control flow
            target.add_instruction(operation, name=name)
            continue
        restricted = {
            qargs: prop for qargs, prop in props.items() if set(qargs) <= keep
        }
        if restricted:
            target.add_instruction(operation, restricted, name=name)
    return target


QUBIT_SET_SOURCE_USER = "user"
QUBIT_SET_SOURCE_REPORTED = "eplg_reported"
QUBIT_SET_SOURCE_TOPOLOGY = "auto:topology"


def _faulty(backend):
    """Return (faulty_qubits, faulty_edges) reported by the backend"""
    try:
        props = backend.properties()
    except Exception:  # noqa: BLE001 - backends without properties
        props = None
    if props is None:
        return set(), set()
    faulty_qubits = set(int(q) for q in props.faulty_qubits())
    faulty_edges = set()
    for gate in props.faulty_gates():
        if len(gate.qubits) == 2:
            faulty_edges.add(tuple(sorted(int(q) for q in gate.qubits)))
    return faulty_qubits, faulty_edges


def _good_edges(backend):
    """Undirected coupling map edges with faulty gates and qubits removed"""
    faulty_qubits, faulty_edges = _faulty(backend)
    edges = set()
    for q0, q1 in backend.coupling_map.get_edges():
        edge = tuple(sorted((int(q0), int(q1))))
        if edge in faulty_edges or q0 in faulty_qubits or q1 in faulty_qubits:
            continue
        edges.add(edge)
    return edges, faulty_qubits


def get_reported_chain(backend, width: int):
    """Return the backend reported layer fidelity chain ``lf_<width>``, or None

    IBM backends publish their best measured chains of each length in
    ``properties().general_qlists`` under the names ``lf_<n>``.
    """
    try:
        qlists = backend.properties().general_qlists
    except Exception:  # noqa: BLE001 - attribute or properties missing
        return None
    for qlist in qlists or []:
        if qlist["name"] == "lf_%d" % width:
            return [int(q) for q in qlist["qubits"]]
    return None


def longest_path_of_length(edges, width: int, faulty_qubits=(), max_steps=200000):
    """Deterministic topology only search for a connected chain of `width` qubits

    Starting qubits are tried in ascending (degree, index) order, neighbours in
    ascending index order. No error data is used. The search is bounded by
    `max_steps` because longest simple path search is exponential in general.

    Returns:
        a chain of `width` qubits, or the longest chain found
    """
    faulty = set(faulty_qubits)
    adj = {}
    for q0, q1 in edges:
        if q0 in faulty or q1 in faulty:
            continue
        adj.setdefault(q0, set()).add(q1)
        adj.setdefault(q1, set()).add(q0)

    best = []
    steps = 0
    for start in sorted(adj, key=lambda q: (len(adj[q]), q)):
        stack = [(start, [start], {start})]
        while stack:
            node, path, visited = stack.pop()
            steps += 1
            if len(path) > len(best):
                best = path
            if len(path) >= width:
                return path[:width]
            if steps > max_steps:
                break
            for neighbor in sorted(adj[node], reverse=True):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], visited | {neighbor}))
        if steps > max_steps:
            break
    return best


def validate_qubit_set(qubits, edges, faulty_qubits=()):
    """Raise ValueError unless `qubits` is a connected set of working qubits

    The set may have any shape, for example a 1D chain or a 2D grid; it only
    needs to be connected through working couplers, since routing is confined
    to the edges inside it. The order is kept as the initial layout.
    """
    qubits = list(qubits)
    if len(set(qubits)) != len(qubits):
        raise ValueError("Qubit set contains repeated qubits: %s" % qubits)
    bad = [q for q in qubits if q in set(faulty_qubits)]
    if bad:
        raise ValueError("Qubit set contains faulty qubits %s" % bad)

    members = set(qubits)
    adj = {q: set() for q in members}
    for q0, q1 in edges:
        if q0 in members and q1 in members:
            adj[q0].add(q1)
            adj[q1].add(q0)
    reached = {qubits[0]}
    frontier = [qubits[0]]
    while frontier:
        for neighbor in adj[frontier.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    if reached != members:
        raise ValueError(
            "Qubit set is not connected through working couplers; unreachable "
            "from qubit %d: %s" % (qubits[0], sorted(members - reached))
        )


def select_qubits(backend, num_qubits: int, qubits: Optional[Sequence[int]] = None):
    """Select the qubits to run on and record where they came from

    In order of precedence:

    1. An explicit set from the caller: `num_qubits` working qubits connected
       through working couplers, in any shape (for example the grid used for
       quantum volume). The given order is used as the initial layout.
    2. The layer fidelity chain of length `num_qubits` reported by the backend.
    3. A topology only chain search over working qubits and couplers.

    Returns:
        (qubits, source) with source one of "user", "eplg_reported" or
        "auto:topology"
    """
    edges, faulty_qubits = _good_edges(backend)

    if qubits is not None:
        chain = [int(q) for q in qubits]
        if len(chain) != num_qubits:
            raise ValueError(
                "Explicit qubits has length %d but log2(quantum_volume) is %d"
                % (len(chain), num_qubits)
            )
        validate_qubit_set(chain, edges, faulty_qubits)
        return chain, QUBIT_SET_SOURCE_USER

    reported = get_reported_chain(backend, num_qubits)
    if reported is not None and len(reported) >= num_qubits:
        return reported[:num_qubits], QUBIT_SET_SOURCE_REPORTED

    chain = longest_path_of_length(edges, num_qubits, faulty_qubits)
    if len(chain) < num_qubits:
        raise ValueError(
            "Insufficient connected qubits for a %d qubit chain, longest found was %d"
            % (num_qubits, len(chain))
        )
    return [int(q) for q in chain], QUBIT_SET_SOURCE_TOPOLOGY


def make_templates(
    backend,
    num_qubits: int,
    num_templates: int,
    qubits: Sequence[int],
    seed: Optional[int] = None,
    optimization_level: int = 3,
):
    """Generate and compile the parameterized QV templates

    Each template is compiled once, onto `qubits`, with routing confined to the
    edges between those qubits. Compilation happens before the timed part of
    the benchmark, as allowed by the protocol.

    Args:
        backend: target backend
        num_qubits: template width and depth D
        num_templates: number of templates M
        qubits: physical qubits, in the order used as the initial layout
        seed: seed for the permutations and the transpiler
        optimization_level: preset pass manager optimization level

    Returns:
        (templates, metadata) where templates is a list of ISA circuits and
        metadata is a list of per template dicts
    """
    if len(qubits) != num_qubits:
        raise ValueError(
            "Need %d qubits for a D=%d template, got %d"
            % (num_qubits, num_qubits, len(qubits))
        )

    rng = np.random.default_rng(seed)
    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        target=chain_target(backend, qubits),
        initial_layout=[int(q) for q in qubits],
        seed_transpiler=seed,
    )

    templates = []
    metadata = []
    for idx in range(num_templates):
        circuit = create_parameterized_qv(num_qubits, rng, name="t%d" % idx)
        isa = pass_manager.run(circuit)
        if isa.num_parameters != circuit.num_parameters:
            raise RuntimeError(
                "Compilation changed the parameter count of template %d (%d -> %d)"
                % (idx, circuit.num_parameters, isa.num_parameters)
            )
        ops = isa.count_ops()
        try:
            duration = isa.estimate_duration(backend.target, unit="s")
        except Exception:  # noqa: BLE001 - durations are informational only
            duration = None
        templates.append(isa)
        metadata.append(
            {
                "num_parameters": isa.num_parameters,
                "num_2q_gates": sum(
                    count
                    for name, count in ops.items()
                    if name in ("cz", "ecr", "cx", "rzz")
                ),
                "depth": isa.depth(),
                "estimated_duration_s": duration,
            }
        )

    return templates, metadata


def _rng_from_bits(bits: np.ndarray) -> np.random.Generator:
    """Seed a generator from a block of measured bits"""
    packed = np.packbits(np.ascontiguousarray(bits, dtype=bool)).tobytes()
    digest = hashlib.sha256(packed).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def initial_parameters(
    num_parameters: int, seed: Optional[int], template_index: int
) -> np.ndarray:
    """Round 0 parameters for one template, drawn from a seeded PRNG"""
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng([int(seed), int(template_index)])
    return rng.uniform(0, 2 * np.pi, num_parameters)


def next_parameters(measurements: np.ndarray, num_parameters: int) -> np.ndarray:
    """Parameters for the next round of a template, seeded by its last output

    The PRNG is seeded from every measured bit of every shot of the previous
    execution, so the new parameters cannot be produced before the previous
    result is available.

    Args:
        measurements: bool array of shape (shots, num_clbits) from the previous
            execution of the template
        num_parameters: number of parameters to draw

    Returns:
        array of `num_parameters` angles in [0, 2*pi)
    """
    return _rng_from_bits(measurements).uniform(0, 2 * np.pi, num_parameters)


def split_buckets(num_templates: int, pipelines: int) -> List[range]:
    """Split template indices into `pipelines` contiguous, near equal groups"""
    if pipelines < 1:
        raise ValueError("pipelines must be at least 1")
    if pipelines > num_templates:
        raise ValueError(
            "pipelines (%d) cannot exceed templates (%d)" % (pipelines, num_templates)
        )
    edges = [round(num_templates * b / pipelines) for b in range(pipelines + 1)]
    return [range(edges[b], edges[b + 1]) for b in range(pipelines)]


def clops_v_label(
    quantum_volume: int, templates: int, parameter_updates: int, shots: int
):
    """Label a CLOPS_v result with its operating point

    An unqualified "CLOPS_v" is reserved for the paper's operating point of
    M=100 templates, K=10 parameter updates and S=100 shots. The quantum
    volume is always part of the label since it sets the circuit size.
    """
    if (
        templates == CANONICAL_TEMPLATES
        and parameter_updates == CANONICAL_PARAMETER_UPDATES
        and shots == CANONICAL_SHOTS
    ):
        return "CLOPS_v(QV%d)" % quantum_volume
    return "CLOPS_v(QV%d, M=%d, K=%d, S=%d)" % (
        quantum_volume,
        templates,
        parameter_updates,
        shots,
    )


def _parse_timestamp(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def is_local_job(job) -> bool:
    """True for jobs run by the local (fake backend) runtime service

    Local jobs report timestamps, but all three are taken when the job object is
    created, so they cannot time a run.
    """
    from qiskit_ibm_runtime.fake_provider.local_runtime_job import LocalRuntimeJob

    return isinstance(job, LocalRuntimeJob)


def job_timestamps(job):
    """Return the server reported (running, finished) timestamps of a job

    Returns (None, None) when the job does not report usable timestamps, as for
    local (fake backend) jobs.
    """
    if is_local_job(job):
        return None, None
    try:
        stamps = job.metrics().get("timestamps", {})
    except Exception:  # noqa: BLE001 - not every job type has metrics()
        return None, None
    return _parse_timestamp(stamps.get("running")), _parse_timestamp(
        stamps.get("finished")
    )


def _chunk_spans(result):
    """Return [(start, stop), ...] for the executed chunks of a result"""
    try:
        timing = result.timing
    except AttributeError:
        return []
    return [(span.start, span.stop) for span in timing]


def _client_location() -> str:
    """Where the benchmark client runs: "qiskit-serverless" or "local"

    Qiskit Serverless sets ENV_JOB_ID_GATEWAY for every program it runs.
    """
    return "qiskit-serverless" if os.environ.get("ENV_JOB_ID_GATEWAY") else "local"


def _environment() -> dict:
    """Versions of the software the result was measured with"""
    versions = {}
    for name in ("qiskit", "qiskit-ibm-runtime", "numpy", "qiskit-serverless"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "client": _client_location(),
        "versions": versions,
    }


def _validate_quantum_volume(quantum_volume: int) -> int:
    num_qubits = int(round(math.log2(quantum_volume))) if quantum_volume > 0 else 0
    if quantum_volume < 4 or 2**num_qubits != quantum_volume:
        raise ValueError(
            "quantum_volume must be a power of two of at least 4, got %s"
            % quantum_volume
        )
    return num_qubits


class ClopsV:
    """Run the CLOPS_v benchmark through the Executor in a Session

    Example usage::

        from qiskit_ibm_runtime import QiskitRuntimeService
        from clops_v import ClopsV

        service = QiskitRuntimeService()
        bench = ClopsV(service, "ibm_boston", quantum_volume=512)
        bench.run()            # blocks until all K rounds have returned
        print(bench.result())  # {"label": "CLOPS_v(QV512)", "clops_v": ..., ...}

    Args:
        service: a QiskitRuntimeService, or None to run directly on `backend`
        backend: a backend name (resolved through `service`) or a backend
            object, e.g. a fake backend for local testing
        quantum_volume: the quantum volume of the device; the templates are
            log2(quantum_volume) qubits wide and deep
        templates: number of templates M, default 100
        parameter_updates: number of parameter rounds K, default 10
        shots: shots per circuit execution S, default 100
        pipelines: number of interleaved job chains P, default 5. The templates
            are split into P groups; each group runs its own chain of K jobs.
            The best value depends on the client's link to the cloud
        qubits: optional explicit set of log2(quantum_volume) connected
            qubits, of any shape, e.g. the grid used to measure quantum volume.
            The order is used as the initial layout. Defaults to the backend
            reported layer fidelity chain of length log2(quantum_volume)
        seed: optional seed for template generation, the transpiler and the
            round 0 parameters
    """

    def __init__(
        self,
        service,
        backend,
        quantum_volume: int,
        templates: int = CANONICAL_TEMPLATES,
        parameter_updates: int = CANONICAL_PARAMETER_UPDATES,
        shots: int = CANONICAL_SHOTS,
        pipelines: int = 5,
        qubits: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ):
        if isinstance(backend, str):
            if service is None:
                raise ValueError("A service is required to resolve a backend name")
            backend = service.backend(backend)
        self.backend = backend

        self.num_qubits = _validate_quantum_volume(quantum_volume)
        if templates < 1 or parameter_updates < 1 or shots < 1:
            raise ValueError("templates, parameter_updates and shots must be positive")

        self.quantum_volume = quantum_volume
        self.templates = templates
        self.parameter_updates = parameter_updates
        self.shots = shots
        self.seed = seed
        self.buckets = split_buckets(templates, pipelines)

        self.qubits, self.qubit_set_source = select_qubits(
            backend, self.num_qubits, qubits
        )
        self.circuits, self.template_metadata = make_templates(
            backend, self.num_qubits, templates, self.qubits, seed=seed
        )

        # rep_delay is not set, so the system default is used
        self.rep_delay = getattr(backend, "default_rep_delay", None)
        if self.rep_delay is None:
            try:
                self.rep_delay = backend.configuration().default_rep_delay
            except AttributeError:
                self.rep_delay = None

        self.jobs = None
        self.session_id = None
        self.client_timing = None
        self.chunk_spans = None

    @property
    def pipelines(self) -> int:
        return len(self.buckets)

    def _program(self, bucket, parameters):
        from qiskit_ibm_runtime import QuantumProgram

        program = QuantumProgram(shots=self.shots)
        for template in bucket:
            program.append_circuit_item(
                self.circuits[template], circuit_arguments=parameters[template]
            )
        return program

    def run(self):
        """Execute the benchmark, blocking until the last result is received

        Returns:
            self, so ``ClopsV(...).run().result()`` works
        """
        from qiskit_ibm_runtime import Executor, Session

        num_rounds = self.parameter_updates
        num_pipes = self.pipelines

        parameters = [
            initial_parameters(circuit.num_parameters, self.seed, idx)
            for idx, circuit in enumerate(self.circuits)
        ]

        jobs = [[None] * num_rounds for _ in range(num_pipes)]
        # chunk timing per job, kept so the breakdown does not re-download results
        chunk_spans = [[[] for _ in range(num_rounds)] for _ in range(num_pipes)]
        timing = {
            "prepare_s": [[0.0] * num_rounds for _ in range(num_pipes)],
            "submit_s": [[0.0] * num_rounds for _ in range(num_pipes)],
            "wait_s": [[0.0] * num_rounds for _ in range(num_pipes)],
        }

        client_start = time.perf_counter()
        session = Session(backend=self.backend)
        try:
            executor = Executor(mode=session)
            for k in range(num_rounds):
                for b, bucket in enumerate(self.buckets):
                    if k > 0:
                        t0 = time.perf_counter()
                        previous = jobs[b][k - 1].result()
                        timing["wait_s"][b][k - 1] = time.perf_counter() - t0
                        chunk_spans[b][k - 1] = _chunk_spans(previous)
                        t0 = time.perf_counter()
                        for item, template in enumerate(bucket):
                            parameters[template] = next_parameters(
                                previous[item]["meas"],
                                self.circuits[template].num_parameters,
                            )
                    else:
                        t0 = time.perf_counter()

                    program = self._program(bucket, parameters)
                    t1 = time.perf_counter()
                    timing["prepare_s"][b][k] = t1 - t0
                    jobs[b][k] = executor.run(program)
                    timing["submit_s"][b][k] = time.perf_counter() - t1

            for b in range(num_pipes):
                t0 = time.perf_counter()
                last = jobs[b][num_rounds - 1].result()
                timing["wait_s"][b][num_rounds - 1] = time.perf_counter() - t0
                chunk_spans[b][num_rounds - 1] = _chunk_spans(last)
        finally:
            self.session_id = getattr(session, "session_id", None)
            session.close()

        timing["client_total_s"] = time.perf_counter() - client_start
        self.jobs = jobs
        self.client_timing = timing
        self.chunk_spans = chunk_spans
        return self

    def result(self, allow_client_clock: bool = False) -> dict:
        """Compute the CLOPS_v result of a completed run

        Args:
            allow_client_clock: when the jobs do not report server timestamps
                (local fake backend runs), fall back to the client side wall
                clock for the run. Such a value is for testing only and is
                flagged in the result as ``timing_source="client"``.

        Returns:
            dict with the CLOPS_v value, its label, the operating point and a
            time breakdown
        """
        if self.jobs is None:
            raise RuntimeError("run() has not been called")

        first_round = [chain[0] for chain in self.jobs]
        last_round = [chain[-1] for chain in self.jobs]
        starts = [job_timestamps(job)[0] for job in first_round]
        ends = [job_timestamps(job)[1] for job in last_round]

        if all(s is not None for s in starts) and all(e is not None for e in ends):
            t_start = min(starts)
            t_end = max(ends)
            total_time = (t_end - t_start).total_seconds()
            timing_source = "server"
        elif allow_client_clock:
            warnings.warn(
                "Jobs report no server timestamps; using the client wall clock. "
                "This is not a valid CLOPS_v measurement.",
                stacklevel=2,
            )
            t_start = t_end = None
            total_time = self.client_timing["client_total_s"]
            timing_source = "client"
        else:
            raise RuntimeError(
                "Jobs report no server running/finished timestamps, so CLOPS_v "
                "cannot be timed. Pass allow_client_clock=True for local testing."
            )

        clops_v = (
            self.templates
            * self.parameter_updates
            * self.shots
            * self.num_qubits
            / total_time
        )

        return {
            "label": clops_v_label(
                self.quantum_volume, self.templates, self.parameter_updates, self.shots
            ),
            "clops_v": round(clops_v),
            "units": "QV layers per second",
            "total_time_s": total_time,
            "timing_source": timing_source,
            "t_start": t_start.isoformat() if t_start else None,
            "t_end": t_end.isoformat() if t_end else None,
            "backend_name": self.backend.name,
            "quantum_volume": self.quantum_volume,
            "num_qubits": self.num_qubits,
            "qubits": list(self.qubits),
            "qubit_set_source": self.qubit_set_source,
            "templates": self.templates,
            "parameter_updates": self.parameter_updates,
            "shots": self.shots,
            "pipelines": self.pipelines,
            "rep_delay": self.rep_delay,
            "execution_interface": "qiskit-ibm-runtime Executor (Session), client: %s"
            % _client_location(),
            "environment": _environment(),
            "session_id": self.session_id,
            "job_ids": [[job.job_id() for job in chain] for chain in self.jobs],
            "breakdown": self.breakdown(total_time),
        }

    def breakdown(self, total_time: Optional[float] = None) -> dict:
        """Approximate split of the run time, after paper section IV

        circuit execution and circuit delay are estimated from the compiled
        template durations and the rep delay; the remainder is attributed to
        runtime compilation and data transfer. QPU busy time comes from the
        Executor's chunk timing where available.
        """
        if self.jobs is None:
            raise RuntimeError("run() has not been called")

        durations = [m["estimated_duration_s"] for m in self.template_metadata]
        executions = self.parameter_updates * self.shots
        circuit_execution = (
            sum(durations) * executions
            if all(d is not None for d in durations)
            else None
        )
        circuit_delay = (
            self.rep_delay * self.templates * executions
            if self.rep_delay is not None
            else None
        )

        qpu_busy = 0.0
        num_spans = 0
        for chain in self.chunk_spans or []:
            for spans in chain:
                num_spans += len(spans)
                qpu_busy += sum((stop - start).total_seconds() for start, stop in spans)

        runtime_and_transfer = None
        if (
            total_time is not None
            and circuit_execution is not None
            and circuit_delay is not None
        ):
            runtime_and_transfer = total_time - circuit_execution - circuit_delay

        client = self.client_timing
        return {
            "circuit_execution_s": circuit_execution,
            "circuit_delay_s": circuit_delay,
            "runtime_compilation_and_data_transfer_s": runtime_and_transfer,
            "qpu_chunk_time_s": qpu_busy if num_spans else None,
            "client_prepare_s": float(np.sum(client["prepare_s"])),
            "client_submit_s": float(np.sum(client["submit_s"])),
            "client_wait_s": float(np.sum(client["wait_s"])),
            "client_total_s": client["client_total_s"],
        }
