"""The steering engine: background-threaded, mid-flight-controllable ASE runs.

ASE optimizers and MD integrators expose ``irun()`` — a generator that yields
once per step (optimizers yield a convergence flag). We drive that generator
ourselves on a worker thread so the MCP tool call returns immediately, recording
a snapshot and draining a control queue after every step. Because the ``Atoms``
object is shared, changing ``fmax`` or switching optimizer is just rebuilding the
driver around the same atoms and continuing — positions and cell carry over.

All optimizers/integrators are created with ``logfile=None``: the default
``'-'`` writes to stdout, which would corrupt the MCP stdio protocol.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optimizer registry
# ---------------------------------------------------------------------------

def _optimizer_classes() -> dict[str, Any]:
    from ase.optimize import BFGS, FIRE, LBFGS

    return {"FIRE": FIRE, "LBFGS": LBFGS, "BFGS": BFGS}


def make_optimizer(name: str, target: Any):
    classes = _optimizer_classes()
    key = name.upper()
    if key not in classes:
        raise ValueError(
            f"unknown optimizer {name!r}; choose from {sorted(classes)}"
        )
    return classes[key](target, logfile=None)


# ---------------------------------------------------------------------------
# Control commands
# ---------------------------------------------------------------------------

@dataclass
class Command:
    kind: str  # pause | resume | abort | set_fmax | switch_optimizer | set_temperature
    params: dict = field(default_factory=dict)


# Terminal vs active statuses.
_ACTIVE = {"pending", "running", "paused"}


class Job:
    """A single relaxation or MD run, controllable while it runs."""

    def __init__(self, job_id: str, atoms: Any, kind: str, config: dict):
        self.id = job_id
        self.atoms = atoms  # single structure (relax/md); None for neb/phonon/minima
        self.kind = kind  # "relax" | "md" | "neb" | "phonon" | "minima"
        self.config = dict(config)

        # Populated by the NEB / phonon runners.
        self.neb: Any = None
        self.images: list[Any] | None = None
        self.ph: Any = None
        self.result: dict | None = None

        # Populated by the minima (multi-minimum search) runner.
        self.template: Any = None  # starting Atoms (positions = x0), no calc
        self.base_calc: Any = None  # the true-PES calculator to bias/polish on
        self.session: Any = None  # to register each found minimum as a structure

        self.status = "pending"
        self.step = 0
        self.error: str | None = None
        self.history: list[dict] = []  # per-snapshot {step, energy, max_force, ...}

        self._lock = threading.Lock()
        self._controls: "queue.Queue[Command]" = queue.Queue()
        self._resume = threading.Event()
        self._resume.set()
        self._thread: threading.Thread | None = None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        runner = {
            "relax": _run_relax,
            "md": _run_md,
            "neb": _run_neb,
            "phonon": _run_phonon,
            "minima": _run_minima,
        }[self.kind]
        self._thread = threading.Thread(
            target=self._guard, args=(runner,), name=f"job-{self.id}", daemon=True
        )
        self._thread.start()

    def _guard(self, runner) -> None:
        try:
            runner(self)
        except Exception:  # noqa: BLE001 - surface any failure to the agent
            with self._lock:
                self.status = "failed"
                self.error = traceback.format_exc()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def is_active(self) -> bool:
        with self._lock:
            return self.status in _ACTIVE

    # -- control ----------------------------------------------------------
    def send(self, kind: str, **params) -> None:
        if kind == "resume":
            self._resume.set()
        self._controls.put(Command(kind, params))

    def _drain_controls(self) -> list[Command]:
        cmds: list[Command] = []
        while True:
            try:
                cmds.append(self._controls.get_nowait())
            except queue.Empty:
                break
        return cmds

    def _wait_while_paused(self) -> bool:
        """Block while paused. Return True if aborted during the pause."""
        with self._lock:
            self.status = "paused"
        self._resume.clear()
        while not self._resume.wait(timeout=0.1):
            for cmd in self._drain_controls():
                if cmd.kind == "abort":
                    return True
                if cmd.kind == "resume":
                    self._resume.set()
        with self._lock:
            if self.status == "paused":
                self.status = "running"
        return False

    # -- snapshots --------------------------------------------------------
    def record(self, **extra) -> None:
        atoms = self.atoms
        energy = float(atoms.get_potential_energy())
        fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        self.record_fields(energy=energy, max_force=fmax, **extra)

    def record_fields(self, **fields) -> None:
        """Append an arbitrary snapshot (used by NEB/phonon runners)."""
        snap = {"step": self.step, **fields}
        with self._lock:
            self.history.append(snap)

    def status_dict(self) -> dict:
        from .diagnostics import analyze_trend

        with self._lock:
            latest = self.history[-1] if self.history else {}
            history = list(self.history)
            status = self.status
            error = self.error
            result = self.result
        out = {
            "job_id": self.id,
            "kind": self.kind,
            "status": status,
            "step": latest.get("step", self.step),
            "energy": latest.get("energy"),
            "max_force": latest.get("max_force"),
            "converged": status == "converged",
            "trend": analyze_trend(history),
        }
        # Surface kind-specific fields from the latest snapshot when present.
        for key in ("barrier", "fraction", "done", "total", "temperature_K",
                    "n_found", "target", "phase"):
            if key in latest:
                out[key] = latest[key]
        if result is not None:
            out["result"] = result
        if error:
            out["error"] = error
        return out

    def trajectory(self, last_n: int = 20) -> dict:
        with self._lock:
            series = list(self.history)
        tail = series[-last_n:] if last_n else series
        return {
            "job_id": self.id,
            "n_snapshots": len(series),
            "snapshots": tail,
        }


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _build_relax_target(atoms: Any, relax_cell: bool):
    if relax_cell:
        from ase.filters import FrechetCellFilter

        return FrechetCellFilter(atoms)
    return atoms


def _run_relax(job: Job) -> None:
    """Drive a relaxation, honoring control commands between steps."""
    atoms = job.atoms
    cfg = job.config
    optimizer_name = cfg.get("optimizer", "FIRE")
    fmax = float(cfg.get("fmax", 0.05))
    max_steps = int(cfg.get("steps", 500))
    relax_cell = bool(cfg.get("relax_cell", False))
    step_delay = float(cfg.get("step_delay", 0.0))  # throttle to give time to react

    with job._lock:
        job.status = "running"

    # Record the starting point so the agent sees step 0.
    job.record()

    while True:
        target = _build_relax_target(atoms, relax_cell)
        opt = make_optimizer(optimizer_name, target)
        restart = False

        for converged in opt.irun(fmax=fmax, steps=max_steps - job.step):
            job.step += 1
            job.record(converged=bool(converged))
            if step_delay:
                time.sleep(step_delay)

            if bool(converged):
                with job._lock:
                    job.status = "converged"
                return

            for cmd in job._drain_controls():
                if cmd.kind == "abort":
                    with job._lock:
                        job.status = "aborted"
                    return
                if cmd.kind == "pause":
                    if job._wait_while_paused():  # aborted during pause
                        with job._lock:
                            job.status = "aborted"
                        return
                elif cmd.kind == "set_fmax":
                    fmax = float(cmd.params["fmax"])
                    restart = True  # re-enter irun so the new fmax takes effect
                elif cmd.kind == "switch_optimizer":
                    optimizer_name = cmd.params["optimizer"]
                    restart = True  # rebuild driver around the same atoms

            if restart:
                break

            if job.step >= max_steps:
                with job._lock:
                    job.status = "finished"  # step budget exhausted, not converged
                return

        if not restart:
            # irun exhausted its step budget without converging.
            with job._lock:
                if job.status == "running":
                    job.status = "finished"
            return


def _run_md(job: Job) -> None:
    """Drive NVT/NVE molecular dynamics, honoring control commands between steps."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet

    atoms = job.atoms
    cfg = job.config
    ensemble = cfg.get("ensemble", "NVT").upper()
    temperature_K = float(cfg.get("temperature_K", 300.0))
    timestep_fs = float(cfg.get("timestep_fs", 1.0))
    max_steps = int(cfg.get("steps", 1000))
    friction = float(cfg.get("friction", 0.01))
    step_delay = float(cfg.get("step_delay", 0.0))

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    if ensemble == "NVE":
        dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs, logfile=None)
    else:
        dyn = Langevin(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=friction,
            logfile=None,
        )

    with job._lock:
        job.status = "running"
    job.record(temperature_K=_instant_temperature(atoms))

    for _ in dyn.irun(steps=max_steps):
        job.step += 1
        job.record(temperature_K=_instant_temperature(atoms))
        if step_delay:
            time.sleep(step_delay)

        for cmd in job._drain_controls():
            if cmd.kind == "abort":
                with job._lock:
                    job.status = "aborted"
                return
            if cmd.kind == "pause":
                if job._wait_while_paused():
                    with job._lock:
                        job.status = "aborted"
                    return
            elif cmd.kind == "set_temperature":
                temperature_K = float(cmd.params["temperature_K"])
                if hasattr(dyn, "set_temperature"):
                    dyn.set_temperature(temperature_K=temperature_K)

    with job._lock:
        if job.status == "running":
            job.status = "finished"


def _instant_temperature(atoms: Any) -> float:
    from ase import units

    ekin = atoms.get_kinetic_energy() / len(atoms)
    return float(ekin / (1.5 * units.kB))


# ---------------------------------------------------------------------------
# NEB (reuses the optimizer steering loop; the optimizer target is the band)
# ---------------------------------------------------------------------------

def _record_neb(job: Job, converged: bool) -> None:
    neb = job.neb
    fmax = float(np.linalg.norm(neb.get_forces(), axis=1).max())
    energies = [float(img.get_potential_energy()) for img in job.images]
    barrier = float(max(energies) - energies[0])
    # energy == barrier so the diagnostics trend tracks barrier convergence.
    job.record_fields(
        energy=barrier, max_force=fmax, barrier=barrier,
        energies=energies, converged=bool(converged),
    )


def _finalize_neb(job: Job, status: str) -> None:
    try:
        from ase.mep import NEBTools

        nt = NEBTools(job.images)
        barrier, delta_e = nt.get_barrier()
        job.result = {
            "barrier": float(barrier),
            "delta_E": float(delta_e),
            "energies": [float(im.get_potential_energy()) for im in job.images],
            "n_images": len(job.images),
        }
    except Exception as exc:  # noqa: BLE001
        job.result = {"error": f"barrier analysis failed: {exc}"}
    with job._lock:
        job.status = status


def _run_neb(job: Job) -> None:
    """Drive a nudged elastic band, steerable like a relaxation.

    Adds one command on top of the relaxation steer set: ``set_climb`` toggles the
    climbing image (turn it on once the band is close to converged for an accurate
    saddle point).
    """
    cfg = job.config
    optimizer_name = cfg.get("optimizer", "FIRE")
    fmax = float(cfg.get("fmax", 0.05))
    max_steps = int(cfg.get("steps", 500))
    step_delay = float(cfg.get("step_delay", 0.0))
    neb = job.neb

    with job._lock:
        job.status = "running"
    _record_neb(job, converged=False)

    while True:
        opt = make_optimizer(optimizer_name, neb)
        restart = False

        for converged in opt.irun(fmax=fmax, steps=max_steps - job.step):
            job.step += 1
            _record_neb(job, converged=bool(converged))
            if step_delay:
                time.sleep(step_delay)

            if bool(converged):
                _finalize_neb(job, "converged")
                return

            for cmd in job._drain_controls():
                if cmd.kind == "abort":
                    _finalize_neb(job, "aborted")
                    return
                if cmd.kind == "pause":
                    if job._wait_while_paused():
                        _finalize_neb(job, "aborted")
                        return
                elif cmd.kind == "set_fmax":
                    fmax = float(cmd.params["fmax"])
                    restart = True
                elif cmd.kind == "switch_optimizer":
                    optimizer_name = cmd.params["optimizer"]
                    restart = True
                elif cmd.kind == "set_climb":
                    neb.climb = bool(cmd.params.get("climb", True))

            if restart:
                break

            if job.step >= max_steps:
                _finalize_neb(job, "finished")
                return

        if not restart:
            with job._lock:
                final = "finished" if job.status == "running" else job.status
            _finalize_neb(job, final)
            return


# ---------------------------------------------------------------------------
# Phonons (not an optimization: a fixed set of finite-displacement force evals)
# ---------------------------------------------------------------------------

# eV -> THz for phonon energies (E = h * nu).
_EV_TO_THZ = 241.79893


def _run_phonon(job: Job) -> None:
    """Drive an ASE phonon calculation displacement-by-displacement.

    We replicate ASE ``Phonons.run()`` so we can report progress (done/total) and
    honor abort/pause between displacements. After the forces are collected we
    read the force constants and report the gamma-point frequencies plus any
    imaginary (unstable) modes.
    """
    cfg = job.config
    step_delay = float(cfg.get("step_delay", 0.0))
    ph = job.ph
    delta = ph.delta

    with job._lock:
        job.status = "running"

    atoms_N = ph.atoms * ph.supercell
    atoms_N.calc = ph.calc
    natoms = len(ph.atoms)
    offset = natoms * ph.offset
    total = 1 + 6 * len(ph.indices)

    def _abort_or_pause() -> bool:
        for cmd in job._drain_controls():
            if cmd.kind == "abort":
                return True
            if cmd.kind == "pause" and job._wait_while_paused():
                return True
        return False

    # Equilibrium structure first.
    eq = ph._eq_disp()
    with ph.cache.lock(eq.name) as handle:
        if handle is not None:
            handle.save(ph.calculate(atoms_N, eq))
    job.step = 1
    job.record_fields(done=1, total=total, fraction=1 / total)

    pos = atoms_N.positions[offset : offset + natoms].copy()
    done = 1
    for a in ph.indices:
        for i in range(3):
            for sign in (-1, 1):
                if _abort_or_pause():
                    with job._lock:
                        job.status = "aborted"
                    return
                disp = ph._disp(a, i, sign)
                with ph.cache.lock(disp.name) as handle:
                    if handle is not None:
                        try:
                            atoms_N.positions[offset + a, i] = pos[a, i] + sign * delta
                            handle.save(ph.calculate(atoms_N, disp))
                        finally:
                            atoms_N.positions[offset + a, i] = pos[a, i]
                done += 1
                job.step = done
                job.record_fields(done=done, total=total, fraction=done / total)
                if step_delay:
                    time.sleep(step_delay)

    job.result = _phonon_results(ph)
    with job._lock:
        job.status = "finished"


def _phonon_results(ph: Any) -> dict:
    ph.read(acoustic=True)
    omega = np.asarray(ph.band_structure([[0, 0, 0]])[0], dtype=float)  # eV
    freqs_thz = omega * _EV_TO_THZ
    tol = 1e-3  # eV; modes below -tol are imaginary (unstable)
    n_imag = int((omega < -tol).sum())
    return {
        "gamma_frequencies_THz": [round(float(f), 4) for f in freqs_thz],
        "n_imaginary_modes": n_imag,
        "min_frequency_THz": round(float(freqs_thz.min()), 4),
        "stable": bool(n_imag == 0),
        "supercell": list(ph.supercell),
    }


# ---------------------------------------------------------------------------
# Minima search (deflation / flooding): repeated steerable relaxations on a
# biased PES, polished on the true PES, deduplicated into distinct minima.
# ---------------------------------------------------------------------------

def _record_minima_step(job: Job, work: Any, n_found: int, target: int, phase: str) -> None:
    energy = float(work.get_potential_energy())
    fmax = float(np.linalg.norm(work.get_forces(), axis=1).max())
    job.record_fields(
        energy=energy, max_force=fmax, n_found=n_found, target=target, phase=phase,
    )


def _perturb_free(atoms: Any, stdev: float, seed: int) -> None:
    """Kick the *unconstrained* atoms by a small random displacement, in place.

    Escape attempts restart from the starting geometry, which can coincide with a
    just-found minimum — exactly where a Gaussian flooding bump has zero gradient,
    so the optimizer would sit on top of it and never escape. A tiny kick breaks
    that symmetry. FixAtoms-constrained atoms (e.g. a frozen substrate) are left
    untouched so the kick perturbs only the degrees of freedom being explored.
    """
    if stdev <= 0:
        return
    from ase.constraints import FixAtoms

    rng = np.random.default_rng(seed)
    disp = rng.normal(scale=stdev, size=atoms.positions.shape)
    fixed: set[int] = set()
    for con in atoms.constraints:
        if isinstance(con, FixAtoms):
            fixed.update(int(i) for i in con.get_indices())
    if fixed:
        disp[sorted(fixed)] = 0.0
    atoms.positions += disp


def _finalize_minima(job: Job, status: str, found: list[dict]) -> None:
    ordered = sorted(found, key=lambda r: r["energy"])
    job.result = {
        "n_found": len(found),
        "minima": [
            {
                "structure_id": r["structure_id"],
                "energy": round(r["energy"], 6),
                "max_force": round(r["max_force"], 6),
            }
            for r in ordered
        ],
    }
    with job._lock:
        job.status = status


def _run_minima(job: Job) -> None:
    """Find multiple distinct relaxed geometries.

    Three methods (``kernel``):

    * ``"flooding"`` / ``"deflation"`` — relax from the starting point on a PES
      biased to repel the minima found so far (a ``DeflatedCalculator``), then
      polish on the true PES. Best when a fixed frame is enforced (adsorbate on a
      frozen slab, anchored conformer): a spatial bump cannot be evaded.
    * ``"basinhopping"`` — random-kick the current structure, relax on the true
      PES, accept by the Metropolis criterion (Wales & Doye). The right tool for
      *free* clusters, whose rigid-body rotation defeats a Cartesian bias.

    Novelty is judged by an energy gate plus a structural ``comparator``:
    ``"rmsd"`` (raw-coordinate RMSD; cheap, frame-dependent) or ``"fingerprint"``
    (sorted pairwise distances; rotation/translation/permutation invariant — use
    for free clusters/molecules). Stops at ``n_minima`` found, after ``patience``
    fruitless attempts, or after ``max_attempts``. Steerable: ``abort`` / ``pause``
    act between/within attempts; ``set_fmax`` / ``switch_optimizer`` on the
    current relaxation.
    """
    from .deflation import DeflatedCalculator, fingerprint, fingerprints_match, rmsd

    cfg = job.config
    base = job.base_calc
    template = job.template
    session = job.session

    method = cfg.get("kernel", "flooding")
    n_minima = int(cfg.get("n_minima", 5))
    patience = int(cfg.get("patience", 6))
    max_attempts = int(cfg.get("max_attempts", 6 * n_minima))
    bias_kw = {
        "sigma": float(cfg.get("sigma", 0.4)),
        "amplitude": float(cfg.get("amplitude", 1.0)),
        "eta": float(cfg.get("eta", 1.0)),
        "power": float(cfg.get("power", 2)),
    }
    energy_tol = float(cfg.get("energy_tol", 0.02))
    struct_tol = float(cfg.get("rmsd_tol", 0.1))  # Å; RMSD or per-distance tol
    comparator = cfg.get("comparator", "rmsd")
    escape_rattle = float(cfg.get("escape_rattle", 0.1))
    bh_step = float(cfg.get("bh_step", 0.4))
    bh_temperature = float(cfg.get("bh_temperature", 0.8))
    seed = int(cfg.get("seed", 0))
    optimizer_name = cfg.get("optimizer", "FIRE")
    fmax = float(cfg.get("fmax", 0.05))
    max_steps = int(cfg.get("steps", 300))
    step_delay = float(cfg.get("step_delay", 0.0))

    centers: list[np.ndarray] = []
    found: list[dict] = []

    with job._lock:
        job.status = "running"

    def _relax(work: Any, phase: str) -> str:
        """Drive one relaxation to convergence, steerable. Returns a status."""
        nonlocal optimizer_name, fmax
        local_step = 0
        while True:
            opt = make_optimizer(optimizer_name, work)
            restart = False
            for converged in opt.irun(fmax=fmax, steps=max_steps - local_step):
                job.step += 1
                local_step += 1
                _record_minima_step(job, work, len(found), n_minima, phase)
                if step_delay:
                    time.sleep(step_delay)
                if bool(converged):
                    return "converged"
                for cmd in job._drain_controls():
                    if cmd.kind == "abort":
                        return "aborted"
                    if cmd.kind == "pause":
                        if job._wait_while_paused():
                            return "aborted"
                    elif cmd.kind == "set_fmax":
                        fmax = float(cmd.params["fmax"])
                        restart = True
                    elif cmd.kind == "switch_optimizer":
                        optimizer_name = cmd.params["optimizer"]
                        restart = True
                if restart:
                    break
                if local_step >= max_steps:
                    return "finished"
            if not restart:
                return "finished"

    def _is_novel(energy: float, work: Any) -> bool:
        positions = work.get_positions()
        fp = fingerprint(work) if comparator == "fingerprint" else None
        for r in found:
            if abs(energy - r["energy"]) > energy_tol:
                continue
            if comparator == "fingerprint":
                if fingerprints_match(fp, r["fingerprint"], struct_tol):
                    return False
            elif rmsd(positions, r["positions"]) <= struct_tol:
                return False
        return True

    def _commit(work: Any, energy: float) -> None:
        positions = work.get_positions()
        max_force = float(np.linalg.norm(work.get_forces(), axis=1).max())
        sid = session.add_structure(work.copy()) if session is not None else None
        found.append({
            "energy": energy,
            "positions": positions,
            "fingerprint": fingerprint(work),
            "max_force": max_force,
            "structure_id": sid,
        })
        centers.append(positions.ravel().copy())
        job.record_fields(
            energy=energy, max_force=max_force, n_found=len(found),
            target=n_minima, phase=f"found#{len(found)}",
        )

    # -- basin-hopping ----------------------------------------------------
    if method == "basinhopping":
        rng = np.random.default_rng(seed)
        cur = template.copy()
        cur.calc = base
        if _relax(cur, "relax#0") == "aborted":
            _finalize_minima(job, "aborted", found)
            return
        e_cur = float(cur.get_potential_energy())
        if _is_novel(e_cur, cur):
            _commit(cur, e_cur)
        stagnant = 0
        for attempt in range(1, max_attempts):
            if len(found) >= n_minima:
                break
            trial = cur.copy()
            trial.calc = base
            _perturb_free(trial, bh_step, seed=seed + attempt)
            if _relax(trial, f"hop#{attempt}") == "aborted":
                _finalize_minima(job, "aborted", found)
                return
            e_new = float(trial.get_potential_energy())
            if _is_novel(e_new, trial):
                _commit(trial, e_new)
                stagnant = 0
            else:
                stagnant += 1
            # Metropolis acceptance moves the walker (exploration), independent
            # of whether the basin was new.
            if e_new <= e_cur or rng.random() < np.exp(
                -(e_new - e_cur) / max(bh_temperature, 1e-9)
            ):
                cur, e_cur = trial, e_new
            if stagnant >= patience:
                break
        _finalize_minima(job, "converged" if found else "finished", found)
        return

    # -- flooding / deflation --------------------------------------------
    stagnant = 0
    for attempt in range(max_attempts):
        work = template.copy()  # always start from x0
        if centers:
            # Escape phase: kick the free atoms (so we don't sit on a bump's
            # zero-gradient peak), relax on the biased PES, then polish on the
            # true one.
            _perturb_free(work, escape_rattle, seed=attempt)
            work.calc = DeflatedCalculator(base, centers, kernel=method, **bias_kw)
            if _relax(work, f"escape#{attempt}") == "aborted":
                _finalize_minima(job, "aborted", found)
                return
            work.calc = base
            phase_status = _relax(work, f"polish#{attempt}")
        else:
            work.calc = base
            phase_status = _relax(work, f"relax#{attempt}")
        if phase_status == "aborted":
            _finalize_minima(job, "aborted", found)
            return

        energy = float(work.get_potential_energy())
        if _is_novel(energy, work):
            _commit(work, energy)
            stagnant = 0
            if len(found) >= n_minima:
                break
        else:
            stagnant += 1
            if stagnant >= patience:
                break

    _finalize_minima(job, "converged" if found else "finished", found)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class JobManager:
    """Creates and starts jobs, enforcing a single-active-job policy.

    Only one relaxation/MD runs at a time so model/GPU access stays serialized.
    """

    def __init__(self, session) -> None:
        self.session = session

    def start(self, atoms: Any, kind: str, config: dict) -> Job:
        job = Job(self.session._new_id("job"), atoms, kind, config)
        return self.start_prepared(job)

    def start_prepared(self, job: Job) -> Job:
        """Launch an already-constructed job (NEB/phonon build their state first)."""
        active = self.session.active_job()
        if active is not None:
            raise RuntimeError(
                f"job {active.id} is still {active.status}; abort or wait for it "
                "before starting another (single-active-job policy)"
            )
        self.session.add_job(job)
        job.start()
        return job

    def steer(self, job: Job, command: str, **params) -> dict:
        valid = {
            "pause",
            "resume",
            "abort",
            "set_fmax",
            "switch_optimizer",
            "set_temperature",
            "set_climb",
        }
        if command not in valid:
            raise ValueError(f"unknown steer command {command!r}; choose from {sorted(valid)}")
        if not job.is_active():
            raise RuntimeError(
                f"job {job.id} is {job.status}; cannot steer a finished job"
            )
        job.send(command, **params)
        # Give the worker a moment to apply and update status (best-effort).
        time.sleep(0.05)
        return job.status_dict()
