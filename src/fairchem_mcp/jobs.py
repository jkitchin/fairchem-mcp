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
        self.atoms = atoms  # single structure (relax/md); None for multi-structure jobs
        self.kind = kind  # relax|md|neb|phonon|minima|eos|elastic|hull|dimer|sella|saddles
        self.config = dict(config)

        # Populated by the NEB / phonon / dimer runners.
        self.neb: Any = None
        self.images: list[Any] | None = None
        self.ph: Any = None
        self.dimer: Any = None  # MinModeAtoms for a dimer saddle search
        self.sella: Any = None  # Sella optimizer for an internal-coordinate TS search
        self.result: dict | None = None

        # Populated by the minima (multi-minimum search) runner.
        self.template: Any = None  # starting Atoms (positions = x0), no calc
        self.base_calc: Any = None  # the true-PES calculator to bias/polish on
        self.session: Any = None  # to register each found minimum as a structure
        self.hull_atoms: list[Any] | None = None  # convex-hull composition set
        self.hull_labels: list[str] | None = None  # one label per hull structure

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
            "eos": _run_eos,
            "elastic": _run_elastic,
            "hull": _run_hull,
            "dimer": _run_dimer,
            "sella": _run_sella,
            "saddles": _run_saddles,
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
                    "n_found", "target", "phase", "volume", "curvature",
                    "lowest_eigenvalue"):
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
# Equation of state (a one-variable scan: isotropic cell strain vs. energy)
# ---------------------------------------------------------------------------

def _finalize_eos(job: Job, status: str, volumes: list, energies: list) -> None:
    """Fit a Birch-Murnaghan-style EOS to (V, E) and report V0, E0, bulk modulus."""
    result: dict = {
        "volumes": [round(v, 6) for v in volumes],
        "energies": [round(e, 6) for e in energies],
        "n_points": len(volumes),
    }
    if len(volumes) >= 5:
        try:
            from ase.eos import EquationOfState
            from ase.units import kJ

            eos = EquationOfState(list(volumes), list(energies))
            v0, e0, B = eos.fit()
            result.update({
                "V0": round(float(v0), 6),
                "E0": round(float(e0), 6),
                "bulk_modulus_GPa": round(float(B / kJ * 1.0e24), 4),
            })
        except Exception as exc:  # noqa: BLE001
            result["fit_error"] = str(exc)
    job.result = result
    with job._lock:
        job.status = status


def _run_eos(job: Job) -> None:
    """Scan isotropic cell strain and (optionally) relax ions at each volume.

    Builds ``n_points`` copies of the structure with the cell scaled so the volume
    spans ``(1 ± strain_range)·V0``, evaluates the energy at each (relaxing the
    ionic positions at fixed cell first when ``relax_ions``), and fits an equation
    of state for the equilibrium volume, energy, and bulk modulus. ``abort`` /
    ``pause`` act between volume points. This is the canonical E–V teaching lab and
    a genuine workhorse for bulk crystals.
    """
    cfg = job.config
    template = job.template
    base = job.base_calc
    n_points = int(cfg.get("n_points", 11))
    strain_range = float(cfg.get("strain_range", 0.05))
    relax_ions = bool(cfg.get("relax_ions", False))
    optimizer_name = cfg.get("optimizer", "FIRE")
    fmax = float(cfg.get("fmax", 0.05))
    max_steps = int(cfg.get("steps", 200))
    step_delay = float(cfg.get("step_delay", 0.0))

    cell0 = np.asarray(template.get_cell(), dtype=float)
    if not all(bool(p) for p in template.pbc) or abs(float(np.linalg.det(cell0))) < 1e-8:
        raise ValueError(
            "EOS scan needs a 3-D periodic cell; this structure is not periodic "
            "in all directions (build a bulk crystal, not a molecule/cluster)"
        )

    # Linear scale factors whose cube (the volume ratio) is evenly spaced.
    vol_ratios = np.linspace(1.0 - strain_range, 1.0 + strain_range, n_points)
    scales = vol_ratios ** (1.0 / 3.0)

    with job._lock:
        job.status = "running"

    volumes: list[float] = []
    energies: list[float] = []
    for i, s in enumerate(scales):
        for cmd in job._drain_controls():
            if cmd.kind == "abort":
                _finalize_eos(job, "aborted", volumes, energies)
                return
            if cmd.kind == "pause" and job._wait_while_paused():
                _finalize_eos(job, "aborted", volumes, energies)
                return

        work = template.copy()
        work.set_cell(cell0 * s, scale_atoms=True)
        work.calc = base
        if relax_ions:
            opt = make_optimizer(optimizer_name, work)
            for _ in opt.irun(fmax=fmax, steps=max_steps):
                pass

        v = float(work.get_volume())
        e = float(work.get_potential_energy())
        volumes.append(v)
        energies.append(e)
        job.step = i + 1
        job.record_fields(
            energy=e, volume=v, done=i + 1, total=n_points,
            fraction=(i + 1) / n_points,
        )
        if step_delay:
            time.sleep(step_delay)

    _finalize_eos(job, "finished", volumes, energies)


# ---------------------------------------------------------------------------
# Elastic constants (the anisotropic analog of the EOS: stress vs. strain).
# Apply small strains along each of the 6 Voigt directions, read the resulting
# stress, and fit C_ij = dσ_i/dε_j. Voigt-Reuss-Hill averaging then gives the
# polycrystalline bulk/shear/Young's moduli plus a Born mechanical-stability check
# — the inputs an alloy-design loop optimizes against.
# ---------------------------------------------------------------------------

# 1 eV/Å³ in GPa (ase.units.GPa is 1 GPa expressed in eV/Å³; divide to convert).
def _voigt_strain_matrix(e: "np.ndarray") -> "np.ndarray":
    """3x3 symmetric strain tensor from a 6-vector in Voigt order (xx,yy,zz,yz,xz,xy)."""
    return np.array([
        [e[0], e[5] / 2.0, e[4] / 2.0],
        [e[5] / 2.0, e[1], e[3] / 2.0],
        [e[4] / 2.0, e[3] / 2.0, e[2]],
    ])


def _elastic_moduli(C: "np.ndarray") -> dict:
    """Voigt-Reuss-Hill moduli (GPa) and derived quantities from a 6x6 C (GPa)."""
    out: dict = {}
    c = C
    # Voigt averages (uniform-strain bound).
    K_v = ((c[0, 0] + c[1, 1] + c[2, 2]) + 2 * (c[0, 1] + c[0, 2] + c[1, 2])) / 9.0
    G_v = (
        (c[0, 0] + c[1, 1] + c[2, 2]) - (c[0, 1] + c[0, 2] + c[1, 2])
        + 3 * (c[3, 3] + c[4, 4] + c[5, 5])
    ) / 15.0
    out["bulk_modulus_voigt_GPa"] = round(float(K_v), 3)
    out["shear_modulus_voigt_GPa"] = round(float(G_v), 3)
    try:
        S = np.linalg.inv(c)  # compliance
        K_r = 1.0 / ((S[0, 0] + S[1, 1] + S[2, 2]) + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
        G_r = 15.0 / (
            4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
            + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
        )
        K_h, G_h = 0.5 * (K_v + K_r), 0.5 * (G_v + G_r)
        out["bulk_modulus_reuss_GPa"] = round(float(K_r), 3)
        out["shear_modulus_reuss_GPa"] = round(float(G_r), 3)
        out["bulk_modulus_GPa"] = round(float(K_h), 3)   # Hill (the usual headline)
        out["shear_modulus_GPa"] = round(float(G_h), 3)
        if 3 * K_h + G_h > 0:
            out["youngs_modulus_GPa"] = round(float(9 * K_h * G_h / (3 * K_h + G_h)), 3)
            out["poisson_ratio"] = round(float((3 * K_h - 2 * G_h) / (2 * (3 * K_h + G_h))), 4)
        if K_h != 0:
            out["pugh_ratio_G_over_K"] = round(float(G_h / K_h), 4)  # <0.57 ~ ductile
    except np.linalg.LinAlgError:
        out["fit_error"] = "stiffness matrix is singular (not invertible)"
    # Born stability: a stable crystal has a positive-definite C.
    eig = np.linalg.eigvalsh(c)
    out["C_eigenvalues_GPa"] = [round(float(x), 3) for x in eig]
    out["mechanically_stable"] = bool((eig > 0).all())
    return out


def _finalize_elastic(job: Job, status: str, C_gpa: "np.ndarray | None") -> None:
    result: dict = {}
    if C_gpa is not None:
        result["C_GPa"] = [[round(float(C_gpa[i, j]), 3) for j in range(6)] for i in range(6)]
        result.update(_elastic_moduli(C_gpa))
    job.result = result
    with job._lock:
        job.status = status


def _run_elastic(job: Job) -> None:
    """Measure the elastic stiffness tensor by straining the cell and reading stress.

    For each of the 6 Voigt strain directions, applies ``n_strains`` magnitudes in
    ``±max_strain`` (relaxing the ions at fixed strained cell when ``relax_ions``),
    records the stress, and least-squares fits C_ij = dσ_i/dε_j. Reports the full
    6x6 C (GPa), Voigt-Reuss-Hill bulk/shear/Young's moduli, Poisson and Pugh
    ratios, and a Born mechanical-stability verdict. ``abort``/``pause`` act between
    deformations. Needs a 3-D periodic crystal.
    """
    from ase.units import GPa

    cfg = job.config
    template = job.template
    base = job.base_calc
    n_strains = int(cfg.get("n_strains", 5))
    max_strain = float(cfg.get("max_strain", 0.01))
    relax_ions = bool(cfg.get("relax_ions", False))
    optimizer_name = cfg.get("optimizer", "FIRE")
    fmax = float(cfg.get("fmax", 0.05))
    max_steps = int(cfg.get("steps", 200))
    step_delay = float(cfg.get("step_delay", 0.0))

    cell0 = np.asarray(template.get_cell(), dtype=float)
    if not all(bool(p) for p in template.pbc) or abs(float(np.linalg.det(cell0))) < 1e-8:
        raise ValueError(
            "elastic-constant scan needs a 3-D periodic cell; this structure is "
            "not periodic in all directions (build a bulk crystal)"
        )
    if n_strains < 3:
        raise ValueError("n_strains must be >= 3 to fit a stress-strain slope")

    grid = np.linspace(-max_strain, max_strain, n_strains)
    eye = np.eye(3)
    total = 6 * n_strains

    with job._lock:
        job.status = "running"

    # stresses[j] -> list of (strain_magnitude, stress_voigt_6) for direction j.
    stresses: list[list[tuple]] = [[] for _ in range(6)]
    done = 0
    for j in range(6):
        for m in grid:
            for cmd in job._drain_controls():
                if cmd.kind == "abort":
                    _finalize_elastic(job, "aborted", None)
                    return
                if cmd.kind == "pause" and job._wait_while_paused():
                    _finalize_elastic(job, "aborted", None)
                    return

            e = np.zeros(6)
            e[j] = float(m)
            work = template.copy()
            work.set_cell(cell0 @ (eye + _voigt_strain_matrix(e)), scale_atoms=True)
            work.calc = base
            if relax_ions:
                opt = make_optimizer(optimizer_name, work)
                for _ in opt.irun(fmax=fmax, steps=max_steps):
                    pass
            sigma = np.asarray(work.get_stress(voigt=True), dtype=float)  # eV/Å³
            stresses[j].append((float(m), sigma))

            done += 1
            job.step = done
            job.record_fields(done=done, total=total, fraction=done / total)
            if step_delay:
                time.sleep(step_delay)

    # Fit C_ij = dσ_i/dε_j (column j from straining direction j), then -> GPa.
    C = np.zeros((6, 6))
    for j in range(6):
        ms = np.array([p[0] for p in stresses[j]])
        sig = np.array([p[1] for p in stresses[j]])  # (n_strains, 6)
        for i in range(6):
            slope = np.polyfit(ms, sig[:, i], 1)[0]
            C[i, j] = slope
    C = 0.5 * (C + C.T) / GPa  # symmetrize and convert eV/Å³ -> GPa
    _finalize_elastic(job, "finished", C)


# ---------------------------------------------------------------------------
# Convex hull of formation energies (phase stability for alloys).
# Given the energies of several compositions plus pure-element references, the
# formation energy per atom is E_f = (E - Σ n_i μ_i)/N. The lower convex hull of
# E_f vs. composition is the set of thermodynamically stable phases; a phase's
# distance above that hull (energy_above_hull) is how metastable it is. The hull
# math is a pure function of (composition, energy) so it can be unit-tested with
# synthetic energies; the job runner just supplies energies from the calculator.
# ---------------------------------------------------------------------------

def _counts(atoms: Any) -> dict:
    """Element -> atom count for a structure."""
    out: dict = {}
    for s in atoms.get_chemical_symbols():
        out[s] = out.get(s, 0) + 1
    return out


def _derive_references(entries: list, overrides: dict | None) -> dict:
    """Reference μ per element: per-atom energy of the lowest pure-element entry.

    ``overrides`` (element -> μ in eV/atom) wins over any pure entry found among
    the inputs. Raises ValueError if an element that appears in a mixed entry has
    no pure reference and no override.
    """
    refs: dict = {}
    for e in entries:
        counts = e["counts"]
        if len(counts) == 1:  # a pure-element entry
            (sym, n), = counts.items()
            mu = e["energy"] / n
            if sym not in refs or mu < refs[sym]:
                refs[sym] = mu
    if overrides:
        refs.update({k: float(v) for k, v in overrides.items()})
    needed = {s for e in entries for s in e["counts"]}
    missing = sorted(needed - set(refs))
    if missing:
        raise ValueError(
            "no reference energy for element(s) "
            f"{missing}: include a pure-element structure for each, or pass "
            "references={element: energy_per_atom}"
        )
    return refs


def _energy_above_hull(comps: "np.ndarray", form: "np.ndarray", tol: float = 1e-4) -> tuple:
    """Distance of each point above the lower convex hull of formation energies.

    ``comps`` is (m, d) composition fractions (rows sum to 1), ``form`` is the
    length-m formation energy per atom. For each point the hull energy is the
    lowest formation energy reachable as a convex combination of all points at the
    same composition — a small linear program — so it works for any number of
    elements. Returns (energy_above_hull[m], on_hull_bool[m]).
    """
    from scipy.optimize import linprog

    m = len(form)
    A_eq = np.vstack([comps.T, np.ones(m)])  # composition rows + (Σλ = 1)
    above = np.zeros(m)
    for j in range(m):
        b_eq = np.append(comps[j], 1.0)
        res = linprog(form, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        hull_e = float(res.fun) if res.success else float(form[j])
        above[j] = max(0.0, float(form[j]) - hull_e)
    on_hull = above <= tol
    return above, on_hull


def _hull_from_entries(entries: list, overrides: dict | None = None) -> dict:
    """Formation energies + convex-hull stability for a set of composition entries.

    ``entries`` is a list of ``{"label": str, "counts": {sym: n}, "energy": eV}``.
    Pure backbone of the convex-hull tool — no ASE/calculator dependency, so it is
    unit-testable with hand-written energies.
    """
    refs = _derive_references(entries, overrides)
    elements = sorted({s for e in entries for s in e["counts"]})
    m = len(entries)
    comps = np.zeros((m, len(elements)))
    form = np.zeros(m)
    for j, e in enumerate(entries):
        counts = e["counts"]
        n = sum(counts.values())
        for i, sym in enumerate(elements):
            comps[j, i] = counts.get(sym, 0) / n
        form[j] = (e["energy"] - sum(counts.get(s, 0) * refs[s] for s in elements)) / n
    above, on_hull = _energy_above_hull(comps, form)

    phases = []
    for j, e in enumerate(entries):
        phases.append({
            "label": e["label"],
            "composition": {elements[i]: round(float(comps[j, i]), 4)
                            for i in range(len(elements)) if comps[j, i] > 0},
            "n_atoms": sum(e["counts"].values()),
            "energy": round(float(e["energy"]), 6),
            "formation_energy_per_atom": round(float(form[j]), 6),
            "energy_above_hull_per_atom": round(float(above[j]), 6),
            "on_hull": bool(on_hull[j]),
        })
    return {
        "elements": elements,
        "references_eV_per_atom": {k: round(float(v), 6) for k, v in refs.items()},
        "phases": phases,
        "stable_phases": [p["label"] for p in phases if p["on_hull"]],
    }


def _run_hull(job: Job) -> None:
    """Evaluate several compositions and build their formation-energy convex hull.

    For each input structure (optionally relaxing ions, and the cell when
    ``relax_cell``) the total energy is computed, then formation energies per atom
    are referenced to the pure elements and the lower convex hull is built. Pure
    elements must be present among the inputs or supplied via ``references``.
    ``abort``/``pause`` act between structures. The classic alloy phase-stability
    picture and the objective an alloy-design loop screens against.
    """
    from ase.filters import FrechetCellFilter

    cfg = job.config
    structures = job.hull_atoms
    labels = job.hull_labels
    base = job.base_calc
    relax = bool(cfg.get("relax", False))
    relax_cell = bool(cfg.get("relax_cell", False))
    optimizer_name = cfg.get("optimizer", "FIRE")
    fmax = float(cfg.get("fmax", 0.05))
    max_steps = int(cfg.get("steps", 200))
    step_delay = float(cfg.get("step_delay", 0.0))
    overrides = cfg.get("references")

    with job._lock:
        job.status = "running"

    entries: list = []
    total = len(structures)
    for i, (atoms0, label) in enumerate(zip(structures, labels)):
        for cmd in job._drain_controls():
            if cmd.kind == "abort":
                _finalize_hull(job, "aborted", None)
                return
            if cmd.kind == "pause" and job._wait_while_paused():
                _finalize_hull(job, "aborted", None)
                return

        work = atoms0.copy()
        work.calc = base
        if relax:
            target = FrechetCellFilter(work) if relax_cell else work
            opt = make_optimizer(optimizer_name, target)
            for _ in opt.irun(fmax=fmax, steps=max_steps):
                pass
        energy = float(work.get_potential_energy())
        entries.append({"label": label, "counts": _counts(work), "energy": energy})

        job.step = i + 1
        job.record_fields(
            energy=energy, done=i + 1, total=total, fraction=(i + 1) / total,
        )
        if step_delay:
            time.sleep(step_delay)

    _finalize_hull(job, "finished", entries, overrides)


def _finalize_hull(job: Job, status: str, entries: list | None,
                   overrides: dict | None = None) -> None:
    result: dict = {}
    if entries:
        try:
            result = _hull_from_entries(entries, overrides)
        except ValueError as exc:
            # All energies are in hand but the references are incomplete: keep them
            # for the user and fail clearly rather than silently dropping the run.
            job.result = {"energies": [{"label": e["label"], "energy": e["energy"]}
                                       for e in entries]}
            with job._lock:
                job.status = "failed"
                job.error = str(exc)
            return
    job.result = result
    with job._lock:
        job.status = status


# ---------------------------------------------------------------------------
# Saddle points, route 1: the dimer method (Hessian-free, single-ended).
# ASE's MinModeTranslate is an Optimizer with irun(), so it drops straight into
# the steering loop. Only forces are needed — robust with noisy ML potentials.
# ---------------------------------------------------------------------------

def _plain_atoms(src: Any) -> Any:
    """A clean ASE Atoms copy (drops calculator and any MinModeAtoms wrapping)."""
    from ase import Atoms

    return Atoms(
        numbers=src.get_atomic_numbers(),
        positions=src.get_positions(),
        cell=src.get_cell(),
        pbc=src.get_pbc(),
    )


def _free_indices(atoms: Any) -> list[int]:
    """Indices of atoms not pinned by a FixAtoms constraint."""
    from ase.constraints import FixAtoms

    fixed: set[int] = set()
    for con in atoms.constraints:
        if isinstance(con, FixAtoms):
            fixed.update(int(i) for i in con.get_indices())
    return [i for i in range(len(atoms)) if i not in fixed]


def _record_dimer_step(job: Job, d: Any, converged: bool = False) -> None:
    energy = float(d.get_potential_energy())
    fmax = float(np.linalg.norm(d.get_forces(), axis=1).max())
    job.record_fields(
        energy=energy, max_force=fmax, curvature=float(d.get_curvature()),
        converged=bool(converged),
    )


def _finalize_dimer(job: Job, status: str) -> None:
    d = job.dimer
    curv = float(d.get_curvature())
    sid = None
    if job.session is not None:
        sid = job.session.add_structure(_plain_atoms(d))
    job.result = {
        "energy": round(float(d.get_potential_energy()), 6),
        "max_force": round(float(np.linalg.norm(d.get_forces(), axis=1).max()), 6),
        "curvature": round(curv, 6),
        "is_index1_saddle": bool(curv < 0.0),
        "structure_id": sid,
    }
    with job._lock:
        job.status = status


def _run_dimer(job: Job) -> None:
    """Drive an ASE dimer (min-mode following) saddle search, steerable.

    The dimer ascends the lowest-curvature mode and descends the rest using only
    forces. A negative converged curvature confirms an index-1 saddle (transition
    state). Steer set: abort / pause / set_fmax (switch_optimizer doesn't apply —
    the dimer translator is the optimizer).
    """
    from ase.mep import MinModeTranslate

    cfg = job.config
    fmax = float(cfg.get("fmax", 0.05))
    max_steps = int(cfg.get("steps", 200))
    step_delay = float(cfg.get("step_delay", 0.0))
    d = job.dimer

    with job._lock:
        job.status = "running"
    _record_dimer_step(job, d)

    local_step = 0
    while True:
        opt = MinModeTranslate(d, logfile=None)
        restart = False
        for converged in opt.irun(fmax=fmax, steps=max_steps - local_step):
            job.step += 1
            local_step += 1
            _record_dimer_step(job, d, converged=bool(converged))
            if step_delay:
                time.sleep(step_delay)
            if bool(converged):
                _finalize_dimer(job, "converged")
                return
            for cmd in job._drain_controls():
                if cmd.kind == "abort":
                    _finalize_dimer(job, "aborted")
                    return
                if cmd.kind == "pause":
                    if job._wait_while_paused():
                        _finalize_dimer(job, "aborted")
                        return
                elif cmd.kind == "set_fmax":
                    fmax = float(cmd.params["fmax"])
                    restart = True
            if restart:
                break
            if local_step >= max_steps:
                _finalize_dimer(job, "finished")
                return
        if not restart:
            _finalize_dimer(job, "finished")
            return


# ---------------------------------------------------------------------------
# Saddle points, route 2: Sella (partitioned rational-function optimization with
# internal coordinates and an approximate Hessian). Sella is an ASE Optimizer, so
# it drops straight into the irun() steering loop like a relaxation — but it
# climbs toward an order-k saddle instead of a minimum. The approximate Hessian it
# maintains lets us report the lowest eigenvalue live and classify the converged
# point by its negative-eigenvalue count.
# ---------------------------------------------------------------------------

def _sella_spectrum(opt: Any) -> tuple[Any, Any]:
    """(lowest_eigenvalue, n_negative) from Sella's approximate Hessian, or (None, None).

    Uses the cached eigenvalues when Sella has already diagonalized the Hessian
    (cheap); otherwise materializes and diagonalizes it once. Returns (None, None)
    before the first diagonalization, so callers must tolerate missing values.
    """
    try:
        H = opt.pes.get_H()
        if H is None:
            return None, None
        evals = getattr(H, "evals", None)
        if evals is None:
            evals = np.linalg.eigvalsh(H.asarray())
        evals = np.asarray(evals, dtype=float)
        if evals.size == 0:
            return None, None
        return float(evals.min()), int((evals < 0.0).sum())
    except Exception:  # noqa: BLE001 - Hessian not ready yet; report nothing
        return None, None


def _record_sella_step(job: Job, opt: Any, converged: bool = False) -> None:
    atoms = job.atoms
    energy = float(atoms.get_potential_energy())
    fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    low, _ = _sella_spectrum(opt)
    fields = {"energy": energy, "max_force": fmax, "converged": bool(converged)}
    if low is not None:
        fields["lowest_eigenvalue"] = round(low, 6)
    job.record_fields(**fields)


def _finalize_sella(job: Job, status: str) -> None:
    opt = job.sella
    atoms = job.atoms
    order = int(job.config.get("order", 1))
    low, n_neg = _sella_spectrum(opt)
    sid = None
    if job.session is not None:
        sid = job.session.add_structure(_plain_atoms(atoms))
    job.result = {
        "energy": round(float(atoms.get_potential_energy()), 6),
        "max_force": round(float(np.linalg.norm(atoms.get_forces(), axis=1).max()), 6),
        "order_requested": order,
        "lowest_eigenvalue": round(low, 6) if low is not None else None,
        "n_negative_eigenvalues": n_neg,
        "is_target_order_saddle": (n_neg == order) if n_neg is not None else None,
        "structure_id": sid,
    }
    try:
        opt.close()
    except Exception:  # noqa: BLE001 - best-effort IOContext cleanup
        pass
    with job._lock:
        job.status = status


def _run_sella(job: Job) -> None:
    """Drive a Sella saddle-point optimization, honoring control commands.

    Sella climbs toward an order-``order`` saddle (1 = transition state) using a
    partitioned rational-function step on an approximate, internal-coordinate
    Hessian. It is an ASE Optimizer, so the loop mirrors ``_run_relax``: abort /
    pause / set_fmax are honored between steps; switch_optimizer does not apply
    (Sella *is* the optimizer, and switching would discard its Hessian). The
    Sella object persists across set_fmax restarts so the Hessian is preserved.
    """
    cfg = job.config
    fmax = float(cfg.get("fmax", 0.05))
    max_steps = int(cfg.get("steps", 200))
    step_delay = float(cfg.get("step_delay", 0.0))
    opt = job.sella

    with job._lock:
        job.status = "running"
    _record_sella_step(job, opt)

    local_step = 0
    while True:
        restart = False
        for converged in opt.irun(fmax=fmax, steps=max_steps - local_step):
            job.step += 1
            local_step += 1
            _record_sella_step(job, opt, converged=bool(converged))
            if step_delay:
                time.sleep(step_delay)
            if bool(converged):
                _finalize_sella(job, "converged")
                return
            for cmd in job._drain_controls():
                if cmd.kind == "abort":
                    _finalize_sella(job, "aborted")
                    return
                if cmd.kind == "pause":
                    if job._wait_while_paused():
                        _finalize_sella(job, "aborted")
                        return
                elif cmd.kind == "set_fmax":
                    fmax = float(cmd.params["fmax"])
                    restart = True  # re-enter irun on the SAME opt (Hessian kept)
            if restart:
                break
            if local_step >= max_steps:
                _finalize_sella(job, "finished")
                return
        if not restart:
            _finalize_sella(job, "finished")
            return


# ---------------------------------------------------------------------------
# Saddle points, route 3: POUNCE find_saddles (multistart eigenvector following,
# full Hessian, explicit Morse-index classification). The active-atom positions
# are the optimization variables; the Hessian is built by central differences of
# the forces, so any ASE calculator works. POUNCE owns the multistart loop, so
# this job runs to completion on the worker thread (no mid-flight steering).
# ---------------------------------------------------------------------------

def _run_saddles(job: Job) -> None:
    """Enumerate index-k saddle points with POUNCE's ``find_saddles``.

    Variables are the Cartesian coordinates of the *active* atoms (default: those
    not held by a FixAtoms constraint); the rest stay frozen, which keeps the
    problem low-dimensional and free of the rigid translational/rotational zero
    modes that confuse a full-atom Hessian. fun/grad come straight from the
    calculator (∇E = −F); the Hessian is a central finite difference of the
    forces. Each found saddle is registered as a structure with its Morse index.
    """
    from pounce import find_saddles

    cfg = job.config
    atoms = job.template
    active = list(cfg["active_indices"])
    delta = float(cfg.get("fd_delta", 0.01))
    index = int(cfg.get("index", 1))
    n_saddles = int(cfg.get("n_saddles", 5))
    grad_tol = float(cfg.get("grad_tol", 1e-3))
    max_step = float(cfg.get("max_step", 0.2))
    dedup = float(cfg.get("dedup", 0.05))
    seed = int(cfg.get("seed", 0))
    local_max_iter = int(cfg.get("local_max_iter", 200))
    max_solves = cfg.get("max_solves")

    base = atoms.get_positions()
    aidx = np.asarray(active, dtype=int)
    x0 = base[aidx].ravel().copy()

    def _set_x(x: np.ndarray) -> None:
        pos = base.copy()
        pos[aidx] = np.asarray(x, float).reshape(-1, 3)
        atoms.set_positions(pos)

    def fun(x):
        _set_x(x)
        return float(atoms.get_potential_energy())

    def grad(x):
        _set_x(x)
        return -atoms.get_forces()[aidx].ravel()

    def hess(x):
        x = np.asarray(x, float)
        n = x.size
        H = np.zeros((n, n))
        for i in range(n):
            xp = x.copy(); xp[i] += delta
            xm = x.copy(); xm[i] -= delta
            H[:, i] = (grad(xp) - grad(xm)) / (2.0 * delta)
        return 0.5 * (H + H.T)

    with job._lock:
        job.status = "running"
    job.record_fields(energy=fun(x0), n_found=0, target=n_saddles, phase="solving")

    res = find_saddles(
        fun, x0, grad=grad, hess=hess, index=index, n_saddles=n_saddles,
        grad_tol=grad_tol, max_step=max_step, dedup=dedup, seed=seed,
        local_max_iter=local_max_iter,
        **({"max_solves": int(max_solves)} if max_solves else {}),
    )

    saddles = []
    for p in res.points:
        _set_x(p.x)
        sid = job.session.add_structure(_plain_atoms(atoms)) if job.session else None
        saddles.append({
            "structure_id": sid,
            "energy": round(float(p.f), 6),
            "morse_index": int(p.index),
            "grad_norm": round(float(p.grad_norm), 6),
            "eigenvalues": [round(float(e), 6) for e in p.eigvalues],
        })
    saddles.sort(key=lambda r: r["energy"])
    job.result = {
        "n_found": len(saddles),
        "solver_status": res.status,
        "n_solves": res.n_solves,
        "saddles": saddles,
    }
    job.record_fields(n_found=len(saddles), target=n_saddles, phase="done")
    with job._lock:
        job.status = "converged" if saddles else "finished"


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
