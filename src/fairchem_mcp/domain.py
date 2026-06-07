"""Domain helpers: structure building, model loading, and job launching.

These wrap ASE and FAIRChem so the MCP tools stay thin. Everything except
:func:`load_model` works with plain ASE calculators (e.g. EMT), so the server is
fully usable for development and tests without FAIRChem/torch installed.
"""

from __future__ import annotations

from typing import Any

from .jobs import Job, JobManager
from .session import Session


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

def build_structure(session: Session, spec: dict) -> dict:
    """Build an ASE Atoms from a small declarative spec.

    spec examples::

        {"kind": "bulk", "name": "Cu", "crystalstructure": "fcc", "a": 3.6,
         "repeat": [2, 2, 2]}
        {"kind": "molecule", "name": "H2O"}
        {"kind": "surface", "symbol": "Pt", "indices": [1,1,1], "layers": 4,
         "vacuum": 10.0}
    """
    kind = spec.get("kind", "bulk")
    atoms = _build(kind, spec)

    rattle = spec.get("rattle")
    if rattle:
        atoms.rattle(float(rattle), seed=int(spec.get("seed", 42)))
    repeat = spec.get("repeat")
    if repeat and kind != "bulk":  # bulk handles repeat below to keep cell sane
        atoms = atoms.repeat(tuple(repeat))

    sid = session.add_structure(atoms)
    return _structure_info(sid, atoms)


def _build(kind: str, spec: dict) -> Any:
    if kind == "bulk":
        from ase.build import bulk

        kwargs = {
            k: spec[k]
            for k in ("crystalstructure", "a", "b", "c", "cubic", "orthorhombic")
            if k in spec
        }
        atoms = bulk(spec["name"], **kwargs)
        repeat = spec.get("repeat")
        if repeat:
            atoms = atoms.repeat(tuple(repeat))
        return atoms
    if kind == "molecule":
        from ase.build import molecule

        atoms = molecule(spec["name"])
        atoms.center(vacuum=float(spec.get("vacuum", 6.0)))
        return atoms
    if kind == "surface":
        from ase.build import fcc111, surface

        if "name" in spec and "indices" not in spec:
            # convenience: fcc111-style
            return fcc111(spec["name"], size=tuple(spec.get("size", (2, 2, 4))),
                          vacuum=float(spec.get("vacuum", 10.0)))
        from ase.build import bulk

        base = bulk(spec["symbol"])
        return surface(base, tuple(spec["indices"]), int(spec.get("layers", 4)),
                       vacuum=float(spec.get("vacuum", 10.0)))
    raise ValueError(f"unknown structure kind {kind!r}")


def load_structure(session: Session, path: str, index: str | int = -1) -> dict:
    from ase.io import read

    atoms = read(path, index=index)
    if isinstance(atoms, list):
        atoms = atoms[-1]
    sid = session.add_structure(atoms)
    return _structure_info(sid, atoms)


def _structure_info(sid: str, atoms: Any) -> dict:
    return {
        "structure_id": sid,
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "pbc": [bool(p) for p in atoms.pbc],
        "cell": atoms.cell.tolist(),
    }


# ---------------------------------------------------------------------------
# Calculators
# ---------------------------------------------------------------------------

def attach_emt(session: Session) -> dict:
    """Attach a fast EMT calculator (no GPU/model needed). Great for testing."""
    from ase.calculators.emt import EMT

    cid = session.add_calculator(EMT(), info={"kind": "emt"})
    return {"calculator_id": cid, "kind": "emt"}


def _ensure_lammps_loadable() -> None:
    """Best-effort fix for the macOS pip-LAMMPS + Homebrew-MPICH dylib mismatch.

    The PyPI ``lammps`` wheel links ``@rpath/libmpi.12.dylib`` and
    ``libpmpi.12.dylib``, which dyld cannot find unless they sit next to
    ``liblammps`` (or ``DYLD_FALLBACK_LIBRARY_PATH`` points at Homebrew's lib).
    If those libs are missing from the ``lammps`` package dir but present in a
    standard Homebrew location, symlink them in (idempotent, non-destructive).
    On other platforms, or if anything is already in place, this is a no-op.
    """
    import sys

    if sys.platform != "darwin":
        return
    import glob
    import importlib.util
    import os

    spec = importlib.util.find_spec("lammps")
    if spec is None or not spec.submodule_search_locations:
        return  # not installed; let the real import error surface to the caller
    pkg_dir = list(spec.submodule_search_locations)[0]

    needed = ["libmpi.12.dylib", "libpmpi.12.dylib"]
    missing = [n for n in needed if not os.path.exists(os.path.join(pkg_dir, n))]
    if not missing:
        return
    search = ["/opt/homebrew/lib", "/usr/local/lib"]
    search += sorted(glob.glob("/opt/homebrew/Cellar/mpich/*/lib"), reverse=True)
    for name in missing:
        for d in search:
            src = os.path.join(d, name)
            if os.path.exists(src):
                try:
                    os.symlink(src, os.path.join(pkg_dir, name))
                except OSError:
                    pass  # racy/read-only; the import below will report clearly
                break


def attach_lammps(
    session: Session,
    pair_style: str,
    pair_coeff: str | list,
    atom_types: dict | None = None,
    extra_cmds: list | None = None,
    lammps_header: list | None = None,
    log_file: str = "none",
    keep_alive: bool = True,
) -> dict:
    """Attach LAMMPS (via ``ase.calculators.lammpslib.LAMMPSlib``) as a calculator.

    LAMMPS is used as the force engine; ASE drives the dynamics, so the resulting
    calculator works with **every** steerable job here — ``start_md``,
    ``start_relaxation``, ``start_neb``, ``start_phonons``, ``start_eos_scan``,
    ``start_minima_search`` — under the same pause/abort/switch_optimizer control.

    ``pair_style`` and ``pair_coeff`` are raw LAMMPS commands (minus the keyword),
    e.g. ``pair_style="lj/cut 7.0"`` with ``pair_coeff="1 1 0.4 2.34"``, or
    ``pair_style="eam/alloy"`` with ``pair_coeff="* * Cu_u3.eam.alloy Cu"`` (a
    potential file the LAMMPS run can find). ``atom_types`` maps element symbols to
    LAMMPS type integers, e.g. ``{"Cu": 1}``. ``log_file="none"`` keeps LAMMPS off
    stdout (required by the MCP stdio protocol).
    """
    _ensure_lammps_loadable()
    from ase.calculators.lammpslib import LAMMPSlib

    coeffs = [pair_coeff] if isinstance(pair_coeff, str) else list(pair_coeff)
    lmpcmds = [f"pair_style {pair_style}"] + [f"pair_coeff {c}" for c in coeffs]
    if extra_cmds:
        lmpcmds += list(extra_cmds)

    kwargs: dict = {"lmpcmds": lmpcmds, "keep_alive": keep_alive, "log_file": log_file}
    if atom_types:
        kwargs["atom_types"] = atom_types
    if lammps_header:
        kwargs["lammps_header"] = lammps_header

    calc = LAMMPSlib(**kwargs)
    info = {"kind": "lammps", "pair_style": pair_style, "pair_coeff": coeffs}
    cid = session.add_calculator(calc, info=info)
    return {"calculator_id": cid, **info}


def list_models() -> dict:
    """List available FAIRChem pretrained models, if FAIRChem is installed."""
    try:
        from fairchem.core.calculate import pretrained_mlip
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"fairchem not importable: {exc}",
            "models": [],
        }
    return {"available": True, "models": list(pretrained_mlip.available_models)}


def load_model(
    session: Session,
    model: str = "uma-s-1p1",
    task: str = "omat",
    device: str = "auto",
) -> dict:
    """Load a FAIRChem pretrained model as an ASE calculator (kept resident)."""
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    resolved = _resolve_device(device)
    calc = FAIRChemCalculator.from_model_checkpoint(
        model, task_name=task, device=resolved
    )
    info = {"kind": "fairchem", "model": model, "task": task, "device": resolved}
    cid = session.add_calculator(calc, info=info)
    return {"calculator_id": cid, **info}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def _attach(session: Session, structure_id: str, calculator_id: str) -> Any:
    atoms = session.get_structure(structure_id)
    atoms.calc = session.get_calculator(calculator_id)
    return atoms


def start_relaxation(
    session: Session,
    structure_id: str,
    calculator_id: str,
    optimizer: str = "FIRE",
    fmax: float = 0.05,
    steps: int = 500,
    relax_cell: bool = False,
    step_delay: float = 0.0,
) -> dict:
    atoms = _attach(session, structure_id, calculator_id)
    config = {
        "optimizer": optimizer,
        "fmax": fmax,
        "steps": steps,
        "relax_cell": relax_cell,
        "step_delay": step_delay,
    }
    job = JobManager(session).start(atoms, "relax", config)
    return {"job_id": job.id, "kind": "relax", "config": config}


def start_md(
    session: Session,
    structure_id: str,
    calculator_id: str,
    ensemble: str = "NVT",
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    steps: int = 1000,
    friction: float = 0.01,
    step_delay: float = 0.0,
) -> dict:
    atoms = _attach(session, structure_id, calculator_id)
    config = {
        "ensemble": ensemble,
        "temperature_K": temperature_K,
        "timestep_fs": timestep_fs,
        "steps": steps,
        "friction": friction,
        "step_delay": step_delay,
    }
    job = JobManager(session).start(atoms, "md", config)
    return {"job_id": job.id, "kind": "md", "config": config}


def start_neb(
    session: Session,
    initial_id: str,
    final_id: str,
    calculator_id: str,
    nimages: int = 5,
    optimizer: str = "LBFGS",
    fmax: float = 0.05,
    climb: bool = False,
    steps: int = 500,
    interpolation: str = "idpp",
    step_delay: float = 0.0,
) -> dict:
    """Build and launch a steerable NEB between two endpoints.

    nimages is the number of *intermediate* images (the band has nimages + 2).
    The endpoints should already be relaxed. Use steer set_climb to enable the
    climbing image once the band is close to converged.
    """
    from ase.mep import NEB

    initial = session.get_structure(initial_id).copy()
    final = session.get_structure(final_id).copy()
    calc = session.get_calculator(calculator_id)

    images = [initial] + [initial.copy() for _ in range(nimages)] + [final]
    for img in images:
        img.calc = calc
    neb = NEB(images, climb=climb, allow_shared_calculator=True)
    neb.interpolate(method=interpolation)

    config = {
        "optimizer": optimizer,
        "fmax": fmax,
        "climb": climb,
        "steps": steps,
        "interpolation": interpolation,
        "step_delay": step_delay,
        "n_images": len(images),
    }
    job = Job(session._new_id("job"), None, "neb", config)
    job.neb = neb
    job.images = images
    JobManager(session).start_prepared(job)

    # Make the band reachable from the execute/inspect escape hatch.
    session.namespace["neb"] = neb
    session.namespace["images"] = images
    return {"job_id": job.id, "kind": "neb", "config": config}


def start_minima_search(
    session: Session,
    structure_id: str,
    calculator_id: str,
    n_minima: int = 5,
    kernel: str = "flooding",
    optimizer: str = "FIRE",
    fmax: float = 0.05,
    steps: int = 300,
    sigma: float = 0.4,
    amplitude: float = 1.0,
    eta: float = 1.0,
    power: float = 2.0,
    energy_tol: float = 0.02,
    rmsd_tol: float = 0.1,
    comparator: str = "rmsd",
    escape_rattle: float = 0.1,
    bh_step: float = 0.4,
    bh_temperature: float = 0.8,
    seed: int = 0,
    patience: int = 6,
    max_attempts: int | None = None,
    step_delay: float = 0.0,
) -> dict:
    """Search for multiple distinct relaxed geometries (local minima of the PES).

    ``kernel`` selects the search:

    * 'flooding' / 'deflation' — relax on a PES biased to repel the minima found
      so far (Gaussian bumps / inverse-distance poles), then polish on the true
      PES. Best for fixed-frame problems (adsorbate on a frozen slab, anchored
      conformer).
    * 'basinhopping' — random-kick + relax + Metropolis accept. Best for *free*
      clusters, whose rigid-body rotation defeats a spatial bias.

    Novelty is judged by energy (``energy_tol`` eV) plus ``comparator``: 'rmsd'
    (raw coords; frame-dependent) or 'fingerprint' (sorted pairwise distances;
    rotation/translation/permutation invariant — use for free clusters/molecules).
    Each new minimum is registered as a structure. Steerable via abort/pause (and
    set_fmax/switch_optimizer on the current relaxation).
    """
    template = session.get_structure(structure_id).copy()
    template.calc = None
    base = session.get_calculator(calculator_id)

    config = {
        "n_minima": n_minima,
        "kernel": kernel,
        "optimizer": optimizer,
        "fmax": fmax,
        "steps": steps,
        "sigma": sigma,
        "amplitude": amplitude,
        "eta": eta,
        "power": power,
        "energy_tol": energy_tol,
        "rmsd_tol": rmsd_tol,
        "comparator": comparator,
        "escape_rattle": escape_rattle,
        "bh_step": bh_step,
        "bh_temperature": bh_temperature,
        "seed": seed,
        "patience": patience,
        "max_attempts": max_attempts if max_attempts is not None else 6 * n_minima,
        "step_delay": step_delay,
    }
    job = Job(session._new_id("job"), None, "minima", config)
    job.template = template
    job.base_calc = base
    job.session = session
    JobManager(session).start_prepared(job)
    return {"job_id": job.id, "kind": "minima", "config": config}


def start_eos_scan(
    session: Session,
    structure_id: str,
    calculator_id: str,
    n_points: int = 11,
    strain_range: float = 0.05,
    relax_ions: bool = False,
    optimizer: str = "FIRE",
    fmax: float = 0.05,
    steps: int = 200,
    step_delay: float = 0.0,
) -> dict:
    """Scan isotropic cell strain and fit an equation of state (V0, E0, B).

    A one-variable lab: the cell volume is the single knob. Builds ``n_points``
    strained copies spanning ``(1 ± strain_range)·V0``, evaluates the energy at
    each (relaxing the ionic positions at fixed cell when ``relax_ions``), and
    fits a Birch-Murnaghan-style EOS for the equilibrium volume, energy, and bulk
    modulus (GPa). Needs a 3-D periodic structure (a bulk crystal). Steerable via
    abort/pause between volume points; poll get_status for done/total.
    """
    template = session.get_structure(structure_id).copy()
    template.calc = None
    base = session.get_calculator(calculator_id)

    config = {
        "n_points": n_points,
        "strain_range": strain_range,
        "relax_ions": relax_ions,
        "optimizer": optimizer,
        "fmax": fmax,
        "steps": steps,
        "step_delay": step_delay,
    }
    job = Job(session._new_id("job"), None, "eos", config)
    job.template = template
    job.base_calc = base
    job.session = session
    JobManager(session).start_prepared(job)
    return {"job_id": job.id, "kind": "eos", "config": config}


def start_saddle_search(
    session: Session,
    structure_id: str,
    calculator_id: str,
    displacement_vector: list | None = None,
    displace_magnitude: float = 0.1,
    dimer_separation: float = 0.01,
    fmax: float = 0.05,
    steps: int = 200,
    seed: int = 0,
    step_delay: float = 0.0,
) -> dict:
    """Find a transition state with the dimer method (Hessian-free, forces only).

    Starts from ``structure_id`` (typically a relaxed minimum, lightly nudged
    toward the saddle), ascends the lowest-curvature mode and descends the rest.
    A negative converged curvature confirms an index-1 saddle. Robust with noisy
    ML potentials since no Hessian is built.

    ``displacement_vector`` (length-3N list, Å) seeds the search direction; if
    omitted, a random kick of ``displace_magnitude`` Å is applied to the
    unconstrained atoms. Steerable via abort/pause/set_fmax. The MinModeAtoms
    object is bound as 'dimer' in the namespace for the execute/inspect hatch.
    """
    import numpy as np
    from ase.mep import DimerControl, MinModeAtoms

    atoms = session.get_structure(structure_id).copy()
    atoms.calc = session.get_calculator(calculator_id)
    n = len(atoms)

    from .jobs import _free_indices

    free = _free_indices(atoms)
    if displacement_vector is not None:
        dvec = np.asarray(displacement_vector, dtype=float).reshape(n, 3)
    else:
        rng = np.random.default_rng(seed)
        dvec = np.zeros((n, 3))
        dvec[free] = rng.normal(scale=displace_magnitude, size=(len(free), 3))

    control = DimerControl(
        initial_eigenmode_method="displacement",
        displacement_method="vector",
        logfile=None,
        dimer_separation=dimer_separation,
    )
    dimer = MinModeAtoms(atoms, control)
    # mask = which atoms participate in the dimer mode (the unconstrained ones);
    # also silences ASE's "couldn't figure out which atoms to displace" warning.
    mask = [i in set(free) for i in range(n)]
    dimer.displace(displacement_vector=dvec, mask=mask)

    config = {
        "fmax": fmax,
        "steps": steps,
        "dimer_separation": dimer_separation,
        "displace_magnitude": displace_magnitude,
        "seed": seed,
        "step_delay": step_delay,
    }
    job = Job(session._new_id("job"), None, "dimer", config)
    job.dimer = dimer
    job.session = session
    JobManager(session).start_prepared(job)

    session.namespace["dimer"] = dimer
    return {"job_id": job.id, "kind": "dimer", "config": config}


def start_sella_search(
    session: Session,
    structure_id: str,
    calculator_id: str,
    order: int = 1,
    fmax: float = 0.05,
    steps: int = 200,
    internal: bool = False,
    displacement_vector: list | None = None,
    displace_magnitude: float = 0.0,
    seed: int = 0,
    step_delay: float = 0.0,
) -> dict:
    """Find an order-``order`` saddle with Sella (RFO + approximate Hessian).

    Sella is an ASE Optimizer that climbs toward a saddle of the requested
    ``order`` (1 = transition state) using a partitioned rational-function step on
    an approximate Hessian it refines as it goes — so it converges in few force
    calls and drops straight into the steering loop (abort/pause/set_fmax; the
    Sella object is kept across set_fmax so its Hessian survives).

    Unlike the dimer it needs no product state and unlike POUNCE it needs no full
    Hessian. ``internal=True`` switches to automatic internal (bond/angle/dihedral)
    coordinates, which helps for molecules. ``displacement_vector`` (length-3N list,
    Å) or a random ``displace_magnitude`` kick on the unconstrained atoms seeds the
    search away from the minimum. The Sella object is bound as 'sella' in the
    namespace for the execute/inspect hatch. Requires `sella` (pip install sella).
    """
    from sella import Sella

    atoms = session.get_structure(structure_id).copy()
    atoms.calc = session.get_calculator(calculator_id)
    n = len(atoms)

    from .jobs import _free_indices

    free = _free_indices(atoms)
    if displacement_vector is not None:
        import numpy as np

        atoms.set_positions(
            atoms.get_positions()
            + np.asarray(displacement_vector, dtype=float).reshape(n, 3)
        )
    elif displace_magnitude:
        import numpy as np

        rng = np.random.default_rng(seed)
        pos = atoms.get_positions()
        pos[free] += rng.normal(scale=displace_magnitude, size=(len(free), 3))
        atoms.set_positions(pos)

    # logfile=None keeps Sella off stdout (the stdio transport reserves it).
    sella = Sella(atoms, order=order, internal=internal, logfile=None)

    config = {
        "order": order,
        "fmax": fmax,
        "steps": steps,
        "internal": internal,
        "displace_magnitude": displace_magnitude,
        "seed": seed,
        "step_delay": step_delay,
    }
    job = Job(session._new_id("job"), atoms, "sella", config)
    job.sella = sella
    job.session = session
    JobManager(session).start_prepared(job)

    session.namespace["sella"] = sella
    return {"job_id": job.id, "kind": "sella", "config": config}


def start_pounce_saddles(
    session: Session,
    structure_id: str,
    calculator_id: str,
    active_indices: list | None = None,
    index: int = 1,
    n_saddles: int = 5,
    fd_delta: float = 0.01,
    grad_tol: float = 1e-3,
    max_step: float = 0.2,
    dedup: float = 0.05,
    local_max_iter: int = 200,
    max_solves: int | None = None,
    seed: int = 0,
    displace_magnitude: float = 0.0,
) -> dict:
    """Enumerate index-``index`` saddle points with POUNCE's ``find_saddles``.

    Multistart eigenvector following (Cerjan-Miller) with a full Hessian and
    explicit Morse-index classification — complements the single-ended dimer when
    you want *several* distinct saddles labeled by index. The optimization
    variables are the Cartesian coordinates of ``active_indices`` (default: atoms
    not held by FixAtoms); the rest stay frozen. fun/grad come from the calculator
    (∇E = −F); the Hessian is a central finite difference of forces (``fd_delta``).

    For float32 ML potentials (e.g. UMA), loosen ``grad_tol`` (default 1e-3) and
    prefer a float64 model — curvature/Morse counting is noise-sensitive. POUNCE
    owns the multistart loop, so this job runs to completion (abort is not honored
    mid-solve).
    """
    atoms = session.get_structure(structure_id).copy()
    atoms.calc = session.get_calculator(calculator_id)

    from .jobs import _free_indices

    if active_indices is None:
        active_indices = _free_indices(atoms)
    if not active_indices:
        raise ValueError(
            "no active atoms to search over (all atoms are fixed); pass "
            "active_indices explicitly"
        )

    if displace_magnitude:
        import numpy as np

        rng = np.random.default_rng(seed)
        pos = atoms.get_positions()
        aidx = np.asarray(active_indices, dtype=int)
        pos[aidx] += rng.normal(scale=displace_magnitude, size=(len(aidx), 3))
        atoms.set_positions(pos)

    config = {
        "active_indices": list(active_indices),
        "index": index,
        "n_saddles": n_saddles,
        "fd_delta": fd_delta,
        "grad_tol": grad_tol,
        "max_step": max_step,
        "dedup": dedup,
        "local_max_iter": local_max_iter,
        "max_solves": max_solves,
        "seed": seed,
    }
    job = Job(session._new_id("job"), None, "saddles", config)
    job.template = atoms
    job.session = session
    JobManager(session).start_prepared(job)
    return {"job_id": job.id, "kind": "saddles", "config": config}


def start_phonons(
    session: Session,
    structure_id: str,
    calculator_id: str,
    supercell: list | tuple = (2, 2, 2),
    delta: float = 0.05,
    step_delay: float = 0.0,
) -> dict:
    """Build and launch a phonon (finite-displacement) calculation.

    Runs 1 + 6*natoms force evaluations in the background, reporting progress.
    On completion, reports gamma-point frequencies and any imaginary (unstable)
    modes. The Phonons object is bound as 'ph' in the namespace for band
    structure / DOS via the execute escape hatch.
    """
    import os
    import tempfile

    from ase.phonons import Phonons

    atoms = session.get_structure(structure_id)
    calc = session.get_calculator(calculator_id)
    cache_dir = tempfile.mkdtemp(prefix="fairchem_mcp_phonon_")
    ph = Phonons(
        atoms, calc, supercell=tuple(supercell), delta=delta,
        name=os.path.join(cache_dir, "phonon"),
    )

    config = {
        "supercell": list(supercell),
        "delta": delta,
        "step_delay": step_delay,
        "cache_dir": cache_dir,
        "n_displacements": 1 + 6 * len(ph.indices),
    }
    job = Job(session._new_id("job"), None, "phonon", config)
    job.ph = ph
    JobManager(session).start_prepared(job)

    session.namespace["ph"] = ph
    return {"job_id": job.id, "kind": "phonon", **config}
