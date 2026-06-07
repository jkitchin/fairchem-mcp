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
