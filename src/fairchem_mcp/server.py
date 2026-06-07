"""The FastMCP server: registers all tools and resources over stdio.

All logging goes to stderr — the stdio transport reserves stdout for the MCP
protocol, so any stray print to stdout would corrupt the stream.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback

from mcp.server.fastmcp import FastMCP

from . import domain, introspect as introspect_mod
from .jobs import JobManager
from .session import Session

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("fairchem-mcp")

mcp = FastMCP("fairchem-mcp")
SESSION = Session()


def _err(exc: Exception) -> dict:
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Models & calculators
# ---------------------------------------------------------------------------

@mcp.tool()
def list_models() -> dict:
    """List FAIRChem pretrained models (UMA, eSEN, ...) if FAIRChem is installed."""
    return domain.list_models()


@mcp.tool()
def load_model(model: str = "uma-s-1p1", task: str = "omat", device: str = "auto") -> dict:
    """Load a FAIRChem model as an ASE calculator, kept resident in memory.

    task is the prediction domain: omat (inorganic), omol (molecules),
    oc20 (catalysis), odac (MOFs), omc (molecular crystals).
    Returns a calculator_id to pass to start_relaxation/start_md.
    """
    try:
        return domain.load_model(SESSION, model=model, task=task, device=device)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def attach_emt() -> dict:
    """Attach a fast EMT calculator (no GPU/model needed). Useful for quick tests
    on Cu/Ag/Au/Ni/Pd/Pt/Al/Pb and method development."""
    return domain.attach_emt(SESSION)


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

@mcp.tool()
def build_structure(spec: dict) -> dict:
    """Build an ASE structure from a declarative spec and register it.

    Examples:
      {"kind":"bulk","name":"Cu","crystalstructure":"fcc","a":3.6,"repeat":[2,2,2]}
      {"kind":"molecule","name":"H2O"}
      {"kind":"surface","name":"Pt","size":[2,2,4],"vacuum":10.0}
    Add "rattle": 0.1 to perturb positions (good for testing relaxation).
    """
    try:
        return domain.build_structure(SESSION, spec)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def load_structure(path: str, index: str = "-1") -> dict:
    """Load a structure from any ASE-readable file (cif, xyz, traj, POSCAR, ...)."""
    try:
        return domain.load_structure(SESSION, path, index=index)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


# ---------------------------------------------------------------------------
# Jobs: start, observe, steer
# ---------------------------------------------------------------------------

@mcp.tool()
def start_relaxation(
    structure_id: str,
    calculator_id: str,
    optimizer: str = "FIRE",
    fmax: float = 0.05,
    steps: int = 500,
    relax_cell: bool = False,
    step_delay: float = 0.0,
) -> dict:
    """Start a geometry optimization in the background; returns immediately.

    Poll get_status to watch progress, and steer() to change fmax, switch
    optimizer, pause or abort mid-run. Set relax_cell=True to also relax the cell.
    step_delay (seconds) throttles steps — useful with fast calculators so you
    have time to observe and react between steps.
    """
    try:
        return domain.start_relaxation(
            SESSION, structure_id, calculator_id,
            optimizer=optimizer, fmax=fmax, steps=steps, relax_cell=relax_cell,
            step_delay=step_delay,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def start_md(
    structure_id: str,
    calculator_id: str,
    ensemble: str = "NVT",
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    steps: int = 1000,
    step_delay: float = 0.0,
) -> dict:
    """Start molecular dynamics (NVT Langevin or NVE) in the background.

    step_delay (seconds) throttles steps so you can observe/react between them.
    """
    try:
        return domain.start_md(
            SESSION, structure_id, calculator_id,
            ensemble=ensemble, temperature_K=temperature_K,
            timestep_fs=timestep_fs, steps=steps, step_delay=step_delay,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def start_neb(
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
    """Start a nudged elastic band (reaction barrier) between two relaxed
    endpoints, in the background. nimages is the number of intermediate images.

    Steerable like a relaxation (pause/resume/abort/set_fmax/switch_optimizer)
    plus steer set_climb to enable the climbing image near convergence. When done,
    get_results returns the forward/reverse barrier and image energies.
    """
    try:
        return domain.start_neb(
            SESSION, initial_id, final_id, calculator_id,
            nimages=nimages, optimizer=optimizer, fmax=fmax, climb=climb,
            steps=steps, interpolation=interpolation, step_delay=step_delay,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def start_phonons(
    structure_id: str,
    calculator_id: str,
    supercell: list[int] | None = None,
    delta: float = 0.05,
    step_delay: float = 0.0,
) -> dict:
    """Start a phonon (finite-displacement) calculation in the background.

    Runs 1 + 6*natoms force evaluations; poll get_status for progress (done/total,
    fraction). When done, get_results returns gamma-point frequencies (THz), the
    count of imaginary modes, and a 'stable' flag. supercell defaults to [2,2,2].
    """
    try:
        return domain.start_phonons(
            SESSION, structure_id, calculator_id,
            supercell=tuple(supercell) if supercell else (2, 2, 2),
            delta=delta, step_delay=step_delay,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def start_minima_search(
    structure_id: str,
    calculator_id: str,
    n_minima: int = 5,
    kernel: str = "flooding",
    optimizer: str = "FIRE",
    fmax: float = 0.05,
    steps: int = 300,
    sigma: float = 0.4,
    amplitude: float = 1.0,
    energy_tol: float = 0.02,
    rmsd_tol: float = 0.1,
    patience: int = 6,
    step_delay: float = 0.0,
) -> dict:
    """Search for multiple distinct relaxed geometries (local minima of the PES).

    Repeatedly relaxes from the starting structure on a PES biased to repel the
    minima found so far, then polishes on the true PES. kernel is 'flooding'
    (Gaussian bumps; sigma in Å, amplitude in eV) or 'deflation' (inverse-distance
    poles). New minima are deduped by energy_tol (eV) + rmsd_tol (Å) and each is
    registered as a new structure. Poll get_status for n_found/target; get_results
    returns the distinct minima (structure_id + energy), sorted by energy.
    Steerable via steer abort/pause (and set_fmax/switch_optimizer mid-relaxation).
    """
    try:
        return domain.start_minima_search(
            SESSION, structure_id, calculator_id,
            n_minima=n_minima, kernel=kernel, optimizer=optimizer, fmax=fmax,
            steps=steps, sigma=sigma, amplitude=amplitude, energy_tol=energy_tol,
            rmsd_tol=rmsd_tol, patience=patience, step_delay=step_delay,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def get_status(job_id: str) -> dict:
    """Get a live snapshot of a job: status, step, energy, max_force, and a
    'trend' verdict (decreasing / plateaued / stuck / diverging) so you can
    decide whether to intervene."""
    try:
        return SESSION.get_job(job_id).status_dict()
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def get_trajectory(job_id: str, last_n: int = 20) -> dict:
    """Return the recent energy/force history of a job (last_n snapshots)."""
    try:
        return SESSION.get_job(job_id).trajectory(last_n=last_n)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def get_results(job_id: str) -> dict:
    """Return a job's final results, if any: NEB barrier/energies, or phonon
    gamma frequencies and stability. None until the job finishes."""
    try:
        return {"job_id": job_id, "result": SESSION.get_job(job_id).result}
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def steer(job_id: str, command: str, value: float | None = None,
          optimizer: str | None = None) -> dict:
    """Steer a running job mid-flight.

    command is one of: pause, resume, abort, set_fmax, switch_optimizer,
    set_temperature, set_climb. Use value for set_fmax / set_temperature,
    optimizer for switch_optimizer (FIRE / LBFGS / BFGS), and set_climb to enable
    the climbing image on a NEB (value 0/1, default on).
    """
    try:
        job = SESSION.get_job(job_id)
        params: dict = {}
        if command == "set_fmax":
            params["fmax"] = value
        elif command == "set_temperature":
            params["temperature_K"] = value
        elif command == "switch_optimizer":
            params["optimizer"] = optimizer
        elif command == "set_climb":
            params["climb"] = True if value is None else bool(value)
        return JobManager(SESSION).steer(job, command, **params)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


# ---------------------------------------------------------------------------
# Code awareness & escape hatch
# ---------------------------------------------------------------------------

@mcp.tool()
def introspect(target: str, live: bool = False) -> dict:
    """Introspect installed code or live objects.

    Static (live=False): dotted path, e.g.
      'fairchem.core.calculate.ase_calculator.FAIRChemCalculator'.
    Live (live=True): an expression over the session namespace, e.g. 'atoms'.
    A trailing '.' (e.g. 'atoms.' or 'ase.build.') lists members/completions.
    Returns signature, docstring and members.
    """
    try:
        return introspect_mod.introspect(target, SESSION.namespace, live=live)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def execute(code: str) -> dict:
    """Execute Python in the persistent session namespace (escape hatch).

    The namespace is shared with all tools: structures, calculators and jobs are
    bound by id (struct_1, calc_1, job_1) plus aliases (atoms, calc, job). Trusted
    local-dev use only. Captured stdout is returned.
    """
    import contextlib
    import io

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, SESSION.namespace)  # noqa: S102 - trusted escape hatch
        return {"ok": True, "stdout": buf.getvalue()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "stdout": buf.getvalue(), **_err(exc)}


@mcp.tool()
def inspect_expr(expr: str) -> dict:
    """Evaluate an expression in the session namespace and return its repr.

    e.g. 'atoms.get_potential_energy()' or 'np.linalg.norm(atoms.get_forces(),axis=1).max()'.
    """
    try:
        value = eval(expr, SESSION.namespace)  # noqa: S307 - trusted escape hatch
        return {"expr": expr, "type": type(value).__qualname__, "value": _jsonable(value)}
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        r = repr(value)
        return r if len(r) <= 1000 else r[:1000] + "…"


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("sim://models")
def models_resource() -> str:
    return json.dumps(domain.list_models(), indent=2)


@mcp.resource("sim://job/{job_id}/status")
def job_status_resource(job_id: str) -> str:
    try:
        return json.dumps(SESSION.get_job(job_id).status_dict(), indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(_err(exc), indent=2)


@mcp.resource("sim://job/{job_id}/trajectory")
def job_trajectory_resource(job_id: str) -> str:
    try:
        return json.dumps(SESSION.get_job(job_id).trajectory(last_n=0), indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(_err(exc), indent=2)


def main() -> None:
    log.info("starting fairchem-mcp server (stdio)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
