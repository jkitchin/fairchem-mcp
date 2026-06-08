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


@mcp.tool()
def attach_lammps(
    pair_style: str,
    pair_coeff: str,
    atom_types: dict | None = None,
    extra_cmds: list[str] | None = None,
    log_file: str = "none",
) -> dict:
    """Attach LAMMPS as a calculator (via ASE LAMMPSlib); LAMMPS computes the
    energy/forces, ASE drives the dynamics.

    Works with every steerable job (start_md, start_relaxation, start_neb,
    start_phonons, start_eos_scan, start_elastic_scan, start_convex_hull,
    start_minima_search). pair_style/pair_coeff are
    raw LAMMPS commands minus the keyword, e.g. pair_style="lj/cut 7.0",
    pair_coeff="1 1 0.4 2.34"; or pair_style="eam/alloy",
    pair_coeff="* * Cu_u3.eam.alloy Cu". atom_types maps symbols to LAMMPS type
    ints, e.g. {"Cu": 1}. Needs the `lammps` package (see README install notes;
    the macOS Homebrew-MPICH dylib fix is applied automatically).
    """
    try:
        return domain.attach_lammps(
            SESSION, pair_style, pair_coeff,
            atom_types=atom_types, extra_cmds=extra_cmds, log_file=log_file,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


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
def start_eos_scan(
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
    """Scan isotropic cell strain and fit an equation of state (background job).

    A one-variable lab over cell volume. Evaluates the energy at n_points volumes
    spanning (1 ± strain_range)·V0 (relaxing ions at fixed cell when relax_ions),
    then fits a Birch-Murnaghan-style EOS. Needs a 3-D periodic structure (a bulk
    crystal). Poll get_status for done/total; get_results returns volumes, energies,
    and the fitted V0, E0, and bulk_modulus_GPa. abort/pause act between points.
    """
    try:
        return domain.start_eos_scan(
            SESSION, structure_id, calculator_id,
            n_points=n_points, strain_range=strain_range, relax_ions=relax_ions,
            optimizer=optimizer, fmax=fmax, steps=steps, step_delay=step_delay,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def start_elastic_scan(
    structure_id: str,
    calculator_id: str,
    n_strains: int = 5,
    max_strain: float = 0.01,
    relax_ions: bool = False,
    optimizer: str = "FIRE",
    fmax: float = 0.05,
    steps: int = 200,
    step_delay: float = 0.0,
) -> dict:
    """Measure the elastic stiffness tensor by stress-vs-strain (background job).

    The anisotropic analog of an EOS: strains the cell by n_strains magnitudes in
    ±max_strain along each of the 6 Voigt directions, reads the stress, and fits
    C_ij = dσ_i/dε_j. get_results returns the 6x6 C_GPa, Voigt-Reuss-Hill
    bulk/shear/Young's moduli, Poisson and Pugh ratios, and a Born mechanical-
    stability verdict. Use relax_ions for the clamped-ion correction (slower).
    Needs a 3-D periodic crystal, ideally pre-relaxed at its equilibrium cell.
    Poll get_status for done/total; abort/pause act between deformations.
    """
    try:
        return domain.start_elastic_scan(
            SESSION, structure_id, calculator_id,
            n_strains=n_strains, max_strain=max_strain, relax_ions=relax_ions,
            optimizer=optimizer, fmax=fmax, steps=steps, step_delay=step_delay,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def start_convex_hull(
    structure_ids: list,
    calculator_id: str,
    relax: bool = False,
    relax_cell: bool = False,
    references: dict | None = None,
    labels: list | None = None,
    optimizer: str = "FIRE",
    fmax: float = 0.05,
    steps: int = 200,
    step_delay: float = 0.0,
) -> dict:
    """Build the formation-energy convex hull over a set of compositions (job).

    Pass structure_ids spanning a chemical system (pure elements + candidate
    compounds/alloys). Each is evaluated (optionally relaxed; relax_cell also
    relaxes the cell), formation energies per atom are referenced to the pure
    elements, and the lower convex hull is built. get_results reports per phase the
    formation_energy_per_atom, energy_above_hull_per_atom (0 = stable), on_hull
    flag, and the list of stable_phases. Pure elements must be among the inputs or
    given as references={element: energy_per_atom}. Works for any element count.
    Poll get_status for done/total; abort/pause act between structures.
    """
    try:
        return domain.start_convex_hull(
            SESSION, structure_ids, calculator_id,
            relax=relax, relax_cell=relax_cell, references=references,
            labels=labels, optimizer=optimizer, fmax=fmax, steps=steps,
            step_delay=step_delay,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def start_saddle_search(
    structure_id: str,
    calculator_id: str,
    displacement_vector: list[float] | None = None,
    displace_magnitude: float = 0.1,
    dimer_separation: float = 0.01,
    fmax: float = 0.05,
    steps: int = 200,
    seed: int = 0,
    step_delay: float = 0.0,
) -> dict:
    """Find a transition state with the dimer method (Hessian-free; forces only).

    Single-ended saddle search from a (usually relaxed) structure: ascends the
    lowest-curvature mode, descends the rest. A negative converged curvature
    confirms an index-1 saddle. Robust with noisy ML potentials (no Hessian).
    displacement_vector (length-3N, Å) seeds the direction; otherwise a random
    displace_magnitude-Å kick is applied to the unconstrained atoms. Poll
    get_status for energy/max_force/curvature; get_results returns the saddle
    energy, curvature, structure_id, and is_index1_saddle. Steer abort/pause/set_fmax.
    """
    try:
        return domain.start_saddle_search(
            SESSION, structure_id, calculator_id,
            displacement_vector=displacement_vector,
            displace_magnitude=displace_magnitude, dimer_separation=dimer_separation,
            fmax=fmax, steps=steps, seed=seed, step_delay=step_delay,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def start_sella_search(
    structure_id: str,
    calculator_id: str,
    order: int = 1,
    fmax: float = 0.05,
    steps: int = 200,
    internal: bool = False,
    displacement_vector: list[float] | None = None,
    displace_magnitude: float = 0.0,
    seed: int = 0,
    step_delay: float = 0.0,
) -> dict:
    """Find an order-`order` saddle with Sella (RFO + refined approximate Hessian).

    A third route to transition states (order=1), complementing the dimer and
    POUNCE. Sella is an ASE optimizer that climbs toward a saddle using a
    partitioned rational-function step on an approximate Hessian it refines as it
    goes, so it converges in few force calls and is fully steerable
    (abort/pause/set_fmax; its Hessian survives set_fmax). Needs no product state
    (unlike NEB) and no full Hessian (unlike POUNCE). internal=True uses automatic
    internal coordinates (good for molecules). displacement_vector (length-3N, Å) or
    a random displace_magnitude-Å kick on the unconstrained atoms seeds the search.
    Poll get_status for energy/max_force/lowest_eigenvalue; get_results returns the
    saddle energy, lowest_eigenvalue, n_negative_eigenvalues, is_target_order_saddle,
    and structure_id. Requires the `sella` package.
    """
    try:
        return domain.start_sella_search(
            SESSION, structure_id, calculator_id,
            order=order, fmax=fmax, steps=steps, internal=internal,
            displacement_vector=displacement_vector,
            displace_magnitude=displace_magnitude, seed=seed, step_delay=step_delay,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@mcp.tool()
def start_pounce_saddles(
    structure_id: str,
    calculator_id: str,
    active_indices: list[int] | None = None,
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
    """Enumerate index-`index` saddles via POUNCE multistart eigenvector following.

    Complements the dimer: returns *several* distinct saddles, each labeled by its
    Morse index (count of negative Hessian eigenvalues). Variables are the
    coordinates of active_indices (default: atoms not held by FixAtoms); the rest
    stay frozen (keeps it low-dimensional and free of rigid zero modes). fun/grad
    from the calculator (∇E=−F); Hessian by central finite difference (fd_delta).
    For float32 ML potentials (UMA), loosen grad_tol and prefer float64. Requires
    the `pounce-solver` package. get_results returns saddles[] with energy,
    morse_index, grad_norm, eigenvalues. Runs to completion (no mid-solve abort).
    """
    try:
        return domain.start_pounce_saddles(
            SESSION, structure_id, calculator_id,
            active_indices=active_indices, index=index, n_saddles=n_saddles,
            fd_delta=fd_delta, grad_tol=grad_tol, max_step=max_step, dedup=dedup,
            local_max_iter=local_max_iter, max_solves=max_solves, seed=seed,
            displace_magnitude=displace_magnitude,
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
    comparator: str = "rmsd",
    optimizer: str = "FIRE",
    fmax: float = 0.05,
    steps: int = 300,
    sigma: float = 0.4,
    amplitude: float = 1.0,
    bh_step: float = 0.4,
    bh_temperature: float = 0.8,
    energy_tol: float = 0.02,
    rmsd_tol: float = 0.1,
    patience: int = 6,
    step_delay: float = 0.0,
) -> dict:
    """Search for multiple distinct relaxed geometries (local minima of the PES).

    kernel selects the method:
      - 'flooding' (Gaussian bumps; sigma Å, amplitude eV) / 'deflation'
        (inverse-distance poles): relax on a biased PES then polish. Best when a
        fixed frame is enforced (adsorbate on a frozen slab, anchored conformer).
      - 'basinhopping' (random kick bh_step Å + relax + Metropolis at bh_temperature
        eV): best for FREE clusters, whose rotation defeats a spatial bias.
    comparator judges novelty alongside energy_tol (eV): 'rmsd' (raw coords) or
    'fingerprint' (rotation/translation/permutation invariant — use for free
    clusters/molecules). rmsd_tol (Å) is the structural tolerance for both.

    Each new minimum is registered as a structure. Poll get_status for n_found/
    target; get_results returns the distinct minima (structure_id + energy) sorted
    by energy. Steerable via steer abort/pause (and set_fmax/switch_optimizer).
    """
    try:
        return domain.start_minima_search(
            SESSION, structure_id, calculator_id,
            n_minima=n_minima, kernel=kernel, comparator=comparator,
            optimizer=optimizer, fmax=fmax, steps=steps, sigma=sigma,
            amplitude=amplitude, bh_step=bh_step, bh_temperature=bh_temperature,
            energy_tol=energy_tol, rmsd_tol=rmsd_tol, patience=patience,
            step_delay=step_delay,
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
    """Return a job's final results, if any: NEB barrier/energies, phonon gamma
    frequencies and stability, distinct minima, or EOS V0/E0/bulk_modulus_GPa.
    None until the job finishes."""
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
