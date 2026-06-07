"""Session state shared across all MCP tools.

A single :class:`Session` holds a persistent Python namespace plus registries of
structures, calculators and jobs. The same objects are reachable both through the
high-level domain tools (which store handles here) and through the
``execute``/``inspect`` escape-hatch tools (which operate on ``namespace``). That
shared namespace is what makes the server "hybrid": the agent can call
``start_relaxation`` and then drop to raw Python on the very same live ``Atoms``.
"""

from __future__ import annotations

import itertools
from typing import Any


class Session:
    """Holds all live objects for one running server process."""

    def __init__(self) -> None:
        # The namespace used by execute()/inspect(). Domain handles are mirrored
        # in here by id (e.g. "struct_1") plus convenience aliases ("atoms").
        self.namespace: dict[str, Any] = {}

        self.structures: dict[str, Any] = {}
        self.calculators: dict[str, Any] = {}
        self.jobs: dict[str, Any] = {}

        self._counters: dict[str, itertools.count] = {
            "struct": itertools.count(1),
            "calc": itertools.count(1),
            "job": itertools.count(1),
        }

        # Seed the namespace with handy imports so execute()/inspect() are useful
        # without boilerplate.
        import ase
        import numpy as np

        self.namespace.update({"ase": ase, "np": np, "session": self})

    def _new_id(self, kind: str) -> str:
        return f"{kind}_{next(self._counters[kind])}"

    def add_structure(self, atoms: Any) -> str:
        sid = self._new_id("struct")
        self.structures[sid] = atoms
        self.namespace[sid] = atoms
        self.namespace["atoms"] = atoms  # most-recent alias
        return sid

    def add_calculator(self, calc: Any, *, info: dict | None = None) -> str:
        cid = self._new_id("calc")
        self.calculators[cid] = calc
        self.namespace[cid] = calc
        self.namespace["calc"] = calc  # most-recent alias
        if info is not None:
            # Stash human-readable metadata (model name, task, device).
            setattr(calc, "_fairchem_mcp_info", info)
        return cid

    def add_job(self, job: Any) -> str:
        self.jobs[job.id] = job
        self.namespace[job.id] = job
        self.namespace["job"] = job  # most-recent alias
        return job.id

    def get_structure(self, sid: str) -> Any:
        if sid not in self.structures:
            raise KeyError(f"unknown structure id {sid!r}")
        return self.structures[sid]

    def get_calculator(self, cid: str) -> Any:
        if cid not in self.calculators:
            raise KeyError(f"unknown calculator id {cid!r}")
        return self.calculators[cid]

    def get_job(self, jid: str) -> Any:
        if jid not in self.jobs:
            raise KeyError(f"unknown job id {jid!r}")
        return self.jobs[jid]

    def active_job(self) -> Any | None:
        """Return the currently running/paused job, if any (single-job policy)."""
        for job in self.jobs.values():
            if job.is_active():
                return job
        return None
