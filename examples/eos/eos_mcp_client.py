"""Equation of state for fcc Cu — driven through the REAL MCP server over stdio.

Unlike `eos_demo.py` (which calls the in-process `domain` functions directly),
this spins up the actual `fairchem-mcp` server as a subprocess and talks to it
the way an agent does: a JSON-RPC handshake, `tools/list`, then a sequence of
`tools/call` requests over the stdio transport. Every line you see on the wire
is a genuine MCP exchange — calculator attach, structure build, the EOS job,
status polling, and the final results — so this faithfully represents what the
MCP server does, not just the engine underneath it.

Calculator is EMT by default (instant, no GPU/model). Point the *server* at a
UMA model by exporting FAIRCHEM_MCP_EXAMPLE_MODEL before running (the client
just calls load_model instead of attach_emt).

    python examples/eos/eos_mcp_client.py                # ENTER between beats
    DEMO_AUTO=1 python examples/eos/eos_mcp_client.py    # auto-advance (recording)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

AUTO = bool(os.environ.get("DEMO_AUTO"))
SPEED = float(os.environ.get("DEMO_SPEED", "1.0"))

B, DIM, GRN, YEL, CYN, MAG, RED, RST = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[36m", "\033[35m",
    "\033[31m", "\033[0m",
)


def _nap(t: float) -> None:
    time.sleep(t * SPEED)


def banner(title: str) -> None:
    line = "─" * 68
    print(f"\n{CYN}{line}{RST}")
    print(f"{CYN}{B}  {title}{RST}")
    print(f"{CYN}{line}{RST}\n")


def gate() -> None:
    if AUTO:
        _nap(1.1)
    else:
        try:
            input(f"\n{DIM}  [ENTER]{RST}")
        except EOFError:
            pass


def _trim(obj, maxlen: int = 88) -> str:
    """One-line JSON, abbreviated, for the response side of the wire."""
    s = json.dumps(obj, separators=(",", ":"))
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


def extract(res):
    """Pull the tool's dict payload out of a CallToolResult."""
    sc = res.structuredContent
    if sc is not None:
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    for c in res.content:
        if getattr(c, "type", None) == "text":
            try:
                return json.loads(c.text)
            except Exception:
                return c.text
    return None


class Wire:
    """A thin MCP client that narrates each tools/call as it crosses the wire."""

    def __init__(self, session: ClientSession) -> None:
        self.session = session

    async def call(self, name: str, args: dict | None = None, *,
                   show: bool = True) -> dict:
        args = args or {}
        if show:
            astr = json.dumps(args, separators=(",", ":")) if args else "{}"
            print(f"  {MAG}→ tools/call{RST} {B}{name}{RST} {DIM}{astr}{RST}")
            _nap(0.35)
        res = await self.session.call_tool(name, args)
        out = extract(res)
        if show:
            tag = f"{RED}error{RST}" if isinstance(out, dict) and "error" in out \
                else f"{GRN}← result{RST}"
            print(f"  {tag} {DIM}{_trim(out)}{RST}")
            _nap(0.45)
        return out


def ev_plot(volumes: list, energies: list, v0: float) -> None:
    width, height = 52, 11
    vmin, vmax = min(volumes), max(volumes)
    emin, emax = min(energies), max(energies)
    espan = (emax - emin) or 1.0
    grid = [[" "] * width for _ in range(height)]
    for v, e in zip(volumes, energies):
        col = int(round((v - vmin) / ((vmax - vmin) or 1.0) * (width - 1)))
        row = int(round((e - emin) / espan * (height - 1)))
        grid[height - 1 - row][col] = f"{CYN}●{RST}"
    col0 = int(round((v0 - vmin) / ((vmax - vmin) or 1.0) * (width - 1)))
    grid[height - 1][min(col0, width - 1)] = f"{YEL}{B}V{RST}"
    print(f"   {DIM}E (eV){RST}")
    for r, line in enumerate(grid):
        print(f"        {DIM}{'┤' if r else '┐'}{RST} " + "".join(line))
    print(f"        {DIM}└{'─' * width}{RST}")
    print(f"         {DIM}{vmin:.1f} Å³{' ' * (width - 14)}volume"
          f"{' ' * 6}{vmax:.1f} Å³{RST}")


async def run() -> None:
    # Launch the actual server as a subprocess; its stderr (INFO logs) is sent
    # to /dev/null so only the protocol conversation shows in the recording.
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "fairchem_mcp.server"], env=os.environ.copy())
    devnull = open(os.devnull, "w")

    banner("fairchem-mcp · equation of state · fcc Cu  (real MCP server, stdio)")
    print(f"  {DIM}spawning: {sys.executable} -m fairchem_mcp.server{RST}")
    _nap(0.5)

    async with stdio_client(params, errlog=devnull) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            si = init.serverInfo
            print(f"  {GRN}✓ initialize{RST}  server={B}{si.name} {si.version}{RST}"
                  f"  {DIM}protocol={init.protocolVersion}{RST}")
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"  {GRN}✓ tools/list{RST}  {B}{len(names)}{RST} tools  "
                  f"{DIM}({', '.join(names[:6])}, …){RST}")
            gate()

            w = Wire(session)

            banner("1 · attach a calculator and build the crystal")
            model = os.environ.get("FAIRCHEM_MCP_EXAMPLE_MODEL")
            if model:
                task = os.environ.get("FAIRCHEM_MCP_EXAMPLE_TASK", "omat")
                calc = (await w.call("load_model", {"model": model, "task": task}))["calculator_id"]
            else:
                calc = (await w.call("attach_emt"))["calculator_id"]
            built = await w.call("build_structure", {"spec": {
                "kind": "bulk", "name": "Cu", "crystalstructure": "fcc",
                "a": 3.6, "cubic": True}})
            sid = built["structure_id"]
            gate()

            banner("2 · start the EOS scan, then poll it like an agent would")
            job = await w.call("start_eos_scan", {
                "structure_id": sid, "calculator_id": calc,
                "n_points": 11, "strain_range": 0.05, "step_delay": 0.25})
            job_id = job["job_id"]
            print(f"  {DIM}background job {job_id} running; polling get_status…{RST}\n")

            seen, total = 0, 11
            while True:
                s = await w.call("get_status", {"job_id": job_id}, show=False)
                total = s.get("total", total) or total
                done = s.get("done", 0) or 0
                if done > seen and s.get("volume") is not None:
                    bar = "█" * done + "·" * (total - done)
                    print(f"  {MAG}→ get_status{RST}  {GRN}{bar}{RST} "
                          f"{done:>2}/{total}   V={s['volume']:7.2f} Å³   "
                          f"E={s['energy']:9.4f} eV")
                    seen = done
                if s.get("status") not in ("running", "pending", "paused"):
                    break
                await asyncio.sleep(0.05)
            gate()

            banner("3 · fetch the fitted results")
            r = (await w.call("get_results", {"job_id": job_id}))["result"]
            print()
            ev_plot(r["volumes"], r["energies"], r["V0"])
            print()
            n = 4  # Cu4 conventional cell
            print(f"  equilibrium volume  V0 : {YEL}{B}{r['V0']:.3f} Å³{RST}"
                  f"   ({r['V0'] / n:.3f} Å³/atom)")
            print(f"  minimum energy      E0 : {YEL}{B}{r['E0']:.4f} eV{RST}")
            print(f"  bulk modulus        B  : {YEL}{B}{r['bulk_modulus_GPa']:.1f} GPa{RST}"
                  f"   {DIM}(expt. fcc Cu ≈ 140 GPa){RST}")
            print(f"\n  {DIM}Every value above arrived over the MCP stdio transport "
                  f"from the real server.{RST}\n")

    devnull.close()


if __name__ == "__main__":
    asyncio.run(run())
