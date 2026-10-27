# Equation of state — fcc Cu

A one-variable lab. The cell volume is the only knob: build a spread of
isotropically strained copies of fcc Cu, evaluate the energy at each, and fit a
Birch–Murnaghan equation of state for the equilibrium volume **V0**, the energy
minimum **E0**, and the **bulk modulus B**.

Runs out of the box on the fast **EMT** calculator (no GPU, no model download).
EMT is qualitative, but here it lands B ≈ 134 GPa against the experimental
≈ 140 GPa for fcc Cu. Point it at a UMA model for publication numbers.

## Two demos — pick by what you want to show

| script | what it exercises | use it to show… |
| --- | --- | --- |
| `eos_mcp_client.py` | spawns the **real `fairchem-mcp` server** and drives it over the **stdio MCP transport** — genuine `initialize`, `tools/list`, and `tools/call` traffic | *what the MCP server actually does* (an agent calling tools, polling a job) |
| `eos_demo.py` | calls the in-process `fairchem_mcp.domain` functions the tools wrap (no protocol) | *the engine* — same physics, fewer moving parts, simplest to run |

Both produce identical numbers (the tool is a ~3-line passthrough to `domain`).
The MCP-client version is the honest "this is the server" recording for slides;
the direct version is the quick engine demo.

## Run them

```bash
# Real MCP server over stdio (recommended for slides):
python examples/eos/eos_mcp_client.py                # ENTER between beats
DEMO_AUTO=1 python examples/eos/eos_mcp_client.py    # auto-advance

# Direct engine call:
python examples/eos/eos_demo.py

# Real numbers on a UMA model (export before running; works for either script):
FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p2 FAIRCHEM_MCP_EXAMPLE_TASK=omat \
  python examples/eos/eos_mcp_client.py
```

`DEMO_SPEED` scales the narration pauses only (the physics compute is unscaled);
`>1` is roomier for a room, `<1` is brisker.

## Recordings (for slides)

Committed [asciinema](https://asciinema.org) casts and rendered GIFs are in
[`recordings/`](recordings/):

- `eos_cu_mcp.cast` / `.gif` — the real MCP-server session
- `eos_cu.cast` / `.gif` — the direct-engine version

Re-record in your own terminal (your font/size/theme) for the crispest result:

```bash
DEMO_AUTO=1 DEMO_SPEED=1.5 asciinema rec examples/eos/recordings/eos_cu_mcp.cast \
  --overwrite --title "fairchem-mcp · EOS · fcc Cu (real MCP server, stdio)" \
  --idle-time-limit 2 --command "python examples/eos/eos_mcp_client.py"

# GIF for Google Slides (drag the .gif straight onto a slide):
agg --font-size 20 --theme monokai \
  examples/eos/recordings/eos_cu_mcp.cast examples/eos/recordings/eos_cu_mcp.gif
```

Play a cast as a narrated fallback: `asciinema play examples/eos/recordings/eos_cu_mcp.cast`
(SPACE pauses, `--speed 0.5` slows it down).
