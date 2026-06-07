# Conference demo

A ~10-minute live demo of `fairchem-mcp` built around the one thing batch
scripts can't do: **watch a simulation and steer it mid-flight.**

| File | What it is |
|---|---|
| [`CUE_CARDS.md`](CUE_CARDS.md) | The run-of-show: timing, narration, and the English prompts to say to Claude. |
| [`conference_demo.py`](conference_demo.py) | Paced driver that runs the real engine (relax + steer, transition state, escape hatch). Your paste-ready script **and** the thing you record. |
| [`RECORDING.md`](RECORDING.md) | How to record/play the terminal with `asciinema` for a fallback. |
| [`recordings/`](recordings/) | A committed baseline cast (re-record in your own terminal for the polished version). |

## Quick start

```bash
pip install -e ".[saddles]"            # dimer needs nothing extra; this adds Sella/POUNCE
python examples/demo/conference_demo.py        # ENTER between beats (presenting live)
DEMO_AUTO=1 python examples/demo/conference_demo.py   # auto-advance (recording / rehearsal)
```

Runs on **EMT** by default (instant, no GPU/model). For a real model:

```bash
FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p2 FAIRCHEM_MCP_EXAMPLE_TASK=omat \
    python examples/demo/conference_demo.py
```

## The three beats

1. **Watch + steer** — a rattled Cu(111) slab relaxes; FIRE crushes the big
   forces then crawls in the tail; switch to LBFGS mid-flight and it converges
   from the same positions. *(The money shot — don't cut this one.)*
2. **Transition state** — a dimer saddle search on an Al adatom hop; the
   curvature flips from positive (uphill) to negative (on the saddle).
3. **Escape hatch** — raw Python (`inspect_expr` / `introspect`) on the same live
   `atoms`, showing the shared namespace.

Every number on screen is from the real engine, so it's reproducible and honest.
