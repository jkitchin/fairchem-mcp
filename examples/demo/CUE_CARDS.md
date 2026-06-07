# Conference demo — cue cards (10 min)

Two ways to run the demo:

- **A. Agent-driven (live, most impressive):** talk to Claude Code in plain
  English; the audience watches it drive the MCP tools. Cards below.
- **B. Scripted driver (reliable, also your recording):**
  `python examples/demo/conference_demo.py` — runs the real engine, paced for a
  room. Press ENTER between beats; or `DEMO_AUTO=1` to auto-advance.

Keep **B** running in a second pane as your safety net. If the live agent wanders,
switch to it without missing a beat.

---

## 0. Before you walk on (pre-flight)
- [ ] Model **pre-loaded** (resident). UMA if the network/GPU is solid; otherwise
      run on **EMT** — the steering story is identical and instant.
- [ ] Terminal: big font, dark theme, **notifications off**, one window.
- [ ] Fallback recording ready: `asciinema play examples/demo/recordings/conference_demo.cast`
      (SPACE pauses, `←/→` skip — see RECORDING.md).
- [ ] Dry-run once on the venue network.

---

## 1. Hook — 45s (slide, no terminal)
> "Today, AI plus simulation is **batch**: the model writes a script, runs it,
> waits, reads the log. No watching, no intervening. fairchem-mcp makes it a
> **conversation** — the model stays loaded, jobs run live, and the agent
> **watches and steers** them mid-flight."

## 2. Setup reveal — 45s
> "It's an MCP server — three lines to register. The model is already loaded and
> **resident**: we pay that cost once, not per run."

```json
{ "mcpServers": { "fairchem": { "command": "fairchem-mcp" } } }
```

## 3. ★ Watch + steer — 4 min  (THE beat; cut others before this one)
Say to Claude:
> "Attach EMT. Build a 4×4×4 Cu(111) slab rattled by 0.25 Å, and start relaxing
> it with FIRE to fmax 0.01 — use a small step delay so we can watch."

Then:
> "Poll the status a few times and tell me the trend — is it still making good
> progress?"

Narrate the live numbers: *forces fall from ~3 to ~0.1, then the tail crawls.*
> "FIRE has no curvature model — it dawdles in the tail. **Switch it to LBFGS,
> without restarting.**"

```
steer(job, "switch_optimizer", optimizer="LBFGS")
```
**Land it:** "It picked up from the **exact same positions** — no restart — and
converged. The agent changed strategy on a live job."

## 4. Transition state — 2.5 min
> "An Al adatom hops between hollow sites on Al(100); the bridge site is the
> transition state. Relax it into the hollow, then run a dimer saddle search
> nudged toward the bridge — and **watch the curvature**."

Narrate: *curvature starts positive (uphill), then flips negative.*
> "Negative curvature = one downhill direction = an index-1 transition state.
> Three routes ship in the server: **dimer, Sella, POUNCE**."

## 5. Escape hatch — 1 min
> "When the tools run out, the agent drops to Python on the **same live atoms** —
> one shared namespace."

```
inspect_expr("np.linalg.norm(atoms.get_forces(), axis=1).max()")
introspect("atoms", live=True)      # real installed API, not stale docs
```

## 6. Close — 1 min
> "Resident model, jobs you watch and steer, a Python escape hatch — that's the
> difference between **scripting** a simulation and **collaborating** on one."

Name-drop the breadth: relaxation, MD, NEB, phonons, **EOS, LAMMPS as a force
engine, three transition-state methods**, minima search — all steerable. EMT out
of the box; point it at UMA for real numbers.

---

### Timing discipline
- The steering beat (§3) is the demo. **If short on time, cut §4 or §5, never §3.**
- Poll 2–3 times per beat; don't read every step aloud.
- If UMA wobbles: *"I'll run this on the built-in classical potential to stay
  fast"* → EMT. Nobody notices.
