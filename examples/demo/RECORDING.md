# Recording the terminal demo

The fallback you play if the live demo misbehaves. Record it **in the same
terminal you'll present from** (same font, size, theme) so it matches the room.

[asciinema](https://asciinema.org) is the right tool — it captures the terminal
as lightweight, replayable text (not a heavy video), and you can pause/scrub
during playback to narrate over it.

## Install

```bash
pip install asciinema          # already installed in this env
# or: brew install asciinema
```

## Record

Auto-advancing (hands-free), paced for a room:

```bash
DEMO_AUTO=1 DEMO_SPEED=2.0 asciinema rec examples/demo/recordings/conference_demo.cast \
  --overwrite --title "fairchem-mcp demo" \
  --command "python examples/demo/conference_demo.py"
```

- `DEMO_SPEED` scales the narration pauses only (2.0 = roomy; 1.0 = brisk). The
  physics compute is unscaled.
- Drop `DEMO_AUTO` to advance by pressing ENTER yourself — then the recorded
  timing matches *your* talking pace, which makes the best narrated fallback.
- A committed baseline cast is already in `recordings/` so you have something to
  play even before you re-record.

## Play back (your fallback during the talk)

```bash
asciinema play examples/demo/recordings/conference_demo.cast
```

- **SPACE** = pause/resume (pause to talk), **`.`** = step while paused,
  **`←/→`** = skip, **Ctrl-C** = quit.
- Slow it down: `asciinema play --speed 0.5 …`; cap dead air:
  `--idle-time-limit 2`.

## Share / embed (optional)

```bash
asciinema upload examples/demo/recordings/conference_demo.cast   # returns a URL
```

For a slide-deck GIF/SVG (no terminal needed to view):

```bash
# GIF:
pip install agg && agg examples/demo/recordings/conference_demo.cast demo.gif
# Crisp SVG:
npx svg-term-cli --in examples/demo/recordings/conference_demo.cast --out demo.svg --window
```

## Tips
- Run a relaxation once before recording to warm caches so the first beat isn't
  sluggish.
- Recording on UMA? Set `FAIRCHEM_MCP_EXAMPLE_MODEL=uma-s-1p2` (and
  `FAIRCHEM_MCP_EXAMPLE_TASK=omat`) before `asciinema rec`.
- 80×24 is the default size when not attached to a real terminal; record from a
  real, generously-sized window for legible playback.
