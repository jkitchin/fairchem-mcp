"""Code-awareness: introspect the *installed* API and *live* objects.

Two modes:

* static (``live=False``) — resolve a dotted path like
  ``fairchem.core.calculate.ase_calculator.FAIRChemCalculator.from_model_checkpoint``
  by importing it, then report its signature, docstring and members. This reads
  the real installed package, not possibly-stale skill docs.
* live (``live=True``) — evaluate the target against the session namespace, so
  the agent can introspect the actual objects it just created (e.g. ``atoms`` or
  ``calc``). Completions use :class:`jedi.Interpreter`, which understands live
  objects and partial expressions.

A target ending in ``.`` (e.g. ``"atoms."`` or ``"ase.build."``) returns member
completions instead of a description.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import jedi


def introspect(target: str, namespace: dict, live: bool = False) -> dict:
    target = target.strip()
    if target.endswith("."):
        return _complete(target[:-1], namespace, live)
    obj = _eval_live(target, namespace) if live else _resolve(target)
    return _describe(obj, target, namespace if live else None)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _resolve(dotted: str) -> Any:
    """Import the longest module prefix of a dotted path, then walk attributes."""
    parts = dotted.split(".")
    obj = None
    rest: list[str] = []
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
            rest = parts[i:]
            break
        except ImportError:
            continue
    if obj is None:
        # Maybe it's a builtin name.
        import builtins

        if hasattr(builtins, parts[0]):
            obj = getattr(builtins, parts[0])
            rest = parts[1:]
        else:
            raise ImportError(f"cannot import any module prefix of {dotted!r}")
    for attr in rest:
        obj = getattr(obj, attr)
    return obj


def _eval_live(expr: str, namespace: dict) -> Any:
    return eval(expr, namespace)  # noqa: S307 - trusted local dev escape hatch


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------

def _describe(obj: Any, target: str, live_ns: dict | None) -> dict:
    out: dict = {"target": target, "type": type(obj).__qualname__}

    if inspect.ismodule(obj):
        out["kind"] = "module"
    elif inspect.isclass(obj):
        out["kind"] = "class"
    elif callable(obj):
        out["kind"] = "callable"
    else:
        out["kind"] = "value"
        out["repr"] = _safe_repr(obj)

    try:
        out["signature"] = str(inspect.signature(obj))
    except (TypeError, ValueError):
        pass

    doc = inspect.getdoc(obj)
    if doc:
        lines = doc.splitlines()
        out["docstring"] = "\n".join(lines[:60])
        if len(lines) > 60:
            out["docstring_truncated"] = True

    if inspect.ismodule(obj) or inspect.isclass(obj):
        out["members"] = [n for n in dir(obj) if not n.startswith("_")][:200]

    return out


def _complete(prefix_expr: str, namespace: dict, live: bool) -> dict:
    """Return public members of an object via jedi (live-namespace aware)."""
    if live:
        ns = namespace
        code = f"{prefix_expr}."
    else:
        obj = _resolve(prefix_expr)
        ns = {"_t": obj}
        code = "_t."

    script = jedi.Interpreter(code, [ns])
    completions = script.complete(1, len(code))
    members = [
        {"name": c.name, "type": c.type}
        for c in completions
        if not c.name.startswith("_")
    ]
    return {"target": f"{prefix_expr}.", "completions": members[:200]}


def _safe_repr(obj: Any) -> str:
    try:
        r = repr(obj)
    except Exception as exc:  # noqa: BLE001
        return f"<unreprable: {exc}>"
    return r if len(r) <= 500 else r[:500] + "…"
