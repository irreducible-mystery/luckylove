"""A registry for arbitrary lines, so that drawing one is recorded, not remembered.

The discipline this enforces is: *keep drawing arbitrary lines, and never stop
saying that is what you are doing.* Not because the lines are avoidable — there
is no partition of a continuum that isn't chosen — but because the failure mode
is amnesia, not arbitrariness.

Prose discipline demonstrably does not survive. Three independent demonstrations,
all in this repository's history:

  1. `paraconsistent/core/labels.py` settles `dc >= 0` in 17 if-statements and
     ~21 crisp comparisons. Not one of them says it is crisp.
  2. julia3claude's own self-audit (§A, `results/CRITIQUE_REPORT.md`) found five
     defensible encodings of one dataset spanning three lattice regions and two
     decisions — "the verdict of the paraconsistent analysis was substantially
     *my* verdict, smuggled in through the encoding."
  3. This module's author wrote "Three is what's forced" one turn after arguing
     that nothing is, and "nobody chose 4" about a number produced entirely by a
     hand-written action vocabulary (Grok Build, `critique/CRITIQUE.md` A5).

Each was a competent party actively trying not to forget. So the remembering is
moved out of prose and into a data structure with a test attached.

A `Choice` is not a config value. It carries what a config value cannot: the
alternatives that were equally defensible, what moves if you pick one of them,
and — the field that matters most — **what this choice makes invisible**. Every
finite representation has a counterfeit theorem; `collapses` is where you write
yours down before someone else finds it.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Choice", "declare", "registry", "get", "unregistered_literals",
           "format_registry"]


@dataclass(frozen=True)
class Choice:
    """One arbitrary line, drawn on purpose and on the record."""

    name: str
    value: Any
    alternatives: tuple[str, ...]
    consequence: str
    collapses: str
    chosen_by: str
    evidence: str = ""

    def __post_init__(self) -> None:
        if not self.alternatives:
            raise ValueError(
                f"Choice {self.name!r}: `alternatives` is empty. If nothing else "
                "was defensible it is not a choice — state the derivation instead."
            )
        for f in ("consequence", "collapses", "chosen_by"):
            if not getattr(self, f).strip():
                raise ValueError(f"Choice {self.name!r}: `{f}` must be non-empty.")

    def __str__(self) -> str:
        alts = "; ".join(self.alternatives)
        return (f"{self.name} = {self.value!r}  [{self.chosen_by}]\n"
                f"    could have been : {alts}\n"
                f"    if it changed   : {self.consequence}\n"
                f"    makes invisible : {self.collapses}"
                + (f"\n    pinned by       : {self.evidence}" if self.evidence else ""))


_REGISTRY: dict[str, Choice] = {}


def declare(name: str, value: Any, *, alternatives, consequence: str,
            collapses: str, chosen_by: str, evidence: str = "") -> Any:
    """Register an arbitrary line and return its value, so it reads inline.

        FTC_DEFAULT = declare("FtC", 0.50, alternatives=..., ...)
    """
    c = Choice(name=name, value=value, alternatives=tuple(alternatives),
               consequence=consequence, collapses=collapses,
               chosen_by=chosen_by, evidence=evidence)
    if name in _REGISTRY and _REGISTRY[name] != c:
        raise ValueError(f"Choice {name!r} already declared with different content")
    _REGISTRY[name] = c
    return value


def registry() -> dict[str, Choice]:
    return dict(_REGISTRY)


def get(name: str) -> Choice:
    return _REGISTRY[name]


def format_registry() -> str:
    if not _REGISTRY:
        return "(no choices declared)"
    return "\n\n".join(str(_REGISTRY[k]) for k in sorted(_REGISTRY))


# ------------------------------------------------------------------ the check

# Literals that carry no modeling content. Everything else in a scanned module
# must be traceable to a declared Choice, or the test fails.
_INNOCENT = {0, 1, 2, -1, 0.0, 1.0, 2.0, -1.0, 0.5, 100}


def unregistered_literals(module) -> list[tuple[str, int, Any]]:
    """Numeric literals in `module` that no declared Choice accounts for.

    This is the build failure. A bare threshold in classification code is
    exactly how an arbitrary line stops looking like one, so a new constant
    either gets a `declare(...)` or it fails the test that calls this.

    Deliberately syntactic and therefore fallible: it cannot see a magic number
    assembled at runtime, and `_INNOCENT` is itself a drawn line (see the
    self-referential entry in the registry). It raises the cost of forgetting;
    it does not abolish it.
    """
    src = inspect.getsource(module)
    declared = {c.value for c in _REGISTRY.values()
                if isinstance(c.value, (int, float))}
    out: list[tuple[str, int, Any]] = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            # Anything inside a declare(...) call is by definition accounted for.
            fn = node.func
            if (isinstance(fn, ast.Name) and fn.id == "declare") or (
                    isinstance(fn, ast.Attribute) and fn.attr == "declare"):
                return
            self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> None:
            v = node.value
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return
            if v in _INNOCENT or v in declared:
                return
            out.append((module.__name__, node.lineno, v))

    V().visit(ast.parse(src))
    return out


# The scanner's own allowlist is a drawn line. Recording it here rather than
# leaving it as a module-private set is the whole point of the exercise.
declare(
    "innocent_literals",
    sorted(_INNOCENT),
    alternatives=[
        "empty set — every literal must be declared, including 0 and 1",
        "allow anything appearing more than once (frequency as innocence)",
        "scan only functions named *classify*/*decide*",
    ],
    consequence="a tighter set fails the build on arithmetic identities; a looser "
                "one lets a real threshold through as 'obviously structural'",
    collapses="the difference between a structural 0.5 (the lattice centre, "
              "forced) and a modeling 0.5 (a default FtC, chosen) — they are the "
              "same token, so this scanner cannot tell them apart",
    chosen_by="claude-opus-5, on Grok Build's A5 finding that the free parameter "
              "changed type rather than vanishing",
    evidence="src/choices.py; tests/test_choices.py",
)
