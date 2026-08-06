"""Running the pipeline under several defensible lines at once.

Not an annotation that a choice was made, and not a checker over the choices.
The disagreement between equally-defensible alternatives is computed and
returned *in the answer itself*, so that where the line matters the output is
visibly plural and cannot be read as a single verdict.

`Ensemble.optimal` returns the same type `Policy.optimal` returns — a frozenset
— and means the same thing one level up. `Policy.optimal` goes plural when two
actions tie under one belief map. `Ensemble.optimal` goes plural when equally
good belief maps disagree about which action wins. Same discipline, same type,
applied to the choice of map rather than the choice of action.

## Why this is not the registry again

The registry that was removed in `24d1c57` emitted a boolean that read as
coverage: a passing test named for accounting for the arbitrary lines, over a
set (numeric literals) that could not contain most of them. Forgetting produced
silence, and silence looked like diligence.

This emits a spread, and the spread has one property that makes it honest:

    **adding a member can only increase measured disagreement, never decrease it**

because disagreement is `|union of the members' argmin sets| > 1`, and unions
grow. So every number here is a LOWER BOUND on how much the choice matters.
A lower bound that is too low invites the obvious repair — add another member —
and `add()` exists so a stranger can do that in one line. A green check invites
nothing.

## What would make this instance five

Say it plainly so it stays checkable:

- A narrow spread read as "the choice doesn't matter" rather than "the members
  I happened to pick agree." Those are different claims and only the second is
  supported. `n_members` is reported next to every spread for that reason.
- The ensemble hardening into a fixed canonical three. The three here are the
  ones Grok Build named in `critique/CRITIQUE.md` A2. They are a starting set,
  not a basis, and nothing distinguishes them except that an outside reader
  proposed them.
- Sensitivity measured on a uniform grid and quoted as if it were deployment
  sensitivity. Grok's A3 is exactly this error one level down, so
  `sensitivity()` takes explicit points and the demo reports both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from paraconsistent import ParaconsistentBlock

from etau_decision import (
    Action, Policy, etau_belief,
    OBSTACLE, CLEAR, FAULT_CONFLICT, FAULT_SILENCE,
)

__all__ = ["Ensemble", "radial_belief", "hard_belief", "DEFAULT_BELIEFS"]

Belief = Callable[[float, float], Mapping[str, float]]


# --- Alternative mass maps. -------------------------------------------------
# Each is as arithmetically clean as `etau_belief`, uses only quantities the
# library already computes, and sums to 1. That is the whole point: none of the
# three is worse-founded than the shipped default, which is why the default was
# never entitled to be called derived.


def radial_belief(mu: float, lam: float) -> dict[str, float]:
    """Split the decided mass by the normalised radial distance D, not by φ."""
    c = ParaconsistentBlock(mu=mu, lam=lam).complete
    decided = 1.0 - c.D
    rest = c.D
    tot = abs(c.dct) or 1.0
    return {
        OBSTACLE: decided * c.muE,
        CLEAR: decided * (1.0 - c.muE),
        FAULT_CONFLICT: rest * max(0.0, c.dct) / tot,
        FAULT_SILENCE: rest * max(0.0, -c.dct) / tot,
    }


def hard_belief(mu: float, lam: float) -> dict[str, float]:
    """Put the decided mass entirely on the side Dc favours (a contradiction reserve)."""
    c = ParaconsistentBlock(mu=mu, lam=lam).complete
    decided = c.phi
    if c.dc > 0:
        obs, clr = decided, 0.0
    elif c.dc < 0:
        obs, clr = 0.0, decided
    else:
        obs = clr = decided / 2.0
    return {
        OBSTACLE: obs,
        CLEAR: clr,
        FAULT_CONFLICT: max(0.0, c.dct),
        FAULT_SILENCE: max(0.0, -c.dct),
    }


DEFAULT_BELIEFS: tuple[Belief, ...] = (etau_belief, radial_belief, hard_belief)


# --- The ensemble. ----------------------------------------------------------


@dataclass
class Ensemble:
    """A policy run under several belief maps, answering plurally where they differ."""

    actions: Sequence[Action]
    beliefs: tuple[Belief, ...] = DEFAULT_BELIEFS

    _policies: list[Policy] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not self.beliefs:
            raise ValueError("an ensemble needs at least one belief map")
        self._policies = [Policy(self.actions, belief=b) for b in self.beliefs]

    @property
    def n_members(self) -> int:
        return len(self.beliefs)

    def add(self, belief: Belief) -> "Ensemble":
        """Add a member. The one-line repair for a spread you suspect is too narrow."""
        return Ensemble(self.actions, tuple(self.beliefs) + (belief,))

    # ---------------------------------------------------------------- answer

    def optimal(self, mu: float, lam: float) -> frozenset[str]:
        """Union of every member's argmin set.

        Singleton iff every member agrees. Plural exactly where the choice of
        belief map changes what you would do — which is the only place it
        matters, and the place an annotation would have been silent.
        """
        out: frozenset[str] = frozenset()
        for p in self._policies:
            out |= p.optimal(mu, lam)
        return out

    def per_member(self, mu: float, lam: float) -> dict[int, frozenset[str]]:
        return {i: p.optimal(mu, lam) for i, p in enumerate(self._policies)}

    def agrees(self, mu: float, lam: float) -> bool:
        return len(self.optimal(mu, lam)) == 1

    # ----------------------------------------------------------- sensitivity

    def sensitivity(self, points: Iterable[tuple[float, float]]) -> float:
        """Fraction of `points` where the members do not agree on the action.

        A LOWER BOUND on how much the belief-map choice matters, monotone
        non-decreasing in the number of members. Report it with `n_members` or
        it reads as a property of the problem rather than of this ensemble.
        """
        n = dis = 0
        for mu, lam in points:
            n += 1
            if len(self.optimal(mu, lam)) > 1:
                dis += 1
        return dis / n if n else float("nan")

    def grid_points(self, steps: int = 101):
        for i in range(steps):
            for j in range(steps):
                yield i / (steps - 1), j / (steps - 1)

    def deployment_sensitivity(self, records) -> float:
        """Sensitivity under the world measure — Grok's A3, one level up.

        Uniform-grid sensitivity is a statement about the annotation square.
        The number that bears on a deployment is this one.
        """
        return self.sensitivity((mu, lam) for mu, lam, *_ in records)
