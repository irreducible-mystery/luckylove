"""Cost-derived partitions of the Eτ annotation square.

The twelve-region readout partitions annotations by the *geometry of the
lattice*: thresholds on Dc and Dct, with FtC drawing the lines. Nothing about
that geometry knows what you are deciding, so FtC is a stipulation — you can
replace it with a different number but you cannot discover it was wrong.

This module partitions annotations by *what you would do about them* instead.
Two annotations fall in the same region iff the same action minimises expected
loss. The boundaries land where expected-cost curves cross. The region count is
then an OUTPUT, not a parameter: an action that is never optimal anywhere
simply does not appear, and you have learned something about your action set.

What you supply instead of FtC is a cost model, and a cost model has a truth
condition. `calibration_report` and `empirical_loss` are how you check whether
yours is wrong. That is the entire trade: a constant that can only be
re-stipulated, for an assumption reality can refute.

Ties are returned, never broken. `Policy.optimal` yields a frozenset of the
argmin actions, so a point sitting exactly on a cost-curve crossing hands you
both and the caller has to look. This is the same discipline the region
classifier in `paraconsistent` 0.2.4 gives up when it returns `str` and settles
`dc >= 0` silently.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from paraconsistent import ParaconsistentBlock

__all__ = [
    "Action", "Policy", "etau_belief", "ftc_policy",
    "empirical_loss", "calibration_report",
    "OBSTACLE", "CLEAR", "FAULT_CONFLICT", "FAULT_SILENCE", "WORLDS",
]

# World states. All four are observable after the fact — the two fault states
# from maintenance logs — which is what keeps the model falsifiable.
OBSTACLE = "obstacle"
CLEAR = "clear"
FAULT_CONFLICT = "fault_conflict"   # channels contradicted; one was broken
FAULT_SILENCE = "fault_silence"     # channels both dead or out of range
WORLDS = (OBSTACLE, CLEAR, FAULT_CONFLICT, FAULT_SILENCE)


# --------------------------------------------------------------- evidence


def etau_belief(mu: float, lam: float) -> dict[str, float]:
    """Map an Eτ annotation to a distribution over world states.

    Built entirely from quantities the logic already computes, and it sums to
    exactly 1 by construction:

        φ = 1 − |Dct|   is the mass that is *decided*, split by μE
        max(0, +Dct)    is the mass of mutually contradicting reports
        max(0, −Dct)    is the mass of mutual silence

        φ + |Dct| = 1

    The point of routing through Dct's sign is that μER cannot do it. μER sends
    p(1,1), p(0,0) and p(0.5,0.5) all to exactly 0.5 — "both channels screaming
    opposite things", "both channels dead" and "both channels calmly reporting a
    coin-flip" become the same number. They call for different actions, so a
    decision layer built on μER alone is structurally unable to choose between
    them.
    """
    c = ParaconsistentBlock(mu=mu, lam=lam).complete
    decided = c.phi
    return {
        OBSTACLE: decided * c.muE,
        CLEAR: decided * (1.0 - c.muE),
        FAULT_CONFLICT: max(0.0, c.dct),
        FAULT_SILENCE: max(0.0, -c.dct),
    }


# ----------------------------------------------------------------- actions


@dataclass(frozen=True)
class Action:
    """An available action and its loss under each world state."""

    name: str
    losses: Mapping[str, float]

    def expected_loss(self, belief: Mapping[str, float]) -> float:
        total = 0.0
        for world, p in belief.items():
            if p:
                try:
                    total += p * self.losses[world]
                except KeyError:
                    raise KeyError(
                        f"action {self.name!r} has no loss for world {world!r}"
                    ) from None
        return total


@dataclass(frozen=True)
class Policy:
    """Argmin-expected-loss over an action set, with ties preserved."""

    actions: Sequence[Action]
    belief: Callable[[float, float], Mapping[str, float]] = etau_belief
    tol: float = 1e-12

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("a policy needs at least one action")

    def optimal(self, mu: float, lam: float) -> frozenset[str]:
        """Every action within `tol` of minimal expected loss.

        Returns a set. On a cost-curve crossing you get both names, because the
        model genuinely does not prefer either and saying otherwise would be
        the thing this module exists to avoid.
        """
        b = self.belief(mu, lam)
        scored = [(a.name, a.expected_loss(b)) for a in self.actions]
        best = min(loss for _, loss in scored)
        return frozenset(n for n, loss in scored if loss - best <= self.tol)

    # -------------------------------------------------------- the partition

    def partition(self, steps: int = 201) -> dict[frozenset[str], int]:
        """Sweep the annotation square; count cells per induced region."""
        cells: Counter[frozenset[str]] = Counter()
        for i in range(steps):
            for j in range(steps):
                cells[self.optimal(i / (steps - 1), j / (steps - 1))] += 1
        return dict(cells)

    def region_count(self, steps: int = 201) -> int:
        """How many regions the cost model actually induces — an output.

        Counts only regions with interior (singleton argmin sets). Tie sets are
        boundaries between regions, not regions of their own.
        """
        return sum(1 for r in self.partition(steps) if len(r) == 1)

    def unreachable(self, steps: int = 201) -> set[str]:
        """Actions that are never optimal anywhere — empirically useless."""
        reached = {n for r in self.partition(steps) for n in r}
        return {a.name for a in self.actions} - reached

    def render(self, steps: int = 41, glyphs: Mapping[str, str] | None = None) -> str:
        """ASCII map of the induced partition over the (μ, λ) square."""
        names = sorted({a.name for a in self.actions})
        glyphs = glyphs or {n: n[0].upper() for n in names}
        rows = []
        for j in range(steps - 1, -1, -1):
            row = []
            for i in range(steps):
                opt = self.optimal(i / (steps - 1), j / (steps - 1))
                row.append(glyphs[next(iter(opt))] if len(opt) == 1 else "·")
            rows.append(f"{j / (steps - 1):4.2f} │" + "".join(row))
        rows.append("     └" + "─" * steps)
        rows.append("      μ=0" + " " * (steps - 9) + "μ=1")
        return "\n".join(rows)


def ftc_policy(ftc: float, act: Action, refrain: Action) -> Callable[[float, float], str]:
    """The baseline this module argues against: threshold μER at FtC.

    Two regions, always, whatever the costs are — the count is stipulated in
    advance by the shape of the rule rather than derived from the problem.
    """
    def decide(mu: float, lam: float) -> str:
        muER = ParaconsistentBlock(mu=mu, lam=lam, FtC=ftc).complete.muER
        return act.name if muER > ftc else refrain.name
    return decide


# ----------------------------------------------------------- falsification


Record = tuple[float, float, str]   # (μ, λ, world that actually obtained)


def empirical_loss(
    decide: Callable[[float, float], str | frozenset[str]],
    records: Iterable[Record],
    actions: Sequence[Action],
) -> float:
    """Mean realised loss of a decision rule on labelled outcomes.

    This is the falsification hook. A cost model that is wrong about the world
    loses here, measurably, against a model that is less wrong — which is the
    property FtC does not have and cannot be given.
    """
    by_name = {a.name: a for a in actions}
    total = n = 0.0
    for mu, lam, world in records:
        chosen = decide(mu, lam)
        if isinstance(chosen, frozenset):
            # Ties cost the average of what they tie between: no free lunch
            # from refusing to choose, and no hidden tie-break either.
            total += sum(by_name[c].losses[world] for c in chosen) / len(chosen)
        else:
            total += by_name[chosen].losses[world]
        n += 1
    return total / n if n else float("nan")


def calibration_report(
    records: Iterable[Record],
    belief: Callable[[float, float], Mapping[str, float]] = etau_belief,
    bins: int = 5,
) -> dict[str, list[tuple[float, float, int]]]:
    """Predicted vs. observed frequency per world state, bucketed.

    If the model is calibrated, predicted ≈ observed in every populated bucket.
    Systematic divergence means the belief function is wrong — the concrete
    sense in which this parameterisation, unlike a threshold, can fail.

    Returns {world: [(mean_predicted, observed_frequency, count), ...]}.
    """
    buckets: dict[str, dict[int, list[tuple[float, bool]]]] = {
        w: defaultdict(list) for w in WORLDS
    }
    for mu, lam, world in records:
        b = belief(mu, lam)
        for w in WORLDS:
            p = b.get(w, 0.0)
            idx = min(int(p * bins), bins - 1)
            buckets[w][idx].append((p, world == w))

    out: dict[str, list[tuple[float, float, int]]] = {}
    for w, by_bin in buckets.items():
        rows = []
        for idx in sorted(by_bin):
            pairs = by_bin[idx]
            pred = sum(p for p, _ in pairs) / len(pairs)
            obs = sum(1 for _, hit in pairs if hit) / len(pairs)
            rows.append((pred, obs, len(pairs)))
        out[w] = rows
    return out


def brier_score(
    records: Iterable[Record],
    belief: Callable[[float, float], Mapping[str, float]] = etau_belief,
) -> float:
    """Multi-class Brier score — lower is better, 0 is perfect."""
    total = n = 0.0
    for mu, lam, world in records:
        b = belief(mu, lam)
        total += sum((b.get(w, 0.0) - (1.0 if world == w else 0.0)) ** 2
                     for w in WORLDS)
        n += 1
    return total / n if n else float("nan")
