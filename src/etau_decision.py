"""Cost-derived partitions of the Eτ annotation square.

The twelve-region readout partitions annotations by the *geometry of the
lattice*: thresholds on Dc and Dct, with FtC drawing the lines. Nothing about
that geometry knows what you are deciding, so FtC is a stipulation — you can
replace it with a different number but you cannot discover it was wrong.

This module partitions annotations by *what you would do about them* instead.
Two annotations fall in the same region iff the same action minimises expected
loss. The boundaries land where expected-cost curves cross.

The region count is then a function of the action/loss parameter rather than a
number painted on a lattice diagram. **It is not "unchosen."** An earlier draft
of this docstring said "nobody chose 4"; that was false about the process that
produced the example, and Grok Build's A5 probe measured it — vocabularies of
1/2/3/4/5 actions induce 1/2/3/4/4 regions. The free parameter did not vanish.
It changed type, from `float FtC` to `(Action, loss matrix)[]`, and that is a
strict upgrade for one reason only: the new type is multi-dimensional and
*checkable*. `empirical_loss`, `calibration_report` and `unreachable_under` are
the checks. FtC has no equivalent, because a threshold makes no claim about the
world that data could contradict.

The arbitrary lines in here are written down in plain comments below. They are
not enforced, deliberately — see the note above `TIE_TOL` for why the mechanism
that used to enforce them was removed.

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
    "Action", "Policy", "etau_belief", "ftc_policy", "HistogramBelief",
    "empirical_loss", "calibration_report", "brier_score", "unreachable_under",
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


# --- Arbitrary lines in this module. -----------------------------------------
#
# Written down, not enforced. An earlier version of this block registered each
# of these with a `Choice` object and an AST scan that failed the build on any
# undeclared numeric literal. That mechanism was removed, because checking what
# it would actually have caught answers the question: of the five BREAKS in
# `critique/CRITIQUE.md`, ZERO are numeric literals. The hourglass topology is
# prose; phi-as-decided-mass is a formula; deployment-vs-annotation is a measure;
# the action vocabulary is a list; belief-as-posterior is a role confusion. The
# scan fired only on housekeeping — a tolerance, three grid sizes, a bin count —
# while creating the appearance that the arbitrary lines were accounted for.
#
# Documented amnesia is worse than undocumented amnesia: it looks like diligence,
# so it does not get questioned. These comments make no coverage claim.

TIE_TOL = 1e-12
# Alternatives: 0.0 (exact equality, ties never fire); 1e-9/1e-6 (wider bands);
# relative tolerance scaled to loss magnitudes.
# Hides: the difference between a genuine tie and two actions differing by less
# than float noise. Too tight and the set-valued type silently degenerates back
# to single-valued — the exact bug it exists to fix. Grok Build's A7.

GRID_STEPS = 201
# Alternatives: 101 (cheaper, misses thin regions); 1001; adaptive refinement;
# exact polytope enumeration — the walls are piecewise-linear, so this is
# computable and would end the sampling question entirely.
# Hides: every region narrower than 1/200 of the square. `region_count` is a
# lower bound and `unreachable` an upper bound, not facts. Grok's A5 saw a fifth
# action fail to carve interior "at this grid" — a resolution artefact.

RENDER_STEPS = 41
# ASCII map only; no computed result depends on it. Hides everything thinner
# than a character cell. The map is a picture of the partition, not evidence.

CALIBRATION_BINS = 5
# Alternatives: 10; quantile bins; no bucketing at all (reliability curve or a
# proper scoring rule alone).
# Hides: miscalibration structure finer than 0.2 of predicted probability, and
# any dependence of calibration on WHERE in the square the prediction came from.
# The 0.55 -> 0.94 drift this module reports against itself is measured here.

# The belief feature map itself — (phi*muE, phi*(1-muE), max(Dct,0), max(-Dct,0))
# — is the largest drawn line in this file, and it is not a number, which is why
# no scanner was ever going to see it. Alternatives: split by the radial distance
# D instead of phi (disagrees on 54% of the square, Grok's A2); hard argmax on
# Dc (21%); any model fit from data (see HistogramBelief). Hides: weight of
# evidence — 1-vs-1, 10-vs-10 and 5000-vs-5000 all land on p(0.5,0.5) and get
# identical mass. Same collapse julia3claude's CRITIQUE_REPORT.md found in muER.


def etau_belief(mu: float, lam: float) -> dict[str, float]:
    """A **feature map** from an Eτ annotation onto the simplex. Not a posterior.

    Arithmetic — it does sum to exactly 1 by construction:

        φ = 1 − |Dct|   split by μE
        max(0, +Dct)    mass of mutually contradicting reports
        max(0, −Dct)    mass of mutual silence

        φ + |Dct| = 1

    What it is good for: routing through Dct's SIGN, which μER cannot do. μER
    sends p(1,1), p(0,0) and p(0.5,0.5) all to exactly 0.5 — "both channels
    screaming opposite things", "both channels dead" and "both channels calmly
    reporting a coin-flip" become one number. This separates them.

    What it is NOT: privileged, or derived, or a good probabilistic model. Two
    independent findings, both kept:

      * It fails its own `calibration_report` on the shipped simulator —
        systematically underconfident, predicting 0.55 where the world delivers
        0.94, because splitting φ by μE is a linear readout of Dc that cannot
        sharpen faster than the evidence separates.
      * A 20×20 histogram of the training set that does not know Eτ exists beats
        it on held-out Brier by three orders of magnitude (0.000075 vs 0.146) and
        on mean loss (−0.741 vs −0.524). Grok Build's A9.

    So: use it as a feature map, or fit `HistogramBelief` and use that. The claim
    that a reparameterisation of the unit square is a probability was wrong.
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
    tol: float = TIE_TOL

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

    def partition(self, steps: int = GRID_STEPS) -> dict[frozenset[str], int]:
        """Sweep the annotation square; count cells per induced region."""
        cells: Counter[frozenset[str]] = Counter()
        for i in range(steps):
            for j in range(steps):
                cells[self.optimal(i / (steps - 1), j / (steps - 1))] += 1
        return dict(cells)

    def region_count(self, steps: int = GRID_STEPS) -> int:
        """How many regions the cost model actually induces — an output.

        Counts only regions with interior (singleton argmin sets). Tie sets are
        boundaries between regions, not regions of their own.
        """
        return sum(1 for r in self.partition(steps) if len(r) == 1)

    def unreachable(self, steps: int = GRID_STEPS) -> set[str]:
        """Actions that are never optimal anywhere — empirically useless."""
        reached = {n for r in self.partition(steps) for n in r}
        return {a.name for a in self.actions} - reached

    def render(self, steps: int = RENDER_STEPS, glyphs: Mapping[str, str] | None = None) -> str:
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
        rows.append("      μ=0" + " " * (steps - len("      μ=0") + 0) + "μ=1")
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
    bins: int = CALIBRATION_BINS,
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


def unreachable_under(policy: "Policy", records: Iterable[Record]) -> set[str]:
    """Actions never optimal under the WORLD measure, not the annotation grid.

    `Policy.unreachable` sweeps a uniform grid over the annotation square. That
    is the wrong measure for deployment: an action can be optimal on a positive
    -measure set of annotations that the actual sensors never produce.

    Grok Build's A3 found the gap by measurement. On a binary-only world where
    faults never obtain, the 4-action policy is **worse** than the 2-action
    policy by ~0.071 mean loss — it spends measure on `recalibrate`/`approach`
    that can never pay — and `unreachable()` does not catch it, because those
    actions genuinely are optimal on some annotations. Annotation-square
    optimality is not deployment optimality. This function measures the latter.
    """
    seen: set[str] = set()
    for mu, lam, _world in records:
        seen |= policy.optimal(mu, lam)
    return {a.name for a in policy.actions} - seen


@dataclass
class HistogramBelief:
    """A belief model fit from data, which does not know Eτ exists.

    Bins the annotation square and reads off empirical world frequencies. Grok
    Build's A9 used exactly this to beat `etau_belief` by three orders of
    magnitude on held-out Brier — the result that demotes the lattice readout
    from posterior to feature map.

    Kept deliberately dumb and dependency-free. It is not proposed as the right
    model; it is here so that "natural from the logic" can never again be
    confused with "good", and so the comparison is one import away.
    """

    bins: int = 20
    prior_strength: float = 1.0
    _table: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    _fallback: dict[str, float] = field(default_factory=dict)

    def _cell(self, mu: float, lam: float) -> tuple[int, int]:
        b = self.bins
        return (min(int(mu * b), b - 1), min(int(lam * b), b - 1))

    def fit(self, records: Iterable[Record]) -> "HistogramBelief":
        counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
        overall: Counter = Counter()
        for mu, lam, world in records:
            counts[self._cell(mu, lam)][world] += 1
            overall[world] += 1
        tot = sum(overall.values()) or 1
        self._fallback = {w: overall[w] / tot for w in WORLDS}
        self._table = {}
        for cell, c in counts.items():
            n = sum(c.values()) + self.prior_strength * len(WORLDS)
            self._table[cell] = {
                w: (c[w] + self.prior_strength * self._fallback.get(w, 0.0)) / n
                for w in WORLDS
            }
        return self

    def __call__(self, mu: float, lam: float) -> dict[str, float]:
        return self._table.get(self._cell(mu, lam), dict(self._fallback))
