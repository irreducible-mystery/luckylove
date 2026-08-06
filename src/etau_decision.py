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

Every arbitrary line in here is registered in `choices.py` and printable with
`format_registry()`. That is the mechanism, replacing a discipline that has
already failed three times in this repository's own history.

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

from choices import declare, format_registry

__all__ = [
    "Action", "Policy", "etau_belief", "ftc_policy", "HistogramBelief",
    "empirical_loss", "calibration_report", "brier_score", "unreachable_under",
    "OBSTACLE", "CLEAR", "FAULT_CONFLICT", "FAULT_SILENCE", "WORLDS",
    "format_registry",
]

# World states. All four are observable after the fact — the two fault states
# from maintenance logs — which is what keeps the model falsifiable.
OBSTACLE = "obstacle"
CLEAR = "clear"
FAULT_CONFLICT = "fault_conflict"   # channels contradicted; one was broken
FAULT_SILENCE = "fault_silence"     # channels both dead or out of range
WORLDS = (OBSTACLE, CLEAR, FAULT_CONFLICT, FAULT_SILENCE)


# --------------------------------------------------------------- evidence


# --- Arbitrary lines in this module, drawn on purpose and on the record. -------

TIE_TOL = declare(
    "tie_tolerance",
    1e-12,
    alternatives=["0.0 — exact float equality only, so ties essentially never fire",
                  "1e-9 / 1e-6 — wider bands, more boundaries reported as ties",
                  "relative tolerance scaled to the loss magnitudes"],
    consequence="sets how wide a cost-curve crossing has to be before `optimal` "
                "returns both actions. Too tight and the set-valued type silently "
                "degenerates back to single-valued — the exact bug it exists to fix.",
    collapses="the difference between a genuine tie and two actions that merely "
              "differ by less than float noise; below this width the module has "
              "no opinion and says so, which is itself a drawn line",
    chosen_by="claude-opus-5; flagged by Grok Build's A7 as a scoring convention "
              "rather than a physical fact",
    evidence="critique/CRITIQUE.md A7; tests/test_decision.py::test_ties_cost_their_average",
)

GRID_STEPS = declare(
    "partition_grid_steps",
    201,
    alternatives=["101 (4x cheaper, misses thin regions)",
                  "1001 (finds slivers, ~25x cost)",
                  "adaptive refinement near detected walls",
                  "exact polytope enumeration — the walls are piecewise-linear, "
                  "so this is computable and would end the sampling question"],
    consequence="a region thinner than one cell is invisible, so `region_count` "
                "and `unreachable` are lower and upper bounds respectively, not "
                "facts. Grok Build's A5 saw a 5th action fail to carve interior "
                "'at this grid' — a resolution artefact, not a finding.",
    collapses="every region narrower than 1/200 of the square, and the entire "
              "question of whether a reported region is genuinely 2-dimensional",
    chosen_by="claude-opus-5, for demo latency",
    evidence="critique/CRITIQUE.md §4 'with grid-resolution caveats documented'",
)

RENDER_STEPS = declare(
    "render_grid_steps",
    41,
    alternatives=["21 (fits narrower terminals)", "81 (needs a wide terminal)"],
    consequence="ASCII map resolution only; no effect on any computed result",
    collapses="everything thinner than one character cell — the map is a picture "
              "of the partition, not evidence about it",
    chosen_by="claude-opus-5, to fit an 80-column terminal",
    evidence="examples/cost_derived_partition.py section 3",
)

CALIBRATION_BINS = declare(
    "calibration_bins",
    5,
    alternatives=["10 (finer, needs more data per bucket)",
                  "quantile bins (equal population, unequal width)",
                  "no bucketing — a reliability curve or a proper scoring rule alone"],
    consequence="coarser bins hide miscalibration inside a bucket; finer bins "
                "make every bucket noisy. The 0.55->0.94 drift this module reports "
                "against itself is measured at this bin count.",
    collapses="all miscalibration structure finer than 0.2 of predicted "
              "probability, and any dependence of calibration on WHERE in the "
              "annotation square the prediction came from",
    chosen_by="claude-opus-5",
    evidence="examples/cost_derived_partition.py section 6",
)


declare(
    "phi_as_decided_mass",
    "(phi*muE, phi*(1-muE), max(Dct,0), max(-Dct,0))",
    alternatives=[
        "radial: split by the normalised distance D instead of phi",
        "hard argmax on Dc with a contradiction reserve",
        "any calibrated model fit from data (see HistogramBelief)",
    ],
    consequence="Grok Build's A2 probe: the radial alternative disagrees with "
                "this one on 54% of the annotation square and the hard-argmax "
                "alternative on 21%, under the same four actions. Region count "
                "stays 4 in all three — the count is stable, the WALLS move.",
    collapses="weight of evidence. 1-vs-1, 10-vs-10 and 5000-vs-5000 all land on "
              "p(0.5,0.5) and get identical mass. Verified against julia3claude's "
              "CRITIQUE_REPORT.md §F, which found the same collapse in muER.",
    chosen_by="claude-opus-5; called 'derived' in the first draft, which was the "
              "same costume this module criticises in FtC",
    evidence="critique/CRITIQUE.md A2; critique/attack_results.json",
)


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

    bins: int = declare(
        "histogram_bins",
        20,
        alternatives=["10 (coarser, less variance, more bias)",
                      "50 (finer, needs far more data)",
                      "adaptive/quantile bins", "isotonic or Platt on (Dc, Dct)"],
        consequence="bin count trades bias against variance; at 20 bins the "
                    "shipped 20k-record simulator gives ~50 records/cell",
        collapses="everything inside a cell — two annotations in the same bin are "
                  "indistinguishable, so this model has a resolution floor exactly "
                  "the way the 12-region readout does, just at a different scale",
        chosen_by="Grok Build's A9 probe; adopted here at its value",
        evidence="critique/CRITIQUE.md A9",
    )
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
