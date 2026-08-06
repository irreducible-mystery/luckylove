#!/usr/bin/env python3
"""Tests pinning the findings in `critique/CRITIQUE.md` (Grok Build, xAI).

These are here so a future edit that re-oversells the argument fails instead of
persuading. They pin measurements, not policy — there is no mechanism here and
no claim that the arbitrary lines in this module are covered. Four of the five
BREAKS are conceded outright and pinned as conceded; A1 is pinned as a partial,
because both facts in it are true and they are different claims.

Runs under pytest, or standalone:  python tests/test_critique_findings.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from etau_decision import (                                          # noqa: E402
    HistogramBelief, Policy, brier_score, empirical_loss, etau_belief,
    unreachable_under, OBSTACLE, CLEAR, FAULT_CONFLICT, FAULT_SILENCE,
)

TOL = 1e-12
def _data():
    from cost_derived_partition import simulate
    return simulate(8000, seed=11), simulate(8000, seed=99)


def test_A9_a_model_ignorant_of_etau_beats_the_lattice_readout():
    """The result that demotes etau_belief from posterior to feature map."""
    train, held = _data()
    hb = HistogramBelief().fit(train)
    assert brier_score(held, hb) < brier_score(held) / 10


def test_A3_deployment_reachability_differs_from_grid_reachability():
    """An action can be grid-reachable and world-useless."""
    from cost_derived_partition import simulate, ALL_ACTIONS
    p = Policy(ALL_ACTIONS)
    binary = simulate(8000, seed=7, fault_rate=0.0)
    assert p.unreachable(101) == set(), "premise: grid says all four are live"
    dead = unreachable_under(p, binary)
    assert dead == {"recalibrate", "approach"}, dead


def test_A5_region_count_tracks_the_action_vocabulary():
    """'Nobody chose N' is false: N is a function of a chosen vocabulary."""
    from cost_derived_partition import PROCEED, HALT, RECALIBRATE, APPROACH
    for actions, expected in [
        ([HALT], 1),
        ([PROCEED, HALT], 2),
        ([PROCEED, HALT, RECALIBRATE], 3),
        ([PROCEED, HALT, RECALIBRATE, APPROACH], 4),
    ]:
        assert Policy(actions).region_count(101) == expected, actions


def test_A2_alternative_mass_maps_move_the_walls():
    """φ-splitting is chosen, not derived: a sibling map disagrees materially."""
    from paraconsistent import ParaconsistentBlock
    from cost_derived_partition import ALL_ACTIONS

    def radial(mu, lam):
        c = ParaconsistentBlock(mu=mu, lam=lam).complete
        decided = 1.0 - c.D
        return {OBSTACLE: decided * c.muE, CLEAR: decided * (1 - c.muE),
                FAULT_CONFLICT: c.D * max(0.0, c.dct) / max(abs(c.dct), 1e-12),
                FAULT_SILENCE: c.D * max(0.0, -c.dct) / max(abs(c.dct), 1e-12)}

    a, b = Policy(ALL_ACTIONS), Policy(ALL_ACTIONS, belief=radial)
    steps = 101
    diff = sum(a.optimal(i / (steps - 1), j / (steps - 1))
               != b.optimal(i / (steps - 1), j / (steps - 1))
               for i in range(steps) for j in range(steps))
    frac = diff / steps ** 2
    assert frac > 0.10, f"expected materially different walls, got {frac:.1%}"


def test_A1_the_pinch_survives_and_so_does_the_diagonal():
    """Both A1 facts hold; they are different claims.

    Grok: the whole μ=λ diagonal is Dcr=0. True.
    Mine: the neutral set is pinched to zero width at exactly p(0.5,0.5). Also
    true — and removing that one point disconnects the set.
    """
    from paraconsistent import ParaconsistentBlock

    def neutral(mu, lam):
        return ParaconsistentBlock(mu=mu, lam=lam).complete.dcr == 0.0

    assert all(neutral(t / 400, t / 400) for t in range(401)), "diagonal"

    def band(mu, n=20000):
        ls = [j / n for j in range(n + 1) if neutral(mu, j / n)]
        return (max(ls) - min(ls)) if ls else None

    assert band(0.5) == 0.0, "pinched at the centre"
    assert len([j / 20000 for j in range(20001) if neutral(0.5, j / 20000)]) == 1
    assert band(0.40) > band(0.45) > band(0.49) > 0.0, "widens away from centre"


# ---------------------------------------------------------------- runner


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
