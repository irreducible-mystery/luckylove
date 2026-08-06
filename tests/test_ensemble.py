#!/usr/bin/env python3
"""Tests for running the pipeline under several defensible lines at once.

The property that makes the spread honest is monotonicity: adding a member can
only raise measured disagreement. That makes every number a lower bound, which
is what separates this from a coverage claim. It is pinned first because if it
ever fails, the whole justification for the module goes with it.

Runs under pytest, or standalone:  python tests/test_ensemble.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from etau_decision import Policy, etau_belief                        # noqa: E402
from etau_ensemble import (                                          # noqa: E402
    DEFAULT_BELIEFS, Ensemble, hard_belief, radial_belief,
)

TOL = 1e-12


def _actions():
    from cost_derived_partition import ALL_ACTIONS
    return ALL_ACTIONS


def _grid(steps=41):
    return [(i / (steps - 1), j / (steps - 1))
            for i in range(steps) for j in range(steps)]


# ------------------------------------------------------- the load-bearing bit


def test_sensitivity_is_monotone_in_members():
    """Adding a member never lowers the spread. This is why it is a lower bound."""
    grid = _grid()
    chains = [
        (etau_belief,),
        (etau_belief, radial_belief),
        (etau_belief, radial_belief, hard_belief),
    ]
    seen = [Ensemble(_actions(), b).sensitivity(grid) for b in chains]
    for a, b in zip(seen, seen[1:]):
        assert b >= a - TOL, f"sensitivity fell when a member was added: {seen}"


def test_adding_a_member_never_shrinks_an_answer():
    """Pointwise version of the same property: the union only grows."""
    small = Ensemble(_actions(), (etau_belief,))
    big = small.add(radial_belief).add(hard_belief)
    for mu, lam in _grid(21):
        assert small.optimal(mu, lam) <= big.optimal(mu, lam), (mu, lam)


def test_union_contains_every_members_choice():
    e = Ensemble(_actions())
    for mu, lam in _grid(21):
        u = e.optimal(mu, lam)
        for choice in e.per_member(mu, lam).values():
            assert choice <= u


def test_singleton_exactly_when_members_agree():
    e = Ensemble(_actions())
    for mu, lam in _grid(21):
        members = list(e.per_member(mu, lam).values())
        all_same = all(m == members[0] and len(m) == 1 for m in members)
        assert e.agrees(mu, lam) == all_same, (mu, lam, members)


def test_one_member_reduces_to_a_plain_policy():
    """With a single belief map the ensemble must be the policy, exactly."""
    e = Ensemble(_actions(), (etau_belief,))
    p = Policy(_actions(), belief=etau_belief)
    for mu, lam in _grid(21):
        assert e.optimal(mu, lam) == p.optimal(mu, lam), (mu, lam)


def test_empty_ensemble_is_rejected():
    try:
        Ensemble(_actions(), ())
    except ValueError:
        return
    raise AssertionError("an ensemble with no members should raise")


# --------------------------------------------------------------- the finding


def test_the_belief_map_choice_changes_the_action_most_of_the_time():
    """The headline. φ-as-decided-mass was never a detail.

    Over half the annotation square, three equally defensible mass maps
    disagree about what to do. An annotation saying "this was a choice" is
    silent about that; the plural answer is not.
    """
    e = Ensemble(_actions())
    s = e.sensitivity(_grid(101))
    assert s > 0.5, f"expected the choice to dominate, measured {s:.3f}"
    assert e.n_members == 3


def test_deployment_sensitivity_is_reported_separately():
    """Grid sensitivity is about the square; deployment sensitivity is about you."""
    from cost_derived_partition import simulate
    e = Ensemble(_actions())
    records = simulate(4000, seed=99)
    d = e.deployment_sensitivity(records)
    assert 0.0 <= d <= 1.0
    # They are genuinely different measures over the same ensemble; asserting
    # only that both exist and are finite. Grok's A3 is the reason both ship.
    assert d == d


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
