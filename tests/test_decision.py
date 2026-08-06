#!/usr/bin/env python3
"""Tests for cost-derived partitions of the Eτ square.

Runs under pytest, or standalone:  python tests/test_decision.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paraconsistent import ParaconsistentBlock                      # noqa: E402
from etau_decision import (                                          # noqa: E402
    Action, Policy, brier_score, empirical_loss, etau_belief,
    OBSTACLE, CLEAR, FAULT_CONFLICT, FAULT_SILENCE, WORLDS,
)

TOL = 1e-12

A = Action("a", {OBSTACLE: 0.0, CLEAR: 1.0, FAULT_CONFLICT: 1.0, FAULT_SILENCE: 1.0})
B = Action("b", {OBSTACLE: 1.0, CLEAR: 0.0, FAULT_CONFLICT: 1.0, FAULT_SILENCE: 1.0})
C = Action("c", {OBSTACLE: 1.0, CLEAR: 1.0, FAULT_CONFLICT: 0.0, FAULT_SILENCE: 1.0})
D = Action("d", {OBSTACLE: 1.0, CLEAR: 1.0, FAULT_CONFLICT: 1.0, FAULT_SILENCE: 0.0})


def grid(steps=21):
    for i in range(steps):
        for j in range(steps):
            yield i / (steps - 1), j / (steps - 1)


# ---------------------------------------------------------------- belief


def test_belief_is_a_distribution():
    """The four masses sum to exactly 1 everywhere, by φ + |Dct| = 1."""
    for mu, lam in grid(41):
        b = etau_belief(mu, lam)
        assert abs(sum(b.values()) - 1.0) < 1e-12, (mu, lam, b)
        assert all(v >= -TOL for v in b.values())


def test_belief_at_the_extremes():
    """Each corner of the lattice puts all mass on one world state."""
    for (mu, lam), world in {
        (1.0, 0.0): OBSTACLE,
        (0.0, 1.0): CLEAR,
        (1.0, 1.0): FAULT_CONFLICT,
        (0.0, 0.0): FAULT_SILENCE,
    }.items():
        b = etau_belief(mu, lam)
        assert abs(b[world] - 1.0) < TOL, (mu, lam, b)


def test_belief_separates_what_muER_conflates():
    """p(1,1), p(0,0) and p(0.5,0.5) share a μER but get distinct beliefs."""
    points = [(1.0, 1.0), (0.0, 0.0), (0.5, 0.5)]
    muERs = {ParaconsistentBlock(mu=m, lam=l).complete.muER for m, l in points}
    assert muERs == {0.5}, "premise: μER cannot tell these apart"

    beliefs = [tuple(round(etau_belief(m, l)[w], 12) for w in WORLDS)
               for m, l in points]
    assert len(set(beliefs)) == 3, "belief model must tell them apart"


def test_contradiction_and_silence_never_coexist():
    """Dct has one sign, so at most one fault mode carries mass."""
    for mu, lam in grid(41):
        b = etau_belief(mu, lam)
        assert b[FAULT_CONFLICT] == 0.0 or b[FAULT_SILENCE] == 0.0


# ---------------------------------------------------------------- policy


def test_ties_are_returned_not_broken():
    """The whole point: a cost-curve crossing yields both names."""
    tie = Action("tie", dict(A.losses))
    p = Policy([A, tie])
    for mu, lam in grid(11):
        assert p.optimal(mu, lam) == frozenset({"a", "tie"}), (mu, lam)


def test_optimal_is_always_nonempty():
    p = Policy([A, B, C, D])
    for mu, lam in grid(41):
        assert p.optimal(mu, lam)


def test_optimal_really_is_the_argmin():
    p = Policy([A, B, C, D])
    by_name = {a.name: a for a in (A, B, C, D)}
    for mu, lam in grid(21):
        b = etau_belief(mu, lam)
        chosen = p.optimal(mu, lam)
        best = min(a.expected_loss(b) for a in (A, B, C, D))
        for name in chosen:
            assert abs(by_name[name].expected_loss(b) - best) < 1e-9
        for a in (A, B, C, D):
            if a.name not in chosen:
                assert a.expected_loss(b) > best


def test_region_count_is_an_output():
    """Four actions each optimal at a corner induce four regions."""
    assert Policy([A, B, C, D]).region_count(101) == 4


def test_dominated_actions_disappear():
    """An action that is never best does not become a region."""
    dominated = Action("never", {w: 99.0 for w in WORLDS})
    p = Policy([A, B, C, D, dominated])
    assert p.unreachable(101) == {"never"}
    assert p.region_count(101) == 4, "the useless action must not add a region"


def test_a_single_action_gives_one_region():
    p = Policy([A])
    assert p.region_count(41) == 1
    assert p.unreachable(41) == set()


def test_scaling_all_losses_preserves_the_partition():
    """argmin depends on order, not on units."""
    p1 = Policy([A, B, C, D])
    scaled = [Action(a.name, {w: 7.0 * v + 3.0 for w, v in a.losses.items()})
              for a in (A, B, C, D)]
    p2 = Policy(scaled)
    for mu, lam in grid(41):
        assert p1.optimal(mu, lam) == p2.optimal(mu, lam), (mu, lam)


def test_empty_action_set_is_rejected():
    try:
        Policy([])
    except ValueError:
        return
    raise AssertionError("empty action set should raise")


def test_missing_loss_is_reported():
    incomplete = Action("partial", {OBSTACLE: 1.0})
    try:
        Policy([incomplete]).optimal(0.5, 0.2)
    except KeyError as exc:
        assert "partial" in str(exc)
        return
    raise AssertionError("a missing world loss should raise")


# ------------------------------------------------------- falsification


def test_empirical_loss_reads_off_the_matrix():
    records = [(1.0, 0.0, OBSTACLE), (0.0, 1.0, CLEAR)]
    assert abs(empirical_loss(lambda m, l: "a", records, [A]) - 0.5) < TOL
    assert abs(empirical_loss(lambda m, l: "b", records, [B]) - 0.5) < TOL


def test_ties_cost_their_average():
    """Refusing to choose buys no free lunch."""
    records = [(1.0, 0.0, OBSTACLE)]                      # A costs 0, B costs 1
    loss = empirical_loss(lambda m, l: frozenset({"a", "b"}), records, [A, B])
    assert abs(loss - 0.5) < TOL


def test_brier_rewards_the_better_model():
    """A model blind to contradiction loses on data containing faults."""
    records = [(1.0, 1.0, FAULT_CONFLICT), (0.0, 0.0, FAULT_SILENCE),
               (1.0, 0.0, OBSTACLE), (0.0, 1.0, CLEAR)]

    def blind(mu, lam):
        c = ParaconsistentBlock(mu=mu, lam=lam).complete
        return {OBSTACLE: c.muE, CLEAR: 1 - c.muE,
                FAULT_CONFLICT: 0.0, FAULT_SILENCE: 0.0}

    assert brier_score(records) < brier_score(records, blind)


def test_perfect_model_scores_zero():
    records = [(1.0, 1.0, FAULT_CONFLICT), (1.0, 0.0, OBSTACLE)]
    assert brier_score(records) < TOL


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
