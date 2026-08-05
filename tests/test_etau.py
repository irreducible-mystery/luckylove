#!/usr/bin/env python3
"""Verification of the downloaded Eτ (PAL2v) implementation.

These tests do not exercise the library against itself — every expected value is
derived independently from the definitions in the Eτ literature (see
docs/paraconsistent-annotated-evidential-logic-etau.md). The point is to
establish that the artefact in vendor/ actually implements the logic it claims
to, not merely that it imports.

Runs under pytest, or standalone:

    python tests/test_etau.py
"""

import math
import random

from paraconsistent import ParaconsistentBlock

TOL = 1e-12


def analyse(mu, lam, **params):
    return ParaconsistentBlock(mu=mu, lam=lam, **params).complete


def grid(steps=21):
    for i in range(steps):
        for j in range(steps):
            yield i / (steps - 1), j / (steps - 1)


# ---------------------------------------------------------------- degrees


def test_core_degrees():
    """Dc = μ − λ and Dct = μ + λ − 1, everywhere on the lattice."""
    for mu, lam in grid():
        c = analyse(mu, lam)
        assert abs(c.dc - (mu - lam)) < TOL, (mu, lam, c.dc)
        assert abs(c.dct - (mu + lam - 1)) < TOL, (mu, lam, c.dct)
        assert -1 - TOL <= c.dc <= 1 + TOL
        assert -1 - TOL <= c.dct <= 1 + TOL


def test_annotation_recoverable_from_degrees():
    """(μ, λ) ↦ (Dc, Dct) is a bijection: μ = (Dc+Dct+1)/2, λ = (Dct−Dc+1)/2."""
    for mu, lam in grid():
        c = analyse(mu, lam)
        assert abs((c.dc + c.dct + 1) / 2 - mu) < TOL
        assert abs((c.dct - c.dc + 1) / 2 - lam) < TOL


def test_extreme_states():
    """The four corners of the lattice classify as the four extreme states."""
    expected = {
        (1.0, 0.0): ("t", 1.0, 0.0),    # True
        (0.0, 1.0): ("f", -1.0, 0.0),   # False
        (1.0, 1.0): ("┬", 0.0, 1.0),    # Inconsistent
        (0.0, 0.0): ("┴", 0.0, -1.0),   # Paracomplete
    }
    for (mu, lam), (label, dc, dct) in expected.items():
        c = analyse(mu, lam)
        assert c.label == label, (mu, lam, c.label, label)
        assert abs(c.dc - dc) < TOL
        assert abs(c.dct - dct) < TOL


def test_centre_is_indefinite():
    """μ = λ = 0.5 sits at the exact origin of the Dc/Dct plane."""
    c = analyse(0.5, 0.5)
    assert c.label == "I"
    assert abs(c.dc) < TOL and abs(c.dct) < TOL


# ---------------------------------------------------------------- geometry


def test_distance_matches_literature_formula():
    """The library's normalised apex distance equals d = √((1−|Dc|)² + Dct²).

    The library computes min(‖(μ,λ)−(1,0)‖, ‖(μ,λ)−(0,1)‖)/√½. Writing
    a = 1−μ, b = λ for the Dc ≥ 0 case, the literature form expands to
    (a+b)² + (b−a)² = 2a² + 2b², and the library form to 2a² + 2b². Identical.
    """
    random.seed(7)
    worst = 0.0
    points = [(random.random(), random.random()) for _ in range(20_000)]
    points += list(grid())
    for mu, lam in points:
        c = analyse(mu, lam)
        literature = math.hypot(1 - abs(c.dc), c.dct)
        worst = max(worst, abs(literature - c.d))
    assert worst < 1e-14, f"max divergence {worst}"
    return worst


def test_real_certainty():
    """D is d clamped to 1; Dcr = (1 − D)·sign(Dc), and stays inside [−1, 1]."""
    for mu, lam in grid():
        c = analyse(mu, lam)
        assert abs(c.D - min(c.d, 1.0)) < TOL
        sign = 1.0 if c.dc >= 0 else -1.0
        assert abs(c.dcr - (1 - c.D) * sign) < TOL
        assert -1 - TOL <= c.dcr <= 1 + TOL


def test_contradiction_discounts_certainty():
    """|Dcr| ≤ |Dc|: real certainty never exceeds nominal certainty."""
    for mu, lam in grid(41):
        c = analyse(mu, lam)
        assert abs(c.dcr) <= abs(c.dc) + TOL, (mu, lam, c.dc, c.dcr)


def test_uncontradicted_evidence_is_undiscounted():
    """On the Dct = 0 diagonal (μ + λ = 1) there is nothing to discount."""
    for i in range(21):
        mu = i / 20
        c = analyse(mu, 1 - mu)
        assert abs(c.dct) < TOL
        assert abs(c.dcr - c.dc) < TOL, (mu, c.dc, c.dcr)


# ---------------------------------------------------------------- evidences


def test_evidence_degrees():
    """μE, μECT, μER and φ follow their definitions and stay in [0, 1]."""
    for mu, lam in grid():
        c = analyse(mu, lam)
        assert abs(c.phi - (1 - abs(c.dct))) < TOL
        assert abs(c.phiE - c.phi) < TOL
        assert abs(c.muE - (c.dc + 1) / 2) < TOL
        assert abs(c.muECT - (c.dct + 1) / 2) < TOL
        assert abs(c.muER - (c.dcr + 1) / 2) < TOL
        for v in (c.phi, c.muE, c.muECT, c.muER):
            assert -TOL <= v <= 1 + TOL


def test_certainty_interval_vanishes_at_extremes():
    """φ = 0 exactly at ⊤ and ⊥ — maximum contradiction leaves no certainty."""
    assert abs(analyse(1.0, 1.0).phi) < TOL
    assert abs(analyse(0.0, 0.0).phi) < TOL
    assert abs(analyse(1.0, 0.0).phi - 1.0) < TOL
    assert abs(analyse(0.0, 1.0).phi - 1.0) < TOL


# ---------------------------------------------------------------- operators


def test_negation():
    """¬p(μ, λ) = p(λ, μ) flips the sign of Dc and preserves Dct."""
    for mu, lam in grid():
        p = analyse(mu, lam)
        neg = analyse(lam, mu)
        assert abs(neg.dc + p.dc) < TOL, (mu, lam)
        assert abs(neg.dct - p.dct) < TOL, (mu, lam)
        assert abs(neg.d - p.d) < TOL


def test_negation_fixes_the_extreme_contradictions():
    """⊤ and ⊥ are fixed points of negation."""
    for mu, lam in ((1.0, 1.0), (0.0, 0.0), (0.5, 0.5)):
        assert analyse(mu, lam).label == analyse(lam, mu).label


# ---------------------------------------------------------------- decision


def test_decision_output():
    """decision_output is 1.0 / 0.0 / 0.5 as μER is above / below / at FtC."""
    for ftc in (0.3, 0.5, 0.7):
        for mu, lam in grid():
            c = analyse(mu, lam, FtC=ftc)
            expected = 1.0 if c.muER > ftc else 0.0 if c.muER < ftc else 0.5
            assert c.decision_output == expected, (mu, lam, ftc, c.muER)


def test_control_limit_raises_the_bar():
    """A higher FtC demands more real evidence before accepting."""
    mu, lam = 0.72, 0.28
    assert analyse(mu, lam, FtC=0.30).decision_output == 1.0
    assert analyse(mu, lam, FtC=0.90).decision_output == 0.0


# ---------------------------------------------------------------- states


def test_all_twelve_states_are_reachable():
    """Sweeping the lattice reaches every one of the 12 states, plus the centre."""
    seen = {analyse(mu, lam).label for mu, lam in grid(41)}
    expected = {
        "t", "f", "┬", "┴",
        "Q┬→t", "Q┬→f", "Q┴→t", "Q┴→f",
        "Qt→┬", "Qt→┴", "Qf→┬", "Qf→┴",
    }
    missing = expected - seen
    assert not missing, f"unreachable states: {missing}"
    assert seen - expected <= {"I"}, f"unexpected states: {seen - expected}"


def test_region_flags_agree_with_label():
    """Exactly one region flag is set, and it is the reported label."""
    for mu, lam in grid():
        c = analyse(mu, lam)
        on = [k for k, v in c.Regions.items() if v]
        assert on == [c.label], (mu, lam, on, c.label)


def test_states_are_symmetric_under_negation():
    """Negation maps each state to its mirror image across the Dct axis.

    Restricted to μ ≠ λ. When μ = λ the annotation is its own negation, so the
    mapping is necessarily the identity there — see the tie-break test below.
    """
    mirror = {
        "t": "f", "f": "t", "┬": "┬", "┴": "┴", "I": "I",
        "Q┬→t": "Q┬→f", "Q┬→f": "Q┬→t",
        "Q┴→t": "Q┴→f", "Q┴→f": "Q┴→t",
        "Qt→┬": "Qf→┬", "Qf→┬": "Qt→┬",
        "Qt→┴": "Qf→┴", "Qf→┴": "Qt→┴",
    }
    for mu, lam in grid(41):
        if mu == lam:
            continue
        got = analyse(lam, mu).label
        want = mirror[analyse(mu, lam).label]
        assert got == want, (mu, lam, got, want)


def test_zero_certainty_tie_breaks_toward_true():
    """On the Dc = 0 line the classifier breaks the tie toward the 'true' side.

    Documents rather than endorses: at μ = λ the evidence is perfectly balanced,
    so neither 'tending to true' nor 'tending to false' is more justified, but
    `classify_12_regions_asymmetric` tests `dc >= 0` and lands on the true
    branch. Pinned here so a future upstream change is noticed.
    """
    assert analyse(0.6, 0.6).label == "Q┬→t"     # Dc = 0, Dct = +0.2
    assert analyse(0.4, 0.4).label == "Q┴→t"     # Dc = 0, Dct = −0.2


def test_zero_contradiction_tie_breaks_toward_inconsistent():
    """On the Dct = 0 line the classifier breaks the tie toward '→ ┬'.

    The same class of quirk as above, on the other axis: Dct = 0 means the
    evidence is perfectly non-contradictory (μ + λ = 1), so 'tending to
    inconsistent' is the less natural reading, but `dct >= 0` selects it.
    """
    c = analyse(0.72, 0.28, FtC=0.90)            # Dc = 0.44, Dct = 0 exactly
    assert abs(c.dct) < TOL
    assert c.label == "Qt→┬"


# ---------------------------------------------------------------- robustness


def test_inputs_are_clamped():
    """Annotations outside [0, 1] are clamped rather than propagating garbage."""
    c = analyse(5.0, -3.0)
    assert c.mu == 1.0 and c.lam == 0.0
    assert c.label == "t"


def test_no_explosion_from_contradiction():
    """A maximally contradictory annotation yields a finite, usable analysis."""
    c = analyse(1.0, 1.0)
    for v in (c.dc, c.dct, c.d, c.D, c.dcr, c.phi, c.muE, c.muECT, c.muER):
        assert math.isfinite(v)
    assert c.label == "┬"
    assert c.phi == 0.0          # no usable certainty ...
    assert c.muER == 0.5         # ... so the real evidence is neutral
    assert c.decision_output == 0.5


def test_recomputation_after_mutation():
    """The lazy cache invalidates when inputs or parameters change."""
    b = ParaconsistentBlock(mu=0.9, lam=0.1)
    assert b.complete.label == "t"
    b.input.lam = 0.9
    assert b.complete.label == "┬", "cache not invalidated on input change"
    b.set_params(FtC=0.99)
    assert b.complete.FtC == 0.99, "cache not invalidated on param change"


# ---------------------------------------------------------------- runner


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            extra = fn()
            note = f"  (max divergence {extra:.3g})" if isinstance(extra, float) else ""
            print(f"  PASS  {name}{note}")
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
