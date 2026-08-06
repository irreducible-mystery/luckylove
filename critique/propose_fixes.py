#!/usr/bin/env python3
"""Minimal constructive counters — not a rewrite of etau_decision.

1. unreachable_under(records) — world-measure, not grid-measure
2. feature_belief — treat lattice quantities as features, isotonic-free
   stub: logistic on (dc, dct) fit by hand on training counts
3. dcr_sign_label — forced three, set-valued on boundaries

Run: PYTHONPATH=src python critique/propose_fixes.py
"""
from __future__ import annotations

import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paraconsistent import ParaconsistentBlock
from etau_decision import (
    Action, Policy, etau_belief, empirical_loss, brier_score,
    OBSTACLE, CLEAR, FAULT_CONFLICT, FAULT_SILENCE, WORLDS,
)


PROCEED = Action("proceed", {OBSTACLE: 3.0, CLEAR: -1.0,
                             FAULT_CONFLICT: 2.0, FAULT_SILENCE: 2.0})
HALT = Action("halt", {OBSTACLE: 0.0, CLEAR: 0.0,
                       FAULT_CONFLICT: 0.0, FAULT_SILENCE: 0.0})
RECALIBRATE = Action("recalibrate", {OBSTACLE: 1.0, CLEAR: 1.0,
                                     FAULT_CONFLICT: -1.5, FAULT_SILENCE: 1.0})
APPROACH = Action("approach", {OBSTACLE: 1.0, CLEAR: 1.0,
                               FAULT_CONFLICT: 1.0, FAULT_SILENCE: -1.5})
ALL4 = [PROCEED, HALT, RECALIBRATE, APPROACH]


def simulate(n, seed=11, fault_rate=0.20):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        u = rng.random()
        if u < fault_rate / 2:
            world = FAULT_CONFLICT
            mu, lam = rng.uniform(0.65, 1.0), rng.uniform(0.65, 1.0)
        elif u < fault_rate:
            world = FAULT_SILENCE
            mu, lam = rng.uniform(0.0, 0.35), rng.uniform(0.0, 0.35)
        elif u < fault_rate + (1 - fault_rate) * 0.45:
            world = OBSTACLE
            mu = rng.uniform(0.60, 0.98)
            lam = min(1.0, max(0.0, 1 - mu + rng.gauss(0, 0.06)))
        else:
            world = CLEAR
            lam = rng.uniform(0.60, 0.98)
            mu = min(1.0, max(0.0, 1 - lam + rng.gauss(0, 0.06)))
        out.append((mu, lam, world))
    return out


def sim_binary(n, seed=7):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        world = OBSTACLE if rng.random() < 0.5 else CLEAR
        truth = 1.0 if world == OBSTACLE else 0.0
        mu = min(1.0, max(0.0, truth + rng.gauss(0, 0.25)))
        lam = min(1.0, max(0.0, (1 - truth) + rng.gauss(0, 0.25)))
        out.append((mu, lam, world))
    return out


# ---------- fix 1: unreachable under a world measure ----------

def unreachable_under(policy: Policy, records, tol=1e-12) -> set[str]:
    """Actions never optimal on the empirical measure of `records`.

    Differs from Policy.unreachable (uniform grid on the annotation square).
    """
    reached: set[str] = set()
    for mu, lam, _ in records:
        reached |= set(policy.optimal(mu, lam))
    return {a.name for a in policy.actions} - reached


# ---------- fix 2: lattice features → histogram posterior ----------

def make_hist_belief(train, bins=15):
    counts = defaultdict(Counter)
    for mu, lam, w in train:
        i = min(int(mu * bins), bins - 1)
        j = min(int(lam * bins), bins - 1)
        counts[(i, j)][w] += 1
    # Laplace-smoothed global prior for empty cells
    global_c = Counter(w for *_, w in train)
    gtot = sum(global_c.values()) or 1
    prior = {w: global_c[w] / gtot for w in WORLDS}

    def belief(mu, lam):
        i = min(int(mu * bins), bins - 1)
        j = min(int(lam * bins), bins - 1)
        c = counts[(i, j)]
        tot = sum(c.values())
        if tot < 5:
            return dict(prior)
        return {w: c[w] / tot for w in WORLDS}
    return belief


# ---------- fix 3: forced three, set-valued ----------

def dcr_regions(mu: float, lam: float, boundary_eps: float = 1e-12) -> frozenset[str]:
    c = ParaconsistentBlock(mu=mu, lam=lam).complete
    if c.D >= 1.0 - boundary_eps:
        return frozenset({"neutral"})
    if c.dcr > 0:
        return frozenset({"for"})
    if c.dcr < 0:
        return frozenset({"against"})
    return frozenset({"neutral"})


def main():
    print("=== propose_fixes: constructive counters ===\n")

    train = simulate(20_000, seed=11)
    held = simulate(10_000, seed=99)
    binary = sim_binary(10_000)

    pol4 = Policy(ALL4)
    pol2 = Policy([PROCEED, HALT])

    print("1. unreachable: grid vs world measure")
    print(f"   grid unreachable (steps=101)     : {pol4.unreachable(101) or '{}'}")
    print(f"   under friend held-out records    : {unreachable_under(pol4, held) or '{}'}")
    print(f"   under binary-only records        : {unreachable_under(pol4, binary) or '{}'}")
    print("   (on binary data, recalibrate/approach often still fire on annotations")
    print("    that *look* like faults — they are grid-reachable and world-wasteful)\n")

    # show waste explicitly
    from etau_decision import ftc_policy
    print("2. mean loss under binary world")
    print(f"   FtC                          : {empirical_loss(ftc_policy(0.5, PROCEED, HALT), binary, [PROCEED, HALT]):.4f}")
    print(f"   cost 2-action                : {empirical_loss(pol2.optimal, binary, [PROCEED, HALT]):.4f}")
    print(f"   cost 4-action                : {empirical_loss(pol4.optimal, binary, ALL4):.4f}")
    print("   → extra actions hurt when their worlds have mass 0\n")

    print("3. lattice readout vs histogram features")
    hist = make_hist_belief(train)
    pol_h = Policy(ALL4, belief=hist)
    print(f"   Brier etau_belief            : {brier_score(held, etau_belief):.4f}")
    print(f"   Brier hist-on-train          : {brier_score(held, hist):.4f}")
    print(f"   loss  etau_belief            : {empirical_loss(pol4.optimal, held, ALL4):.4f}")
    print(f"   loss  hist belief            : {empirical_loss(pol_h.optimal, held, ALL4):.4f}\n")

    print("4. forced-three coverage (steps=101)")
    c = Counter()
    steps = 101
    for i in range(steps):
        for j in range(steps):
            c[frozenset(dcr_regions(i / (steps - 1), j / (steps - 1)))] += 1
    total = steps * steps
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"   {str(sorted(k)):20s} {v/total:6.1%}")
    print("\n   No FtC. No losses. Forced by D = min(d,1).")
    print("   Cost-derived sits on top of a belief map; this sits under both.")


if __name__ == "__main__":
    main()
