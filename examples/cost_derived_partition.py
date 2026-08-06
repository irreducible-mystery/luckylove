#!/usr/bin/env python3
"""Cost-derived partitions of the Eτ square, versus the FtC threshold.

Run:  python examples/cost_derived_partition.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paraconsistent import ParaconsistentBlock                      # noqa: E402
from etau_decision import (                                          # noqa: E402
    Action, Policy, brier_score, calibration_report, empirical_loss,
    etau_belief, ftc_policy,
    OBSTACLE, CLEAR, FAULT_CONFLICT, FAULT_SILENCE, WORLDS,
)


def rule(title):
    print(f"\n{title}\n{'─' * len(title)}")


# Losses. Negative is a gain: proceeding on a genuinely clear path is the only
# way the robot makes progress. Halting is the do-nothing baseline at 0.
PROCEED = Action("proceed", {OBSTACLE: 3.0, CLEAR: -1.0,
                             FAULT_CONFLICT: 2.0, FAULT_SILENCE: 2.0})
HALT = Action("halt", {OBSTACLE: 0.0, CLEAR: 0.0,
                       FAULT_CONFLICT: 0.0, FAULT_SILENCE: 0.0})
RECALIBRATE = Action("recalibrate", {OBSTACLE: 1.0, CLEAR: 1.0,
                                     FAULT_CONFLICT: -1.5, FAULT_SILENCE: 1.0})
APPROACH = Action("approach", {OBSTACLE: 1.0, CLEAR: 1.0,
                               FAULT_CONFLICT: 1.0, FAULT_SILENCE: -1.5})

ALL_ACTIONS = [PROCEED, HALT, RECALIBRATE, APPROACH]
GLYPHS = {"proceed": "P", "halt": "H", "recalibrate": "R", "approach": "A"}


def simulate(n, seed=11, fault_rate=0.20):
    """(μ, λ, world) records from a two-channel sensor that sometimes breaks."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        u = rng.random()
        if u < fault_rate / 2:
            world = FAULT_CONFLICT                     # both channels assert
            mu, lam = rng.uniform(0.65, 1.0), rng.uniform(0.65, 1.0)
        elif u < fault_rate:
            world = FAULT_SILENCE                      # both channels dead
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


def main():
    # ------------------------------------------------------------------
    rule("1. What μER cannot see")

    print("  Three annotations that μER maps to the same number:\n")
    print(f"  {'p(μ, λ)':>14} {'μER':>6} │ " +
          " ".join(f"{w:>15}" for w in WORLDS))
    for mu, lam in [(1.0, 1.0), (0.0, 0.0), (0.5, 0.5)]:
        muER = ParaconsistentBlock(mu=mu, lam=lam).complete.muER
        b = etau_belief(mu, lam)
        print(f"  p({mu:.2f}, {lam:.2f}) {muER:>6.2f} │ " +
              " ".join(f"{b[w]:>15.2f}" for w in WORLDS))

    print("\n  'Both channels screaming opposite things', 'both channels dead'")
    print("  and 'both calmly reporting a coin-flip' are one number to μER and")
    print("  three distinct beliefs here. Any decision layer built on μER alone")
    print("  is structurally unable to treat them differently.")

    # ------------------------------------------------------------------
    rule("2. The induced partition — region count is an output")

    policy = Policy(ALL_ACTIONS)
    part = policy.partition(steps=201)

    print(f"  actions supplied     : {len(ALL_ACTIONS)}")
    print(f"  regions induced      : {policy.region_count(201)}")
    print(f"  never optimal        : {policy.unreachable(201) or '(none)'}\n")

    total = sum(part.values())
    for region, cells in sorted(part.items(), key=lambda kv: -kv[1]):
        kind = "region " if len(region) == 1 else "boundary"
        print(f"  {kind}  {'+'.join(sorted(region)):<24} {cells / total:>6.1%}")

    print("\n  Nobody chose 4. Four actions happened to each be optimal")
    print("  somewhere. Had one not been, it would simply be absent — and that")
    print("  absence is a finding about the action set, not a tuning decision.")

    # ------------------------------------------------------------------
    rule("3. The map")

    print("  P proceed   H halt   R recalibrate   A approach   · tie\n")
    print(policy.render(steps=41, glyphs=GLYPHS))

    # ------------------------------------------------------------------
    rule("4. Change the costs, the count changes with them")

    for cost in (-1.5, -0.5, 0.5, 1.5):
        recal = Action("recalibrate", {OBSTACLE: 1.0, CLEAR: 1.0,
                                       FAULT_CONFLICT: cost, FAULT_SILENCE: 1.0})
        p = Policy([PROCEED, HALT, recal, APPROACH])
        gone = p.unreachable(101)
        print(f"  recalibrate cost when right = {cost:<4} -> "
              f"{p.region_count(101)} regions"
              f"{'   (recalibrate never optimal)' if gone else ''}")

    print("\n  The boundaries move because the costs moved. FtC moves because")
    print("  somebody moved it.")

    # ------------------------------------------------------------------
    rule("5. Falsification: does the cost model actually pay?")

    train = simulate(20_000, seed=11)
    held_out = simulate(20_000, seed=99)

    baseline = ftc_policy(0.5, PROCEED, HALT)
    two_action = Policy([PROCEED, HALT])

    rows = [
        ("FtC = 0.50 threshold (proceed/halt)", baseline, [PROCEED, HALT]),
        ("cost-derived, same 2 actions", two_action.optimal, [PROCEED, HALT]),
        ("cost-derived, all 4 actions", policy.optimal, ALL_ACTIONS),
    ]
    print(f"  {'decision rule':<38} {'mean loss':>10}")
    for name, decide, acts in rows:
        loss = empirical_loss(decide, held_out, acts)
        print(f"  {name:<38} {loss:>10.4f}")

    print("\n  The comparison is the point, not the ordering. These numbers exist")
    print("  and can come out against the model — that is what a cost model buys")
    print("  you and a stipulated threshold cannot.")

    # ------------------------------------------------------------------
    rule("6. Is the belief model itself wrong?")

    def skewed(mu, lam):
        """A deliberately wrong model: ignores contradiction entirely."""
        c = ParaconsistentBlock(mu=mu, lam=lam).complete
        return {OBSTACLE: c.muE, CLEAR: 1 - c.muE,
                FAULT_CONFLICT: 0.0, FAULT_SILENCE: 0.0}

    print(f"  Brier score, etau_belief : {brier_score(held_out):.4f}")
    print(f"  Brier score, μE-only     : {brier_score(held_out, skewed):.4f}"
          "   (lower is better)\n")

    print("  Calibration of etau_belief — predicted vs. observed frequency:\n")
    report = calibration_report(train, bins=5)
    for world in WORLDS:
        rows = [r for r in report[world] if r[2] >= 50]
        cells = "  ".join(f"{p:.2f}->{o:.2f}" for p, o, _ in rows)
        print(f"    {world:<16} {cells}")

    print("\n  Read the drift: where etau_belief says 0.55 the world says 0.94,")
    print("  and where it says 0.12 the world says 0.00. It is systematically")
    print("  UNDERCONFIDENT — and explicably so. It splits the decided mass φ by")
    print("  μE, which is a linear readout of Dc, so it can never sharpen faster")
    print("  than the evidence separates. On this simulator it is wrong.")
    print()
    print("  That is the hook working, not the hook failing. The default model")
    print("  shipped in this module just failed its own check, on the first")
    print("  dataset it was pointed at. No table like this exists for FtC —")
    print("  there is no claim about the world inside a threshold to check.")
    print()
    print("  Note also that the miscalibrated model still beat the FtC baseline")
    print("  in section 5. argmin only needs the ORDER of expected losses, so a")
    print("  monotone distortion of the probabilities can leave every decision")
    print("  boundary where it was. Calibration and decision quality are")
    print("  different properties; a model can be visibly wrong and still act")
    print("  well, which is its own reason to measure both.")


if __name__ == "__main__":
    main()
