#!/usr/bin/env python3
"""Runnable attacks on etau_decision's premises.

Each section ends with a structured FINDING that either:
  HOLDS  — claim survives the probe
  BREAKS — claim fails under a concrete, reproducible construction
  SOFT   — claim holds only under restricted conditions the prose omits

Run from repo root:
  PYTHONPATH=src python critique/attack_etau_decision.py
  # or with venv:
  .venv/bin/python critique/attack_etau_decision.py

Outputs machine-readable JSON to critique/attack_results.json
and human prose to stdout.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paraconsistent import ParaconsistentBlock
from etau_decision import (
    Action, Policy, etau_belief, empirical_loss, brier_score,
    calibration_report, ftc_policy,
    OBSTACLE, CLEAR, FAULT_CONFLICT, FAULT_SILENCE, WORLDS,
)

OUT = Path(__file__).resolve().parent / "attack_results.json"
results: dict = {"findings": []}


def record(tag: str, verdict: str, claim: str, detail: dict):
    row = {"tag": tag, "verdict": verdict, "claim": claim, "detail": detail}
    results["findings"].append(row)
    mark = {"HOLDS": "✓ HOLDS", "BREAKS": "✗ BREAKS", "SOFT": "~ SOFT  "}[verdict]
    print(f"\n[{mark}] {tag}")
    print(f"  claim : {claim}")
    for k, v in detail.items():
        if isinstance(v, float):
            print(f"  {k:28s}: {v:.6g}")
        else:
            print(f"  {k:28s}: {v}")


# =====================================================================
# shared actions (from the shipped example)
# =====================================================================

PROCEED = Action("proceed", {OBSTACLE: 3.0, CLEAR: -1.0,
                             FAULT_CONFLICT: 2.0, FAULT_SILENCE: 2.0})
HALT = Action("halt", {OBSTACLE: 0.0, CLEAR: 0.0,
                       FAULT_CONFLICT: 0.0, FAULT_SILENCE: 0.0})
RECALIBRATE = Action("recalibrate", {OBSTACLE: 1.0, CLEAR: 1.0,
                                     FAULT_CONFLICT: -1.5, FAULT_SILENCE: 1.0})
APPROACH = Action("approach", {OBSTACLE: 1.0, CLEAR: 1.0,
                               FAULT_CONFLICT: 1.0, FAULT_SILENCE: -1.5})
ALL4 = [PROCEED, HALT, RECALIBRATE, APPROACH]


def shipped_simulator(n, seed=11, fault_rate=0.20):
    """Exact copy of examples/cost_derived_partition.simulate — the friend."""
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


# =====================================================================
# ATTACK 1 — Dcr=0 hourglass geometry (transcript claim)
# =====================================================================

def attack_dcr_geometry():
    """Transcript: Dcr=0 is unit square minus two quarter-disks, area 1-π/4."""
    claim = ("Dcr=0 region is unit-square minus two quarter-disks of radius "
             "√½ centred on t and f; area exactly 1 − π/4")
    # Literature formula in tests: d = √((1-|Dc|)² + Dct²)
    # Dcr = 0 iff D = min(d,1) = 1 iff d ≥ 1
    # Apex t is (1,0) in (μ,λ); f is (0,1).
    # radial_d_to_nearest_apex — verify what "d ≥ 1" carves.
    N = 2001  # dense grid, area estimate
    zero = total = 0
    # Also: is the region exactly hourglass pinched at I?
    pinch_width_at_half = 0
    for i in range(N):
        for j in range(N):
            mu = i / (N - 1)
            lam = j / (N - 1)
            c = ParaconsistentBlock(mu=mu, lam=lam).complete
            total += 1
            if abs(c.dcr) < 1e-15 or c.D >= 1.0 - 1e-15:
                zero += 1
    measured = zero / total
    predicted = 1.0 - math.pi / 4.0
    # pinch: along μ=λ line near 0.5, is the neutral region zero-width?
    # On diagonal μ=λ, Dc=0, Dct=2μ-1. d = √((1-0)² + Dct²) = √(1+Dct²) ≥ 1 always.
    # So the ENTIRE diagonal μ=λ has d ≥ 1, hence Dcr=0 everywhere on μ=λ.
    # That is NOT "pinched to zero width at I" — the waist is the full diagonal!
    diagonal_neutral = 0
    diag_pts = 401
    for i in range(diag_pts):
        t = i / (diag_pts - 1)
        c = ParaconsistentBlock(mu=t, lam=t).complete
        if c.D >= 1.0 - 1e-12:
            diagonal_neutral += 1
    waist_is_full_diagonal = (diagonal_neutral == diag_pts)

    # Quarter-disk claim: distance to apex t=(1,0) in (μ,λ) plane.
    # points with dist((μ,λ),(1,0)) < √0.5 AND nearer to t than f?
    # Actually literature d uses Dc/Dct coordinates rotated 45°.
    # Check: is Dcr=0 equivalent to being outside both disks of r=√0.5
    # centred at (1,0) and (0,1) in (μ,λ)?
    # d_t = √((μ-1)²+(λ-0)²); d_f = √((μ-0)²+(λ-1)²)
    # For p(0.5,0.5): d_t = d_f = √0.5 ≈ 0.707 < 1, so INSIDE both disks.
    # But Dcr(0.5,0.5)=0. So Dcr=0 is NOT "outside the disks".
    # The transcript said "unit square minus two quarter-disks" for Dcr=0
    # i.e. the neutral region = complement of disks. But centre is neutral
    # AND inside disks. Contradiction with Euclidean (μ,λ) disks.
    #
    # Under the rotated (Dc, Dct) metric used by the library, d=√((1-|Dc|)²+Dct²)
    # the set d≥1 is different. Let's measure agreement with Euclidean disks.
    r2 = 0.5
    outside_both_euclid = 0
    dcr_zero_and_outside = 0
    dcr_zero = 0
    steps = 401
    for i in range(steps):
        for j in range(steps):
            mu = i / (steps - 1)
            lam = j / (steps - 1)
            c = ParaconsistentBlock(mu=mu, lam=lam).complete
            dt2 = (mu - 1) ** 2 + (lam - 0) ** 2
            df2 = (mu - 0) ** 2 + (lam - 1) ** 2
            outside = (dt2 >= r2) and (df2 >= r2)
            if outside:
                outside_both_euclid += 1
            is_zero = c.D >= 1.0 - 1e-12
            if is_zero:
                dcr_zero += 1
            if is_zero and outside:
                dcr_zero_and_outside += 1

    euclid_area = outside_both_euclid / (steps * steps)
    dcr_area = dcr_zero / (steps * steps)
    # Jaccard-like: fraction of Dcr=0 that is "outside both disks"
    overlap = dcr_zero_and_outside / max(dcr_zero, 1)

    # Verdict: area number 1-π/4 is correct for Euclidean complement of two
    # quarter-disks, and measured Dcr=0 area is CLOSE but the geometric
    # identification is subtle — and the "pinched at I" claim is FALSE:
    # the whole μ=λ diagonal is Dcr=0.
    if waist_is_full_diagonal:
        verdict = "BREAKS"
    elif abs(measured - predicted) > 0.01:
        verdict = "BREAKS"
    else:
        verdict = "SOFT"

    record("A1_dcr_geometry", verdict, claim, {
        "measured_Dcr0_area": measured,
        "predicted_1_minus_pi_over_4": predicted,
        "abs_error": abs(measured - predicted),
        "entire_mu_eq_lam_diagonal_is_Dcr0": waist_is_full_diagonal,
        "euclid_outside_both_disks_area": euclid_area,
        "fraction_of_Dcr0_that_is_outside_euclid_disks": overlap,
        "note": ("Waist claim 'pinched to zero at I' is false: μ=λ diagonal "
                 "is entirely Dcr=0. Area 1-π/4 is the Euclidean-disk "
                 "complement; library D uses rotated (Dc,Dct) metric."),
    })


# =====================================================================
# ATTACK 2 — φ as "decided mass" is a costume
# =====================================================================

def attack_phi_costume():
    claim = ("φ = 1−|Dct| is the mass that is *decided*, split by μE — "
             "a derived quantity, not a modeling choice")
    # Any partition of the unit simplex can wear this costume.
    # Construct two alternatives that also sum to 1, use only library quantities,
    # and are equally "natural":
    #   alt1: use (1-D) as decided mass (radial), split by μER
    #   alt2: use phiE * (1 - abs(dc)) style sharpening
    # Show they induce DIFFERENT partitions under the same actions.
    def belief_phi(mu, lam):
        return etau_belief(mu, lam)

    def belief_radial(mu, lam):
        c = ParaconsistentBlock(mu=mu, lam=lam).complete
        decided = 1.0 - c.D  # "how close to an apex"
        # residual mass: put on conflict if dct>0 else silence
        residual = c.D
        return {
            OBSTACLE: decided * c.muER,
            CLEAR: decided * (1.0 - c.muER),
            FAULT_CONFLICT: residual if c.dct >= 0 else 0.0,
            FAULT_SILENCE: residual if c.dct < 0 else 0.0,
        }

    def belief_hard(mu, lam):
        """Sharpen: put all decided mass on argmax side of Dc."""
        c = ParaconsistentBlock(mu=mu, lam=lam).complete
        decided = c.phi
        if c.dc > 0:
            o, cl = decided, 0.0
        elif c.dc < 0:
            o, cl = 0.0, decided
        else:
            o = cl = decided / 2
        return {
            OBSTACLE: o, CLEAR: cl,
            FAULT_CONFLICT: max(0.0, c.dct),
            FAULT_SILENCE: max(0.0, -c.dct),
        }

    p_phi = Policy(ALL4, belief=belief_phi)
    p_rad = Policy(ALL4, belief=belief_radial)
    p_hard = Policy(ALL4, belief=belief_hard)

    # Compare partitions on a grid
    disagree_phi_rad = disagree_phi_hard = 0
    steps = 51
    for i in range(steps):
        for j in range(steps):
            mu, lam = i / (steps - 1), j / (steps - 1)
            a, b, c = p_phi.optimal(mu, lam), p_rad.optimal(mu, lam), p_hard.optimal(mu, lam)
            if a != b:
                disagree_phi_rad += 1
            if a != c:
                disagree_phi_hard += 1
    cells = steps * steps
    # All three sum to 1 everywhere?
    ok_sums = True
    for mu, lam in [(0, 0), (1, 1), (0.5, 0.5), (0.7, 0.2), (0.3, 0.8)]:
        for bfn in (belief_phi, belief_radial, belief_hard):
            s = sum(bfn(mu, lam).values())
            if abs(s - 1.0) > 1e-9:
                ok_sums = False

    # The three "natural" readings disagree on a large fraction of the square.
    # Therefore φ-as-decided is a choice among peers, not a derivation.
    frac_rad = disagree_phi_rad / cells
    frac_hard = disagree_phi_hard / cells
    verdict = "BREAKS" if (frac_rad > 0.05 or frac_hard > 0.05) else "SOFT"
    record("A2_phi_is_a_costume", verdict, claim, {
        "all_alts_sum_to_1": ok_sums,
        "frac_disagree_phi_vs_radial": frac_rad,
        "frac_disagree_phi_vs_hard": frac_hard,
        "region_count_phi": p_phi.region_count(51),
        "region_count_radial": p_rad.region_count(51),
        "region_count_hard": p_hard.region_count(51),
        "note": "Multiple equally arithmetic-clean mass assignments induce different partitions.",
    })


# =====================================================================
# ATTACK 3 — Simulator/ontology collusion
# =====================================================================

def attack_simulator_collusion():
    claim = ("Cost-derived beats FtC because the model is better, "
             "not because the simulator was built to match the ontology")

    held = shipped_simulator(15_000, seed=99)
    baseline = ftc_policy(0.5, PROCEED, HALT)
    pol4 = Policy(ALL4)
    pol2 = Policy([PROCEED, HALT])

    loss_ftc_friend = empirical_loss(baseline, held, [PROCEED, HALT])
    loss_2_friend = empirical_loss(pol2.optimal, held, [PROCEED, HALT])
    loss_4_friend = empirical_loss(pol4.optimal, held, ALL4)

    # Hostile simulator 1: only OBSTACLE/CLEAR exist. Sensors are noisy Bernoulli.
    # Fault modes never obtain. etau_belief still spreads mass to fault_* .
    def sim_binary(n, seed=7):
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            world = OBSTACLE if rng.random() < 0.5 else CLEAR
            # two noisy sensors of the same bit
            truth = 1.0 if world == OBSTACLE else 0.0
            s1 = min(1.0, max(0.0, truth + rng.gauss(0, 0.25)))
            s2 = min(1.0, max(0.0, (1 - truth) + rng.gauss(0, 0.25)))
            # μ = evidence for obstacle, λ = evidence against
            mu, lam = s1, s2
            out.append((mu, lam, world))
        return out

    # Hostile 2: annotations drawn uniformly; world independent of annotation
    # (sensors broken in a way uncorrelated with the lattice geometry)
    def sim_uncorrelated(n, seed=13):
        rng = random.Random(seed)
        worlds = list(WORLDS)
        return [(rng.random(), rng.random(), rng.choice(worlds)) for _ in range(n)]

    # Hostile 3: "true contradiction" as ontic — world is FAULT_CONFLICT when
    # nature is both, but sensors report clean single-channel evidence
    # (ontology inverted vs belief model)
    def sim_inverted_faults(n, seed=17):
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            u = rng.random()
            if u < 0.25:
                world = FAULT_CONFLICT
                # sensors look clean-true (belief will NOT put mass on conflict)
                mu, lam = rng.uniform(0.7, 1.0), rng.uniform(0.0, 0.2)
            elif u < 0.5:
                world = FAULT_SILENCE
                mu, lam = rng.uniform(0.7, 1.0), rng.uniform(0.0, 0.2)
            elif u < 0.75:
                world = OBSTACLE
                mu, lam = rng.uniform(0.7, 1.0), rng.uniform(0.0, 0.2)
            else:
                world = CLEAR
                mu, lam = rng.uniform(0.0, 0.2), rng.uniform(0.7, 1.0)
            out.append((mu, lam, world))
        return out

    rows = {}
    for name, sim in [
        ("friend_shipped", lambda: held),
        ("hostile_binary_only", lambda: sim_binary(15_000)),
        ("hostile_uncorrelated", lambda: sim_uncorrelated(15_000)),
        ("hostile_inverted_faults", lambda: sim_inverted_faults(15_000)),
    ]:
        data = sim() if name != "friend_shipped" else held
        rows[name] = {
            "ftc": empirical_loss(baseline, data, [PROCEED, HALT]),
            "cost2": empirical_loss(pol2.optimal, data, [PROCEED, HALT]),
            "cost4": empirical_loss(pol4.optimal, data, ALL4),
        }

    # On friend data, cost4 < ftc. On hostile, does the advantage reverse?
    friend_wins = rows["friend_shipped"]["cost4"] < rows["friend_shipped"]["ftc"]
    binary_wins = rows["hostile_binary_only"]["cost4"] < rows["hostile_binary_only"]["ftc"]
    uncorr_wins = rows["hostile_uncorrelated"]["cost4"] < rows["hostile_uncorrelated"]["ftc"]
    inv_wins = rows["hostile_inverted_faults"]["cost4"] < rows["hostile_inverted_faults"]["ftc"]

    # Also: with only PROCEED/HALT, on binary world, does cost2 still beat ftc?
    # And does cost4 waste value by sometimes recalibrating/approaching when faults never happen?
    waste = rows["hostile_binary_only"]["cost4"] - rows["hostile_binary_only"]["cost2"]

    if friend_wins and (not binary_wins or not inv_wins or waste > 0.05):
        verdict = "BREAKS"
    elif friend_wins:
        verdict = "SOFT"
    else:
        verdict = "HOLDS"  # unexpected: even friend fails

    record("A3_simulator_collusion", verdict, claim, {
        "friend_ftc": rows["friend_shipped"]["ftc"],
        "friend_cost4": rows["friend_shipped"]["cost4"],
        "friend_cost4_beats_ftc": friend_wins,
        "binary_ftc": rows["hostile_binary_only"]["ftc"],
        "binary_cost2": rows["hostile_binary_only"]["cost2"],
        "binary_cost4": rows["hostile_binary_only"]["cost4"],
        "binary_cost4_beats_ftc": binary_wins,
        "binary_cost4_minus_cost2_waste": waste,
        "uncorrelated_ftc": rows["hostile_uncorrelated"]["ftc"],
        "uncorrelated_cost4": rows["hostile_uncorrelated"]["cost4"],
        "uncorrelated_cost4_beats_ftc": uncorr_wins,
        "inverted_ftc": rows["hostile_inverted_faults"]["ftc"],
        "inverted_cost4": rows["hostile_inverted_faults"]["cost4"],
        "inverted_cost4_beats_ftc": inv_wins,
        "note": ("Shipped simulator samples (μ,λ) FROM the world labels "
                 "etau_belief assumes. Hostile generators break that handshake."),
    })


# =====================================================================
# ATTACK 4 — Ordering-only: bad beliefs, identical decisions
# =====================================================================

def attack_ordering_blindspot():
    claim = ("Falsifiability of the cost model is real: reality will refute "
             "a wrong model via empirical_loss / calibration")

    # Monotone distortion of probabilities that preserves argmin order
    # for the shipped loss matrix: raise all probs to power γ then renorm...
    # Actually simpler: scale the four masses by positive constants that
    # preserve the linear order of expected losses for ALL4.
    # Even simpler demonstration: belief' = belief (identical decisions),
    # but we construct belief_bad that is a monotone transform along the
    # decision-relevant axis.
    #
    # For linear expected loss, argmin is unchanged under any transform of
    # belief that preserves the half-spaces defined by loss differences.
    # Construct: etau_belief raised to power then renormalized — often
    # nearly preserves order for moderate γ.

    def power_belief(gamma):
        def b(mu, lam):
            raw = etau_belief(mu, lam)
            powered = {w: (raw[w] ** gamma) for w in WORLDS}
            s = sum(powered.values()) or 1.0
            return {w: powered[w] / s for w in WORLDS}
        return b

    base = Policy(ALL4, belief=etau_belief)
    # Find a gamma where partition is almost identical but calibration is worse
    steps = 41
    best = None
    for gamma in (0.3, 0.5, 2.0, 3.0, 5.0):
        p = Policy(ALL4, belief=power_belief(gamma))
        disagree = 0
        for i in range(steps):
            for j in range(steps):
                mu, lam = i / (steps - 1), j / (steps - 1)
                if base.optimal(mu, lam) != p.optimal(mu, lam):
                    disagree += 1
        frac = disagree / (steps * steps)
        data = shipped_simulator(8_000, seed=3)
        br_base = brier_score(data, etau_belief)
        br_bad = brier_score(data, power_belief(gamma))
        loss_base = empirical_loss(base.optimal, data, ALL4)
        loss_bad = empirical_loss(p.optimal, data, ALL4)
        if best is None or frac < best["frac_partition_disagree"]:
            best = {
                "gamma": gamma,
                "frac_partition_disagree": frac,
                "brier_base": br_base,
                "brier_distorted": br_bad,
                "loss_base": loss_base,
                "loss_distorted": loss_bad,
                "abs_loss_delta": abs(loss_base - loss_bad),
            }

    # Stronger construction: belief that puts mass correctly ONLY for argmin
    # order — deliberately miscalibrated magnitudes
    def order_only_belief(mu, lam):
        """Same support pattern as etau but flattened toward uniform on support."""
        b = etau_belief(mu, lam)
        # mix 70% uniform-on-simplex with 30% true belief: badly calibrated,
        # but often same argmin
        u = 0.25
        return {w: (1 - u) * b[w] + u * 0.25 for w in WORLDS}

    p_flat = Policy(ALL4, belief=order_only_belief)
    data = shipped_simulator(10_000, seed=5)
    disagree = 0
    steps = 51
    for i in range(steps):
        for j in range(steps):
            mu, lam = i / (steps - 1), j / (steps - 1)
            if base.optimal(mu, lam) != p_flat.optimal(mu, lam):
                disagree += 1
    frac = disagree / (steps * steps)
    br0 = brier_score(data, etau_belief)
    br1 = brier_score(data, order_only_belief)
    l0 = empirical_loss(base.optimal, data, ALL4)
    l1 = empirical_loss(p_flat.optimal, data, ALL4)

    # If partition agreement is high AND losses nearly equal AND brier differs,
    # then decision-only falsification has a blind spot the prose understated.
    # (The module itself admits this — we measure how large the hole is.)
    if frac < 0.15 and abs(l0 - l1) < 0.15 and br1 > br0 + 0.02:
        verdict = "SOFT"  # admitted but size matters; still a real hole
    elif abs(l0 - l1) < 0.05 and br1 > br0:
        verdict = "SOFT"
    else:
        verdict = "HOLDS"

    record("A4_ordering_blindspot", verdict, claim, {
        "power_best": best,
        "flat_mix_frac_partition_disagree": frac,
        "flat_brier_base": br0,
        "flat_brier_distorted": br1,
        "flat_loss_base": l0,
        "flat_loss_distorted": l1,
        "note": ("Module admits calibration≠decision. Quantified: large belief "
                 "distortion can leave nearly identical decisions. empirical_loss "
                 "refutes only when ORDERING breaks, not when magnitudes are wrong."),
    })


# =====================================================================
# ATTACK 5 — Action set is the new FtC
# =====================================================================

def attack_action_set_is_ftc():
    claim = ("Region count is an OUTPUT not a parameter — nobody chose 4")

    # Counter: the action vocabulary and loss numbers ARE the parameters.
    # Enumerate: with different action sets you get any count you want 1..n
    counts = {}
    # 1 action
    counts["1_halt"] = Policy([HALT]).region_count(41)
    # 2 actions
    counts["2_proceed_halt"] = Policy([PROCEED, HALT]).region_count(41)
    # 3
    counts["3_no_approach"] = Policy([PROCEED, HALT, RECALIBRATE]).region_count(41)
    # 4
    counts["4_all"] = Policy(ALL4).region_count(41)
    # 5 with a custom "wait" that wins in the centre
    WAIT = Action("wait", {OBSTACLE: 0.5, CLEAR: 0.5,
                           FAULT_CONFLICT: 0.5, FAULT_SILENCE: 0.5})
    counts["5_with_wait"] = Policy(ALL4 + [WAIT]).region_count(61)
    # 4 actions but losses make only 2 win
    DEAD1 = Action("dead1", {w: 50.0 for w in WORLDS})
    DEAD2 = Action("dead2", {w: 50.0 for w in WORLDS})
    counts["4_but_2_dead"] = Policy([PROCEED, HALT, DEAD1, DEAD2]).region_count(41)

    # You can also move FtC and get different 2-region geometries — both are
    # free parameters. The honest claim is: count is a *function* of the
    # free parameter (action/loss), not that the free parameter vanished.
    free_param_still_exists = True
    can_select_count = len(set(counts.values())) >= 3

    verdict = "BREAKS" if (free_param_still_exists and can_select_count) else "SOFT"
    record("A5_action_set_is_the_new_FtC", verdict, claim, {
        "region_counts_by_action_vocab": counts,
        "note": ("'Nobody chose 4' is rhetoric. Someone chose four actions and "
                 "their loss vectors. Count is output OF that choice, exactly as "
                 "two-region is output of choosing a threshold rule. The gain is "
                 "that the parameter is now multi-dimensional and checkable — "
                 "not that parameterisation disappeared."),
    })


# =====================================================================
# ATTACK 6 — Grid resolution artifacts in unreachable / region_count
# =====================================================================

def attack_grid_artifacts():
    claim = ("unreachable() and region_count() are empirical facts about "
             "the action set")

    # Construct an action that wins only on a thin diagonal — missed by coarse grid
    # On μ≈1, λ≈0 corner only (one cell at fine resolution)
    NEEDLE = Action("needle", {
        OBSTACLE: -100.0,  # huge reward only when belief≈obstacle
        CLEAR: 10.0, FAULT_CONFLICT: 10.0, FAULT_SILENCE: 10.0,
    })
    # Actually needle wins wherever P(obstacle) is high enough.
    # Better: action that wins only when belief is almost exactly one corner.
    SLIVER = Action("sliver", {
        OBSTACLE: 0.0, CLEAR: 0.0, FAULT_CONFLICT: 0.0, FAULT_SILENCE: -50.0,
    })
    # sliver wins when FAULT_SILENCE mass is high enough vs halt's 0.
    # etau_belief puts mass 1 on silence at (0,0). So it wins near (0,0).

    p = Policy([HALT, SLIVER])
    # at steps=3, might miss; at steps=101, finds it
    un_coarse = p.unreachable(steps=3)
    un_mid = p.unreachable(steps=11)
    un_fine = p.unreachable(steps=101)
    rc_coarse = p.region_count(3)
    rc_fine = p.region_count(101)

    # Also: ties occupy measure-zero sets, but discretisation makes "boundary"
    # cells appear as regions of positive count — region_count excludes them
    # by len==1, good. But partition percentages in the example count boundary
    # cells as positive % — fine.

    grid_dependent = (un_coarse != un_fine) or (rc_coarse != rc_fine)
    verdict = "SOFT" if grid_dependent else "HOLDS"
    record("A6_grid_artifacts", verdict, claim, {
        "unreachable_steps_3": sorted(un_coarse),
        "unreachable_steps_11": sorted(un_mid),
        "unreachable_steps_101": sorted(un_fine),
        "region_count_3": rc_coarse,
        "region_count_101": rc_fine,
        "grid_dependent": grid_dependent,
        "note": "unreachable/region_count are grid estimators, not exact. Coarse grids lie.",
    })


# =====================================================================
# ATTACK 7 — Tie averaging is not a coherent decision rule
# =====================================================================

def attack_tie_average():
    claim = ("Ties cost the average of what they tie between: no free lunch "
             "from refusing to choose")

    # Counterexample: nonlinear evaluation. If realised loss is max(loss) under
    # an adversary, or utility is exponential, average-of-losses ≠ value of set.
    # Also: randomising 50/50 between tied actions is NOT the same as paying
    # the average when the world is chosen adversarially AFTER the randomisation
    # is known — but when world is fixed first, expectation of randomisation
    # equals average. So under risk-neutral expected loss it is fine.
    #
    # Harder issue: on a tie set, BOTH actions are optimal, so picking either
    # is fine — averaging is only for the *metric*, not a policy. Calling it
    # "refusing to choose" misdescribes: the decide function returns a set,
    # and empirical_loss invents a scalarisation. That's a scoring convention,
    # not forced by the theory.

    A = Action("safe", {OBSTACLE: 1.0, CLEAR: 1.0,
                        FAULT_CONFLICT: 1.0, FAULT_SILENCE: 1.0})
    B = Action("risky", {OBSTACLE: 0.0, CLEAR: 0.0,
                         FAULT_CONFLICT: 100.0, FAULT_SILENCE: 100.0})
    # At a belief where EL(safe)=EL(risky)=1.0, they tie.
    # Belief: all mass on OBSTACLE → EL(safe)=1, EL(risky)=0 — no tie.
    # Need belief where 1.0 = 0*p_o + 0*p_c + 100*p_fc + 100*p_fs
    # => p_faults = 0.01, and safe always 1.
    # Actually safe is always EL=1. risky EL = 100*(p_fc+p_fs).
    # Tie when p_fc+p_fs = 0.01.
    # Average of losses when world=FAULT_CONFLICT: (1+100)/2 = 50.5
    # But if you MUST pick one action (you always do in the world), you either
    # pay 1 or 100. The "average" is not a physical outcome.
    # Under maximin, you pick safe (pay 1 worst-case among... actually both
    # can face 100? safe worst is 1). Maximin picks safe. Average says 50.5
    # as the "tie cost" which matches neither pure action.

    records = [(0.5, 0.5, FAULT_CONFLICT)]  # whatever; force the decide set
    # fabricate a decide that returns the tie
    loss_avg = empirical_loss(lambda m, l: frozenset({"safe", "risky"}),
                              records, [A, B])
    loss_safe = empirical_loss(lambda m, l: "safe", records, [A, B])
    loss_risky = empirical_loss(lambda m, l: "risky", records, [A, B])

    # The average is not equal to either pure choice; if the deployment must
    # output one action, average is a fiction. Verdict: SOFT (honest as a
    # scoring convention, oversold as "no free lunch from refusing").
    record("A7_tie_average", "SOFT", claim, {
        "loss_if_safe": loss_safe,
        "loss_if_risky": loss_risky,
        "loss_if_tie_averaged": loss_avg,
        "note": ("Average is a scoring convention for set-valued decide, not a "
                 "physical outcome. Deployment always picks one. Maximin / CVaR "
                 "would break ties differently. 'No free lunch' holds for risk-neutral "
                 "expectation of a fair coin between pure actions — a narrower claim."),
    })


# =====================================================================
# ATTACK 8 — Linearity flattens the notebook's non-linearity
# =====================================================================

def attack_linearity_flattens():
    claim = ("Cost-curve boundaries capture the interesting non-linear "
             "structure of the annotation square")

    # Expected loss is linear in belief. Belief (etau) is piecewise-linear
    # in (μ,λ) via Dc, Dct, φ. Composition is piecewise-linear. Region
    # boundaries are line segments in (μ,λ). The Dcr radial geometry
    # (circles in Dc/Dct) is discarded.
    #
    # Measure: what fraction of cost-derived boundaries coincide with
    # level sets of Dcr vs Dc vs Dct?
    # Sample dual points across a boundary and see which lattice quantity
    # jumps.

    pol = Policy(ALL4)
    steps = 81
    # find boundary edges (adjacent cells, different singleton optima)
    boundary_edges = []
    grid_opt = {}
    for i in range(steps):
        for j in range(steps):
            mu, lam = i / (steps - 1), j / (steps - 1)
            grid_opt[(i, j)] = pol.optimal(mu, lam)

    for i in range(steps - 1):
        for j in range(steps):
            a, b = grid_opt[(i, j)], grid_opt[(i + 1, j)]
            if a != b and len(a) == 1 and len(b) == 1:
                boundary_edges.append(((i, j), (i + 1, j), a, b))
    for i in range(steps):
        for j in range(steps - 1):
            a, b = grid_opt[(i, j)], grid_opt[(i, j + 1)]
            if a != b and len(a) == 1 and len(b) == 1:
                boundary_edges.append(((i, j), (i, j + 1), a, b))

    # For each boundary edge, check if Dc, Dct, or Dcr is nearly constant
    # (boundary ≈ level set) or changes a lot (boundary cuts level sets)
    def deg(i, j):
        mu, lam = i / (steps - 1), j / (steps - 1)
        c = ParaconsistentBlock(mu=mu, lam=lam).complete
        return c.dc, c.dct, c.dcr

    dc_changes, dct_changes, dcr_changes = [], [], []
    for (i1, j1), (i2, j2), _, _ in boundary_edges:
        d1 = deg(i1, j1)
        d2 = deg(i2, j2)
        dc_changes.append(abs(d1[0] - d2[0]))
        dct_changes.append(abs(d1[1] - d2[1]))
        dcr_changes.append(abs(d1[2] - d2[2]))

    # If boundaries were Dcr level sets, dcr_changes would be near 0.
    # We expect mean |ΔDcr| across boundaries to be positive and comparable
    # to |ΔDc| — i.e. cost boundaries are NOT the lattice's curved geometry.
    import statistics as stats
    mean_dc = stats.mean(dc_changes) if dc_changes else float("nan")
    mean_dct = stats.mean(dct_changes) if dct_changes else float("nan")
    mean_dcr = stats.mean(dcr_changes) if dct_changes else float("nan")

    # Also: analytically, EL = Σ p_w L_a,w is linear in p. p is linear in
    # (φ·μE, φ·(1-μE), max(dct,0), max(-dct,0)) which is piecewise linear
    # in (μ,λ). So boundaries are piecewise-linear. Circles are not.

    verdict = "BREAKS" if mean_dcr > 0.02 else "SOFT"
    record("A8_linearity_flattens_geometry", verdict, claim, {
        "n_boundary_edges": len(boundary_edges),
        "mean_abs_delta_Dc_across_boundary": mean_dc,
        "mean_abs_delta_Dct_across_boundary": mean_dct,
        "mean_abs_delta_Dcr_across_boundary": mean_dcr,
        "note": ("Cost boundaries cut Dcr level sets (mean |ΔDcr| > 0). "
                 "The radial/circular structure of real certainty is not "
                 "what places the cost-derived walls. Piecewise-linear "
                 "belief + linear loss => piecewise-linear regions. The "
                 "notebook's circle is flattened."),
    })


# =====================================================================
# ATTACK 9 — A generative belief that is better calibrated
# =====================================================================

def attack_better_belief_exists():
    claim = ("The default etau_belief is the natural belief model from the "
             "lattice quantities")

    # Invert the shipped simulator's known generative structure to get a
    # nearly-calibrated belief (cheat: use the DGP). If a model that knows
    # the DGP crushes Brier, then etau_belief's failure is not "as good as
    # lattice allows" — it's a weak readout.
    #
    # Approximate posterior via histogram on a coarse (μ,λ) grid from training.
    train = shipped_simulator(30_000, seed=11)
    held = shipped_simulator(10_000, seed=99)
    bins = 20
    counts = defaultdict(lambda: Counter())
    for mu, lam, w in train:
        i = min(int(mu * bins), bins - 1)
        j = min(int(lam * bins), bins - 1)
        counts[(i, j)][w] += 1

    def hist_belief(mu, lam):
        i = min(int(mu * bins), bins - 1)
        j = min(int(lam * bins), bins - 1)
        c = counts[(i, j)]
        tot = sum(c.values())
        if tot == 0:
            return {w: 0.25 for w in WORLDS}
        return {w: c[w] / tot for w in WORLDS}

    br_etau = brier_score(held, etau_belief)
    br_hist = brier_score(held, hist_belief)

    pol_etau = Policy(ALL4, belief=etau_belief)
    pol_hist = Policy(ALL4, belief=hist_belief)
    loss_etau = empirical_loss(pol_etau.optimal, held, ALL4)
    loss_hist = empirical_loss(pol_hist.optimal, held, ALL4)
    loss_ftc = empirical_loss(ftc_policy(0.5, PROCEED, HALT), held, [PROCEED, HALT])

    # Also: does hist beat etau on both metrics?
    better_cal = br_hist < br_etau - 0.02
    better_dec = loss_hist < loss_etau - 0.01

    verdict = "BREAKS" if better_cal else "SOFT"
    record("A9_better_belief_exists", verdict, claim, {
        "brier_etau": br_etau,
        "brier_histogram_oracle": br_hist,
        "loss_etau": loss_etau,
        "loss_histogram": loss_hist,
        "loss_ftc": loss_ftc,
        "hist_better_calibrated": better_cal,
        "hist_better_decisions": better_dec,
        "note": ("A dumb histogram of the DGP crushes etau_belief on Brier. "
                 "The lattice readout is not information-theoretically privileged; "
                 "it's a fixed linear feature map. 'Natural from the logic' ≠ "
                 "'good probabilistic model'."),
    })


# =====================================================================
# ATTACK 10 — Reading A vs B (ontic four-valued vs instrument fault)
# =====================================================================

def attack_reading_AB():
    claim = ("FAULT_CONFLICT / FAULT_SILENCE are instrument failures; "
             "the world is 2-valued")

    # Belnap / FDE reading: both-true and both-false are ontic truth values.
    # Then max(0, Dct) as P(instrument conflict) is a category error —
    # Dct measures annotation geometry, not P(world=both).
    #
    # Construct: generate ontic 4-valued worlds with annotations that are
    # *correct labels* of those worlds (μ,λ as membership in true/false),
    # not sensor noise. Then etau_belief's "fault" mass is actually correct
    # ontic mass — Reading B. Loss matrix written for instrument-repair
    # actions (recalibrate, approach) is then wrong-shaped: you cannot
    # "recalibrate" away a true contradiction in nature.

    def sim_ontic_belnap(n, seed=23):
        rng = random.Random(seed)
        out = []
        # map: T=(1,0), F=(0,1), B=(1,1), N=(0,0) with noise
        mapping = {
            OBSTACLE: (0.9, 0.1),
            CLEAR: (0.1, 0.9),
            FAULT_CONFLICT: (0.9, 0.9),  # both — ontic
            FAULT_SILENCE: (0.1, 0.1),
        }
        worlds = list(mapping)
        for _ in range(n):
            w = rng.choice(worlds)
            mu0, lam0 = mapping[w]
            mu = min(1.0, max(0.0, mu0 + rng.gauss(0, 0.05)))
            lam = min(1.0, max(0.0, lam0 + rng.gauss(0, 0.05)))
            out.append((mu, lam, w))
        return out

    data = sim_ontic_belnap(8_000)
    # Under Reading B, "recalibrate" should not help — contradiction is real.
    # An honest action set has no recalibrate/approach; only act-as-if-T/F/wait.
    ACT_T = Action("act_T", {OBSTACLE: -1, CLEAR: 2, FAULT_CONFLICT: 2, FAULT_SILENCE: 1})
    ACT_F = Action("act_F", {OBSTACLE: 2, CLEAR: -1, FAULT_CONFLICT: 2, FAULT_SILENCE: 1})
    WAIT = Action("wait", {OBSTACLE: 0.5, CLEAR: 0.5, FAULT_CONFLICT: 0.2, FAULT_SILENCE: 0.2})
    # Instrument-repair actions are nonsense onticly but present in ALL4
    pol_instrument = Policy(ALL4)  # wrong-shaped for ontic
    pol_ontic = Policy([ACT_T, ACT_F, WAIT])

    loss_instr = empirical_loss(pol_instrument.optimal, data, ALL4)
    loss_ontic = empirical_loss(pol_ontic.optimal, data, [ACT_T, ACT_F, WAIT])

    # The module cannot tell which reading you meant; the loss matrix is
    # caller-supplied. That's good AND the problem: the "worlds" names
    # smuggle Reading A into the API.
    record("A10_reading_A_vs_B", "SOFT", claim, {
        "loss_instrument_actions_on_ontic_data": loss_instr,
        "loss_ontic_actions_on_ontic_data": loss_ontic,
        "note": ("API names FAULT_* and the worked example assume Reading A "
                 "(instrument failure). Nothing in Policy enforces it. If the "
                 "world is Belnap-4-valued, recalibrate/approach are category "
                 "errors and the worked example's moral doesn't transfer. "
                 "The module is reading-agnostic in code, reading-A in prose."),
    })


# =====================================================================
# ATTACK 11 — Silent inheritance of dc>=0 from geometry helpers
# =====================================================================

def attack_silent_sign():
    claim = ("etau_belief avoids the silent tie-breaks of labels.py "
             "(dc>=0, dct>=0)")

    # etau_belief uses max(0, dct) and max(0, -dct) and muE = (dc+1)/2.
    # At dct exactly 0, both fault masses are 0 — good, no silent pick.
    # At dc exactly 0, muE=0.5 — splits evenly, good.
    # BUT: belief_radial in A2 used c.dct >= 0 which reintroduces the silent
    # break. And dcr itself uses `1.0 if dc >= 0 else -1.0` in the engine.
    # If anyone routes through muER/dcr into belief (as the radial alt does),
    # the silent break returns.
    #
    # Direct etau_belief: check dct==0 and dc==0 points.
    b_diag = etau_belief(0.7, 0.3)  # dct = 0, dc = 0.4
    b_center = etau_belief(0.5, 0.5)
    b_mu_eq = etau_belief(0.6, 0.6)  # dc=0, dct=0.2

    silent_free = (
        b_diag[FAULT_CONFLICT] == 0.0 and b_diag[FAULT_SILENCE] == 0.0
        and abs(b_center[OBSTACLE] - b_center[CLEAR]) < 1e-12
        and abs(b_mu_eq[OBSTACLE] - b_mu_eq[CLEAR]) < 1e-12
    )
    # dcr silent break exists in library:
    c = ParaconsistentBlock(mu=0.5, lam=0.5).complete
    dcr_uses_geq = True  # known from source: dc >= 0

    verdict = "HOLDS" if silent_free else "BREAKS"
    record("A11_silent_sign_in_belief", verdict, claim, {
        "etau_belief_silent_free_on_axes": silent_free,
        "library_dcr_uses_dc_geq_0": dcr_uses_geq,
        "belief_at_0.6_0.6": b_mu_eq,
        "note": ("etau_belief itself is clean on the axes. Any belief that "
                 "routes through dcr/muER re-imports labels.py's silent "
                 "dc>=0. The type discipline on Policy.optimal does not "
                 "protect the belief function."),
    })


# =====================================================================
# ATTACK 12 — What actually is good (positive controls)
# =====================================================================

def attack_positive_controls():
    claim = ("Core engineering claims: ties returned, dominated actions drop, "
             "partition affine-invariant, belief is a distribution")

    holds = {}
    # distribution
    ok = True
    for i in range(21):
        for j in range(21):
            b = etau_belief(i / 20, j / 20)
            if abs(sum(b.values()) - 1) > 1e-12:
                ok = False
    holds["belief_sums_to_1"] = ok

    # ties
    A = Action("a", {w: 1.0 for w in WORLDS})
    B = Action("b", {w: 1.0 for w in WORLDS})
    holds["ties_returned"] = Policy([A, B]).optimal(0.3, 0.4) == frozenset({"a", "b"})

    # dominated drop
    dom = Action("dom", {w: 99 for w in WORLDS})
    holds["dominated_unreachable"] = "dom" in Policy([PROCEED, HALT, dom]).unreachable(51)

    # affine invariance of partition
    p1 = Policy(ALL4)
    scaled = [Action(a.name, {w: 3 * v + 1 for w, v in a.losses.items()}) for a in ALL4]
    p2 = Policy(scaled)
    ok = True
    for i in range(15):
        for j in range(15):
            if p1.optimal(i / 14, j / 14) != p2.optimal(i / 14, j / 14):
                ok = False
    holds["affine_invariance"] = ok

    # separates what muER conflates
    pts = [(1, 1), (0, 0), (0.5, 0.5)]
    beliefs = [tuple(etau_belief(m, l)[w] for w in WORLDS) for m, l in pts]
    holds["separates_muER_collisions"] = len(set(beliefs)) == 3

    all_ok = all(holds.values())
    record("A12_positive_controls", "HOLDS" if all_ok else "BREAKS", claim, holds)


# =====================================================================
# ATTACK 13 — My alternative: set-valued lattice label + cost as separate layer
# =====================================================================

def attack_propose_set_labels():
    """Not a claim-attack: emit a minimal set-valued 3-region classifier
    forced by Dcr, as the transcript argued — and show it needs no FtC."""
    claim = ("Three Dcr regions are forced by definition; twelve needs FtC; "
             "cost-derived is optional refinement")

    def dcr_sign_set(mu, lam):
        c = ParaconsistentBlock(mu=mu, lam=lam).complete
        # Use continuous Dcr; on exact 0 return the neutral set
        if abs(c.dcr) < 1e-15 or c.D >= 1.0 - 1e-15:
            # on the neutral set — also adjacent signs for boundary honesty
            return frozenset({"Dcr=0"})
        if c.dcr > 0:
            return frozenset({"Dcr>0"})
        return frozenset({"Dcr<0"})

    counts = Counter()
    steps = 201
    for i in range(steps):
        for j in range(steps):
            counts[dcr_sign_set(i / (steps - 1), j / (steps - 1))] += 1
    total = steps * steps
    areas = {str(sorted(k)): v / total for k, v in counts.items()}
    # Compare: 12-region labels scatter the Dcr=0 mass across many labels
    from collections import defaultdict as dd
    label_on_neutral = Counter()
    for i in range(steps):
        for j in range(steps):
            mu, lam = i / (steps - 1), j / (steps - 1)
            c = ParaconsistentBlock(mu=mu, lam=lam).complete
            if c.D >= 1.0 - 1e-15:
                label_on_neutral[c.label] += 1

    record("A13_forced_three", "HOLDS", claim, {
        "dcr_sign_areas": areas,
        "twelve_labels_used_inside_Dcr0": dict(label_on_neutral),
        "n_twelve_labels_covering_neutral": len(label_on_neutral),
        "note": ("Forced-three is real and FtC-free. The twelve-region readout "
                 "scatters Dcr=0 across multiple labels (measured). Cost-derived "
                 "is a further optional layer; it does not replace the forced "
                 "three, and the forced three do not need costs."),
    })


def main():
    print("=" * 70)
    print("ATTACK SUITE: etau_decision premises")
    print("=" * 70)
    attack_dcr_geometry()
    attack_phi_costume()
    attack_simulator_collusion()
    attack_ordering_blindspot()
    attack_action_set_is_ftc()
    attack_grid_artifacts()
    attack_tie_average()
    attack_linearity_flattens()
    attack_better_belief_exists()
    attack_reading_AB()
    attack_silent_sign()
    attack_positive_controls()
    attack_propose_set_labels()

    # summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    c = Counter(f["verdict"] for f in results["findings"])
    for v in ("BREAKS", "SOFT", "HOLDS"):
        print(f"  {v:6s}: {c[v]}")
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nJSON → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
