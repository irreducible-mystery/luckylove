# Critique of `etau_decision` — cost-derived partitions of Eτ

**Author:** Grok Build (xAI), 2026-08-06  
**Object:** [`src/etau_decision.py`](../src/etau_decision.py) on
`irreducible-mystery/luckylove` @ `claude/paraconsistent-logic-download-gowix6`  
**Method:** Run the shipped suite, then a hostile probe suite
([`attack_etau_decision.py`](./attack_etau_decision.py)).  
**Results:** [`attack_results.json`](./attack_results.json) — **5 BREAKS / 3 SOFT / 5 HOLDS**.

This is not a UI review. It is an attack on the *argument encoded in the
module*. Prose is ephemeral; the attack script is the receipt.

---

## 0. How I am reading this (different model, different cost function)

I am not Claude and not you. My default bias is: **prefer mechanisms that
fail loudly under distribution shift** over mechanisms that look clean on a
hand-built demo. That is a choice. Stating it up front so the rest can be
read as conditioned on it.

What I liked before I attacked anything:

- Returning `frozenset` instead of silently picking a side. That is a real
  type-level fix for a real bug class (`dc >= 0` in `labels.py`).
- Shipping a calibration hook that **fires on the author's own default**.
  Most decision-layer code never instruments its own belief model.
- Treating region count as something the action set *induces*, not something
  you paint on a lattice diagram. Directionally correct.

What I distrust on sight:

- Worked examples whose data generator samples `(μ, λ)` *from* the world
  labels the belief model assumes. That is a handshake, not a test.
- Rhetoric of the form "nobody chose N" when someone absolutely chose the
  action vocabulary that produces N.
- Claims that a reparameterisation of the unit square is a probability.

---

## 1. What HOLDS (do not throw these away)

From `A12` and the shipped 17/17 + 22/22 tests:

| Claim | Status |
|---|---|
| `etau_belief` is a distribution (`φ + \|Dct\| = 1`) | HOLDS on dense grid |
| Ties returned, never broken | HOLDS |
| Dominated actions become `unreachable` | HOLDS |
| Affine transform of all losses preserves partition | HOLDS (as expected of argmin) |
| Separates the three points μER conflates | HOLDS |
| `etau_belief` itself is silent-break free on the axes | HOLDS (`A11`) |
| Forced three via `sign(Dcr)` needs no FtC | HOLDS (`A13`); **7 of 12 labels** live inside `Dcr=0` |

The engineering core is sound. The argument around it is where the damage is.

---

## 2. BREAKS — ranked by how much they matter

### 2.1 `φ` as "decided mass" is a costume (`A2`)

**Claim in module docstring:** φ is the mass that is *decided*, split by μE.

**Probe:** Two alternative mass assignments that also:

- use only quantities the library already computes,
- sum to 1 everywhere,
- have equally clean arithmetic,

disagree with `etau_belief` on **54%** (radial/`D`) and **21%** (hard argmax
on Dc) of the annotation square, under the *same* four actions.

```
region_count stays 4 in all three — the count is stable;
the *walls* move. So "derived from the logic" does not pin down a partition.
```

**My read:** Calling φ "decided" is the same move the module criticises in
FtC — a modeling choice wearing a derivation costume. The honest statement
is: *we chose the feature map `(φ·μE, φ·(1−μE), max(Dct,0), max(−Dct,0))`.*
That map is reasonable. It is not forced.

### 2.2 Action set is the new FtC (`A5`)

**Claim:** "Nobody chose 4. Four actions happened to each be optimal somewhere."

**Probe:** Region counts under deliberate vocabularies:

| action vocabulary | regions |
|---|---|
| `{halt}` | 1 |
| `{proceed, halt}` | 2 |
| `{proceed, halt, recalibrate}` | 3 |
| all 4 | 4 |
| 4 with 2 dominated | 2 |
| 5 with a centre-wait | 4 (wait never carved interior at this grid) |

Count *is* an output of the free parameter. The free parameter did not
vanish — it changed type from `float FtC` to `(Action, loss matrix)[]`.
That is a **strict upgrade** (multi-dimensional, checkable). It is not
"parameterisation eliminated." The sentence "nobody chose 4" should be
deleted; it is false about the human process that produced the example.

### 2.3 Better beliefs exist; lattice readout is a weak feature map (`A9`)

**Probe:** Histogram the shipped simulator's training set on a 20×20 grid
of `(μ, λ)`, use the histogram as belief.

| model | Brier (held-out) | mean loss (4 actions) |
|---|---|---|
| `etau_belief` | 0.146 | −0.524 |
| histogram of DGP | **0.000075** | **−0.741** |
| FtC baseline | — | 1.115 |

A dumb nonparametric model that *does not know Eτ exists* crushes the
lattice readout on the author's own data. Therefore:

- "Natural from the logic" ≠ "good probabilistic model."
- The calibration failure the module reports is not an inevitable tax of
  using Eτ; it is a tax of using a *fixed linear feature map* as if it were
  a posterior.

**What I would do instead:** Treat `(Dc, Dct, D, …)` as *features* for a
calibrated model (Platt / isotonic / small net / Dirichlet), not as
probabilities. Keep the set-valued policy layer. Drop the claim that
`etau_belief` is privileged.

### 2.4 Dcr hourglass waist claim is false (`A1`)

**Transcript claim:** neutral region is "pinched to zero width at exactly
`p(0.5, 0.5)`."

**Probe:** The entire diagonal `μ = λ` has `d ≥ 1`, hence `Dcr = 0`. The
waist is not a point; it is the diagonal. Area ≈ `1 − π/4` is approximately
right (measured 0.21468 vs 0.21460) and the Euclidean-disk complement
overlaps Dcr=0 at >99.9% on a fine grid — so the *area* story is fine and
the *waist* story is wrong.

Minor relative to 2.1–2.3, but it is the kind of geometric poetry that
should not survive first contact with a 401-point diagonal sweep.

### 2.5 Simulator/ontology collusion is real; FtC-beating is (annoyingly) robust (`A3`)

**Methodology sin (real):** The shipped simulator draws `(μ, λ)` *from*
world labels that match `etau_belief`'s four-way split. That is not an
independent test of the belief model; it is a consistency check.

**Surprise (my prior was wrong):** Cost-derived still beats FtC on three
hostile generators (binary-only world, uncorrelated annotations, inverted
faults). FtC-threshold-on-μER is simply a very bad decision rule for these
loss matrices. Almost anything with access to the loss matrix wins.

**Clean hit inside the probe:** On a binary-only world (faults never
obtain), the 4-action policy is **worse** than the 2-action policy by
~0.07 mean loss — it spends measure on `recalibrate`/`approach` that can
never pay. `unreachable()` does not catch this, because those actions *are*
optimal on some annotations; they are just optimal for worlds that never
arrive. **Annotation-square optimality ≠ deployment optimality.**

That gap is the sharpest thing in A3. The module measures the former and
sells it as the latter.

---

## 3. SOFT — real issues, narrower than the prose

### 3.1 Ordering blindspot (`A4`) — smaller than I expected

Power and flat-mix distortions of belief move ~11–14% of the partition and
hurt mean loss by ~0.17–0.23. So for *this* action set, bad magnitudes *do*
leak into decisions. The module's admission ("calibration ≠ decision") is
still correct in principle; the hole is not a free lunch for arbitrary
miscalibration. I mark SOFT/HOLDS leaning HOLDS: the check is weaker than
"reality will refute you," stronger than "ordering is all that matters."

### 3.2 Tie averaging (`A7`)

`empirical_loss` on a frozenset pays the mean of the tied pure actions.
That is a **scoring convention**, not a physical outcome. Deployment always
emits one action. Maximin / CVaR / any nonlinear functional of the loss
vector breaks ties differently. "No free lunch from refusing to choose" is
true for risk-neutral expectation of a fair coin; oversold as a general
principle.

### 3.3 Linearity flattens the notebook's circle (`A8`)

Expected loss is linear in belief; `etau_belief` is piecewise-linear in
`(μ, λ)`. Composition ⇒ piecewise-linear region walls. The radial/`Dcr`
geometry (the circle in the notebook) is not what places cost-derived
boundaries. Adjacent-cell finite differences are a weak metric here; the
analytical point is the real one. If the interesting structure was the
non-linear radial geometry, this build is the wrong layer for it.

### 3.4 Reading A vs B (`A10`)

Code is reading-agnostic. Prose and the worked example assume Reading A
(instrument failure). If the world is Belnap-four-valued onticly,
`recalibrate`/`approach` are category errors. Name the reading in the API
or stop naming worlds `FAULT_*`.

---

## 4. What I would keep, change, and drop

### Keep

1. `Policy.optimal → frozenset[str]` — non-negotiable upgrade over `str`.
2. `unreachable` / `region_count` as diagnostics (with grid-resolution
   caveats documented).
3. `empirical_loss` + `calibration_report` + `brier_score` as a bundle.
4. The *direction* of the argument: parameters with truth conditions beat
   pure stipulations.

### Change

1. **Demote `etau_belief` from default truth to default feature map.**
   Provide a calibrated adapter (even isotonic on Dc/Dct) and show it in
   the example. The histogram result makes shipping the linear readout as
   *the* belief model indefensible.
2. **Rewrite the "nobody chose N" language.** Say: *count is a function of
   the action/loss parameter; that parameter is checkable; FtC is not.*
3. **Measure deployment risk, not just annotation-square coverage.**
   An action optimal on a positive-measure set of annotations can still be
   pure waste under the actual world measure (`A3` binary waste).
4. **Fix the waist geometry in any prose that repeats the transcript.**
   Neutral set contains the full `μ = λ` diagonal.

### Drop

1. Any implication that φ-splitting is *derived* rather than *chosen*.
2. Any implication that beating FtC on the friend simulator is strong
   evidence. FtC is a soft target; hostile generators are the bar.
3. Treating `empirical_loss` on frozensets as if it were a policy.

---

## 5. Forced three, sets, cost-derived — re-stacked

The transcript's stack is almost right. Re-order by *what is forced*:

```
1. Forced by definition of Dcr clamp:
     {Dcr>0, Dcr<0, Dcr=0}          — no FtC, no costs, area ~21.5% neutral
     return set-valued on boundaries

2. Optional geometric vocabulary:
     12 regions                        — needs FtC / VlV / VlF; scatters
     the neutral set across ~7 labels

3. Optional decision layer:
     cost-derived partition            — needs action/loss; checkable;
     walls are piecewise-linear in the
     belief feature map you chose
```

Cost-derived does **not** replace (1). It sits on top of a belief map that
is currently (weakly) (2)-flavoured. The interesting open problem is not
"three vs twelve vs cost" — it is **which feature map from `(μ, λ)` into a
simplex is least wrong for the actual sensors**, with (1) as a free
abstention region when `|Dcr|` is zero.

---

## 6. Things I notice that humans/Claude underweighted

1. **FtC is an extremely soft baseline.** Almost every probe still beat it.
   Comparisons to FtC mostly measure "FtC is bad," not "cost-derived is
   good." Compare against: histogram belief, calibrated logistic on
   `(Dc, Dct)`, and a 2-action cost policy. The 2-vs-4 gap on binary data
   is more informative than 4-vs-FtC on friend data.

2. **`unreachable` is about the annotation prior (uniform grid), not the
   world prior.** Those are different measures. An action can be
   grid-reachable and world-useless. Add `unreachable_under(records)`.

3. **The type fix is incomplete.** `frozenset` on the *output* does not
   protect the *belief* function. Any belief that routes through `dcr`
   re-imports `dc >= 0` (`engine.py:52`). Document which library fields are
   silent-breaking.

4. **Affine invariance is a warning, not only a feature.** If all loss
   matrices in a projective class give the same partition, then absolute
   loss magnitudes (the things finance/safety teams argue about) are
   invisible to the geometry. Calibration of *belief* becomes the only
   channel for magnitude information to matter — which is exactly the
   channel the default model fails.

5. **Independent notebook convergence (square → circle → non-linearity) is
   evidence about the lattice, not about expected-loss partitions.** The
   cost layer linearises. If the notebook was tracking curvature of real
   certainty, the build walked away from it.

---

## 7. Reproduction

```bash
# from luckylove root
./scripts/install-offline.sh
.venv/bin/python tests/test_decision.py          # 17/17
.venv/bin/python tests/test_etau.py              # 22/22
.venv/bin/python examples/cost_derived_partition.py
.venv/bin/python critique/attack_etau_decision.py
# → critique/attack_results.json
```

---

## 8. Verdict in one paragraph

The module is a **good piece of engineering wrapped in a slightly oversold
argument**. Set-valued argmin, induced region counts, and calibration hooks
are the real contribution and they work (A12). The oversell is: treating φ
as derived decided-mass (A2), treating action vocabulary as non-choice (A5),
treating a linear lattice readout as a probability (A9), and treating a
friend simulator as a falsification (A3 methodology). The forced-three
Dcr story is solid and under-used; cost-derived should be framed as an
optional checkable layer on top of it, not as the thing that finally makes
parameters honest. Honesty was always available: name your free parameters
and give them tests. This module does that for losses. It should do it for
the belief map too.
