# Paraconsistent Annotated Evidential Logic Eτ

A downloaded, verified, offline-installable copy of an Eτ (PAL2v) implementation,
plus reference documentation for the logic itself.

Eτ is a non-classical logic that reasons over propositions annotated with **two
independent degrees of evidence** — one favourable, one contrary — so that
contradictory and incomplete information can be carried through inference
without trivialising the system.

## What is here

| Path | Contents |
|---|---|
| `docs/paraconsistent-annotated-evidential-logic-etau.md` | The logic: lattice, degrees, the 12 states, operators, bibliography |
| `vendor/paraconsistent/` | The downloaded distributions (wheel + sdist) with provenance and licence |
| `requirements.txt` | Hash-pinned dependency for reproducible installs |
| `scripts/install-offline.sh` | Installs from `vendor/` with no network access |
| `examples/etau_demo.py` | Runnable demo of the para-analyser and block chaining |
| `tests/test_etau.py` | Verification of the downloaded artefact against the literature |
| `src/etau_decision.py` | Cost-derived partitions: regions induced by what you'd *do*, not by `FtC` |
| `examples/cost_derived_partition.py` | The above, worked through against an `FtC` baseline |
| `tests/test_decision.py` | Tests for the decision layer |
| `src/etau_ensemble.py` | Runs the pipeline under several defensible belief maps and answers plurally where they disagree |
| `tests/test_ensemble.py` | Tests, including the monotonicity that makes the spread a lower bound |
| `tests/test_critique_findings.py` | Pins Grok Build's findings (`luckylove#1`) |

## Cost-derived partitions

`src/etau_decision.py` is an argument in code. The twelve regions partition
annotations by the geometry of the lattice, with `FtC` drawing the lines —
a stipulation you can replace but never discover to be wrong. The alternative:
partition by *what you would do*, so two annotations share a region iff the same
action minimises expected loss, and the boundaries land where cost curves cross.

Three consequences, all demonstrated in the example:

- **The region count follows the action vocabulary rather than a threshold.**
  Supply four actions, get however many are optimal somewhere; cheapening one
  action's payoff drops the induced count from 4 to 3. An earlier draft of this
  file said "nobody chose 4." That was false — I chose the vocabulary, and Grok
  Build measured it (`critique/CRITIQUE.md` A5: 1/2/3/4/5 actions → 1/2/3/4/4
  regions). The free parameter changed type, from `float FtC` to
  `(Action, loss matrix)[]`. It did not vanish.
- **Ties are returned, not broken.** `Policy.optimal` yields a `frozenset` of
  argmin actions, so a point on a crossing hands you both names. This is the
  discipline `paraconsistent` 0.2.4 gives up by returning `str` and settling
  `dc >= 0` silently.
- **The parameter acquires a truth condition.** `calibration_report`,
  `brier_score` and `unreachable_under` can catch the model being wrong. There's
  no equivalent table for `FtC`, because a threshold makes no claim about the
  world. This is weaker than "reality will refute you": `argmin` needs only the
  order of expected losses, so reality refutes you where the ordering breaks and
  is silent elsewhere.

The belief model routes through `Dct`'s sign, which `μER` cannot do: `μER` maps
`p(1,1)`, `p(0,0)` and `p(0.5,0.5)` all to `0.5`, collapsing "channels
contradict", "channels dead" and "channels agree on a coin-flip" into one
number. They warrant different actions.

It is a **feature map, not a posterior** — I chose the split
`(φ·μE, φ·(1−μE), max(Dct,0), max(−Dct,0))`, and calling it "derived from the
logic" in the first draft was the same costume this module criticises in `FtC`.
A 20×20 histogram that does not know Eτ exists beats it on held-out Brier by
orders of magnitude (`HistogramBelief`). Use it as features, or fit from data.

Two results worth flagging, since both cut against the module:

1. **The shipped default belief model fails its own calibration check.** On the
   example's simulator it is systematically underconfident — predicts 0.55 where
   the world delivers 0.94 — because it splits the decided mass `φ` by `μE`, a
   linear readout of `Dc` that cannot sharpen faster than evidence separates.
   That's the hook working, not the hook failing.
2. **It still beats the `FtC` baseline anyway.** `argmin` needs only the *order*
   of expected losses, so a monotone distortion can leave every boundary intact.
   Calibration and decision quality are different properties — measure both.

## Quick start

Offline, from the vendored artefacts:

```bash
./scripts/install-offline.sh          # into ./.venv
.venv/bin/python examples/etau_demo.py
.venv/bin/python tests/test_etau.py
```

Or from PyPI with hash verification:

```bash
pip install -r requirements.txt
```

## The 30-second version of the logic

A proposition is annotated `p(μ, λ)` with `μ, λ ∈ [0, 1]`:

- **μ** — degree of *favourable* evidence
- **λ** — degree of *contrary* evidence

`μ` and `λ` are independent, which is the whole point: `p(1, 1)` means "fully
supported *and* fully refuted" (inconsistent) and `p(0, 0)` means "neither
supported nor refuted" (paracomplete). Neither one blows up the system.

Two derived degrees drive everything else:

```
Degree of certainty      Dc  = μ − λ        ∈ [−1, 1]
Degree of contradiction  Dct = μ + λ − 1    ∈ [−1, 1]
```

`Dc` says how true/false the proposition is; `Dct` says how much you should
trust that answer. Together they place the proposition in one of **12 logical
states** — four extreme (true, false, inconsistent, paracomplete) and eight
transitional.

```
                    Dct = +1  ⊤ (inconsistent)
                          │
        Qf→⊤              │              Qt→⊤
              Q⊤→f        │        Q⊤→t
                          │
Dc = −1  f ───────────────┼─────────────── t  Dc = +1
                          │
              Q⊥→f        │        Q⊥→t
        Qf→⊥              │              Qt→⊥
                          │
                    Dct = −1  ⊥ (paracomplete)
```

See `docs/paraconsistent-annotated-evidential-logic-etau.md` for the full
treatment.

## What was downloaded

[`paraconsistent` 0.2.4](https://pypi.org/project/paraconsistent/) — "Paraconsistent-Lib",
an MIT-licensed, zero-dependency Python implementation of PAL2v. It is the
library described in *Paraconsistent-Lib: an intuitive PAL2v algorithm Python
Library* ([arXiv:2511.20700](https://arxiv.org/abs/2511.20700)), whose authors
include **João Inácio da Silva Filho** — the originator of the PAL2v algorithms.

Both distributions were downloaded from PyPI and their SHA-256 digests checked
against PyPI's published values. See `vendor/paraconsistent/PROVENANCE.md`.

## Verification performed

The download was not taken on trust. `tests/test_etau.py` checks the installed
package against values derived independently from the literature:

- the four extreme states and the centre point classify correctly;
- `Dc` and `Dct` match their definitions;
- the library's radial distance agrees with the literature formula
  `d = √((1 − |Dc|)² + Dct²)` to within 4.5e-16 over 20 000 random points
  (the library computes it as a normalised distance to the nearest apex — the
  two forms are algebraically identical, see the docs);
- `Dcr`, `φ`, `μE`, `μECT`, `μER` and the decision output match their
  definitions;
- negation `¬p(μ, λ) = p(λ, μ)` flips the sign of `Dc` and preserves `Dct`.

## Known issues in the upstream package

Found while verifying; recorded here so they are not rediscovered later.

1. **The shipped example is broken.** `paraconsistent/examples/demo_wiring.py`
   imports `from mrn.paraconsistent.block import ParaconsistentBlock` — a stale
   module path from before the package was renamed — and then uses
   `config.FL` and `complete.gc`, neither of which exists. Importing it raises
   `ModuleNotFoundError: No module named 'mrn'`. `examples/etau_demo.py` here is
   a working replacement.
2. **No logical connectives.** The package implements the *para-analyser*
   (degrees, geometry, state classification) but not Eτ's `¬`, `∧`, `∨`. Those
   are one-liners over the annotation pair — see the docs for the definitions.
3. **No `LICENSE` file in the distributions.** `pyproject.toml` declares
   `license = { file = "LICENSE" }` but no such file is packaged; the MIT text
   lives only in section 9 of the README. It is extracted to
   `vendor/paraconsistent/LICENSE` for convenience.
4. **Documentation is in Portuguese.** The docs here are in English.
5. **Arbitrary tie-breaking on the axes.** The classifier tests `dc >= 0` and
   `dct >= 0`, so exact zeros count as positive: annotations with `μ = λ` always
   resolve to the *true* side (`p(0.6, 0.6)` is `Q┬→t`), and annotations on the
   perfectly non-contradictory diagonal `μ + λ = 1` resolve to the
   *inconsistent* side (`Qt→┬` even though `Dct = 0`). Both are pinned in the
   tests so an upstream change would be caught.

## Sources that could not be downloaded

This environment's egress policy allows `pypi.org` and `files.pythonhosted.org`
but blocks `arxiv.org`, `github.com`, and the publisher sites, so the papers
themselves could not be fetched. `docs/paraconsistent-annotated-evidential-logic-etau.md`
carries a full bibliography with DOIs and URLs for retrieval from an unrestricted
network.

## How much the lines matter

`src/etau_ensemble.py` runs the decision pipeline under three equally defensible
belief maps at once — the ones Grok Build named in `critique/CRITIQUE.md` A2 —
and returns the *union* of their argmin sets. Where they agree you get a
singleton. Where the choice of map changes what you'd do, the answer is plural
and cannot be read as a single verdict.

Measured, with the shipped four-action vocabulary:

| members | grid | deployment |
|---|---|---|
| 1 (`etau_belief` only) | 0.012 | 0.000 |
| 2 (+ radial) | 0.538 | 0.537 |
| 3 (+ hard argmax) | **0.551** | **0.538** |

**Over half the annotation square, and over half of actual traffic, the choice
of mass map changes the action.** That number is what an annotation reading
"this was a choice" is silent about.

Two things to hold onto, or this becomes the next thing that needs backing out:

1. **These are lower bounds.** Disagreement is `|union| > 1` and unions only
   grow, so adding a member can never reduce the measured spread
   (`test_sensitivity_is_monotone_in_members`). A narrow spread means "the
   members I picked agree", never "the choice doesn't matter" — which is why
   `n_members` is reported beside every figure and `Ensemble.add()` is a
   one-liner.
2. **The ensemble is itself a drawn line.** Three maps, proposed by one outside
   reader. Not a basis, not canonical. The repair for a suspiciously narrow
   spread is to add a member, and that repair is available to anyone.
