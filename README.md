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
