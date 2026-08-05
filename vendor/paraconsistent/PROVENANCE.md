# Provenance — `paraconsistent` 0.2.4

## What this is

"Paraconsistent-Lib" — a zero-dependency Python implementation of Paraconsistent
Annotated Evidential Logic Eτ (PAL2v). It provides the para-analyser: the
degrees of certainty and contradiction, the derived geometry, the resulting
evidence degrees, and classification into the 12 logical states.

## Retrieval

| | |
|---|---|
| Downloaded on | 2026-08-05T06:04Z |
| Source index | PyPI (`https://pypi.org/simple/`) |
| Project page | <https://pypi.org/project/paraconsistent/> |
| Version | `0.2.4` — the latest of 14 releases (`0.1.0` … `0.2.4`) |
| Method | direct HTTPS `GET` from `files.pythonhosted.org` |

## Artefacts and digests

Both digests were compared against the values published in PyPI's JSON API
(`https://pypi.org/pypi/paraconsistent/json`) and match.

### `paraconsistent-0.2.4-py3-none-any.whl`

```
size    13590 bytes
sha256  b5b69a1893d5ba5eff0df5a10e7851cdc5e4788cbb85e41d2ed2c215a67c6b9f
url     https://files.pythonhosted.org/packages/26/16/44cde68657e145e0842042341777bde745553441f081c69d0c53f0454223/paraconsistent-0.2.4-py3-none-any.whl
uploaded 2025-11-09T12:17:55Z
```

### `paraconsistent-0.2.4.tar.gz`

```
size    15469 bytes
sha256  2aa537d4048dec80fc7503c0bc167dc0088da241c1a0ff754b79bbb697c5e440
url     https://files.pythonhosted.org/packages/89/1e/7e9fe1dd5adf653ca260a72d6efd5db5394b3175d683df2ee0e13156900b/paraconsistent-0.2.4.tar.gz
uploaded 2025-11-09T12:17:56Z
```

Re-verify at any time:

```bash
sha256sum -c SHA256SUMS
```

## Package metadata (from `PKG-INFO`)

```
Name           paraconsistent
Version        0.2.4
Summary        Blocos de lógica paraconsistente com API simples.
Author-email   Diego <diego263526352635@gmail.com>
Keywords       logic, paraconsistent, LPA2v, reasoning, fuzzy
Requires-Python >=3.9
Dependencies   none (standard library only)
License        MIT (classifier: "License :: OSI Approved :: MIT License")
```

The README shipped in the distribution gives the upstream source repository as
<https://github.com/dirgogoo/Paraconsistent-Lib>; the project's documentation
site is <https://eailab-ifsp.github.io/Paraconsistent-Lib/> (EAILab, IFSP).

## Associated publication

de Carvalho Junior, A.; da Cruz, D. O.; Alves, B. da S.; da Silva Paulo Junior,
F.; **da Silva Filho, J. I.** — *"Paraconsistent-Lib: an intuitive PAL2v
algorithm Python Library"*, arXiv:2511.20700, 24 Nov 2025.
<https://arxiv.org/abs/2511.20700>

João Inácio da Silva Filho, a co-author, is the originator of the PAL2v
algorithms, which is the strongest available signal that this implementation
faithfully reflects the intended formalism.

## Licence

MIT. The distributions declare `license = { file = "LICENSE" }` in
`pyproject.toml` but **do not actually package a `LICENSE` file** — the licence
text appears only in section 9 of the bundled `README.md`. That text is
extracted verbatim to `LICENSE` in this directory.

## Contents of the sdist

```
paraconsistent-0.2.4/
├── MANIFEST.in
├── PKG-INFO
├── README.md
├── pyproject.toml
├── setup.cfg
└── src/paraconsistent/
    ├── __init__.py
    ├── py.typed
    ├── blocks/
    │   ├── __init__.py
    │   └── block.py          ParaconsistentBlock — input/config/complete proxies, lazy cache
    └── core/
        ├── __init__.py
        ├── config.py         BlockParams (FtC, VlV, VlF, L), validation and clamping
        ├── engine.py         ParaconsistentEngine — the analysis pipeline
        ├── labels.py         12-region classifier
        ├── metrics.py        clamp01, radial distance to nearest apex
        └── types.py          Complete TypedDict
```

901 lines total across the Python and Markdown files. No compiled code, no
network access, no subprocess use, no filesystem writes — inspected before
installation.

## Integrity checks performed after download

1. SHA-256 of both artefacts matched PyPI's published digests.
2. The wheel installed offline (`pip install --no-index --find-links .`) into a
   clean virtualenv, confirming it is self-sufficient.
3. Numerical behaviour was verified against the literature — see
   `tests/test_etau.py` in the repository root.
