# Paraconsistent Annotated Evidential Logic Eτ

> Also written **PAL Eτ**, **LPAe τ** (Portuguese: *Lógica Paraconsistente
> Anotada Evidencial Eτ*), and — for the algorithmic form — **PAL2v**
> (*Paraconsistent Annotated Logic with annotation of 2 values*, Portuguese
> **LPA2v**). The names are used interchangeably in the literature; PAL2v tends
> to denote the computational/algorithmic treatment of Eτ.

---

## 1. Why it exists

Classical logic is *explosive*: from a contradiction, everything follows.

```
p, ¬p ⊢ q      for any q
```

That is fatal for any system that fuses evidence from multiple real sensors,
experts, or databases, because such sources routinely disagree. A classical
system given `sensor_A: obstacle` and `sensor_B: ¬obstacle` derives every
proposition in its language and becomes useless.

A **paraconsistent** logic rejects explosion: contradictions are contained
rather than catastrophic. Eτ goes further than merely tolerating contradiction —
it *measures* it, and makes the measurement available as a first-class output
that downstream control logic can act on.

Eτ is simultaneously:

- **paraconsistent** — it tolerates `p` and `¬p` both being well supported;
- **paracomplete** — it tolerates neither `p` nor `¬p` being supported;
- **non-alethic** — it is both of the above at once.

## 2. Lineage

| Year | Contribution |
|---|---|
| 1963 | **Newton C. A. da Costa** introduces the paraconsistent calculi `C_n`, founding the field. |
| 1987 | **V. S. Subrahmanian** introduces *annotated logics* in the setting of logic programming. |
| ~1990 | **da Costa**, with Subrahmanian and Vago, formalises annotated logics; a Hasse lattice carrying **two** values per annotation is proposed. |
| 1998 | **João Inácio da Silva Filho** gives the first practical PAL2v application — the decision-making process of the **Emmy** autonomous mobile robot. |
| 1999 | Da Silva Filho's USP doctoral thesis develops PAL2v methods, algorithms and electronic-circuit implementations. |
| 2000s– | **Jair Minoro Abe**, **Kazumi Nakamatsu** and Da Silva Filho develop and apply Eτ across robotics, control, medical diagnosis, pattern recognition and networks; Abe and Nakamatsu fix the label *PAL2v* (2009). |
| 2024 | A comprehensive review consolidates three decades of algorithms, applications and open problems. |

## 3. The annotation lattice

An atomic formula of Eτ is

```
p(μ, λ)        μ, λ ∈ [0, 1]
```

read as *"the proposition p has favourable evidence μ and contrary evidence λ."*

- **μ** — degree of favourable evidence (*grau de crença*)
- **λ** — degree of contrary evidence (*grau de descrença*)

The two are **independent**. This is the structural difference from probability
(`P(¬p) = 1 − P(p)` is forced) and from a single-valued fuzzy membership. In Eτ,
`μ + λ` may be anything in `[0, 2]`, and the deviation from 1 is itself
meaningful.

The annotation constants form the lattice `τ = ⟨[0, 1]², ≤⟩` with the
component-wise order, drawn as the unit square — in the literature the
**QUPC** (*Quadrado Unitário do Plano Cartesiano*, Unit Square of the Cartesian
Plane). Its four corners are the extreme states:

```
 λ
 1 │ (0,1) F ─────────────── ⊤ (1,1)
   │  false                   inconsistent
   │
   │
 0 │ (0,0) ⊥ ─────────────── V (1,0)
   │  paracomplete            true
   └──────────────────────────────── μ
     0                        1
```

| Annotation | State | Meaning |
|---|---|---|
| `p(1, 0)` | **V** / **t** — True | fully supported, not refuted |
| `p(0, 1)` | **F** / **f** — False | fully refuted, not supported |
| `p(1, 1)` | **⊤** — Inconsistent | fully supported *and* fully refuted |
| `p(0, 0)` | **⊥** — Paracomplete | neither supported nor refuted |

`⊤` is the top of the lattice and `⊥` the bottom. Crucially, `⊤` does **not**
trivialise the theory — that is exactly what paraconsistency buys.

## 4. The two derived degrees

Rotating the QUPC by 45° gives the **USCP** (*Unit Square of the Cartesian
Plane* in Dc/Dct coordinates, Portuguese **QUPC** rotated — often called the
*lattice of the para-analyser*). Its axes are:

### Degree of certainty

```
Dc = μ − λ                      Dc ∈ [−1, 1]
```

How *decided* the proposition is. `Dc = +1` → true; `Dc = −1` → false;
`Dc = 0` → the evidence for and against is balanced.

### Degree of contradiction

```
Dct = μ + λ − 1                 Dct ∈ [−1, 1]
```

How *conflicted* the evidence is. `Dct = +1` → both sources maximally assert
(inconsistent); `Dct = −1` → neither asserts (paracomplete); `Dct = 0` → the
evidence is "well-behaved", `μ + λ = 1`, the case that a probability would
describe.

The map `(μ, λ) ↦ (Dc, Dct)` is a bijection — nothing is lost:

```
μ = (Dc + Dct + 1) / 2
λ = (Dct − Dc + 1) / 2
```

### Certainty interval

```
φ = 1 − |Dct|                   φ ∈ [0, 1]
```

The width of the interval over which the certainty is meaningful. `φ = 0` at the
`⊤` and `⊥` corners — maximum contradiction leaves no usable certainty.

### Degree of real certainty

`Dc` alone overstates confidence when `Dct` is large. The *real* certainty
discounts it by the distance to the nearest ideal (non-contradictory) vertex:

```
d   = √( (1 − |Dc|)² + Dct² )

Dcr = 1 − d      if Dc > 0
      d − 1      if Dc < 0
      0          if Dc = 0
```

`Dcr ∈ [−1, 1]`. A proposition with high `Dc` but also high `Dct` gets pulled
back toward 0 — it *looks* decided but the underlying evidence is in conflict.

> **Note on the implementation.** `paraconsistent` 0.2.4 computes this as a
> normalised distance to the nearest apex:
> `d = min(‖(μ,λ) − (1,0)‖, ‖(μ,λ) − (0,1)‖) / √½`. The two forms are
> algebraically identical. Writing `a = 1 − μ`, `b = λ` for the `Dc ≥ 0` case:
>
> ```
> literature:  (1 − Dc)² + Dct² = (1 − μ + λ)² + (μ + λ − 1)² = (a+b)² + (b−a)² = 2a² + 2b²
> library:     ((1−μ)² + λ²) / ½ = 2(1−μ)² + 2λ²             = 2a² + 2b²
> ```
>
> Verified numerically to 4.4e-16 over 20 000 random points — see
> `tests/test_etau.py::test_distance_matches_literature_formula`.
>
> The library clamps the *normalised* `D = min(d, 1)` before computing
> `Dcr = (1 − D) · sign(Dc)`, so `Dcr` never leaves `[−1, 1]` even in the
> high-contradiction region where `d > 1`.

### Resulting evidence degrees

Each degree can be mapped back onto `[0, 1]` so it can be fed into another
block as a `μ`:

```
μE   = (Dc  + 1) / 2            resulting evidence from certainty
μECT = (Dct + 1) / 2            resulting evidence from contradiction
μER  = (Dcr + 1) / 2            resulting REAL evidence
```

`μER` is the normal output of a paraconsistent analysis node, and the normal
input to the next node in a network.

## 5. The twelve logical states

Two control limits partition the USCP. In `paraconsistent` 0.2.4 both derive
from a single parameter `FtC` (*Fator de Tolerância à Certeza*, the **Certainty
Control Limit**, default `0.50`):

```
VSSC  =  FtC          upper certainty control value
VICC  = −FtC          lower certainty control value
VSSCT =  1 − FtC      upper contradiction control value
VICCT =  FtC − 1      lower contradiction control value
```

Regions, in the order the classifier tests them:

### Extreme states (4)

| Condition | Label | Name |
|---|---|---|
| `Dc ≥ VSSC` | `t` | True |
| `Dc ≤ VICC` | `f` | False |
| `Dct ≥ VSSCT` | `┬` | Inconsistent |
| `Dct ≤ VICCT` | `┴` | Paracomplete / Indeterminate |

### Non-extreme states (8)

Inside the central square, whichever of `|Dc|` and `|Dct|` dominates names the
state, and the sign of the other gives the direction of the arrow:

| Label | Name |
|---|---|
| `Q┬→t` | Quasi-inconsistent tending to True |
| `Q┬→f` | Quasi-inconsistent tending to False |
| `Q┴→t` | Quasi-paracomplete tending to True |
| `Q┴→f` | Quasi-paracomplete tending to False |
| `Qt→┬` | Quasi-true tending to Inconsistent |
| `Qt→┴` | Quasi-true tending to Paracomplete |
| `Qf→┬` | Quasi-false tending to Inconsistent |
| `Qf→┴` | Quasi-false tending to Paracomplete |

### Plus the centre

The exact origin `Dc = Dct = 0` (i.e. `μ = λ = 0.5`) is labelled `I` —
*indefinite*. It is not one of the classical twelve; it is the degenerate point
where no region has a claim.

> **Character encoding.** The library uses the box-drawing characters
> `┬` (U+252C) and `┴` (U+2534), **not** the mathematical `⊤` (U+22A4) and
> `⊥` (U+22A5) used in most papers. String comparisons against labels must use
> the box-drawing forms.

> **Tie-breaking on the axes.** The classifier tests `dc >= 0` and `dct >= 0`,
> so a value of exactly zero counts as positive:
>
> - on `Dc = 0` (i.e. `μ = λ`) the state resolves to the *true* side —
>   `p(0.6, 0.6)` is `Q┬→t`, never `Q┬→f`;
> - on `Dct = 0` (i.e. `μ + λ = 1`, the perfectly non-contradictory diagonal)
>   the state resolves to the *inconsistent* side — a point with `Dc = 0.44`,
>   `Dct = 0` is labelled `Qt→┬`.
>
> Neither is wrong so much as arbitrary: on those lines the two candidate
> labels are equally (un)justified. It does mean the state labels are not quite
> symmetric under negation along `μ = λ`, which is unavoidable there since such
> annotations are their own negation. Both behaviours are pinned in
> `tests/test_etau.py` so an upstream change would be caught.

## 6. Operators

Eτ's connectives act on the annotation pair. The implementation in
`paraconsistent` 0.2.4 does **not** provide them — it implements the
para-analyser only — but they are one-liners:

### Negation

```
¬p(μ, λ)  =  p(λ, μ)
```

Negation swaps the evidences. Consequences: it flips the sign of `Dc`
(`λ − μ = −Dc`) and **preserves** `Dct` (`λ + μ − 1 = Dct`). Negating a
proposition does not change how contradictory its evidence is — which is
exactly right, and is checked in `tests/test_etau.py::test_negation`.

Note that `⊤` and `⊥` are fixed points: `¬p(1,1) = p(1,1)`, `¬p(0,0) = p(0,0)`.

### Conjunction and disjunction

```
p(μ₁, λ₁) ∧ p(μ₂, λ₂)  =  p( min(μ₁, μ₂),  max(λ₁, λ₂) )
p(μ₁, λ₁) ∨ p(μ₂, λ₂)  =  p( max(μ₁, μ₂),  min(λ₁, λ₂) )
```

Favourable evidence combines pessimistically, contrary evidence optimistically
— the lattice meet and join.

## 7. Paraconsistent analysis networks

The reason `μER` is normalised back to `[0, 1]` is composition. A
**paraconsistent analysis node** (PAN) takes `(μ, λ)`, runs the analysis, and
emits `μER`; that output becomes an input evidence of the next node. Chaining
nodes gives a **PANnet**, and a PANnet with learning gives a **paraconsistent
artificial neural network** (PANN).

A common wiring, and the one shown in `examples/etau_demo.py`:

```python
b2.input.mu  = b1.complete.muER      # real evidence forward
b2.input.lam = 1 - b1.complete.phi   # contradiction becomes contrary evidence
```

The second line is the interesting one: the *first* node's contradiction is
injected as the *second* node's contrary evidence, so conflict propagates
through the network as diminished support instead of being silently discarded.

Named algorithms built on this substrate include the **Para-analyser**,
**ParaExtrCTX** (contradiction extractor), the **PAL2v Filter**, **PANnet** and
**PNN**.

## 8. Relationship to other uncertainty formalisms

| | Represents | Contradiction | Ignorance |
|---|---|---|---|
| Probability | one number, `P(¬p) = 1 − P(p)` | cannot express | conflated with `P = 0.5` |
| Fuzzy sets | one membership `μ` | cannot express | conflated with `μ = 0.5` |
| Dempster–Shafer | mass over subsets | conflict `K`, but Dempster's rule can behave counter-intuitively at high conflict | `m(Θ)` |
| Intuitionistic fuzzy | `(μ, ν)` with `μ + ν ≤ 1` | **excluded by the constraint** | `1 − μ − ν` |
| **Eτ / PAL2v** | `(μ, λ)`, unconstrained | `Dct > 0`, measured explicitly | `Dct < 0`, measured explicitly |

The dropped constraint `μ + λ ≤ 1` is precisely what separates Eτ from
intuitionistic fuzzy sets, and it is what lets the upper half of the square
represent genuine contradiction rather than being an error state.

Practically, Eτ's appeal in engineering is its cost: the whole analysis is a
handful of additions, a subtraction, a square root and some comparisons — no
matrix algebra, no iteration, no training. That makes it viable on
microcontrollers, in edge computing and in TinyML settings where Bayesian or
Dempster–Shafer fusion would not fit.

## 9. Applications reported in the literature

- **Robotics** — the Emmy family of autonomous mobile robots; obstacle
  avoidance and corridor navigation for terrestrial mobile robots; autonomous
  robots in logistics centres; learning-from-demonstration for linear Cartesian
  robots.
- **Networks** — determination of operating parameters and performance analysis
  of computer networks; management and control of communication-network routes.
- **Security** — web application firewall enhancement (*Mogura Taiji Eτ*);
  optimising data-loss-prevention levels.
- **Medicine** — diagnostic support and analysis of clinical evidence.
- **Signal and pattern analysis** — characterisation and classification of
  numerical data patterns; statistical process control.

## 10. Bibliography

Sorted roughly by usefulness as an entry point. None of these could be
downloaded in this environment (see the note in `README.md`); DOIs and URLs are
given for retrieval elsewhere.

### Reviews and surveys — start here

1. de Carvalho Junior, A. *et al.* **"A comprehensive review on paraconsistent
   annotated evidential logic: Algorithms, Applications, and Perspectives."**
   *Engineering Applications of Artificial Intelligence* **127** Part B (2024).
   DOI [10.1016/j.engappai.2023.107342](https://doi.org/10.1016/j.engappai.2023.107342)

2. Abe, J. M.; Nakamatsu, K.; da Silva Filho, J. I. **"Three decades of
   paraconsistent annotated logics: a review paper on some applications."**
   *Procedia Computer Science* **159** (2019), 1175–1181.
   DOI [10.1016/j.procs.2019.09.283](https://doi.org/10.1016/j.procs.2019.09.283)

### Foundations

3. da Silva Filho, J. I.; Abe, J. M.; Lambert-Torres, G. **"Paraconsistent
   Annotated Evidential Logic Eτ and Applications in Automation and Robotics."**
   In *The Handbook on Reasoning-Based Intelligent Systems*, World Scientific
   (2013), ch. 13.
   DOI [10.1142/9789814329484_0013](https://doi.org/10.1142/9789814329484_0013)

4. **"Paraconsistent Annotated Evidential Logic Eτ."** Springer chapter (2018).
   DOI [10.1007/978-3-319-74110-9_2](https://doi.org/10.1007/978-3-319-74110-9_2)

5. da Costa, N. C. A. **"On the theory of inconsistent formal systems."**
   *Notre Dame Journal of Formal Logic* **15**(4) (1974), 497–510. — the origin
   of the `C_n` paraconsistent calculi.

6. Subrahmanian, V. S. **"On the semantics of quantitative logic programs."**
   *Proc. 4th IEEE Symposium on Logic Programming* (1987). — the origin of
   annotated logics.

7. da Silva Filho, J. I. **"Métodos de aplicações da Lógica Paraconsistente
   Anotada de anotação com dois valores LPA2v com construção de algoritmo e
   implementação de circuitos eletrônicos."** Doctoral thesis, Universidade de
   São Paulo (1999).

### The implementation vendored here

8. de Carvalho Junior, A.; da Cruz, D. O.; Alves, B. da S.; da Silva Paulo
   Junior, F.; da Silva Filho, J. I. **"Paraconsistent-Lib: an intuitive PAL2v
   algorithm Python Library."** arXiv:2511.20700 (24 Nov 2025).
   <https://arxiv.org/abs/2511.20700>

9. Paraconsistent-Lib documentation.
   <https://eailab-ifsp.github.io/Paraconsistent-Lib/>

### Applications

10. **"Analysis of the displacement of terrestrial mobile robots in corridors
    using paraconsistent annotated evidential logic Eτ."** arXiv:2009.14192.
    <https://arxiv.org/abs/2009.14192>

11. **"Application of Paraconsistent Annotated Evidential Logic Eτ for a
    Terrestrial Mobile Robot to Avoid Obstacles."** *Procedia Computer Science*
    (2021). DOI [10.1016/j.procs.2021.08.077](https://doi.org/10.1016/j.procs.2021.08.077)

12. **"Paraconsistent Annotated Evidential Logic Eτ Applied to Autonomous
    Robots in Logistic Center."** Springer (2021).
    DOI [10.1007/978-3-030-85906-0_58](https://doi.org/10.1007/978-3-030-85906-0_58)

13. **"Determination of Operating Parameters and Performance Analysis of
    Computer Networks with Paraconsistent Annotated Evidential Logic Eτ."**
    Springer (2017). DOI [10.1007/978-3-319-51133-7_1](https://doi.org/10.1007/978-3-319-51133-7_1)

14. **"Paraconsistent Annotated Logic Algorithms Applied in Management and
    Control of Communication Network Routes."** *Sensors* (2021).
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC8234040/>

15. **"Optimizing the Data Loss Prevention Level Using Logic Paraconsistent
    Annotated Evidential Eτ."** Springer (2023).
    DOI [10.1007/978-3-031-35759-6_9](https://doi.org/10.1007/978-3-031-35759-6_9)

16. **"Mogura Taiji Eτ — A Paraconsistent Annotated Evidential Logic Eτ
    approach to enhance WAF."** *Procedia Computer Science* (2025).
    <https://www.sciencedirect.com/science/article/pii/S1877050925031084>

17. **"Process of Learning from Demonstration with Paraconsistent Artificial
    Neural Cells for Application in Linear Cartesian Robots."** *Robotics*
    **12**(3):69 (2023). DOI [10.3390/robotics12030069](https://doi.org/10.3390/robotics12030069)
