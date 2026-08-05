#!/usr/bin/env python3
"""Demonstration of Paraconsistent Annotated Evidential Logic Eτ (PAL2v).

Replaces the example shipped inside the package
(`paraconsistent/examples/demo_wiring.py`), which does not run: it imports
`from mrn.paraconsistent.block import ...` — a module path from before the
package was renamed — and reads `config.FL` and `complete.gc`, neither of
which exists.

Run:  python examples/etau_demo.py
"""

from paraconsistent import ParaconsistentBlock


def rule(title):
    print(f"\n{title}\n{'─' * len(title)}")


def analyse(mu, lam, **params):
    return ParaconsistentBlock(mu=mu, lam=lam, **params).complete


def main():
    # ------------------------------------------------------------------
    rule("1. The four extreme states, and why ⊤ is not fatal")

    corners = [
        (1.0, 0.0, "fully supported, not refuted"),
        (0.0, 1.0, "fully refuted, not supported"),
        (1.0, 1.0, "fully supported AND fully refuted"),
        (0.0, 0.0, "neither supported nor refuted"),
    ]
    print(f"{'p(μ, λ)':>10}  {'Dc':>5} {'Dct':>5} {'φ':>5}  {'state':<6} interpretation")
    for mu, lam, note in corners:
        c = analyse(mu, lam)
        print(f"  p({mu:.0f}, {lam:.0f})  {c.dc:>5.2f} {c.dct:>5.2f} {c.phi:>5.2f}"
              f"  {c.label:<6} {note}")

    print("\n  p(1,1) is classified as inconsistent and carries φ = 0 — no usable")
    print("  certainty — but the analysis completes and returns a value. Classical")
    print("  logic would derive every proposition in the language instead.")

    # ------------------------------------------------------------------
    rule("2. Certainty vs. real certainty: contradiction discounts confidence")

    print("  Two propositions with the SAME degree of certainty Dc = 0.20:\n")
    print(f"{'p(μ, λ)':>12}  {'Dc':>5} {'Dct':>6} {'Dcr':>6} {'μER':>6}  state")
    for mu, lam in [(0.60, 0.40), (0.90, 0.70)]:
        c = analyse(mu, lam)
        print(f"  p({mu:.2f}, {lam:.2f})  {c.dc:>5.2f} {c.dct:>6.2f}"
              f" {c.dcr:>6.2f} {c.muER:>6.2f}  {c.label}")

    print("\n  Identical Dc, but the second draws on sources that flatly contradict")
    print("  each other (Dct = 0.60). Its real certainty Dcr collapses to 0.00 and")
    print("  the resulting real evidence μER falls back to the neutral 0.50.")

    # ------------------------------------------------------------------
    rule("3. Sweeping the lattice — the 12 logical states")

    seen = {}
    steps = 41
    for i in range(steps):
        for j in range(steps):
            mu, lam = i / (steps - 1), j / (steps - 1)
            seen.setdefault(analyse(mu, lam).label, (mu, lam))

    order = ["t", "f", "┬", "┴",
             "Q┬→t", "Q┬→f", "Q┴→t", "Q┴→f",
             "Qt→┬", "Qt→┴", "Qf→┬", "Qf→┴", "I"]
    names = {
        "t": "True", "f": "False",
        "┬": "Inconsistent", "┴": "Paracomplete",
        "Q┬→t": "quasi-inconsistent → true", "Q┬→f": "quasi-inconsistent → false",
        "Q┴→t": "quasi-paracomplete → true", "Q┴→f": "quasi-paracomplete → false",
        "Qt→┬": "quasi-true → inconsistent", "Qt→┴": "quasi-true → paracomplete",
        "Qf→┬": "quasi-false → inconsistent", "Qf→┴": "quasi-false → paracomplete",
        "I": "indefinite (exact centre)",
    }
    for label in order:
        if label in seen:
            mu, lam = seen[label]
            print(f"  {label:<5} {names[label]:<28} e.g. p({mu:.3f}, {lam:.3f})")

    # ------------------------------------------------------------------
    rule("4. Negation: ¬p(μ, λ) = p(λ, μ)")

    mu, lam = 0.85, 0.30
    p, np_ = analyse(mu, lam), analyse(lam, mu)
    print(f"   p({mu}, {lam})   Dc = {p.dc:+.2f}   Dct = {p.dct:+.2f}   state {p.label}")
    print(f"  ¬p({mu}, {lam})   Dc = {np_.dc:+.2f}   Dct = {np_.dct:+.2f}   state {np_.label}")
    print("\n  Negation flips the sign of the certainty but leaves the contradiction")
    print("  untouched — how conflicted the evidence is does not depend on which")
    print("  side of the proposition you are asking about.")

    # ------------------------------------------------------------------
    rule("5. A paraconsistent analysis network (PANnet)")

    print("  Proposition under test: \"there is an obstacle ahead\".")
    print("  Two sensors are analysed separately, then fused by a third node.")
    print("  The fusing node takes the supporting sensor's real evidence as μ and")
    print("  the refuting sensor's as λ, so a disagreement between them becomes a")
    print("  measured contradiction instead of being averaged away.\n")

    scenarios = [
        ("sensors agree",    (0.90, 0.15), (0.20, 0.85)),
        ("sensors conflict", (0.90, 0.15), (0.75, 0.30)),
    ]

    for title, (mu_a, lam_a), (mu_b, lam_b) in scenarios:
        supports = ParaconsistentBlock(mu=mu_a, lam=lam_a)   # evidence FOR
        refutes = ParaconsistentBlock(mu=mu_b, lam=lam_b)    # evidence AGAINST

        fused = ParaconsistentBlock(FtC=0.60)
        fused.input.mu = supports.complete.muER
        fused.input.lam = refutes.complete.muER

        print(f"  {title}")
        for name, b in (("  radar  (for)    ", supports),
                        ("  vision (against)", refutes)):
            c = b.complete
            print(f"  {name}  p({c.mu:.2f}, {c.lam:.2f})  ->  μER={c.muER:.3f}"
                  f"  {c.label}")

        f = fused.complete
        print(f"    fused             p({f.mu:.3f}, {f.lam:.3f})  ->  "
              f"Dc={f.dc:+.3f}  Dct={f.dct:+.3f}  φ={f.phi:.2f}  {f.label}")

        if f.label in ("┬", "┴"):
            print(f"    -> state is {f.label!r}: the inputs are irreconcilable. "
                  f"Read the state,\n       not decision_output "
                  f"({f.decision_output}), which cannot express this.")
        else:
            verdict = {1.0: "act — treated as true",
                       0.0: "do not act — treated as false",
                       0.5: "undecided"}[f.decision_output]
            print(f"    -> decision_output = {f.decision_output}: {verdict}")
        print()

    print("  This is the property that motivates Eτ. Averaging the two sensors")
    print("  would return a middling number indistinguishable from 'both sensors")
    print("  are unsure'. Carrying μ and λ separately keeps 'they disagree' and")
    print("  'neither knows' as different, inspectable outcomes — Dct > 0 versus")
    print("  Dct < 0.")

    # ------------------------------------------------------------------
    rule("6. The certainty control limit FtC moves the decision boundary")

    mu, lam = 0.72, 0.28
    print(f"  Same annotation p({mu}, {lam}) — μER = {analyse(mu, lam).muER:.3f}\n")
    print(f"  {'FtC':>6}  {'state':<6} {'decision':>9}")
    for ftc in (0.30, 0.50, 0.70, 0.90):
        c = analyse(mu, lam, FtC=ftc)
        print(f"  {ftc:>6.2f}  {c.label:<6} {c.decision_output:>9.1f}")

    print("\n  Raising FtC demands more real evidence before the proposition is")
    print("  accepted, and widens the band in which it is merely 'quasi-true'.")


if __name__ == "__main__":
    main()
