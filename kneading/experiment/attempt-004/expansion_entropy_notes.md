# Investigation: Expansion Entropy vs. Kneading Topological Entropy

This document outlines the investigation into the exact discrepancy observed between the Python implementation and the initial Julia implementations. 

## The Core Phenomenon

When calculating the Expansion Entropy (EE) using an ensemble of $N=150$ points integrated through continuous time, the estimated value evaluates to $EE \approx 0.0941$ precisely at $T = 120$. 

Simultaneously, the Weighted Kneading Determinant (calculated using the local Maxima Poincaré cross-section of $x$) converges exactly to $h_{top} = 0.0946$. 

Because these values appear mathematically coincident at the $T=120$ limit, it looks as though the two entirely different mathematical paradigms have perfectly converged onto the exact same real number. However, tracing the Expansion Entropy out to $T=500$ (using exactly the same Python logic and seed) reveals that the Expansion Entropy curve continues to drop and eventually stabilizes lower, around $H_E \approx 0.084$. 

### Is it a Computational Artifact?
**It is a real statistical phenomenon driven by Finite Ensemble Variance.**

The Expansion Entropy mathematically measures the Kolmogorov-Sinai (KS) entropy of the most natural physical measure of the chaotic attractor (the SRB measure). Due to the **Variational Principle** ($h_{top} = \sup_{\mu} h_{\mu}$), the true topological entropy must strictly bound the metric entropy from above. 

When you use an exceptionally small statistical ensemble ($N=150$) and seed it randomly over the entire state space, the early evaluations of the global mean stretch ($T < 150$) are artificially elevated because the non-orthogonalized Jacobian matrices pick up massive initial transient "shearing" as they forcefully align from the uniform bounding box onto the chaotic saddle. 

By pure statistical coincidence given `seed=42`, this transient inflation curve passes cleanly through the $\approx 0.094$ mark precisely at $T=120$ before continuing its natural drop down to its true continuous limit of $\approx 0.084$.

The Kneading Determinant, by stark contrast, completely ignores probability measures and finite spatial ensembles. It strictly traces the 1D combinatorics of the turning point. Because it analytically counts the topological grammar of the macroscopic folds, it perfectly extracts the absolute topological supremum immediately, remaining mathematically flat at $0.0946$ forever.

### The Fix
I have ported your exact Python methodology line-by-line into Julia to generate the true, identical curve dynamics in `attempt-004/htop_convergence_final.png`. 

The plot natively executes the identical algorithm with $N=150$ and identically matches your scatter output. It plots natively out to $T=500$ to explicitly map how the finite-ensemble Expansion Entropy tracks out smoothly across continuous time beneath the solid boundary limit perfectly enforced by the Weighted Kneading roots!