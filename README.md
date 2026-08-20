# HP_model
Implementing the HP lattice protein folding model to evaluate and compare advanced discrete optimization algorithms.

## Motivation

Protein folding is fundamentally an optimization problem: given an amino acid sequence, find a low-energy three-dimensional conformation among an enormous number of possible structures.

This project uses the simplified **Hydrophobic–Polar (HP) lattice protein model** as a small, interpretable environment for studying this problem.

The goal is a **starting project to understand the computational principles behind protein folding** and how different optimization strategies explore a difficult discrete search space.

The project will be used to experiment with and compare methods such as:

* Genetic Algorithms
* Ant Colony Optimization
* Monte Carlo methods
* Other evolutionary and metaheuristic optimization techniques

By implementing these methods on the same HP folding problem, we can study how different search strategies handle:

* Large combinatorial search spaces
* Local optima
* Exploration vs. exploitation
* Population diversity
* Fitness landscapes
* Constraints and self-collisions
* Convergence behavior

---

## Why the HP Model?

Real protein folding involves complex physical interactions between many different atoms and molecules. Directly modelling these interactions is computationally expensive and introduces significant complexity.

The HP model provides a much simpler abstraction.

Each amino acid is represented as either:

* **H** — Hydrophobic
* **P** — Polar

The protein is folded on a lattice, and hydrophobic residues receive favorable interactions when they form non-consecutive contacts.

This gives us a simple but meaningful optimization problem where we can focus on the **search algorithm rather than the complexity of molecular physics**.

---

## Problem Formulation

Given a protein sequence:

```text
HHPHHPHPH
```

the objective is to find a valid lattice conformation that maximizes favorable hydrophobic contacts while avoiding self-overlapping structures.

A candidate fold is represented as a sequence of lattice moves.

For a protein of length `n`, there are `n-1` moves, with each move corresponding to one of the six lattice directions:

```text
+x, -x, +y, -y, +z, -z
```

This creates a discrete combinatorial search problem.

The theoretical search space grows exponentially with sequence length, making exhaustive enumeration impractical even for relatively small proteins.
