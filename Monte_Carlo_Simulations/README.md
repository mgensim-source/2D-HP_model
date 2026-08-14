# Monte Carlo Simulation, Explained

[Monte Carlo Simulation](https://www.youtube.com/watch?v=WjmNedsX1T0)

Applied to the space of problems where the space of possible outcomes is vast and continuous.

Eg: Simulation of particle scattering when they collide - quantum randomness determining every interaction, 100 dimension integral where traditional grid-based approaches [divide continuous space into grids and find the value of function at each grid point] fails because it requires evaluating grid function at every single point which is more than atoms in observable universe.

Monte Carlo idea avoids enumeration on all possibilities, it instead tries to find the expected value of the function as the law of large numbers.

It works by:

- Randomly sample possibilities
- Compute the outcomes of each sample
- Average them

It implements law of large numbers to guarantee convergence - with large number of possible samples the value is closer to the expected value/the average.

---

## Expected Value

E(X) = Σ (x · P(x)) if discrete else integration not summation, where p(x) is the probability of the x.

The expected value of a single roll of a fair six-sided die is 3.5.

Single 6-sided die (d6):

E(X) = (1/6 + 2/6 + 3/6 + 4/6 + 5/6 + 6/6) = 3.5

---

## Monte Carlo Error Rate

The Monte Carlo method error rate is written using Big-O notation as:

O(1/√N)

This means that the statistical error of a Monte Carlo simulation shrinks in proportion to the inverse square root of the number of random samples (N) you generate.

### Error Relationship

```text
Error ∝ 1 / sqrt(N)

Why Does the Error Scale as 1/√N?

This relationship follows from statistical properties described by the Central Limit Theorem (CLT).

1. Variance of the Mean

For N independent samples, the variance of the sample mean decreases as:

Var(X̄) = σ²/N

where σ² is the population variance.

2. Standard Error

Statistical error is measured by the standard deviation of the mean:

SE = √(σ²/N)

Therefore:

SE = σ/√N

3. The Result

Since σ is a constant for a given problem:

Error ∝ 1/√N
```

# Implementation
Note: This Runs several independent annealing chains, each starting from `starting_coords` (or a fresh straight chain if not given), and
keeps the best fold found across all of them.

A single Metropolis chain can get trapped near a local-minimum energy and never escape once temperature has cooled, so for landscapes with
a narrow global optimum (as HP folding often has), multiple independent restarts find the true minimum far more reliably than one long run does. Only the history from the run that produced the global best is kept (for plotting); other runs' histories are
discarded to save memory.

(a) Define an Initial fold - which is straight line in our case.

(b) Generate random fold - this is done using def generate_random_fold(coords) which randomly selects the index of the sequence and applies transformations as defined below to the entire half chain
    after the index identified.
    
    _TRANSFORMATIONS = [
    lambda x, y: (-y, x),   # rotate 90
    lambda x, y: (-x, -y),  # rotate 180
    lambda x, y: (y, -x),   # rotate 270
    lambda x, y: (x, -y),   # reflect across x-axis
    lambda x, y: (-x, y),   # reflect across y-axis
    lambda x, y: (y, x),    # reflect across main diagonal
    lambda x, y: (-y, -x),  # reflect across anti-diagonal
]
  Handel corner cases where transformation lies on the same grid or overlaps so discard that perturbations.

(c) Calculate the Energy of the new folded sequence

(d) Call metropolis_accept func to decide if the newly folded sequence should be kept or discarded! -  Metropolis criterion: accept a move if it lowers energy, or with
    probability. 
    
    exp(-delta_energy / temperature) if it raises energy. 
    
    If accepted it is added to the protein.update(candidate_coords) and accepted moves.

Continue till Defined Folding steps.....

note: The annealing helps the algorithms by starting at a high "temperature" that accepts worse moves and slowly cooling down, 
it lets the model explore freely before settling into the best answer.
