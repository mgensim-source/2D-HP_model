## Evolutionary Algorithms : 
Family of optimisation and search tecniques inspired by biological evolution - they iteratively evolve a population of candidate solutiosn toward high fitness using operators that mimic natural selection, recombination and mutation.

Optimisation Problems - There's always an objetive function that you want to optimise ie. find optimal solution from the feasible solution space [solution that satifies all the constraints for input variables.]

https://www.youtube.com/watch?v=19s8THlAhB8

> OPtimisation problems can be linear vs non-linear and input variables can be continous vs discrete (combinatorial) - knapsack problem; travellinfg salesmap problem, graph optimisation etc
> Solution MEthods
    > Analytical
    > Numerical (Gradient Descent)
    > (MEta)heuristic, greedy approaches, Evolutionary algorithms


Numerical(Gradient Descent):
- Minimization in machine learning most of the time the objective function is a quadratic function (mean square of the diff between current and target solution)
- This give rise to parabolic like function in 3d space and its very easy to find the minimum (a)most of the cases there's only one minimum - global minimum. (b)you can use gradient descent or it's variants to find minimum
- Gradient descent - ()select start point ()move it in gradient direction - travel along the path where function value reduces fastest with respect to current location.
- Gradent descent is very subjective to the starting point so if there can many local minima it can get trapped in one of it and might not reach globla minimum.
- Gradient descent work for conitnious spaces so discrete, conbinatorial it doesn't work directly - you will have to convert categorical to numerical like one hot encoding etc and then use/apply gradient descent.....


Evolutionary Algorithms:
It also needs epistatic separation or low epistasis, meaning genes do not interfere too much with each other. Key Mathematical PropertiesContinuity: Small shifts in input variables produce small changes in the fitness function.Low Epistasis: Independent traits or genes can be optimized separately without ruining other parts of the solution.

Components of Evolutionary Algorithms:
* Individuals, Genomics - potential solution in search space
* Population - group of potential solution
* Fitness Function - function to decide which ones are good slolutions
* Selection - way to select the most fit ones among all solutions
* Genetic Operators
    Mutations - abrupt, slight change in the potential solutions
    Crossover(Recombination) - like child from parent; two potential solutions combining to form the new one.
* Generation - next generation of possible/potential solution through selection, Mutation, Crossover etc.
* Termination Criteria.

* Niching Methods (this is set of methods to maintain diversity of potential solutions to avoid premature convergence).
* Elitism (keep the best individuals from current generation)

Genetic Algorithm Process:
Information Encoding: The information (like: DNA string, vertex of graphs, set of kanpsack) needs to be encoded to be fed in the genetic algorithm.
    - Three methods: [some way to encode the input information]
        (a) Binary encoding
        (b) Integer encoding: [ACGT]->[0,1,2,3]
        (c) Real number encoding
    - Input Data structure
        (a) List(vector)
        (b) Set (knapsack problem)
        (c) Graph
(a) Generate the random population (set of potential solutiosn in search space) It can be random or predefined prior.
(b) Compute fitness function values for each of those generated solutions.
(c) A Selection operator selects the best k intermedium solutions.
(d) Perform crossover and mutations to get new set of solutions.
    (d1.) Crossover - create off springs
        (d1.1) methods 1-point [choose one point and swap between parts], Multi-point, Uniform crossover
        (d.1.2) diversity is they key here.
    (d.2) Mutation - induce randomness to avoid stucking, change a bit
        (d.2.1) bit flip in binary encofing
        (d.2.2) Random tweaking in real number encoding like: addind guassian noise to the real number 
        (d.2.3) swapping in integer or permutation encoding.
        This is to balance the tradeoff between overfittign and underfitting.
(e) you can either append/add this new mutated solutions to the previous one or replace them completely.
    (e.1) Survivor selection is about choosing whether to keep parents or not (mu, lambda)
        (mu, lambda )stratergy = lambda offspring from mu parents, select among lambda offspring onlu
        (mu + lambda) = consider both parents and offspiring while choosinfg next generation.
(f) repeate the process.


## How it works

Encoding. Each candidate fold is a list of n-1 moves (n = sequence length), one of 6 lattice directions {+x,-x,+y,-y,+z,-z} per move. Immediate reversals are disallowed everywhere moves are generated (construction, crossover repair, mutation) since a reversal always causes an instant self-collision — this prunes the search space from 6^(n-1) down to roughly 5^(n-2).

1. fitness = (# non-consecutive H-H contacts) − COLLISION_PENALTY × (# self-overlaps)
2. Collisions are penalized rather than rejected outright, so the GA can still explore near-miss folds instead of being hard-blocked.
3. Evolution loop, each generation:
  
    1. Score every individual in the population.
  
    2. Carry the top n_elite individuals forward unchanged (elitism).
  
    3. Fill the rest of the next generation by repeatedly:
     
        1. picking two parents via tournament selection, recombining them with single-point crossover (+ repair for any reversal introduced at the splice point),
  
        2. mutating the children,
  
        3. adding them to the new population.
  
        4. Repeat for GENERATIONS generations, tracking the best fold seen so far.
