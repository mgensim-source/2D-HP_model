# Ant Colony Optimization (ACO), Explained

**Metaheuristic that models real ant foraging behavior for Protein Folding problems.**

Two well-known approaches that can be applied, and how ACO addresses the bottleneck of each:

**(a) Greedy Algorithm** -  Makes the **single best local choice** at each exact step without looking ahead or ever changing its mind. It is fast, but it often gets trapped in **poor, sub-optimal end results (local minima).**

**(b) Heuristic-Based** - A **rule of thumb or educated guess** used to quickly find a decent solution for a complex problem (like choosing the shortest visible path next).

**Ant Colony Optimization**, on the other hand, tries to account for **local best heuristics as well as previous history** through pheromone trails to explore many paths and iteratively improve them over time.

> A simple idea is for every path the ant explores it deposits pheromones on the way, over the time the best optimal paths will have more pheromones deposits leading to selection of those paths by the ants.

- Hence, there's a deposition function (addition of new pheromones) and an evaporation function (reduction of deposited pheromones) - the evaporation part is important to ensure both exploitation and explorations of the paths at the same time.
- In our case, i've considered deposition of pheromone as a function of Q*abs(E) where E is the Energy which is governed by the number of non-contact H-H in the folded protein. Whereas, evaporation is (1 - rho)*prev_pheromone

**Broadly**, the algorithm can be written in the following steps:
- [1.] For every H/P added in the folding grid[one of the two valid position/locations], identify the valid move out of four which is based on the probabilities.
    - [1.a] For every one of the four position solve for probabilities
      tau = pheromone[residue_index][direction] #The pheromone associated with choosing direction when placing residue_index.
      new_contacts = count_new_hh_contacts(
      eta = 1.0 + new_contacts
      value = (tau ** alpha) * (eta ** beta)
      values.append((direction, new_pos, value))
      total = sum(value for _, _, value in values)
      probabilities.append(
            (direction, pos, value / total)
        )
   - [1.b] The move out of four is not chosen with one that has more probability, it is chosen randomly but weighted by the probabilities ie. more the probability more likely to be choosen.
   - [1.c] continue this for num_iterations, after every iterations change the pheromone for the sequence, evaporate the pheromone and energy....
