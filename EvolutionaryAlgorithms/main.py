"""
Self-avoiding walk on a lattice that maximizes the number
of *topological* (non-consecutive) H-H contacts -- a proxy for the
hydrophobic core packing that drives real protein folding.

This script folds a given HP sequence on the 3D simple-cubic lattice
using a genetic algorithm (GA).

Encoding
--------
An individual is a list of n-1 moves (n = sequence length), each move
drawn from the 6 lattice directions {+x,-x,+y,-y,+z,-z}. Immediate
reversals (which would make the chain double back on itself) are
disallowed at construction/mutation/repair time -- this alone removes
a large share of guaranteed self-overlaps and shrinks the search space
from 6^(n-1) to roughly 5^(n-2).

Fitness
-------
    fitness = (#H-H contacts) - COLLISION_PENALTY * (#self-overlaps)

"""
import math
import random
from config import SEQUENCE, POPULATION_SIZE, GENERATIONS, MUTATION_RATE, SEED, COLLISION_PENALTY
from plots import plot_fold

DIRS = {
    0: (1, 0, 0),
    1: (-1, 0, 0),
    2: (0, 1, 0),
    3: (0, -1, 0),
    4: (0, 0, 1),
    5: (0, 0, -1),
}
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
ALL_DIRS = list(DIRS.keys())


def random_move(prev=None):
    """Pick a random direction, excluding the immediate reverse of `prev`.

        If residue i+1 moved, say, +x from residue i, then residue i+2 moving -x 
        would land it right back on top of residue i. That's an immediate self-collision
    
    """
    if prev is None:
        return random.choice(ALL_DIRS)
    choices = [d for d in ALL_DIRS if d != OPPOSITE[prev]]
    return random.choice(choices)


# ---------------------------------------------------------------------------
# Decoding + fitness
# ---------------------------------------------------------------------------
def decode(moves):
    """Turn a move list into 3D coordinates; count self-overlaps."""
    coords = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    collisions = 0
    for m in moves:
        dx, dy, dz = DIRS[m]
        x, y, z = coords[-1]
        nxt = (x + dx, y + dy, z + dz)
        if nxt in occupied:
            collisions += 1
        occupied.add(nxt)
        coords.append(nxt)
    return coords, collisions


def hh_contacts(coords, seq):
    """Count non-consecutive H residues that are lattice-adjacent."""
    n = len(seq)
    pos_index = {}
    for i, c in enumerate(coords):
        pos_index.setdefault(c, []).append(i)

    contacts = 0
    seen = set()
    for i in range(n):
        if seq[i] != "H":
            continue
        x, y, z = coords[i]
        for dx, dy, dz in DIRS.values():
            for j in pos_index.get((x + dx, y + dy, z + dz), []):
                if seq[j] == "H" and abs(i - j) > 1:
                    key = (i, j) if i < j else (j, i)
                    if key not in seen:
                        seen.add(key)
                        contacts += 1
    return contacts


def fitness(moves, seq):
    coords, collisions = decode(moves)
    contacts = hh_contacts(coords, seq)
    score = contacts - COLLISION_PENALTY * collisions #since the random generation might lead to self overlaps, we penalize the score by COLLISION_PENALTY * collisions
    return score, collisions, contacts

def random_individual(n):
    moves, prev = [], None
    for _ in range(n - 1):
        m = random_move(prev)
        moves.append(m)
        prev = m
    return moves


def repair(moves):
    """Re-roll any move that reverses the previous one (post-crossover)."""
    fixed, prev = [], None
    for m in moves:
        if prev is not None and m == OPPOSITE[prev]:
            m = random_move(prev)
        fixed.append(m)
        prev = m
    return fixed


def tournament_select(pop, scores, k=3):
    best_idx, best_score = None, -math.inf
    for _ in range(k):
        i = random.randrange(len(pop))
        if scores[i] > best_score:
            best_idx, best_score = i, scores[i]
    return pop[best_idx]


def crossover(p1, p2):
    if len(p1) < 2:
        return p1[:], p2[:]
    pt = random.randint(1, len(p1) - 1)
    c1 = repair(p1[:pt] + p2[pt:])
    c2 = repair(p2[:pt] + p1[pt:])
    return c1, c2


def mutate(moves, rate):
    out, prev = [], None
    for m in moves:
        if random.random() < rate:
            m = random_move(prev)
        elif prev is not None and m == OPPOSITE[prev]:
            m = random_move(prev)
        out.append(m)
        prev = m
    return out


def run_ga(seq, pop_size=300, generations=400, mutation_rate=0.08,
           elite_frac=0.05, tournament_k=3, seed=None, verbose=True):
    if seed is not None:
        random.seed(seed)

    n = len(seq)
    pop = [random_individual(n) for _ in range(pop_size)]  # Random_individual(n) generates one random candidate fold: a list of n-1 random moves (using the no-immediate-reversal rule from before) that encodes some arbitrary, unoptimized way of laying the protein chain out on the 3D lattice. for _ in range(pop_size) repeats that pop_size times (default 400) ie. complete population of random folds.
    n_elite = max(1, int(pop_size * elite_frac)) #"elitism" in GA algorithm - how many of the best individuals get carried over unchanged to the next generation

    best_moves, best_score = None, -math.inf
    history = []
    for gen in range(generations):
        scored = [fitness(ind, seq) for ind in pop] #genetate fitness scores for each individual in the population
        scores = [s[0] for s in scored]
        order = sorted(range(pop_size), key=lambda i: scores[i], reverse=True) #sort the population by fitness score, highest first, and return the indices of the sorted order

        if scores[order[0]] > best_score:
            best_score = scores[order[0]]
            best_moves = pop[order[0]][:]
        history.append(best_score)

        if verbose and (gen % 25 == 0 or gen == generations - 1):
            _, col, con = scored[order[0]]
            print(f"gen {gen:4d} | best fitness {scores[order[0]]:6.1f} "
                  f"| H-H contacts {con:3d} | collisions {col}")

        new_pop = [pop[i][:] for i in order[:n_elite]] #carry over the best n_elite individuals unchanged to the next generation
        while len(new_pop) < pop_size: #fill the rest of the new population by selecting parents, crossing them over, and mutating the offspring
            p1 = tournament_select(pop, scores, tournament_k) #p1 and p2 are just short for "parent 1" and "parent 2" — two individuals picked from the current population (pop) to be bred together, like picking two parent folds to combine into offspring.
            p2 = tournament_select(pop, scores, tournament_k) #tournament_select(pop, scores, tournament_k) is a function that implements a selection mechanism called "tournament selection." It randomly picks k individuals from the population and returns the one with the highest fitness score. This way, better individuals have a higher chance of being selected as parents for the next generation.
            c1, c2 = crossover(p1, p2) 
            new_pop.append(mutate(c1, mutation_rate))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(c2, mutation_rate))
        pop = new_pop

    return best_moves, best_score, history



def main():
    seq =  SEQUENCE
    population_size =  POPULATION_SIZE
    generations =  GENERATIONS
    mutation_rate =  MUTATION_RATE
    seed =  SEED
    best_moves, best_score, history = run_ga(
        seq, pop_size=population_size, generations=generations,
        mutation_rate=mutation_rate, seed=seed, verbose=True
    )
    coords, collisions = decode(best_moves)
    contacts = hh_contacts(coords, seq)
    print(f"\nBest fold found:")
    print(f"  fitness       = {best_score:.1f}")
    print(f"  H-H contacts  = {contacts}")
    print(f"  self-overlaps = {collisions}  {'(valid SAW)' if collisions == 0 else '(INVALID)'}")

    try:
        plot_fold(coords, seq, 'output.png', contacts, collisions)
        print(f"  plot saved to output.png")
    except Exception as e:
        print(f"  (plotting skipped: {e})")

    return best_moves, best_score, coords, history



if __name__ == "__main__":
    main()