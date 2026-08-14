from utils import Protein, calculate_energy
import random
import math

_TRANSFORMATIONS = [
    lambda x, y: (-y, x),   # rotate 90
    lambda x, y: (-x, -y),  # rotate 180
    lambda x, y: (y, -x),   # rotate 270
    lambda x, y: (x, -y),   # reflect across x-axis
    lambda x, y: (-x, y),   # reflect across y-axis
    lambda x, y: (y, x),    # reflect across main diagonal
    lambda x, y: (-y, -x),  # reflect across anti-diagonal
]
def generate_random_fold(coords):
    """
    Pivot move: pick a random non-terminal residue, apply a random lattice
    symmetry to everything downstream of it. Returns None if the result
    is not self-avoiding (caller should treat that as a rejected move).
    """
    n = len(coords) #perturbations for longer length (more than 2)
    if n < 3:
        return coords[:]

    pivot_index = random.randint(1, n - 2) #get a random pivot index between 1 and n-2 (inclusive) that has to be perturbed...
    pivot_x, pivot_y = coords[pivot_index]
    #One single transformation, applied to every point after pivot_index, all relative to the pivot.
    #One transformation is picked once, outside the loop. Same rotation/reflection gets reused for every residue in the tail.
    transform = random.choice(_TRANSFORMATIONS)

    new_coords = coords[: pivot_index + 1]
    occupied = set(new_coords)

    for i in range(pivot_index + 1, n):
        dx = coords[i][0] - pivot_x #find the scale of the trasnformation in x direction
        dy = coords[i][1] - pivot_y #find the scale of the transformation in y direction 
        tdx, tdy = transform(dx, dy)
        new_pos = (pivot_x + tdx, pivot_y + tdy)

        if new_pos in occupied: #note it should not overlap with any of the previous residues, otherwise it is not a valid fold
            return None

        occupied.add(new_pos)
        new_coords.append(new_pos)

    return new_coords

def metropolis_accept(delta_energy, temperature):
    """
    Metropolis criterion: accept a move if it lowers energy, or with
    probability exp(-delta_energy / temperature) if it raises energy.
    """
    if delta_energy < 0:
        return True

    acceptance_prob = math.exp(-delta_energy / temperature)
    return random.random() < acceptance_prob



def run_monte_carlo(
    protein,
    folding_steps=5000,
    initial_temperature=5.0,
    annealing=True,
    seed=None,
    history_every=1,
):
    """
    Runs one Metropolis / simulated-annealing chain, mutating `protein`
    in place to its final fold, and returns a dict with the best fold
    found on this chain, and history for plotting.
    """
    if seed is not None:
        random.seed(seed)

    best = protein.copy()

    energy_history = [protein.energy]
    compactness_history = [protein.compactness]
    fold_history = [list(protein.coords)]
    accepted_moves = 0
    temperature = initial_temperature

    for step in range(folding_steps):
        if annealing:
            temperature = max(initial_temperature * (1 - step / folding_steps), 1e-6)

        candidate_coords = generate_random_fold(protein.coords)

        if candidate_coords is not None:
            candidate_energy = calculate_energy(protein.sequence, candidate_coords)
            delta = candidate_energy - protein.energy

            if metropolis_accept(delta, temperature):
                protein.update(candidate_coords) #this also changes the energy and compactness of the protein object
                accepted_moves += 1

                if protein.energy < best.energy:
                    best = protein.copy()

        if step % history_every == 0:
            energy_history.append(protein.energy)
            compactness_history.append(protein.compactness)
            fold_history.append(list(protein.coords))
            

    return {
        "final": protein,
        "best": best,
        "energy_history": energy_history,
        "compactness_history": compactness_history,
        "fold_history": fold_history,
        "accepted_moves": accepted_moves,
    }





def run_monte_carlo_multistart(
    sequence,
    num_runs=20,
    folding_steps=5000,
    initial_temperature=5.0,
    annealing=True,
    seed=None,
    history_every=1,
    starting_coords=None,):
    """
    Runs several independent annealing chains, each starting from
    `starting_coords` (or a fresh straight chain if not given), and
    keeps the best fold found across all of them.

    A single Metropolis chain can get trapped near a local-minimum energy
    and never escape once temperature has cooled, so for landscapes with
    a narrow global optimum (as HP folding often has), multiple
    independent restarts find the true minimum far more reliably than
    one long run does. Only the history from the run that produced the
    global best is kept (for plotting); other runs' histories are
    discarded to save memory.

    Returns the same shape as run_monte_carlo, plus 'run_energies': the
    best energy found by each individual run, useful for judging how
    hard-to-find the optimum is (a tight cluster of run_energies near the
    best suggests the search converges reliably; a wide spread means the
    landscape has a narrow optimum that most single runs miss).
    """
    best_overall = None
    best_result = None
    run_energies = []

    for run_index in range(num_runs):
        run_seed = None if seed is None else seed + run_index
        protein = Protein(sequence, starting_coords)

        result = run_monte_carlo(
            protein,
            folding_steps=folding_steps,
            initial_temperature=initial_temperature,
            annealing=annealing,
            seed=run_seed,
            history_every=history_every,
        )

        run_energies.append(result["best"].energy)

        if best_overall is None or result["best"].energy < best_overall:
            best_overall = result["best"].energy
            best_result = result

    best_result["run_energies"] = run_energies
    return best_result