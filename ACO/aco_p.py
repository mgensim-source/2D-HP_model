import random
from utils import calculate_energy, get_valid_moves

def initialize_pheromone(n, initial_pheromone=1.0):
    pheromone = []

    for _ in range(n):
        pheromone.append({
            "U": initial_pheromone,
            "D": initial_pheromone,
            "L": initial_pheromone,
            "R": initial_pheromone
        })

    return pheromone

#The heuristic - the immediate knowledge
#Note: Pheromone provides a long-term memory of the search process, while the heuristic provides immediate knowledge about the problem. The heuristic is used to guide the ants in their search for solutions, helping them make decisions based on the current state of the problem.
#Heuristic calulates if the immediate addition causes H-H contact or not.
#η=1+number of new H-H contacts

def count_new_hh_contacts(sequence, index, new_pos, coords): #coords: already occupied coordinates, new_pos is new possiblly identified position
    if sequence[index] != "H": 
        return 0

    occupied = {pos: i for i, pos in enumerate(coords)}

    x, y = new_pos

    neighbors = [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1)
    ]

    contacts = 0

    for pos in neighbors:
        if pos in occupied:
            j = occupied[pos]

            if sequence[j] == "H" and abs(index - j) != 1: #index is current position/pointer in the string, if it is H and it is not consecutive in the sequence given/input sequence
                contacts += 1 #index is index of the pointer/new one in the sequence, j is the index of the neighbor in the sequence. If both are H and not consecutive, it counts as a new contact.

    return contacts

#ACO Probability assosicaed with the nove
#Pi​(d)=∑k​τi,kα​ηi,kβ​τi,dα​ηi,dβ​​
#Given all legal moves for the next protein residue, calculate the probability of choosing each move using pheromone + heuristic information.
def calculate_probabilities(
    moves, #moves is the legal next moves available ("U", (1,2))
    pheromone,
    residue_index, #residue index is the index of the sequence we'tr trying to place in the grid
    sequence,
    coords, #this includes coordinates of the already placed residues
    alpha=1.0, #Controls how strongly pheromone influences the decision.
    beta=2.0 #Controls how strongly the heuristic influences the decision.
):
    values = [] #raw score for every possible move. eg: ("U", (1, 2), 5.4), 
    for direction, new_pos in moves: #For each possible move, we calculate: pheromone+heuristic
        tau = pheromone[residue_index][direction] #The pheromone associated with choosing direction when placing residue_index.
        new_contacts = count_new_hh_contacts(
            sequence,
            residue_index,
            new_pos,
            coords
        )
        eta = 1.0 + new_contacts

        value = (tau ** alpha) * (eta ** beta)
        values.append((direction, new_pos, value))

    total = sum(value for _, _, value in values)
    probabilities = []
    for direction, pos, value in values:
        probabilities.append(
            (direction, pos, value / total)
        )

    return probabilities

#We don't simply choose the highest probability.
#We randomly sample according to the probabilities.
#That's important because ACO needs exploration.
def choose_move(probabilities):
    directions = [x[0] for x in probabilities]
    positions = [x[1] for x in probabilities]
    weights = [x[2] for x in probabilities]

    selected = random.choices(
        range(len(probabilities)),
        weights=weights,
        k=1
    )[0]

    return directions[selected], positions[selected]


def construct_solution(
    sequence,
    pheromone,
    alpha=1.0,
    beta=2.0
):
    coords = [(0, 0)]
    occupied = {(0, 0)}

    moves_used = []

    for i in range(1, len(sequence)):

        current = coords[-1]

        valid_moves = get_valid_moves(
            current,
            occupied
        )

        if not valid_moves:
            return None

        probabilities = calculate_probabilities(
            valid_moves,
            pheromone,
            i,
            sequence,
            coords,
            alpha,
            beta
        )

        direction, new_pos = choose_move(probabilities)

        coords.append(new_pos)
        occupied.add(new_pos)
        moves_used.append(direction)

    energy = calculate_energy(sequence, coords)

    return {
        "coords": coords,
        "moves": moves_used,
        "energy": energy
    }

def evaporate_pheromone(pheromone, rho=0.1):

    for i in range(len(pheromone)):
        for direction in pheromone[i]:
            pheromone[i][direction] *= (1 - rho)

def deposit_pheromone(pheromone, solution, Q=2.0, tau_max=30.0):
    """The actual learning step. Capped at tau_max so no direction can run away
    and make the search deterministic/premature."""
    energy = solution["energy"]
    quality = Q * abs(energy)
 
    for i, direction in enumerate(solution["moves"], start=1):
        pheromone[i][direction] = min(tau_max, pheromone[i][direction] + quality)
 


def aco_hp(
    sequence,
    num_ants=50,
    iterations=200,
    alpha=1.0,
    beta=2.0,
    rho=0.1,
    Q=10.0,
    tau_max=30.0
):

    n = len(sequence)
    pheromone = initialize_pheromone(n)
    best_solution = None

    for iteration in range(iterations):
        solutions = []
        for _ in range(num_ants):

            solution = construct_solution(
                sequence,
                pheromone,
                alpha,
                beta
            )

            if solution is not None:
                solutions.append(solution)

                if (
                    best_solution is None
                    or solution["energy"]
                    < best_solution["energy"]
                ):
                    best_solution = solution

        # --------------------------------
        # Evaporation
        # --------------------------------

        evaporate_pheromone(
            pheromone,
            rho
        )

        for solution in solutions:
            deposit_pheromone(pheromone, solution, Q, tau_max)

        if iteration % 10 == 0:

            if best_solution:
                print(
                    f"Iteration {iteration}: "
                    f"Best energy = "
                    f"{best_solution['energy']}"
                )

    return best_solution