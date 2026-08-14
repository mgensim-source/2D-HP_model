

DIRECTIONS = {
    "U": (0, 1),
    "D": (0, -1),
    "L": (-1, 0),
    "R": (1, 0)
}
def get_valid_moves(current, occupied):
    x, y = current
      
    valid = []

    for direction, (dx, dy) in DIRECTIONS.items():
        new_pos = (x + dx, y + dy)

        if new_pos not in occupied:
            valid.append((direction, new_pos))

    return valid


def calculate_energy(sequence, coords):
    occupied = {pos: i for i, pos in enumerate(coords)}
    
    hh_contacts = 0

    for i in range(len(sequence)):
        if sequence[i] != "H":
            continue

        x, y = coords[i]
 
        neighbors = [
            (x + 1, y),
            (x - 1, y), 
            (x, y + 1),
            (x, y - 1)
        ]

        for pos in neighbors:
            if pos in occupied:
                j = occupied[pos]

                # Don't count consecutive residues
                if abs(i - j) != 1 and sequence[j] == "H":
                    hh_contacts += 1

    # Each contact gets counted twice, as symmetry
    hh_contacts //= 2

    return -hh_contacts
