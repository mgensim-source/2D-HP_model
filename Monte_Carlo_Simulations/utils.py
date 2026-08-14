#It is just a properly or a feature to be understood, the code performed unchanged it removed - has no inifulence in the search process....
def calculate_compactness(coords):
    """
    Fraction of the fold's bounding box occupied by the chain.
    1.0 = chain fills its bounding box completely (maximally compact).
    """
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]

    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1
    box_area = width * height

    if box_area == 0:
        return 0.0 

    return len(coords) / box_area

def calculate_energy(sequence, coords):
    """
    E(fold) = -(number of unique, non-consecutive H-H lattice contacts).
    Lower (more negative) is more stable. 
    """
    occupied = {pos: i for i, pos in enumerate(coords)}

    hh_contacts = 0

    for i in range(len(sequence)):
        if sequence[i] != "H":
            continue

        x, y = coords[i]
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        for pos in neighbors:
            if pos in occupied:
                j = occupied[pos]
                if abs(i - j) != 1 and sequence[j] == "H":
                    hh_contacts += 1

    hh_contacts //= 2  # each contact counted from both ends
    return -hh_contacts


class Protein:

    def __init__(self, sequence, coords=None):
        self.sequence = sequence
        self.coords = coords if coords is not None else self._straight_chain(len(sequence))
        self.energy = calculate_energy(self.sequence, self.coords)
        self.compactness = calculate_compactness(self.coords)

    @staticmethod
    def _straight_chain(n):
        return [(i, 0) for i in range(n)]

    def update(self, new_coords):
        """Replace the current fold and recompute energy/compactness."""
        self.coords = new_coords
        self.energy = calculate_energy(self.sequence, self.coords)
        self.compactness = calculate_compactness(self.coords)

    def copy(self):
        return Protein(self.sequence, list(self.coords))

    def __repr__(self):
        return (
            f"Protein(len={len(self.sequence)}, "
            f"energy={self.energy}, compactness={self.compactness:.2f})"
        )
