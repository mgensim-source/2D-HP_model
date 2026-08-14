from aco_p import aco_hp

sequence = "HPHPPHHPHPPHPHHPPHPH"

best = aco_hp(
    sequence,
    num_ants=50, #now many attempts of folding you want to make in one attempt (in parallel)
    iterations=500 
)

print("\nBest energy:", best["energy"])
print("Coordinates:")
print(best["coords"])

print("Moves:")
print(best["moves"])

import matplotlib.pyplot as plt
def plot_protein(sequence, coords, energy):

    x = [p[0] for p in coords]
    y = [p[1] for p in coords]

    plt.figure(figsize=(7, 7))

    # Protein backbone
    plt.plot(
        x,
        y,
        color="black",
        linewidth=2,
        zorder=1
    )

    # Residues
    for i, (px, py) in enumerate(coords):

        color = "red" if sequence[i] == "H" else "blue"

        plt.scatter(
            px,
            py,
            color=color,
            s=400,
            edgecolors="black",
            zorder=2
        )

        plt.text(
            px,
            py,
            str(i),
            ha="center",
            va="center",
            color="white",
            fontweight="bold"
        )

    plt.title(f"2D HP Folding — Energy = {energy}")

    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    plt.show()

plot_protein(
    sequence,
    best["coords"],
    best["energy"]
)
