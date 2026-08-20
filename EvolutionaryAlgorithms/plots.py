import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_fold(coords, seq, path, contacts, collisions):
    
    xs, ys, zs = zip(*coords)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xs, ys, zs, color="gray", linewidth=1.5, zorder=1)

    for i, (x, y, z) in enumerate(coords):
        color = "#d62728" if seq[i] == "H" else "#1f77b4"
        ax.scatter(x, y, z, color=color, s=140, edgecolor="black",
                   linewidth=0.6, zorder=2)
        ax.text(x, y, z, str(i), fontsize=6, zorder=3)

    ax.scatter([], [], color="#d62728", label="H (hydrophobic)")
    ax.scatter([], [], color="#1f77b4", label="P (polar)")
    ax.set_title(f"3D HP fold  |  H-H contacts = {contacts}  |  overlaps = {collisions}")
    ax.legend(loc="upper right")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)