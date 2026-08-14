import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving files without a display
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_protein(sequence, coords, energy, title_prefix="2D HP Folding", save_path=None):
    x = [p[0] for p in coords]
    y = [p[1] for p in coords]

    fig = plt.figure(figsize=(7, 7))
    plt.plot(x, y, color="black", linewidth=2, zorder=1)

    for i, (px, py) in enumerate(coords):
        color = "red" if sequence[i] == "H" else "blue"
        plt.scatter(px, py, color=color, s=400, edgecolors="black", zorder=2)
        plt.text(px, py, str(i), ha="center", va="center", color="white", fontweight="bold")

    plt.title(f"{title_prefix} — Energy = {energy}")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_energy_evolution(energy_history, save_path=None):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(energy_history, color="darkred", linewidth=1)
    plt.xlabel("Recorded step")
    plt.ylabel("Energy")
    plt.title("Energy evolution")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_compactness_evolution(compactness_history, save_path=None):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(compactness_history, color="darkgreen", linewidth=1)
    plt.xlabel("Recorded step")
    plt.ylabel("Compactness")
    plt.title("Compactness evolution")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def create_fold_gif(sequence, fold_history, save_path="output/evolution.gif", step_stride=None, fps=10, max_frames=150):
    """
    Animates the fold evolving across the recorded history.
    Rendering costs roughly 0.1-0.2s per frame, so by default this
    subsamples fold_history down to `max_frames` frames rather than
    rendering every recorded step (pass step_stride explicitly to
    override this and use a fixed stride instead).
    """
    if step_stride is None:
        step_stride = max(1, len(fold_history) // max_frames)

    frames = fold_history[::step_stride]

    fig, ax = plt.subplots(figsize=(6, 6))

    all_x = [p[0] for frame in frames for p in frame]
    all_y = [p[1] for frame in frames for p in frame]
    pad = 1
    xlim = (min(all_x) - pad, max(all_x) + pad)
    ylim = (min(all_y) - pad, max(all_y) + pad)

    def _draw(i):
        ax.clear()
        frame_coords = frames[i]
        x = [p[0] for p in frame_coords]
        y = [p[1] for p in frame_coords]

        ax.plot(x, y, color="black", linewidth=2, zorder=1)
        for j, (px, py) in enumerate(frame_coords):
            color = "red" if sequence[j] == "H" else "blue"
            ax.scatter(px, py, color=color, s=250, edgecolors="black", zorder=2)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Step {i * step_stride}")

    anim = animation.FuncAnimation(fig, _draw, frames=len(frames))

    try:
        anim.save(save_path, writer="pillow", fps=fps)
    finally:
        plt.close(fig)
