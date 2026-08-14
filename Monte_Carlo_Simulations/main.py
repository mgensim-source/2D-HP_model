
import sys
import os
import yaml
from utils import Protein
from plots import plot_protein, plot_energy_evolution, plot_compactness_evolution, create_fold_gif
from simulations import run_monte_carlo_multistart



def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config_path=None):
    project_root = os.path.dirname(os.path.abspath(__file__))
    print("=======", project_root)

    if config_path is None:
        config_path = os.path.join(project_root, "config.yaml")

    config = load_config(config_path)
    sequence = config["sequence"]

    structure_cfg = config.get("structure", {}) or {}
    initial_coords = None
    if structure_cfg.get("use_structure"):
        initial_coords = [tuple(c) for c in structure_cfg["coordinates"]]

    seed = config.get("seed")
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    protein = Protein(sequence, initial_coords)
    plot_protein(
        sequence, protein.coords, protein.energy,
        title_prefix="Initial fold",
        save_path=os.path.join(output_dir, "initial.png"),
    )

    sim_cfg = config.get("simulation", {}) or {}
    result = run_monte_carlo_multistart(
        sequence,
        num_runs=sim_cfg.get("num_runs", 20),
        folding_steps=sim_cfg.get("folding_steps", 5000),
        initial_temperature=sim_cfg.get("temperature", 5.0),
        annealing=sim_cfg.get("annealing", True),
        seed=seed,
        starting_coords=initial_coords,
    )
    best = result["best"]
    plot_energy_evolution(
        result["energy_history"],
        save_path=os.path.join(output_dir, "energy_evolution.png"),
    )
    plot_compactness_evolution(
        result["compactness_history"],
        save_path=os.path.join(output_dir, "compactness_evolution.png"),
    )

    if (config.get("plot", {}) or {}).get("create_gif", False):
        create_fold_gif(
            sequence, result["fold_history"],
            save_path=os.path.join(output_dir, "evolution.gif"),
        )

    print(f"Final energy (best run): {result['final'].energy}")
    print(f"Best energy found: {best.energy}")
    print(f"Best compactness: {best.compactness:.3f}")
    print(f"Per-run best energies: {result['run_energies']}")
    print(f"Best coordinates: {best.coords}")

    plot_protein(
        sequence, best.coords, best.energy,
        title_prefix="Min-energy fold",
        save_path=os.path.join(output_dir, "min_energy.png"),
    )

    print(f"\nPlots written to: {output_dir}")


if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_arg)