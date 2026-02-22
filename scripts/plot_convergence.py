import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(pinn_file, fem_file, navami_files, vpinn_files, deepritz_files, output_file):
    # Load data
    pinn_data = np.loadtxt(pinn_file, skiprows=1)
    fem_data = np.loadtxt(fem_file, skiprows=1)
    navami_data = np.loadtxt(navami_files, skiprows=1)
    vpinn_data = np.loadtxt(vpinn_files, skiprows=1)
    deepritz_data = np.loadtxt(deepritz_files, skiprows=1)

    pinn_dof = pinn_data[:, 0]
    pinn_losses = pinn_data[:, 1]

    fem_dof = fem_data[:, 0]
    fem_losses = fem_data[:, 1]

    navami_dof = navami_data[:, 0]
    navami_losses = navami_data[:, 1]

    vpinn_dof = vpinn_data[:, 0]
    vpinn_losses = vpinn_data[:, 1]

    deepritz_dof = deepritz_data[:, 0]
    deepritz_losses = deepritz_data[:, 1]

    # print slopes
    pinn_slope = np.polyfit(np.log(pinn_dof), np.log(pinn_losses), 1)[0]
    fem_slope = np.polyfit(np.log(fem_dof), np.log(fem_losses), 1)[0]
    navami_slope = np.polyfit(np.log(navami_dof), np.log(navami_losses), 1)[0]
    vpinn_slope = np.polyfit(np.log(vpinn_dof), np.log(vpinn_losses), 1)[0]
    deepritz_slope = np.polyfit(np.log(deepritz_dof), np.log(deepritz_losses), 1)[0]

    print(f"PINN slope: {pinn_slope}, FEM slope: {fem_slope}, Navami slope: {navami_slope}, VPINN slope: {vpinn_slope}, DeepRitz slope: {deepritz_slope}")
    plt.loglog(pinn_dof, pinn_losses, label="PINN", marker='o', linewidth=2)
    plt.loglog(fem_dof, fem_losses, label="FEM", marker='o', linewidth=2)
    plt.loglog(navami_dof, navami_losses, label="Navami", marker='o', linewidth=2)
    plt.loglog(vpinn_dof, vpinn_losses, label="hp-VPINN", marker='o', linewidth=2)
    plt.loglog(deepritz_dof, deepritz_losses, label="Deep Ritz", marker='o', linewidth=2)

    plt.xlabel("Total Trainable Parameters")
    plt.gca().set_facecolor('#f0f0f0')  # set gray bg color
    plt.grid(color='white', linestyle='-', linewidth=1.5)
    
    plt.legend()

    x_ = [pinn_dof[0],pinn_dof[len(pinn_dof)//2], pinn_dof[-1]]
    plt.xticks(ticks=x_, labels=['{:.1e}'.format(i) for i in x_], fontsize=10)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    print(f"Saved comparison convergence plot to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2D comparison convergence curves")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Project root directory")
    parser.add_argument("--example", default="10", help="Example id used in example<id> directory")
    parser.add_argument("--pde", default="poisson")
    parser.add_argument("--surface", default="2d")
    parser.add_argument("--bc", default="dirichlet")
    parser.add_argument("--domain", default="domain_-1.0to1.0")
    parser.add_argument("--output-dir", default="scripts/plots", help="Output directory relative to root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    example_dir = (
        root
        / "test_exact"
        / f"{args.pde}_results"
        / args.surface
        / args.bc
        / args.domain
        / f"example{args.example}"
    )

    pinn_file = example_dir / "pinn_convergence.txt"
    fem_file = example_dir / "fem_convergence.txt"
    navami_file = example_dir / "navami_convergence.txt"
    vpinn_file = example_dir / "vpinn_convergence.txt"
    deepritz_file = example_dir / "deepritz_convergence.txt"

    missing = [
        str(path)
        for path in [pinn_file, fem_file, navami_file, vpinn_file, deepritz_file]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing convergence text files:\n" + "\n".join(missing))

    output_file = (
        root
        / args.output_dir
        / f"convergence_plot_{args.pde}_{args.surface}_{args.bc}_example{args.example}.png"
    )

    plot_convergence(
        pinn_file=pinn_file,
        fem_file=fem_file,
        navami_files=navami_file,
        vpinn_files=vpinn_file,
        deepritz_files=deepritz_file,
        output_file=output_file,
    )