import os
import sys
import numpy as np
import subprocess
import json
from scipy.stats import linregress
import pandas as pd

def run_plot(example_idx):
    """Run the plotting routine for a given example."""
    print(f"Running plot for example {example_idx}...")
    cwd = os.getcwd()
    file_path = os.path.join(cwd, "test_parameterization.py") 
    result = subprocess.run(["python3", file_path, str(example_idx)])
    if result.returncode != 0:
        print(f"Error running example {example_idx}:\n{result.stderr}")
        raise RuntimeError(f"Plotting failed for example {example_idx}")

def analyze_convergence(dof, loss):
    """Compute slope, correlation, and convergence status."""
    logx, logy = np.log(dof), np.log(loss)
    slope, intercept, r_value, _, _ = linregress(logx, logy)
    converged = (slope < -0.3) and (abs(r_value) > 0.5)
    return slope, r_value, converged

def main():
    examples = range(1, 16)  # change range as needed
    results = []

    for i in examples:
        run_plot(i)

    if config["form"] == "weak":
        save_dir = f"poisson_results/{config['surface']}/{config['NN']}/{config['bc']}/{config['form']}/{config['basis_type']}/domain_{config['domain']['min']}to{config['domain']['max']}"
    else:
        save_dir = f"poisson_results/{config['surface']}/{config['NN']}/{config['bc']}/{config['form']}/domain_{config['domain']['min']}to{config['domain']['max']}"
    
    # --- Now read combined CSV once ---
    output_csv = os.path.join(save_dir, "output.csv")

    df = pd.read_csv(output_csv)

    # Each column (after 'dof') corresponds to an example’s loss
    dof = df["dof"].to_numpy()
    cols = [c for c in df.columns if c != "dof"]

    for c in cols:
        loss = df[c].to_numpy()
        slope, r_value, converged = analyze_convergence(dof, loss)
        results.append({
            "example": c,
            "slope": slope,
            "correlation": r_value,
            "converged": int(converged)
        })
        print(f"{c}: slope={slope:.3f}, corr={r_value:.3f}, converged={converged}")

    # --- Global convergence check ---
    num_converged = sum(r["converged"] for r in results)
    total = len(results)
    frac_converged = num_converged / total if total > 0 else 0.0
    overall_converged = frac_converged >= 0.8

    print("\n===== Summary =====")
    print(f"{num_converged}/{total} examples converged "
          f"({frac_converged*100:.1f}%)")
    print(f"Overall convergence: {overall_converged}")

    # Save results
    output_json_path = os.path.join(save_dir, "convergence_summary.json")
    with open(output_json_path, "w") as f:
        json.dump({
            "individual_results": results,
            "num_converged": num_converged,
            "total_examples": total,
            "fraction_converged": frac_converged,
            "overall_converged": int(overall_converged)
        }, f, indent=4)

    print("Saved summary to convergence_summary.json")

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
        if config["operation"] != "plot":
            raise ValueError("Config operation must be 'plot' to run this script.")
    main()
