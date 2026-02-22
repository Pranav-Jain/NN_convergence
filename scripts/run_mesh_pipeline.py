#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def run(cmd, cwd=None, env=None):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def update_config(config_path: Path, *, pde: str, surface: str, bc: str, experiment: str, operation: str, k: float, NN: str = None):
    with config_path.open("r") as f:
        config = json.load(f)

    config["pde"] = pde
    config["surface"] = surface
    config["bc"] = bc
    config["experiment"] = experiment
    config["operation"] = operation
    config["k"] = k
    if NN is not None:
        config["NN"] = NN

    with config_path.open("w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")

def scripts_render_output(root: Path, target_rel: Path) -> Path:
    output_stem = str(target_rel).replace("/", "|")
    if output_stem.endswith(".npy"):
        output_stem = output_stem[:-4]
    return root / "scripts" / "images" / f"{output_stem}.png"


def copy_plot_artifacts(root: Path, test_dir: Path, *, pde: str, surface: str, bc: str, example: int, NN: str = None) -> Path:
    if NN is not None:
        save_dir = test_dir / f"{pde}_results" / surface / NN / bc / "domain_-1.0to1.0"
    elif surface == "2d":
        save_dir = test_dir / f"{pde}_results" / surface / bc / "domain_-1.0to1.0"
    else:
        save_dir = test_dir / f"{pde}_results" / surface / bc
    dest_dir = root / "scripts" / "plots"
    dest_dir.mkdir(parents=True, exist_ok=True)

    convergence_png = save_dir / f"convergence_example_{example}.png"
    if not convergence_png.exists():
        raise FileNotFoundError(f"Missing convergence plot: {convergence_png}")

    if NN is not None:
        output_plot = dest_dir / f"convergence_example_{pde}_{surface}_{NN}_{bc}_example{example}.png"
    else:
        output_plot = dest_dir / f"convergence_example_{pde}_{surface}_{bc}_example{example}.png"
    shutil.copy2(convergence_png, output_plot)
    print(f"Copied final convergence plot to {output_plot}")
    return output_plot


def generate_and_render_surface_mesh(root: Path, surface: str, k: float):
    if surface == "cylinder":
        render_mesh_name = "cylinder_minimal_surface"
    elif surface == "moai":
        render_mesh_name = f"moai_smooth_{k}"
    else:
        return None

    computed_param_dir = root / "test_computed_parameterization"
    run(["python3", "minimal_surface.py", surface, str(int(k))], cwd=str(computed_param_dir))
    run(["python3", str(root / "scripts" / "render_mesh.py"), render_mesh_name], cwd=str(root))
    return render_mesh_name


def generate_2d_comparison_plot(root: Path, *, pde: str, surface: str, bc: str, example: str) -> Path:
    output_dir = root / "scripts" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    run([
        "python3",
        str(root / "scripts" / "plot_convergence.py"),
        "--root",
        str(root),
        "--example",
        str(example),
        "--pde",
        pde,
        "--surface",
        surface,
        "--bc",
        bc,
        "--output-dir",
        "scripts/plots",
    ], cwd=str(root))

    output_plot = output_dir / f"convergence_plot_{pde}_{surface}_{bc}_example{example}.png"
    if not output_plot.exists():
        raise FileNotFoundError(f"Expected 2D comparison plot was not created: {output_plot}")
    return output_plot


def main():
    parser = argparse.ArgumentParser(description="Run train+plot+render pipeline for one mesh")
    parser.add_argument("--surface", required=True)
    parser.add_argument("--example", default=1)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--pde", default="poisson")
    parser.add_argument("--bc", default="dirichlet")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--k", type=float, default=-100)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--copy-plot", action="store_true", help="Copy convergence plot to scripts/plots")
    parser.add_argument("--render", action="store_true", help="Render pred/error/true images to scripts/images")
    parser.add_argument("--render-only", action="store_true", help="Skip test_convergence and only run rendering from existing npy files")
    parser.add_argument("--NN", required=False, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.NN is not None:
        test_dir = root / "test_exact_parameterization"
    elif args.surface == "2d":
        test_dir = root / "test_exact"
    else:
        test_dir = root / "test_computed_parameterization"
    config_path = test_dir / "config.json"

    normal_model = root / "data" / f"model_{args.surface}_normal.pth"
    if not normal_model.exists() and args.NN is None and args.surface != "2d":
        print(f"Normal model not found at {normal_model}, training...")
        run(["python3", "train_NN_mesh_normals.py", args.surface], cwd=str(test_dir))

    if not args.render_only:
        update_config(
            config_path,
            pde=args.pde,
            surface=args.surface,
            bc=args.bc,
            experiment=args.experiment,
            operation="plot",
            k=args.k,
            NN=args.NN,
        )
        env = os.environ.copy()
        env["NO_POLYSCOPE_SHOW"] = "1"
        run(["python3", "test_convergence.py", str(args.example)], cwd=str(test_dir), env=env)

        if args.surface == "cylinder":
            env = os.environ.copy()
            env["NO_POLYSCOPE_SHOW"] = "1"
            run(["python3", "test_convergence.py", str(int(args.example) + 1)], cwd=str(test_dir), env=env)
            run(["python3", "test_convergence.py", str(int(args.example) + 2)], cwd=str(test_dir), env=env)

        if args.surface == "moai":
            env = os.environ.copy()
            env["NO_POLYSCOPE_SHOW"] = "1"
            split = args.example.split('_')
            run(["python3", "test_convergence.py", str(int(split[0]) + 1)+ '_' + str(split[1])], cwd=str(test_dir), env=env)
            run(["python3", "test_convergence.py", str(int(split[0]) + 2)+ '_' + str(split[1])], cwd=str(test_dir), env=env)

    plot_file = None
    comparison_plot_file = None
    if args.surface == "2d" and args.experiment == "comparison":
        comparison_plot_file = generate_2d_comparison_plot(
            root,
            pde=args.pde,
            surface=args.surface,
            bc=args.bc,
            example=str(args.example),
        )

    if args.copy_plot:
        plot_file = copy_plot_artifacts(
            root,
            test_dir,
            pde=args.pde,
            surface=args.surface,
            bc=args.bc,
            example=args.example,
            NN=args.NN,
        )

    scripts_images_dir = None
    rendered_mesh_name = None
    size, layers = args.size, args.layers
    should_render = args.render or args.render_only
    if should_render:
        if args.surface in {"cylinder", "moai"}:
            rendered_mesh_name = generate_and_render_surface_mesh(root, args.surface, args.k)
            scripts_images_dir = root / "scripts" / "images"
            scripts_images_dir.mkdir(parents=True, exist_ok=True)
        else:
            if args.NN is not None:
                result_dir = test_dir / f"{args.pde}_results" / args.surface / args.NN / args.bc / f"domain_-1.0to1.0" / f"example{args.example}" / "npy_files"
            elif args.surface == "2d":
                result_dir = test_dir / f"{args.pde}_results" / args.surface / args.bc / f"domain_-1.0to1.0" / f"example{args.example}" / "npy_files"
            else:
                result_dir = test_dir / f"{args.pde}_results" / args.surface / args.bc / f"example{args.example}" / "npy_files"

            render_targets = [
                result_dir / f"pred_{size}_{layers}.npy",
                result_dir / f"error_{size}_{layers}.npy",
                result_dir / f"true_{size}_{layers}.npy",
            ]

            if args.surface == "dziuk" and args.pde == "helmholtz":
                render_targets.append(result_dir / f"pwos_estimate.npy")
                render_targets.append(result_dir / f"pwos_error.npy")

            if args.surface == "2d" and args.experiment == "comparison":
                render_targets.append(result_dir / f"deepritz_error_73701.npy")
                render_targets.append(result_dir / f"error_fem_256_256.npy")
                render_targets.append(result_dir / f"navami_poisson_abs_error_49.93k_10000.npy")
                render_targets.append(result_dir / f"vpinn_error_vpinn_45901.npy")

            if args.surface == "2d" and args.experiment == "comparison":
                render_targets.append(result_dir / f"deepritz_error_73701.npy")
                render_targets.append(result_dir / f"error_fem_256_256.npy")
                render_targets.append(result_dir / f"navami_poisson_abs_error_49.93k_10000.npy")
                render_targets.append(result_dir / f"vpinn_error_vpinn_45901.npy")
                

            scripts_images_dir = root / "scripts" / "images"
            scripts_images_dir.mkdir(parents=True, exist_ok=True)

            for target in render_targets:
                if not target.exists():
                    raise FileNotFoundError(f"Missing render input: {target}")

                target_rel = target.relative_to(root)
                scripts_png = scripts_render_output(root, target_rel)

                run([
                    "python3",
                    str(root / "scripts" / "render_vertexscalars.py"),
                    str(target_rel),
                    str(scripts_images_dir),
                ], cwd=str(root))

                if not scripts_png.exists():
                    raise RuntimeError(
                        f"Render finished but expected output was not created: {scripts_png}. "
                        "Check Blender logs for Python import/runtime errors."
                    )

    msg = f"Done for {args.surface} (size={size}, layers={layers})."
    if scripts_images_dir is not None:
        msg += f" Images saved to {scripts_images_dir}."
    else:
        msg += " Rendering skipped."
    if args.render_only:
        msg += " (Render-only mode)"
    if plot_file is not None:
        msg += f" Plot saved to {plot_file}"
    if comparison_plot_file is not None:
        msg += f" Comparison plot saved to {comparison_plot_file}"
    if rendered_mesh_name is not None:
        msg += f" Mesh rendered with render_mesh.py for {rendered_mesh_name}."
    print(msg)


if __name__ == "__main__":
    main()
