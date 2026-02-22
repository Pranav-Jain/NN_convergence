import argparse
import glob
import os
import sys

import gpytoolbox as gpy
import numpy as np
import torch

sys.path.append('../src')
from siren import MLP


def _build_surface_spec(surface: str, k: float = None):
    if surface == "cylinder":
        return {
            "save_dir": "poisson_results/cylinder/dirichlet",
            "mesh_in": "../data/cylinder.obj",
            "mesh_out": "../data/cylinder_minimal_surface.obj",
            "example_indices": range(1, 4),
            "output_offset": 1,
        }
    if surface == "moai":
        return {
            "save_dir": "helmholtz_results/moai/dirichlet",
            "mesh_in": "../data/moai.obj",
            "mesh_out": f"../data/moai_smooth_{k}.obj",
            "example_indices": range(13, 16),
            "output_offset": 13,
        }
    raise ValueError(f"Unsupported surface '{surface}'. Expected one of: cylinder, moai")


def generate_mesh(surface: str, k: float = None):
    spec = _build_surface_spec(surface, k)
    v_mesh, f_mesh = gpy.read_mesh(spec["mesh_in"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_vertices = np.zeros(v_mesh.shape)
    for i in spec["example_indices"]:
        if surface == "moai":
            filenames = glob.glob(f"{spec['save_dir']}/example{i}_{abs(int(k))}/*.pth")
        else:
            filenames = glob.glob(f"{spec['save_dir']}/example{i}/*.pth")
        if not filenames:
            raise FileNotFoundError(f"No model checkpoints found in {spec['save_dir']}/example{i}_{abs(int(k))}")

        for file in filenames:
            n_layers = int(file.split("_")[-1].split(".")[0])
            size_layer = int(file.split("_")[-2])

            model = MLP(n=size_layer, n_layers=n_layers, in_dim=3, out_dim=1)
            model.to(device=device)
            model.load_state_dict(torch.load(file, weights_only=True, map_location=device))

            pred = model(torch.Tensor(v_mesh).to(device=device)).squeeze().detach().cpu().numpy()
            final_vertices[:, i - spec["output_offset"]] = pred

    if surface == "moai":
        final_vertices*= abs(k) 
    gpy.write_mesh(spec["mesh_out"], final_vertices, f_mesh)
    print(f"Saved generated mesh to {spec['mesh_out']}")


def main():
    parser = argparse.ArgumentParser(description="Generate derived meshes for selected surfaces")
    parser.add_argument("surface", choices=["cylinder", "moai"], help="Surface to process")
    parser.add_argument("k", type=float, help="K parameter for moai surface (ignored for cylinder)")
    args = parser.parse_args()
    generate_mesh(args.surface, args.k)


if __name__ == "__main__":
    main()