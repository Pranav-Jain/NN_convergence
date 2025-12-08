import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import polyscope as ps
import gpytoolbox as gpy
import torch
from tqdm import tqdm
import glob
import os
import torch.nn as nn
import scipy
from scipy.sparse.linalg import spsolve
import json
import pandas as pd
import sys

sys.path.append('../src')
from siren import MLP, MLP_normals
from surface_laplacian import get_surface_laplacian

with open("config.json", "r") as f:
    config = json.load(f)

usage_msg = "Usage: python test_parameterization.py [1|2|3|...]"

# Interpolate function f on mesh
def f(v):
    f = get_interpolated_values(u_mesh, v, v_mesh, f_mesh)
    return f.squeeze()

# Laplacian of f (torch version)
def laplacian_f(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    if sys.argv[1] == "1":
        lap_f = torch.sin(np.pi*x)*torch.sin(np.pi*y)*torch.sin(np.pi*z)
    elif sys.argv[1] == "2":
        lap_f = torch.sin(2*np.pi*x)*torch.sin(-np.pi*y)*torch.sin(-3*np.pi*z)
    elif sys.argv[1] == "3":
        lap_f = torch.sin(-3*np.pi*x)*torch.sin(6*np.pi*y)*torch.sin(8*np.pi*z)
    elif sys.argv[1] == "4":
        lap_f = torch.sin(np.pi*x)**2*torch.sin(np.pi*y)*torch.sin(np.pi*z)**3
    elif sys.argv[1] == "5":
        lap_f = torch.sin(np.pi*x)**5*torch.sin(np.pi*y)**(-2)*torch.sin(np.pi*z)
        
    else:
        raise NotImplementedError()

    return lap_f.squeeze()

# Laplacian of f (numpy version)
def laplacian_f_np(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    if sys.argv[1] == "1":
        lap_f = np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
    elif sys.argv[1] == "2":
        lap_f = np.sin(2*np.pi*x)*np.sin(-np.pi*y)*np.sin(-3*np.pi*z)
    elif sys.argv[1] == "3":
        lap_f = np.sin(-3*np.pi*x)*np.sin(6*np.pi*y)*np.sin(8*np.pi*z)
    elif sys.argv[1] == "4":
        lap_f = np.sin(np.pi*x)**2*np.sin(np.pi*y)*np.sin(np.pi*z)**3
    elif sys.argv[1] == "5":
        lap_f = np.sin(np.pi*x)**5*np.sin(np.pi*y)**(-2)*np.sin(np.pi*z)
        
    else:
        raise NotImplementedError()

    return lap_f.squeeze()
    
def get_random_points(v_mesh, f_mesh, n):
    # Sample points on mesh
    v_mesh_rdm, _, _ = gpy.random_points_on_mesh(v_mesh, f_mesh, n, return_indices=True)

    return v_mesh_rdm

def get_interpolated_values(f, v, v_mesh, f_mesh):
    # Compute nearest triangle & barycentric coordinates
    _, idx, bary = gpy.signed_distance(v, v_mesh, f_mesh, use_cpp=True)

    # Gather triangle values of f
    # f_mesh[idx] gives shape (n, 3), then indexing into f → (n, 3, fdim)
    face_vals = f[f_mesh[idx]]

    # Weighted sum
    if len(f.shape) == 2:
        f_v = (face_vals * bary[:, :, None]).sum(axis=1).squeeze()
    else:
        f_v = (face_vals * bary[:, :]).sum(axis=1).squeeze()
    return f_v

def sample_in_domain(n):
    v_mesh_rnd = get_random_points(v_mesh, f_mesh, n)
    return v_mesh_rnd

def get_normals(v, S_theta=None):
    n_pred = S_theta(v)
    n = n_pred / torch.linalg.norm(n_pred, dim=1, keepdim=True)

    return n

def train_strong_form(l_model, device, n, size_layer, n_layers):
    optimizer = torch.optim.Adam(l_model.parameters(), lr=config["architecture"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=config["architecture"]["scheduler_patience"])

    S_theta = MLP_normals(n=64, n_layers=5, in_dim=3, out_dim=3)
    S_theta.to(device=device)
    S_theta.load_state_dict(torch.load(f"../data/model_{config['surface']}_normal.pth", weights_only=True, map_location=device))
    S_theta.requires_grad_(True)
    
    losses = []
    for i in (pbar:= tqdm(range(config["architecture"]["max_iter"]))):
        try:
            optimizer.zero_grad()

            if i % 5000 == 0:
                v_mesh_global = sample_in_domain(int(1e6))
                v_mesh_global = torch.tensor(v_mesh_global, dtype=torch.float32, device=device).requires_grad_(True)

            idx = torch.randint(0, int(1e6), (n,), device=device)
            v_cart = v_mesh_global[idx]

            normals = get_normals(v_cart, S_theta)
            laplacian_pred = get_surface_laplacian(l_model, v_cart, normals)

            true_lap = laplacian_f(v_cart)
            true_lap = true_lap - mean_f

            loss = torch.linalg.norm(laplacian_pred - true_lap, 2)**2

            ### BOUNDARY CONDITIONS ###
            if len(boundary_edges) > 0:
                ## DIRICHLET BC ##
                if config["bc"] == "dirichlet":

                    # Get boundary points
                    bdry_edges_rdm = boundary_edges[torch.randint(0, boundary_edges.shape[0], (n,), device=device)]
                    w = torch.rand((n, 1), device=device) # linear interpolation weights
                    bdry_points = (1 - w) * v_mesh_torch[bdry_edges_rdm[:, 0]] + w * v_mesh_torch[bdry_edges_rdm[:, 1]]

                    u_bdry_pred = l_model(bdry_points).squeeze()
                    w = w.squeeze()
                    u_bdry_true = (1 - w) * u_mesh_torch[bdry_edges_rdm[:, 0]] + w * u_mesh_torch[bdry_edges_rdm[:, 1]]

                    loss_bdry = torch.linalg.norm(u_bdry_pred - u_bdry_true, 2)**2
                    loss = loss + 100*loss_bdry
            
            loss.backward()
            if not torch.isnan(loss):
                torch.nn.utils.clip_grad_norm_(l_model.parameters(), config["architecture"]["grad_clip"])  # Gradient clipping
                optimizer.step()
                scheduler.step(loss)

                loss_value = loss.item()
                pbar.set_description(f"Loss: {loss_value}")
                losses.append(loss_value)

        except KeyboardInterrupt:
            exit()

    plt.figure()
    plt.plot(np.log(losses))
    plt.title("Loss")
    plt.xlabel("Iteration")
    plt.ylabel("log(Loss)")
    plt.title(f"surface = {config['surface']}, {size_layer} x {n_layers}")
    plt.savefig(f"{save_dir}/example{sys.argv[1]}/loss_{size_layer}_{n_layers}.png")

    return l_model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_layers = [config["architecture"]["num_layers"]]
    size_layer = np.arange(30, 160, 20)

    V = sample_in_domain(100000)
    true = f(V)

    for i in n_layers:
        for j in size_layer:
            print(f"Processing model with {i} layers of {j} size")

            lap_model = MLP(n=j, n_layers=i)
            lap_model.to(device=device)

            lap_model = train_strong_form(lap_model, device, n=config["architecture"]["num_samples"], size_layer=j, n_layers=i)

            pred = lap_model(torch.Tensor(V).to(device=device)).squeeze().detach().cpu().numpy()
            
            l2_loss = np.linalg.norm(true - pred, 2) / np.linalg.norm(true, 2)
            print(l2_loss)

            # Save the model
            torch.save(lap_model.state_dict(), f"{save_dir}/example{sys.argv[1]}/model_{j}_{i}.pth")

def plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        filenames = glob.glob(f"{save_dir}/example{sys.argv[1]}/*.pth")
    except:
        print(usage_msg)
        exit()
    dof = []
    losses = []

    V = v_mesh

    for file in filenames:
        print(file)
        n_layers = int(file.split("_")[-1].split(".")[0])
        size_layer = int(file.split("_")[-2])

        model = MLP(n=size_layer, n_layers=n_layers, in_dim=3, out_dim=1)
        model.to(device=device)
        model.load_state_dict(torch.load(file, weights_only=True, map_location=device))

        pred = model(torch.Tensor(V).to(device=device)).squeeze().detach().cpu().numpy()
        true = f(V)

        const = np.mean(true - pred)
        pred = pred + const

        l2_loss = np.linalg.norm(pred - true, 2) / np.linalg.norm(true, 2)
        print(l2_loss, n_layers, size_layer)
        losses.append(l2_loss)

        total_trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        dof.append(total_trainable_params)

    dof = np.array(dof)
    losses = np.array(losses)

    dof_sorted = np.sort(dof)
    losses_sorted = losses[np.argsort(dof)]

    # Create DataFrame for this run
    example_name = f"example{sys.argv[1]}"
    output_path = f"{save_dir}/output.csv"
    data = pd.DataFrame({
        "dof": dof_sorted,
        example_name: losses_sorted
    })
    if sys.argv[1] == "1":
        data.to_csv(output_path, index=False)
    else:
        existing = pd.read_csv(output_path)
        merged = pd.merge(existing, data, on="dof", how="outer")
        merged = merged.sort_values(by="dof")
        merged.to_csv(output_path, index=False)

    x = dof_sorted
    y = losses_sorted

    #fit a line to the data
    coeffs = np.polyfit(np.log(x), np.log(y), 1)

    plt.loglog(x, y, label="PINN", marker='o')
    plt.loglog(x, 1/x, label="1/x", color="red", linestyle='--')
    plt.loglog(x, (1/x)**2, label="1/x^2", color="green", linestyle='--')

    plt.xlabel("Total Trainable Parameters")
    plt.ylabel("Relative L2 loss")
    plt.title(f"Slope: {coeffs[0]}")
    plt.legend()
    plt.savefig(f"{save_dir}/convergence_example_{sys.argv[1]}.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    S_theta = MLP_normals(n=64, n_layers=5, in_dim=3, out_dim=3)
    S_theta.to(device=device)
    S_theta.load_state_dict(torch.load(f"../data/model_{config['surface']}_normal.pth", weights_only=True, map_location=device))
    S_theta.requires_grad_(True)

    v_mesh, f_mesh = gpy.read_mesh(f"../data/{config['surface']}.obj")

    # Discrete Laplace-Beltrami
    L = gpy.cotangent_laplacian(v_mesh, f_mesh)                # stiffness matrix
    L = scipy.sparse.csc_matrix(L)
    M = gpy.massmatrix(v_mesh, f_mesh, type='barycentric')     # area weights
    M = -scipy.sparse.csc_matrix(M)

    # Discrete Laplacian of f
    lap_f_ = laplacian_f_np(v_mesh)
    
    # Enforce discrete compatibility: 1^T M f = 0
    area = M.sum()
    mean_f = np.ones_like(lap_f_) @ (M @ lap_f_) / area
    lap_f_ -= mean_f

    # Discrete surface Laplacian of u
    u_mesh = spsolve(L, M @ lap_f_)   # equivalent to -M^{-1} L u

    # Get boundary edges
    boundary_edges = gpy.boundary_edges(f_mesh)
    boundary_edges = torch.tensor(boundary_edges, dtype=torch.long, device=device)

    v_mesh_torch = torch.tensor(v_mesh, dtype=torch.float32, device=device).requires_grad_(True)
    u_mesh_torch = torch.tensor(u_mesh, dtype=torch.float32, device=device).requires_grad_(True)
    
    save_dir = f"poisson_results/{config['surface']}/{config['bc']}/domain_{config['domain']['min']}to{config['domain']['max']}"

    if config["operation"] == "train":
        # If directory doesn't exist, create it
        if not os.path.exists(f"{save_dir}/example{sys.argv[1]}"):
            os.makedirs(f"{save_dir}/example{sys.argv[1]}")
        
        json.dump(config, open(f"{save_dir}/example{sys.argv[1]}/config.json", 'w'), indent=4)
        
        train()
    elif config["operation"] == "plot":
        plot()
    else:
        print("Mention operation")
        raise NotImplementedError()