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
from scipy.sparse.linalg import eigsh

sys.path.append('../src')
from siren import MLP, MLP_normals
from surface_laplacian import get_surface_laplacian

with open("config.json", "r") as f:
    config = json.load(f)

usage_msg = "Usage: python test_parameterization.py [1|2|3|...]"

# Interpolate function f on mesh
def u_numpy(v):
    f = get_interpolated_values(u_mesh, v, v_mesh, f_mesh)
    return f.squeeze()

# Laplacian of f (torch version)
def rhs(v):
    if not have_bdry:
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        if sys.argv[1] == "1":
            lap_u = torch.sin(9*np.pi*x)*torch.sin(10*np.pi*y)*torch.sin(7*np.pi*z)
        elif sys.argv[1] == "2":
            lap_u = torch.sin(10*np.pi*x)*torch.sin(13*np.pi*y)*torch.sin(5*np.pi*z)
        elif sys.argv[1] == "3":
            lap_u = torch.sin(15*np.pi*x)*torch.sin(12*np.pi*y)*torch.sin(8*np.pi*z)
        elif sys.argv[1] == "4":
            lap_u = torch.sin(8*np.pi*x)**2*torch.sin(15*np.pi*y)*torch.sin(6*np.pi*z)**3
        elif sys.argv[1] == "5":
            lap_u = x - y + z
        elif sys.argv[1] == "6":
            lap_u = 4*x + 2*y - 3*z
        elif sys.argv[1] == "7":
            lap_u = -1*x + 2*y**2 - z**3
        elif sys.argv[1] == "8":
            lap_u = 100*x + 234*y - 457*z
        elif sys.argv[1] == "9":
            lap_u = 100*x**2 + 234*y**3 + 457*z**4
        elif sys.argv[1] == "10":
            lap_u = -100*x**8 + 234*y**6 + 457*z**9
        else:
            raise NotImplementedError()

    return lap_u.squeeze()

# Laplacian of f (numpy version)
def rhs_np(v):
    if not have_bdry:
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        if sys.argv[1] == "1":
            lap_u = np.sin(9*np.pi*x)*np.sin(10*np.pi*y)*np.sin(7*np.pi*z)
        elif sys.argv[1] == "2":
            lap_u = np.sin(10*np.pi*x)*np.sin(13*np.pi*y)*np.sin(5*np.pi*z)
        elif sys.argv[1] == "3":
            lap_u = np.sin(15*np.pi*x)*np.sin(12*np.pi*y)*np.sin(8*np.pi*z)
        elif sys.argv[1] == "4":
            lap_u = np.sin(8*np.pi*x)**2*np.sin(15*np.pi*y)*np.sin(6*np.pi*z)**3
        elif sys.argv[1] == "5":
            lap_u = x - y + z
        elif sys.argv[1] == "6":
            lap_u = 4*x + 2*y - 3*z
        elif sys.argv[1] == "7":
            lap_u = -1*x + 2*y**2 - z**3
        elif sys.argv[1] == "8":
            lap_u = 100*x + 234*y - 457*z
        elif sys.argv[1] == "9":
            lap_u = 100*x**2 + 234*y**3 + 457*z**4
        elif sys.argv[1] == "10":
            lap_u = -100*x**8 + 234*y**6 + 457*z**9
        else:
            raise NotImplementedError()
        
    else:
        lap_u = get_interpolated_values(rhs_mesh, v, v_mesh, f_mesh)

    return lap_u.squeeze()
    
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
    return n_pred

def train_strong_form(l_model, device, n, size_layer, n_layers):
    optimizer = torch.optim.Adam(l_model.parameters(), lr=config["architecture"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=config["architecture"]["scheduler_patience"])

    surf_normal_model = MLP_normals(n=64, n_layers=5, in_dim=3, out_dim=3)
    surf_normal_model.to(device=device)
    surf_normal_model.load_state_dict(torch.load(f"../data/model_{config['surface']}_normal.pth", weights_only=True, map_location=device))
    surf_normal_model.requires_grad_(True)
    
    losses = []
    for i in (pbar:= tqdm(range(config["architecture"]["max_iter"]))):
        try:
            optimizer.zero_grad()

            if i % 5000 == 0:
                v_mesh_global = sample_in_domain(int(1e6))
                v_mesh_global = torch.tensor(v_mesh_global, dtype=torch.float32, device=device).requires_grad_(True)
                true_lap_global = rhs_np(v_mesh_global.detach().cpu().numpy())
                true_lap_global = torch.tensor(true_lap_global, dtype=torch.float32, device=device)

            idx = torch.randint(0, int(1e6), (n,), device=device)
            v_cart = v_mesh_global[idx]

            normals = get_normals(v_cart, surf_normal_model)
            laplacian_pred = get_surface_laplacian(l_model, v_cart, normals)

            if not have_bdry:
                true_lap = rhs(v_cart)
                # Enforce zero-mean RHS
                true_lap = true_lap - mean_f

            else:
                true_lap = true_lap_global[idx]
                true_lap = torch.tensor(true_lap, dtype=torch.float32, device=device)

            loss = torch.linalg.norm(laplacian_pred - true_lap, 2)**2
            

            ### BOUNDARY CONDITIONS ###
            if have_bdry:
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

                elif config["bc"] == "neumann":
                    bdry_normal_model = MLP_normals(n=64, n_layers=5, in_dim=3, out_dim=3)
                    bdry_normal_model.to(device=device)
                    bdry_normal_model.load_state_dict(torch.load(f"../data/model_{config['surface']}_bdry_normal.pth", weights_only=True, map_location=device))
                    bdry_normal_model.requires_grad_(True)
                    
                    # Get boundary points
                    bdry_edges_rdm = boundary_edges[torch.randint(0, boundary_edges.shape[0], (n,), device=device)]
                    w = torch.rand((n, 1), device=device) # linear interpolation weights
                    bdry_points = (1 - w) * v_mesh_torch[bdry_edges_rdm[:, 0]] + w * v_mesh_torch[bdry_edges_rdm[:, 1]]

                    w = w.squeeze()
                    bdry_normal = get_normals(bdry_points, bdry_normal_model)

                    grad_bdry = torch.autograd.grad(l_model(bdry_points).squeeze(), bdry_points, torch.ones_like(bdry_points[:, 0]), create_graph=True, retain_graph=True)[0]
                    n_surf_bdry = get_normals(bdry_points, surf_normal_model)
                    grad_bdry_surface = grad_bdry - torch.sum(grad_bdry * n_surf_bdry, dim=1, keepdim=True) * n_surf_bdry

                    flux_pred = torch.sum(grad_bdry_surface * bdry_normal, dim=1)

                    loss_bdry = torch.linalg.norm(flux_pred, 2)**2
                    loss = loss + 100*loss_bdry

                else:
                    raise NotImplementedError("BC not implemented")
            
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
    true = u_numpy(V)

    for i in n_layers:
        for j in size_layer:
            print(f"Processing model with {i} layers of {j} size")

            lap_model = MLP(n=j, n_layers=i)
            lap_model.to(device=device)

            lap_model = train_strong_form(lap_model, device, n=config["architecture"]["num_samples"], size_layer=j, n_layers=i)

            pred = lap_model(torch.Tensor(V).to(device=device)).squeeze().detach().cpu().numpy()

            const = np.mean(true - pred)
            pred = pred + const
            
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
        true = u_numpy(V)

        const = np.mean(true - pred)
        pred = pred + const

        l2_loss = np.linalg.norm(pred - true, 2) / np.linalg.norm(true, 2)
        print(l2_loss, n_layers, size_layer)
        losses.append(l2_loss)

        total_trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        dof.append(total_trainable_params)

        # Plot the error plots using polyscope
        ps.init()
        ps.set_screenshot_extension(".png")
        ps.register_point_cloud("Surface Points", V)
        ps.get_point_cloud("Surface Points").add_scalar_quantity("True Solution", true, enabled=True)
        ps.screenshot(f"{save_dir}/example{sys.argv[1]}/true_{size_layer}_{n_layers}.png")

        ps.get_point_cloud("Surface Points").add_scalar_quantity("Predicted Solution", pred, enabled=True)
        ps.screenshot(f"{save_dir}/example{sys.argv[1]}/pred_{size_layer}_{n_layers}.png")

        ps.get_point_cloud("Surface Points").add_scalar_quantity("Error", np.abs(true - pred), enabled=True)
        ps.screenshot(f"{save_dir}/example{sys.argv[1]}/error_{size_layer}_{n_layers}.png")

    fem_loss = []
    fem_dof = []
    for iter in range(4):
        v_mesh_fem, f_mesh_fem = gpy.read_mesh(f"../data/{config['surface']}_{iter}.obj")
        print(f"FEM Mesh {iter}: {v_mesh_fem.shape[0]} vertices, {f_mesh_fem.shape[0]} faces")
        if not have_bdry:
            lap_u_fem = rhs_np(v_mesh_fem)
            lap_u_fem -= mean_f

            M_fem = gpy.massmatrix(v_mesh_fem, f_mesh_fem)
            M_fem = -sp.sparse.csc_matrix(M_fem)
            
            L_fem = gpy.cotangent_laplacian(v_mesh_fem, f_mesh_fem)

            u_fem = spsolve(L_fem, M_fem @ lap_u_fem)

            const = np.mean(u_numpy(v_mesh_fem) - u_fem)
            u_fem = u_fem + const
        
        else:
            lap_u_fem = rhs_np(v_mesh_fem)

            M_mesh_fem = gpy.massmatrix(v_mesh_fem, f_mesh_fem)
            M_mesh_fem = sp.sparse.csc_matrix(M_mesh_fem)
            f_ = -M_mesh_fem @ lap_u_fem

            BV = gpy.boundary_vertices(f_mesh_fem)
            
            lap_mesh = gpy.cotangent_laplacian(v_mesh_fem, f_mesh_fem)

            # Apply Dirichlet Boundary
            # Section 4.3 - https://web.stanford.edu/class/energy281/FiniteElementMethod.pdf
            if config["bc"] == "dirichlet":
                for i in BV:
                    lap_mesh[i, :] = 0
                    lap_mesh[i, i] = 1
                f_[BV] = u_numpy(v_mesh_fem[BV])

            u_fem = sp.sparse.linalg.spsolve(lap_mesh, f_)

            if config["bc"] == "neumann":
                const = np.mean(u_numpy(v_mesh_fem) - u_fem)
                u_fem = u_fem + const
        
        l2_loss = np.linalg.norm(u_numpy(v_mesh_fem) - u_fem, 2) / np.linalg.norm(u_numpy(v_mesh_fem), 2)
        print("FEM Loss:", l2_loss)
        fem_loss.append(l2_loss)
        fem_dof.append(v_mesh_fem.shape[0])

    dof = np.array(dof)
    losses = np.array(losses)
    fem_dof = np.array(fem_dof)
    fem_loss = np.array(fem_loss)

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
    plt.loglog(fem_dof, fem_loss, label="FEM", marker='o')
    plt.loglog(x, 1/x, label="1/x", color="red", linestyle='--')
    plt.loglog(x, (1/x)**2, label="1/x^2", color="green", linestyle='--')

    plt.xlabel("Total Trainable Parameters")
    plt.ylabel("Relative L2 loss")
    plt.title(f"Slope: {coeffs[0]}")
    plt.legend()
    plt.savefig(f"{save_dir}/convergence_example_{sys.argv[1]}.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v_mesh, f_mesh = gpy.read_mesh(f"../data/{config['surface']}.obj")

     # Discrete Laplace-Beltrami
    L = gpy.cotangent_laplacian(v_mesh, f_mesh)                # stiffness matrix
    L = scipy.sparse.csc_matrix(L)
    M = gpy.massmatrix(v_mesh, f_mesh, type='barycentric')     # area weights
    M = scipy.sparse.csc_matrix(M)

    # Get boundary loop
    bdry_loop = gpy.boundary_loops(f_mesh)

    if len(bdry_loop) == 0:
        have_bdry = False
        # Discrete Laplacian of f
        lap_u_ = rhs_np(v_mesh)
        
        # Enforce discrete compatibility: 1^T M f = 0
        area = M.sum()
        mean_f = np.ones_like(lap_u_) @ (M @ lap_u_) / area
        lap_u_ -= mean_f

        # Discrete surface Laplacian of u
        u_mesh = spsolve(L, -M @ lap_u_)   # equivalent to -M^{-1} L u

    elif len(bdry_loop) == 1: # single boundary loop
        bdry_loop = bdry_loop[0]
        have_bdry = True
        boundary_edges = gpy.boundary_edges(f_mesh)
        boundary_edges = torch.tensor(boundary_edges, dtype=torch.long, device=device)

        k = int(sys.argv[1])
        eigvals, eigvecs = eigsh(L, M=M, k=k+1, sigma=0.0)

        u_mesh = eigvecs[:,int(sys.argv[1])] # eigenfunction
        lam = eigvals[int(sys.argv[1])]

        rhs_mesh = -lam * u_mesh

    else:
        print("Multiple boundary loops not supported")
        raise NotImplementedError()
    
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