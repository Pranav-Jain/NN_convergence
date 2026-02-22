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
from scipy.spatial import cKDTree

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
        elif sys.argv[1] == "13":
            lap_u = torch.sin(30*np.pi*x)*torch.sin(23*np.pi*y)*torch.sin(50*np.pi*z)
        else:
            raise NotImplementedError()

    return lap_u.squeeze()

# Laplacian of f (numpy version)
def rhs_np(v):
    if not have_bdry and config['experiment'] != 'montecarlo':
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
        elif sys.argv[1] == "11" and config['surface'] == 'cat' and config['pde'] == 'helmholtz' and config['experiment'] == 'heat': # For the heat equation test
            lap_u = np.zeros(v.shape[0])
            src_indices1 = np.array([31754, 31737, 31714, 31697, 31670, 7911, 31671, 31700, 31717, 31742, 31756, 7938, 31740, 7925, 31698, 31699, 7926, 31741, 31716])
            src_indices2 = np.array([3102, 16906, 2866, 16217, 2645, 15859, 2646, 16221, 2868, 16907, 3103, 17234, 16905, 16531, 16219, 16218, 16532, 16908, 2867])
            src_indices3 = np.array([12273, 12276, 12275, 12495, 12264, 12250, 1353, 11831, 11652, 11651, 11453, 11454, 11452, 11656, 11848, 12065, 1362, 12066, 1439, 12060, 11841, 1299, 11657, 1224, 11850, 11851, 12059])    
            source_indices = np.concatenate([src_indices1, src_indices2, src_indices3])
            A = 0.1
            sigma = 0.01

            tree = cKDTree(v)

            for i in source_indices:
                center = v_mesh[i]
                idx = tree.query_ball_point(center, r=sigma)
                d2 = np.sum((v[idx] - center)**2, axis=1)
                weights = np.exp(-d2 / (2 * sigma**2))
                weights /= np.sum(weights)
                lap_u[idx] = config['k'] * A * weights

        elif sys.argv[1] == "12" and config['surface'] == 'mushroom' and config['pde'] == 'helmholtz' and config['experiment'] == 'sound': # For the sound propagation test
            lap_u = np.zeros(v.shape[0])
            c = 1000
            A = 3*1e7
            sigma = 0.2
            source_indices = np.array([184164, 184166, 38770, 135718, 135335, 38661, 184165])

            tree = cKDTree(v)

            for i in source_indices:
                center = v_mesh[i]
                idx = tree.query_ball_point(center, r=sigma)
                d2 = np.sum((v[idx] - center)**2, axis=1)
                weights = np.exp(-d2 / (2 * sigma**2))
                weights /= np.sum(weights)
                lap_u[idx] = -c**(-2) * A * weights

        elif sys.argv[1] == "13" and config['pde'] == 'helmholtz' and config['experiment'] == 'heat':
            lap_u = x
        elif sys.argv[1] == "14" and config['pde'] == 'helmholtz' and config['experiment'] == 'heat':
            lap_u = y
        elif sys.argv[1] == "15" and config['pde'] == 'helmholtz' and config['experiment'] == 'heat':
            lap_u = z
            
        else:
            raise NotImplementedError()
        
    else:
        lap_u = get_interpolated_values(rhs_mesh, v, v_mesh, f_mesh)

    return lap_u.squeeze()

def get_random_points_Niloy(v_mesh, f_mesh, v_emb, f_emb, n):
    v_emb_rdm, ind_emb_rdm, bary_emb_rdm  = gpy.random_points_on_mesh(v_emb, f_emb, n, return_indices=True)
    v_mesh_rdm = np.zeros_like(v_emb_rdm)
    for i in range(v_emb_rdm.shape[0]):
        v = v_mesh[f_emb[ind_emb_rdm[i]]]
        for j in range(v.shape[0]):
            v_mesh_rdm[i] += v[j] * bary_emb_rdm[i][j]

    return v_mesh_rdm, v_emb_rdm
    
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

def get_normals_Niloy(v_c, v_emb, S_theta=None):

    J = torch.vmap(torch.func.jacrev(S_theta))(v_emb)
    
    cross = torch.linalg.cross(J[:, :, 0], J[:, :, 1])
    n = cross / torch.linalg.norm(cross, dim=1, keepdim=True)

    return n


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

    if config['experiment'] == 'Niloy':
        emb_model = MLP(n=64, n_layers=5, in_dim=3, out_dim=3)
        emb_model.to(device=device)
        emb_model.load_state_dict(torch.load(f"../data/model_sphere_{config['surface']}.pth", weights_only=True, map_location=device))
        emb_model.requires_grad_(True)
    
    losses = []
    for i in (pbar:= tqdm(range(config["architecture"]["max_iter"]))):
        try:
            optimizer.zero_grad()

            if i % 5000 == 0:
                if config['experiment'] == 'Niloy':
                    v_mesh_global, v_emb_global = get_random_points_Niloy(v_mesh, f_mesh, v_emb, f_emb, int(1e6))
                    v_emb_global = torch.tensor(v_emb_global, dtype=torch.float32, device=device).requires_grad_(True)
                else:
                    v_mesh_global = sample_in_domain(int(1e6))
                v_mesh_global = torch.tensor(v_mesh_global, dtype=torch.float32, device=device).requires_grad_(True)
                true_lap_global = rhs_np(v_mesh_global.detach().cpu().numpy())
                true_lap_global = torch.tensor(true_lap_global, dtype=torch.float32, device=device)

            idx = torch.randint(0, int(1e6), (n,), device=device)
            v_cart = v_mesh_global[idx]

            if config['experiment'] == 'heat' and config['surface'] == 'cat':
                src_indices1 = np.array([31754, 31737, 31714, 31697, 31670, 7911, 31671, 31700, 31717, 31742, 31756, 7938, 31740, 7925, 31698, 31699, 7926, 31741, 31716])
                src_indices2 = np.array([3102, 16906, 2866, 16217, 2645, 15859, 2646, 16221, 2868, 16907, 3103, 17234, 16905, 16531, 16219, 16218, 16532, 16908, 2867])
                src_indices3 = np.array([12273, 12276, 12275, 12495, 12264, 12250, 1353, 11831, 11652, 11651, 11453, 11454, 11452, 11656, 11848, 12065, 1362, 12066, 1439, 12060, 11841, 1299, 11657, 1224, 11850, 11851, 12059])    
                source_indices = np.concatenate([src_indices1, src_indices2, src_indices3])
                v_mesh_src = torch.tensor(v_mesh[source_indices], dtype=torch.float32, device=device)
                v_cart = torch.cat([v_cart, v_mesh_src], dim=0)

            if config['experiment'] == 'sound' and config['surface'] == 'mushroom':
                source_indices = np.array([184164, 184166, 38770, 135718, 135335, 38661, 184165])
                v_mesh_src = torch.tensor(v_mesh[source_indices], dtype=torch.float32, device=device)
                v_cart = torch.cat([v_cart, v_mesh_src], dim=0)

            if config['experiment'] == 'Niloy':
                normals = get_normals_Niloy(v_cart, v_emb_global[idx],  emb_model)
            else:
                normals = get_normals(v_cart, surf_normal_model)
            laplacian_pred = get_surface_laplacian(l_model, v_cart, normals)

            if config['pde'] == "helmholtz":
                laplacian_pred = laplacian_pred + (config['k'] * l_model(v_cart).squeeze())

            if not have_bdry and (config['experiment'] != 'heat' and config['experiment'] != 'sound' and config['experiment'] != 'montecarlo'):
                true_lap = rhs(v_cart)
                if config['pde'] == "poisson":
                    # Enforce zero-mean RHS
                    true_lap = true_lap - mean_f

            else:
                true_lap = true_lap_global[idx]
                true_lap = torch.tensor(true_lap, dtype=torch.float32, device=device)

                if config['experiment'] == 'heat' and config['surface'] == 'cat':
                    src_indices1 = np.array([31754, 31737, 31714, 31697, 31670, 7911, 31671, 31700, 31717, 31742, 31756, 7938, 31740, 7925, 31698, 31699, 7926, 31741, 31716])
                    src_indices2 = np.array([3102, 16906, 2866, 16217, 2645, 15859, 2646, 16221, 2868, 16907, 3103, 17234, 16905, 16531, 16219, 16218, 16532, 16908, 2867])
                    src_indices3 = np.array([12273, 12276, 12275, 12495, 12264, 12250, 1353, 11831, 11652, 11651, 11453, 11454, 11452, 11656, 11848, 12065, 1362, 12066, 1439, 12060, 11841, 1299, 11657, 1224, 11850, 11851, 12059])    
                    source_indices = np.concatenate([src_indices1, src_indices2, src_indices3])
                    v_mesh_src = torch.tensor(rhs_np(v_mesh[source_indices]), dtype=torch.float32, device=device)
                    true_lap = torch.cat([true_lap, v_mesh_src], dim=0)

                if config['experiment'] == 'sound' and config['surface'] == 'mushroom':
                    source_indices = np.array([184164, 184166, 38770, 135718, 135335, 38661, 184165])
                    v_mesh_src = torch.tensor(rhs_np(v_mesh[source_indices]), dtype=torch.float32, device=device)
                    true_lap = torch.cat([true_lap, v_mesh_src], dim=0)

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
    plt.plot(np.log(losses), linewidth=2)
    plt.title("Loss")
    plt.xlabel("Iteration")
    plt.ylabel("log(Loss)")
    plt.gca().set_facecolor('#f0f0f0')  # set gray bg color
    plt.grid(color='white', linestyle='-', linewidth=1.5)
    plt.title(f"surface = {config['surface']}, {size_layer} x {n_layers}")
    plt.savefig(f"{save_dir}/example{sys.argv[1]}/loss_{size_layer}_{n_layers}.png")

    return l_model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_layers = [config["architecture"]["num_layers"]]
    # size_layer = np.arange(30, 160, 20)
    size_layer = [150]

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
    dof_depth = []
    losses = []
    losses_depth = []

    V = v_mesh

    min_pred = []
    max_pred = []
    min_err = []
    max_err = []
    for file in filenames:
        print(file)
        n_layers = int(file.split("_")[-1].split(".")[0])
        size_layer = int(file.split("_")[-2])

        model = MLP(n=size_layer, n_layers=n_layers, in_dim=3, out_dim=1)
        model.to(device=device)
        model.load_state_dict(torch.load(file, weights_only=True, map_location=device))

        pred = model(torch.Tensor(V).to(device=device)).squeeze().detach().cpu().numpy()
        true = u_numpy(V)

        if config['bc']=='neumann' or not have_bdry:
            const = np.mean(true - pred)
            pred = pred + const

        l2_loss = np.linalg.norm(pred - true, 2) / np.linalg.norm(true, 2)
        print(l2_loss, n_layers, size_layer)

        total_trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        if n_layers != 3 and config["surface"] == "mushroom":
            dof_depth.append(total_trainable_params)
            losses_depth.append(l2_loss)
        else:
            dof.append(total_trainable_params)
            losses.append(l2_loss)
        if n_layers == 3 and size_layer == 90 and config["surface"] == "mushroom":
            dof_depth.append(total_trainable_params)
            losses_depth.append(l2_loss)
        
        # output numpy array of true, pred and error. Required for blender rendering
        if not os.path.exists(f"{save_dir}/example{sys.argv[1]}/npy_files/"):
            os.makedirs(f"{save_dir}/example{sys.argv[1]}/npy_files/")
        np.save(f"{save_dir}/example{sys.argv[1]}/npy_files/true_{size_layer}_{n_layers}.npy", true)
        np.save(f"{save_dir}/example{sys.argv[1]}/npy_files/pred_{size_layer}_{n_layers}.npy", pred)
        np.save(f"{save_dir}/example{sys.argv[1]}/npy_files/error_{size_layer}_{n_layers}.npy", np.abs(true - pred))
        np.save(f"{save_dir}/example{sys.argv[1]}/npy_files/rhs.npy", rhs_np(V))

        # Output min/max pred and error across all models for color mapping in blender
        min_pred.append(true.min())
        max_pred.append(true.max())
        min_err.append(np.abs(true - pred).min())
        max_err.append(np.abs(true - pred).max())
    
    np.save(f"{save_dir}/example{sys.argv[1]}/npy_files/min_max_pred.npy", np.array([max(min_pred), min(max_pred)]))
    np.save(f"{save_dir}/example{sys.argv[1]}/npy_files/min_max_err.npy", np.array([max(min_err), min(max_err)]))

    fem_loss = []
    fem_dof = []
    for iter in range(6):
        try:
            v_mesh_fem, f_mesh_fem = gpy.read_mesh(f"../data/{config['surface']}_{iter}.obj")
        except:
            break
        print(f"FEM Mesh {iter}: {v_mesh_fem.shape[0]} vertices, {f_mesh_fem.shape[0]} faces")
        if not have_bdry:
            lap_u_fem = rhs_np(v_mesh_fem)
            
            if config['pde'] == "poisson":
                lap_u_fem -= mean_f

            M_fem = gpy.massmatrix(v_mesh_fem, f_mesh_fem)
            M_fem = sp.sparse.csc_matrix(M_fem)
            
            L_fem = gpy.cotangent_laplacian(v_mesh_fem, f_mesh_fem)

            if config['pde'] == "helmholtz":
                L_fem = L_fem - (config['k'] * M_fem)

            u_fem = spsolve(L_fem, -M_fem @ lap_u_fem)

            const = np.mean(u_numpy(v_mesh_fem) - u_fem)
            u_fem = u_fem + const

            np.save(f"{save_dir}/example{sys.argv[1]}/npy_files/fem_pred_{iter}.npy", u_fem)
            np.save(f"{save_dir}/example{sys.argv[1]}/npy_files/fem_error_{iter}.npy", np.abs(u_numpy(v_mesh_fem) - u_fem))
        
        else:
            lap_u_fem = rhs_np(v_mesh_fem)

            M_mesh_fem = gpy.massmatrix(v_mesh_fem, f_mesh_fem)
            M_mesh_fem = sp.sparse.csc_matrix(M_mesh_fem)
            f_ = -M_mesh_fem @ lap_u_fem

            BV = gpy.boundary_vertices(f_mesh_fem)
            
            lap_mesh = gpy.cotangent_laplacian(v_mesh_fem, f_mesh_fem)

            if config['pde'] == "helmholtz":
                lap_mesh = lap_mesh + (config['k'] * -M_mesh_fem)

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
    print(dof_sorted, losses_sorted)

    if config["surface"] == "mushroom":
        dof_depth = np.array(dof_depth)
        losses_depth = np.array(losses_depth)

        dof_depth_sorted = np.sort(dof_depth)
        losses_depth_sorted = losses_depth[np.argsort(dof_depth)]


    # Create DataFrame for this run
    try:
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
    except:
        print("Error saving CSV")

    x = dof_sorted
    y = losses_sorted

    #fit a line to the data
    coeffs = np.polyfit(np.log(x), np.log(y), 1)

    plt.loglog(x, y, label="PINN", marker='o', linewidth=2)

    if config["surface"] == "mushroom":
        plt.loglog(dof_depth_sorted, losses_depth_sorted, label="PINN (depth)", marker='o', linewidth=2, color='green')

    plt.loglog(fem_dof, fem_loss, label="FEM", marker='o', linewidth=2)
    plt.loglog(x, 0.9*x[0]*y[0]*1/x, color='red', label="1/x", linestyle='--')

    plt.xlabel("Total Trainable Parameters")
    plt.gca().set_facecolor('#f0f0f0')  # set gray bg color
    plt.grid(color='white', linestyle='-', linewidth=1.5)
    
    # plt.ylabel("Relative L2 loss")
    plt.title(f"Slope: {coeffs[0]:.2f}")
    x_ = [x[0],x[len(x)//2], x[-1]]
    y_ = [y[0],y[len(y)//2], y[-1]]
    plt.xticks(ticks=x_, labels=['{:.1e}'.format(i) for i in x_], fontsize=10)
    plt.yticks(ticks=y_, labels=['{:.1e}'.format(i) for i in y_], fontsize=10)
    plt.savefig(f"{save_dir}/convergence_example_{sys.argv[1]}.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v_mesh, f_mesh = gpy.read_mesh(f"../data/{config['surface']}.obj")

    if config['experiment'] == 'Niloy':
        v_emb, f_emb = gpy.read_mesh(f"../data/sphere_{config['surface']}.obj")

     # Discrete Laplace-Beltrami
    L = gpy.cotangent_laplacian(v_mesh, f_mesh)                # stiffness matrix
    L = scipy.sparse.csc_matrix(L)
    M = gpy.massmatrix(v_mesh, f_mesh, type='barycentric')     # area weights
    M = scipy.sparse.csc_matrix(M)

    # Get boundary loop
    bdry_loop = gpy.boundary_loops(f_mesh)

    if len(bdry_loop) == 0:
        have_bdry = False

        if config['experiment'] == 'montecarlo' and config['pde'] == 'helmholtz': # monte carlo test
            eigvals, eigvecs = eigsh(L, M=M, k=100, sigma=0.0)
            if np.any(np.abs(eigvals - np.abs(config['k'])) < 1e-1):
                print("k too close to an eigenvalue for Helmholtz PDE")
                exit()
            config['k'] = -1
            x, y, z = v_mesh[:, 0], v_mesh[:, 1], v_mesh[:, 2]
            u_mesh = x*y
            rhs_mesh = y*(-16*x**5*z**4 - 8*x**5*z**2 - x**5 + 64*x**4*z**6 + 64*x**4*z**4 + 12*x**4*z**2 + 4*x**4 - 8*x**3*y**2*z**2 - 2*x**3*y**2 - 96*x**3*z**8 - 144*x**3*z**6 - 54*x**3*z**4 - 32*x**3*z**2 - 6*x**3 + 16*x**2*y**2*z**4 + 12*x**2*y**2*z**2 + 4*x**2*y**2 + 64*x**2*z**10 + 128*x**2*z**8 + 76*x**2*z**6 + 74*x**2*z**4 + 32*x**2*z**2 - x*y**4 - 8*x*y**2*z**6 - 10*x*y**2*z**4 - 16*x*y**2*z**2 - 6*x*y**2 - 16*x*z**12 - 40*x*z**10 - 33*x*z**8 - 60*x*z**6 - 43*x*z**4 - 6*x*z**2 + 6*y**2*z**4 + 4*y**2*z**2 + 14*z**8 + 16*z**6 + 4*z**4)/(16*x**4*z**4 + 8*x**4*z**2 + x**4 - 64*x**3*z**6 - 64*x**3*z**4 - 12*x**3*z**2 + 8*x**2*y**2*z**2 + 2*x**2*y**2 + 96*x**2*z**8 + 144*x**2*z**6 + 54*x**2*z**4 + 2*x**2*z**2 - 16*x*y**2*z**4 - 12*x*y**2*z**2 - 64*x*z**10 - 128*x*z**8 - 76*x*z**6 - 12*x*z**4 + y**4 + 8*y**2*z**6 + 10*y**2*z**4 + 2*y**2*z**2 + 16*z**12 + 40*z**10 + 33*z**8 + 10*z**6 + z**4)
        # Discrete Laplacian of f
        lap_u_ = rhs_np(v_mesh)
        
        if config['pde'] == "poisson":
            # Enforce discrete compatibility: 1^T M f = 0
            area = M.sum()
            mean_f = np.ones_like(lap_u_) @ (M @ lap_u_) / area
            lap_u_ -= mean_f

            # Discrete surface Laplacian of u
            u_mesh = spsolve(L, -M @ lap_u_)   # equivalent to -M^{-1} L u

        elif config['pde'] == "helmholtz" and config['experiment'] != 'montecarlo':
            eigvals, eigvecs = eigsh(L, M=M, k=100, sigma=0.0)
            if np.any(np.abs(eigvals - np.abs(config['k'])) < 1e-1):
                print("k too close to an eigenvalue for Helmholtz PDE")
                exit()
            A = L - config['k'] * M
            u_mesh = spsolve(A, -M @ lap_u_)

    elif len(bdry_loop) > 0:
        have_bdry = True
        boundary_edges = gpy.boundary_edges(f_mesh)
        boundary_edges = torch.tensor(boundary_edges, dtype=torch.long, device=device)

        if config['experiment'] == 'minimal_surface' and config['pde'] == 'poisson': # minimal surface test
                u_mesh = v_mesh[:, int(sys.argv[1])-1]
                rhs_mesh = np.zeros(v_mesh.shape[0])

        elif config['experiment'] == 'interpolation' and config['pde'] == 'poisson' and config['surface'] == 'plane': # interpolation test
            bdry_loops = gpy.boundary_loops(f_mesh)
            rhs_mesh = np.zeros(v_mesh.shape[0])

            u_mesh = np.zeros(v_mesh.shape[0])
            tree = cKDTree(v_mesh)
            for idx, loop in enumerate(bdry_loops):
                u_mesh[loop] = 2*idx + 1
                # spread the boundary values to the interior using gaussian filter
                
                sigma = 0.08
                for i in loop:
                    center = v_mesh[i]
                    idxs = tree.query_ball_point(center, r=sigma)
                    u_mesh[idxs] = 2*idx + 1
        else:
            k = int(sys.argv[1])
            eigvals, eigvecs = eigsh(L, M=M, k=k+1, sigma=0.0)

            u_mesh = eigvecs[:,int(sys.argv[1])] # eigenfunction
            lam = eigvals[int(sys.argv[1])]

            if config['pde'] == "poisson":
                rhs_mesh = -lam * u_mesh
            elif config['pde'] == "helmholtz":
                if np.abs(lam - config['k']) < 1e-1:
                    print("k too close to an eigenvalue for Helmholtz PDE")
                    exit()
                rhs_mesh = (-lam + config['k']) * u_mesh
            else:
                raise ValueError("Unsupported PDE type")

    else:
        print("Multiple boundary loops not supported")
        raise NotImplementedError()
    
    v_mesh_torch = torch.tensor(v_mesh, dtype=torch.float32, device=device).requires_grad_(True)
    u_mesh_torch = torch.tensor(u_mesh, dtype=torch.float32, device=device).requires_grad_(True)
        
    save_dir = f"{config['pde']}_results/{config['surface']}/{config['bc']}"

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