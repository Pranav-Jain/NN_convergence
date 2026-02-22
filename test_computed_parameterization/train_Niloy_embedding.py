import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import gpytoolbox as gpy
import torch
from tqdm import tqdm
import math
import torch.nn as nn
import sys

sys.path.append('../src')
from siren import MLP

def get_spherical_coordinates(v, c):
    r = np.ones(v.shape[0])
    theta = np.arccos((v[:, 2] - c[2]) / r)
    phi = np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0])

    return np.stack([r, theta, phi], axis=1)

def get_spherical_coordinates_torch(v):
    r = torch.ones(v.shape[0], device=v.device)
    theta = torch.arccos((v[:, 2]) / r)
    if torch.isnan(theta).any():
        theta = torch.nan_to_num(theta)
    phi = torch.arctan2(v[:, 1], v[:, 0])

    r = r.requires_grad_(True)
    theta = theta.requires_grad_(True)
    phi = phi.requires_grad_(True)

    return torch.stack([r, theta, phi], dim=1)

def get_rotation_matrix(theta, phi):
    R = torch.zeros(theta.shape[0], 3, 2).to(theta.device)
    R[:, 0, 0] = torch.cos(theta) * torch.cos(phi)
    R[:, 0, 1] = -torch.sin(phi)
    R[:, 1, 0] = torch.cos(theta) * torch.sin(phi)
    R[:, 1, 1] = torch.cos(phi)
    R[:, 2, 0] = -torch.sin(theta)

    return R

def get_random_points(v_mesh, f_mesh, v_emb, f_emb, n):
    # Sample points on *embedding* mesh
    v_emb_rdm, ind_emb_rdm, bary_emb_rdm = gpy.random_points_on_mesh(
        v_emb, f_emb, n, return_indices=True
    )

    # Gather corresponding vertices on the *other* mesh
    # Shape: (n, 3, dim)
    tri_vertices = v_mesh[f_emb[ind_emb_rdm]]

    # Apply barycentric weights
    # Shape: (n, 3) -> (n, 3, 1), broadcast multiply & sum over vertices
    v_mesh_rdm = (tri_vertices * bary_emb_rdm[:, :, None]).sum(axis=1)

    return v_mesh_rdm, v_emb_rdm

def get_interpolated_values(f, v, v_mesh, f_mesh):
    # Compute nearest triangle & barycentric coordinates
    _, idx, bary = gpy.signed_distance(v, v_mesh, f_mesh, use_cpp=True)

    # Gather triangle values of f
    # f_mesh[idx] gives shape (n, 3), then indexing into f → (n, 3, fdim)
    face_vals = f[f_mesh[idx]]

    # Weighted sum
    f_v = (face_vals * bary[:, :, None]).sum(axis=1)
    return f_v

def loss(model, v, v_emb, true_n):
    v_pred = model(v_emb)
    l = torch.mean(torch.linalg.norm(v_pred - v, ord=2, axis=1)**2)

    v_sp = get_spherical_coordinates_torch(v_emb)
    J = torch.vmap(torch.func.jacrev(model))(v_emb)

    R = get_rotation_matrix(v_sp[:, 1], v_sp[:, 2])
    J_local = torch.bmm(J, R)
    cross = torch.linalg.cross(J_local[:, :, 0], J_local[:, :, 1])
    n = cross / torch.linalg.norm(cross, dim=1, keepdim=True)

    l = l + torch.mean(torch.linalg.norm(n - true_n, ord=2, axis=1)**2)

    return l

def train_mesh(v_mesh, f_mesh, v_emb, f_emb, n_layers = 5, size_layer = 64, lr=1e-4, max_iter=5000000, n_samples=10000, tol=1e-8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MLP(n=size_layer, n_layers=n_layers, in_dim=3, out_dim=3)
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    normals = gpy.per_vertex_normals(v_mesh, f_mesh)

    v_mesh_rnd_global, v_emb_rnd_global = get_random_points(v_mesh, f_mesh, v_emb, f_emb, int(1e6))
    normals_rnd_global = get_interpolated_values(normals, v_mesh_rnd_global, v_mesh, f_mesh)

    v_mesh_rnd_global = torch.tensor(v_mesh_rnd_global, dtype=torch.float32, device=device)
    v_emb_rnd_global = torch.tensor(v_emb_rnd_global, dtype=torch.float32, device=device)
    normals_rnd_global = torch.tensor(normals_rnd_global, dtype=torch.float32, device=device)

    # Train the model
    losses = []
    for i in (pbar:= tqdm(range(max_iter))):
        try:
            optimizer.zero_grad()

            idx = torch.randint(0, int(1e6), (n_samples,), device=device)
            v_mesh_rnd = v_mesh_rnd_global[idx]
            v_emb_rnd = v_emb_rnd_global[idx]
            normals_rnd = normals_rnd_global[idx]

            l = loss(model, v_mesh_rnd, v_emb_rnd, normals_rnd)

            l.backward()
            optimizer.step()
            
            loss_value = l.item()
            pbar.set_description(f"Loss: {loss_value}")
            losses.append(loss_value)

        except:
            break

    plt.figure()
    plt.plot(np.log(losses), linewidth=2)
    plt.title("Loss")
    plt.xlabel("Iteration")
    plt.ylabel("log(Loss)")
    plt.gca().set_facecolor('#f0f0f0')  # set gray bg color
    plt.grid(color='white', linestyle='-', linewidth=1.5)
    plt.savefig(f"../example_{sys.argv[1]}_loss_Niloy.png")

    return model

if __name__ == "__main__":
    v_emb, f_emb = gpy.read_mesh(f"../data/sphere_{sys.argv[1]}.obj")
    v_mesh, f_mesh = gpy.read_mesh(f"../data/{sys.argv[1]}.obj")

    # model = train_mesh(v_mesh, f_mesh, v_emb, f_emb)
    # # Save the model
    # torch.save(model.state_dict(), f"../data/model_sphere_{sys.argv[1]}.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(n=64, n_layers=5, in_dim=3, out_dim=3)
    model.load_state_dict(torch.load(f"../data/model_sphere_{sys.argv[1]}.pth", map_location=device))
    model.to(device=device)
    # _, v = get_random_points(v_mesh, f_mesh, v_emb, f_emb, 100000)
    v = v_emb
    v = torch.tensor(v, dtype=torch.float32, device=device)
    v_emb_ = get_spherical_coordinates_torch(v)

    v_pred = model(v).detach().cpu().numpy()

    v_pred[:, 0] += 1
    ps.init()
    ps.register_point_cloud("surface", v_pred)
    ps.register_surface_mesh("Embedding", v_emb, f_emb)
    ps.register_surface_mesh("Original", v_mesh, f_mesh)
    ps.register_point_cloud("embedding points", v.detach().cpu().numpy())
    ps.show()

        