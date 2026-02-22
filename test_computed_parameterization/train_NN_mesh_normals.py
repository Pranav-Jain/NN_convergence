import numpy as np
import gpytoolbox as gpy
import torch
from tqdm import tqdm
import sys

sys.path.append('../src')
from siren import MLP, MLP_normals


def get_random_points(v_mesh, f_mesh, n):
    # Sample points on mesh
    v_mesh_rdm, ind_mesh_rdm, bary_mesh_rdm = gpy.random_points_on_mesh(v_mesh, f_mesh, n, return_indices=True)

    return v_mesh_rdm

def get_interpolated_values(f, v, v_mesh, f_mesh):
    # Compute nearest triangle & barycentric coordinates
    _, idx, bary = gpy.signed_distance(v, v_mesh, f_mesh, use_cpp=True)

    # Gather triangle values of f
    # f_mesh[idx] gives shape (n, 3), then indexing into f → (n, 3, fdim)
    face_vals = f[f_mesh[idx]]

    # Weighted sum
    f_v = (face_vals * bary[:, :, None]).sum(axis=1)
    return f_v

def loss(model, v, true_n):
    # Predict normals
    n_pred = model(v)

    cos = torch.sum(n_pred * true_n, dim=1)
    loss = torch.sum((1 - cos)**2)

    return loss

def train_mesh(v_mesh, f_mesh, n_layers = 5, size_layer = 64, lr=1e-4, max_iter=100000000, n_samples=10000, tol=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MLP_normals(n=size_layer, n_layers=n_layers, in_dim=3, out_dim=3)
    model.load_state_dict(torch.load(f"../data/model_{sys.argv[1]}_normal.pth", map_location=device))
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    normals = gpy.per_vertex_normals(v_mesh, f_mesh)
    
    # Train the model
    for i in (pbar:= tqdm(range(max_iter))):
        try:
            optimizer.zero_grad()

            # Since the sampling is expensive, we sample 1 million points once every 10000 iterations
            if i % 10000 == 0:
                v_mesh_rnd_global = get_random_points(v_mesh, f_mesh, int(1e6))
                normals_rnd_global = get_interpolated_values(normals, v_mesh_rnd_global, v_mesh, f_mesh)
                normals_rnd_global = normals_rnd_global / np.linalg.norm(normals_rnd_global, ord=2, axis=1, keepdims=True)

                v_mesh_rnd_global = torch.tensor(v_mesh_rnd_global, dtype=torch.float32, device=device)
                normals_rnd_global = torch.tensor(normals_rnd_global, dtype=torch.float32, device=device)

            idx = torch.randint(0, int(1e6), (n_samples,), device=device)
            v_mesh_rnd = v_mesh_rnd_global[idx]
            normals_rnd = normals_rnd_global[idx]

            l = loss(model, v_mesh_rnd, normals_rnd)

            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # Gradient clipping
            optimizer.step()
            
            loss_value = l.item()
            pbar.set_description(f"Loss: {loss_value}")

            if loss_value < tol:
                break
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            break
        
    return model

def train_bdry_normal(v_mesh, f_mesh, boundary_edges, n_layers = 3, size_layer = 64, lr=1e-4, max_iter=10000000, n_samples=10000, tol=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bdry_model = MLP_normals(n=size_layer, n_layers=n_layers, in_dim=3, out_dim=3)
    bdry_model.to(device=device)

    optimizer = torch.optim.Adam(bdry_model.parameters(), lr=lr)

    surf_normals = gpy.per_vertex_normals(v_mesh, f_mesh)
    surf_normals_torch = torch.tensor(surf_normals, dtype=torch.float32, device=device)
    boundary_edges = torch.tensor(boundary_edges, dtype=torch.long, device=device)
    v_mesh_torch = torch.tensor(v_mesh, dtype=torch.float32, device=device)

    edge_vec = v_mesh_torch[boundary_edges[:, 1]] - v_mesh_torch[boundary_edges[:, 0]]
    edge_vec = edge_vec / torch.norm(edge_vec, p=2, dim=1, keepdim=True)

    bdry_normal = torch.cross(surf_normals_torch[boundary_edges[:, 0]], edge_vec, dim=1)
    bdry_normal = bdry_normal / torch.norm(bdry_normal, p=2, dim=1, keepdim=True)
    
    # Train the model
    for i in (pbar:= tqdm(range(max_iter))):
        try:
            optimizer.zero_grad()

            idx = torch.randint(0, boundary_edges.shape[0]-1, (n_samples,), device=device)
            bdry_edges_rdm = boundary_edges[idx]
            w = torch.rand((n_samples, 1), device=device) # linear interpolation weights
            bdry_points = (1 - w) * v_mesh_torch[bdry_edges_rdm[:, 0]] + w * v_mesh_torch[bdry_edges_rdm[:, 1]]
            bdry_normal_rdm = bdry_normal[idx]
            bdry_normal_rdm = bdry_normal_rdm / torch.norm(bdry_normal_rdm, p=2, dim=1, keepdim=True)
            
            l = loss(bdry_model, bdry_points, bdry_normal_rdm)
            l.backward()
            torch.nn.utils.clip_grad_norm_(bdry_model.parameters(), 10.0)  # Gradient clipping
            optimizer.step()
            
            loss_value = l.item()
            pbar.set_description(f"Loss: {loss_value}")

            if loss_value < tol:
                break
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            break
        
    return bdry_model

if __name__ == "__main__":
    v_mesh, f_mesh = gpy.read_mesh(f"../data/{sys.argv[1]}.obj")
    model = train_mesh(v_mesh, f_mesh)

    # Save the model
    torch.save(model.state_dict(), f"../data/model_{sys.argv[1]}_normal.pth")

    boundary_edges = gpy.boundary_edges(f_mesh)
    if boundary_edges.shape[0] == 0:
        print("No boundary edges found in the mesh.")
        sys.exit(0)
    model_bdry = train_bdry_normal(v_mesh, f_mesh, boundary_edges)
    torch.save(model_bdry.state_dict(), f"../data/model_{sys.argv[1]}_bdry_normal.pth")