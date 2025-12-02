import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import gpytoolbox as gpy
import torch
from tqdm import tqdm
import math
import torch.nn as nn
import sys

def get_random_points(v_mesh, f_mesh, v_emb, f_emb, n):
    # Sample points on embedding mesh
    v_emb_rdm, ind_emb_rdm, bary_emb_rdm = gpy.random_points_on_mesh(
        v_emb, f_emb, n, return_indices=True
    )

    # Gather corresponding vertices on the mesh
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


# Create a MLP model with SIREN layers
class SIRENLayer(nn.Module):
    def __init__(self, in_dim, out_dim, is_first=False, w0=30.0):
        super().__init__()
        self.is_first = is_first
        self.in_dim = in_dim
        self.w0 = w0
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # first layer: U(-1/in_dim, 1/in_dim)
                self.linear.weight.uniform_(-1 / self.in_dim, 1 / self.in_dim)
            else:
                # deeper layers: U(-sqrt(6/in_dim)/w0, sqrt(6/in_dim)/w0)
                bound = np.sqrt(6 / self.in_dim) / self.w0
                self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.fill_(0.0)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))
    
class MLP(nn.Module):
    def __init__(self, n=512, n_layers=3, in_dim=3, out_dim=1, w0=30.0):
        super().__init__()
        layers = []

        # First SIREN layer (high-frequency)
        layers.append(SIRENLayer(in_dim, n, is_first=True, w0=w0))

        # Hidden layers
        for _ in range(n_layers):
            layers.append(SIRENLayer(n, n, is_first=False, w0=1.0))

        self.trunk = nn.Sequential(*layers)

        self.final_layer = nn.Linear(n, out_dim, bias=True)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        features = self.trunk(x)
        output = self.final_layer(features)

        return output.squeeze()
    
def sample_in_sphere(n):
    vec = np.random.uniform(-1, 1, (n, 3))
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)  # Normalize to unit length
    return vec

def parameterize_sphere_ellipsoid(v_e):
    v_c = v_e.clone()
    v_c[:, 0] = v_e[:, 0] * 3.0
    v_c[:, 1] = v_e[:, 1] * 2.0
    v_c[:, 2] = v_e[:, 2] * 1.0

    return v_c

def loss(model, v, true_n=None):
    n_pred = model(v)

    if true_n is None:
        n_true = v.clone()
        n_true[:, 0] = 2*v[:, 0]/9.0
        n_true[:, 1] = 2*v[:, 1]/4.0
        n_true[:, 2] = 2*v[:, 2]/1.0
        true_n = n_true / torch.linalg.norm(n_true, dim=1, keepdim=True)

    l = torch.mean(torch.linalg.norm(n_pred - true_n, ord=2, axis=1)**2)

    return l

# Train using exact normals
def train_exact(n_layers = 3, size_layer = 64, lr=1e-4, max_iter=500000, n_samples=10000, tol=1e-8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the model
    model = MLP(n=size_layer, n_layers=n_layers, in_dim=3, out_dim=3)
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for i in (pbar:= tqdm(range(max_iter))):
        optimizer.zero_grad()

        v_emb_rnd = sample_in_sphere(n_samples)
        v_emb_rnd = torch.tensor(v_emb_rnd, dtype=torch.float).to(device=device).requires_grad_(True)
        v_rnd = parameterize_sphere_ellipsoid(v_emb_rnd)

        l = loss(model, v_rnd)
        l.backward()
        optimizer.step()
        
        loss_value = l.item()
        pbar.set_description(f"Loss: {loss_value}")

    return model

# Train using mesh normals
def train_mesh(v_mesh, f_mesh, v_emb, f_emb, n_layers = 3, size_layer = 64, lr=1e-4, max_iter=500000, n_samples=10000, tol=1e-8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MLP(n=size_layer, n_layers=n_layers, in_dim=3, out_dim=3)
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    normals = gpy.per_vertex_normals(v_mesh, f_mesh)

    v_mesh_rnd_global, _ = get_random_points(v_mesh, f_mesh, v_emb, f_emb, int(1e6))
    normals_rnd_global = get_interpolated_values(normals, v_mesh_rnd_global, v_mesh, f_mesh)

    v_mesh_rnd_global = torch.tensor(v_mesh_rnd_global, dtype=torch.float32, device=device)
    normals_rnd_global = torch.tensor(normals_rnd_global, dtype=torch.float32, device=device)

    # Train the model
    for i in (pbar:= tqdm(range(max_iter))):
        optimizer.zero_grad()

        idx = torch.randint(0, int(1e6), (n_samples,), device=device)
        v_mesh_rnd = v_mesh_rnd_global[idx]
        normals_rnd = normals_rnd_global[idx]

        l = loss(model, v_mesh_rnd, normals_rnd)

        l.backward()
        optimizer.step()
        
        loss_value = l.item()
        pbar.set_description(f"Loss: {loss_value}")
        
    return model

if __name__ == "__main__":
    if sys.argv[1] == "exact":
        model = train_exact()
        # Save the model
        torch.save(model.state_dict(), "../data/model_ellipsoid_normal.pth")
    elif sys.argv[1] == "mesh":
        v_emb, f_emb = gpy.read_mesh("../data/sphere_ellipsoid.obj")

        v_mesh = np.copy(v_emb)
        v_mesh[:, 0] = v_mesh[:, 0] * 3.0
        v_mesh[:, 1] = v_mesh[:, 1] * 2.0
        v_mesh[:, 2] = v_mesh[:, 2] * 1.0
        f_mesh = np.copy(f_emb)

        model = train_mesh(v_mesh, f_mesh, v_emb, f_emb)
        
        # Save the model
        torch.save(model.state_dict(), "../data/model_ellipsoid_normal_mesh.pth")

        