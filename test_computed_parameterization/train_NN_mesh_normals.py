import numpy as np
import gpytoolbox as gpy
import torch
from tqdm import tqdm
import torch.nn as nn
import sys


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

# Create a MLP model
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

# MLP with SIREN layers
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


def loss(model, v, true_n):
    # Predict normals
    n_pred = model(v)
    n_pred = n_pred / torch.linalg.norm(n_pred, dim=1, keepdim=True)

    # Normal loss
    loss = nn.functional.mse_loss(n_pred, true_n)

    return loss

def train_mesh(v_mesh, f_mesh, n_layers = 5, size_layer = 256, lr=1e-3, max_iter=1000000, n_samples=1000, tol=1e-6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MLP(n=size_layer, n_layers=n_layers, in_dim=3, out_dim=3)
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

            l = loss(model, v_mesh_rnd, normals_rnd, )

            l.backward()
            optimizer.step()
            
            loss_value = l.item()
            pbar.set_description(f"Loss: {loss_value}")

            if loss_value < tol:
                break
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            break
        
    return model

if __name__ == "__main__":
    v_mesh, f_mesh = gpy.read_mesh(f"../data/{sys.argv[1]}.obj")
    model = train_mesh(v_mesh, f_mesh)

    # Save the model
    torch.save(model.state_dict(), f"../data/model_{sys.argv[1]}.pth")