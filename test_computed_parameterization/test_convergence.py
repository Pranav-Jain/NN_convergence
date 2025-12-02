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
import sys

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
    
# Deine SIREN layer
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
    
# Define MLP with SIREN layers
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

def get_surface_laplacian(model, v_cart, S_theta=None):

    def model_scalar(v_):
        return model(v_).view(-1).requires_grad_(True)  # Ensure scalar output
    
    f_ = model_scalar(v_cart)

    n = get_normals(v_cart, S_theta)

    grad_f = torch.autograd.grad(f_, v_cart, torch.ones_like(f_), create_graph=True, retain_graph=True)[0]

    # Surface gradient (tangential component)
    grad_f_surf = grad_f - torch.sum(grad_f * n, dim=1, keepdim=True) * n

    # Compute divergence of surface gradient
    def compute_divergence(grad_, v_):
        # Compute derivatives of each component of grad_f_surf
        div_x = torch.autograd.grad(grad_[:, 0], v_, torch.ones_like(grad_[:, 0]), create_graph=True, retain_graph=True)[0]
        div_x = div_x - torch.sum(div_x * n, dim=1, keepdim=True) * n
        div_y = torch.autograd.grad(grad_[:, 1], v_, torch.ones_like(grad_[:, 1]), create_graph=True, retain_graph=True)[0]
        div_y = div_y - torch.sum(div_y * n, dim=1, keepdim=True) * n
        div_z = torch.autograd.grad(grad_[:, 2], v_, torch.ones_like(grad_[:, 2]), create_graph=True, retain_graph=True)[0]
        div_z = div_z - torch.sum(div_z * n, dim=1, keepdim=True) * n
        
        # Build Hessian matrix
        hessian = torch.zeros(v_cart.shape[0], 3, 3).to(v_cart.device)
        hessian[:, 0, :] = div_x
        hessian[:, 1, :] = div_y
        hessian[:, 2, :] = div_z

        # Sum diagonal terms for divergence
        divF = hessian[:, 0, 0] + hessian[:, 1, 1] + hessian[:, 2, 2]

        return divF, hessian

    div_grad_f_surf, hessians = compute_divergence(grad_f_surf, v_cart)

    # Compute Hessian applied to normal: H n
    hessian_dot_n = torch.bmm(hessians, n.unsqueeze(-1)).squeeze()

    # Compute normal term: n^T (H n)
    normals_term = torch.sum(n * hessian_dot_n, dim=1)

    # Final Laplace-Beltrami operator
    lap_beltrami = div_grad_f_surf - normals_term

    return lap_beltrami

def get_bdry_points(n, device):
    if config["surface"] == "hemisphere":
        phi = torch.rand(n, device=device) * 2 * np.pi - np.pi
        theta = torch.ones_like(phi, device=device) * np.pi/2
        theta[n//2:] = 0.0
        # theta = torch.zeros_like(phi, device=device)
        r = torch.ones_like(phi, device=device)
        bdry_points = torch.stack([r, theta, phi], dim=1)

    elif config["surface"] == "heightfield":
        b = torch.tensor(np.random.uniform(config["domain"]["min"], config["domain"]["max"], (n))).to(device=device).requires_grad_(True).float()
        b1 = torch.stack([b, torch.ones_like(b)*config["domain"]["max"]], dim=1)
        b2 = torch.stack([b, torch.ones_like(b)*config["domain"]["min"]], dim=1)
        b3 = torch.stack([torch.ones_like(b)*config["domain"]["max"], b], dim=1)
        b4 = torch.stack([torch.ones_like(b)*config["domain"]["min"], b], dim=1)
        bdry_points = torch.cat((b1, b2, b3, b4), dim=0)

        z = 0.5 * (bdry_points[:, 0]**2 + bdry_points[:, 1]**2)

        bdry_points = torch.stack([bdry_points[:, 0], bdry_points[:, 1], z], dim=1)

    return bdry_points.requires_grad_(True)


def train_strong_form(l_model, device, n, size_layer, n_layers):
    optimizer = torch.optim.Adam(l_model.parameters(), lr=config["architecture"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=config["architecture"]["scheduler_patience"])

    S_theta = MLP(n=256, n_layers=5, in_dim=3, out_dim=3)
    S_theta.to(device=device)
    S_theta.load_state_dict(torch.load(f"../data/model_{config['surface']}.pth", weights_only=True, map_location=device))
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

            laplacian_pred = get_surface_laplacian(l_model, v_cart, S_theta)

            true_lap = laplacian_f(v_cart)
            true_lap = true_lap - mean_f

            loss = torch.linalg.norm(laplacian_pred - true_lap, 2)**2

            # TODO: Add boundary conditions

            if config["debug"]:
                print(f"Loss: {loss.item()}")
                exit()
            
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

    # V, _ = sample_in_domain(100000)
    V = v_mesh

    for file in filenames:
        print(file)
        n_layers = int(file.split("_")[-2])
        size_layer = int(file.split("_")[-3])

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

    S_theta = MLP(n=256, n_layers=5, in_dim=3, out_dim=3)
    S_theta.to(device=device)
    S_theta.load_state_dict(torch.load(f"../data/model_{config['surface']}.pth", weights_only=True, map_location=device))
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