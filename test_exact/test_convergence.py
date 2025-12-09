import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
import sys
import glob
import os
import gpytoolbox as gpy
import scipy as sp
import json
import polyscope as ps
import pandas as pd

sys.path.append('../src')
from siren import MLP
from surface_laplacian import get_surface_laplacian

with open("config.json", "r") as f:
    config = json.load(f)

def relative_l2_loss(pred, true):
    return np.linalg.norm(pred - true, 2) / (np.linalg.norm(true, 2))

# Define the functions
def u_torch(v):
    if config["dimension"] == 2: # 2D case (on a square)
        x, y = v[:, 0], v[:, 1]
        if sys.argv[1] == "1":
            f = torch.sin(x)*torch.cos(y) / (-2)
        elif sys.argv[1] == "2":
            f = torch.sin(np.pi*x)*torch.cos(np.pi*y) / (-2*np.pi**2)
        elif sys.argv[1] == "3":
            f = torch.sin(2*x)*torch.cos(3*y) / (-13)
        elif sys.argv[1] == "4":
            f = torch.sin(10*x)*torch.cos(20*y) / (-500)
        elif sys.argv[1] == "5":
            f = torch.sin(10*np.pi*x)*torch.cos(20*np.pi*y) / (-500*np.pi**2)
        elif sys.argv[1] == "6":
            f = (23*x**3 + 43*y) / 3090.0
        elif sys.argv[1] == "7":
            f = (23*x**3 - 43*y**5) / 137250.0
        elif sys.argv[1] == "8":
            f = (105*x**2 + 324*y**3) / 43125.0
        elif sys.argv[1] == "9":
            f = (-765*x**5 + 78*y**3) / 2400375.0
        elif sys.argv[1] == "10":
            f = (-685*x**8 - 546*y**9) / 1333984375.0
        elif sys.argv[1] == "11":
            f = torch.sin(np.pi*x)*torch.sin(np.pi*y) / (-2*np.pi**2)
        elif sys.argv[1] == "12":
            f = torch.sin(2*np.pi*x)*torch.sin(np.pi*y) / (-5*np.pi**2)
        elif sys.argv[1] == "13":
            f = torch.sin(2*np.pi*x)*torch.sin(3*np.pi*y) / (-13*np.pi**2)
        elif sys.argv[1] == "14":
            f = torch.sin(5*np.pi*x)*torch.sin(5*np.pi*y) / (-50*np.pi**2)
        elif sys.argv[1] == "15":
            f = torch.sin(8*np.pi*x)*torch.sin(10*np.pi*y) / (-164*np.pi**2)
        else:
            raise NotImplementedError()
        
    elif config["dimension"] == 3: # 3D case (on sphere)
        r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

        if sys.argv[1] == "1":
            f = r**2 * (-0.5*torch.cos(theta))
        elif sys.argv[1] == "2":
            f = r**2 * (-0.5*torch.sin(theta)*torch.cos(phi))
        elif sys.argv[1] == "3":
            f = r**2 * (-0.5*torch.sin(theta)*torch.sin(phi))
        elif sys.argv[1] == "4":
            f = r**2 * torch.sin(theta)**2 * torch.cos(theta) * torch.cos(2*phi)
        else:
            raise NotImplementedError()

    return f.squeeze()

# Laplacian of f
def rhs(v):
    if config["dimension"] == 2:
        x, y = v[:, 0], v[:, 1]
        if sys.argv[1] == "1":
            lap_u = torch.cos(y)*torch.sin(x)
        elif sys.argv[1] == "2":
            lap_u = torch.cos(np.pi*y)*torch.sin(np.pi*x)
        elif sys.argv[1] == "3":
            lap_u = torch.cos(3*y)*torch.sin(2*x)
        elif sys.argv[1] == "4":
            lap_u = torch.cos(20*y)*torch.sin(10*x)
        elif sys.argv[1] == "5":
            lap_u = torch.cos(20*np.pi*y)*torch.sin(10*np.pi*x)
        elif sys.argv[1] == "6":
            lap_u = (138*x) / 3090.0
        elif sys.argv[1] == "7":
            lap_u = ((138*x)-(860*y**3)) / 137250.0
        elif sys.argv[1] == "8":
            lap_u = (6*(35+324*y)) / 43125.0
        elif sys.argv[1] == "9":
            lap_u = ((-15300*x**3)+468*y) / 2400375.0
        elif sys.argv[1] == "10":
            lap_u = (-56*((685*x**6)+(702*y**7))) / 1333984375.0
        elif sys.argv[1] == "11":
            lap_u = torch.sin(np.pi*x)*torch.sin(np.pi*y)
        elif sys.argv[1] == "12":
            lap_u = torch.sin(2*np.pi*x)*torch.sin(np.pi*y)
        elif sys.argv[1] == "13":
            lap_u = torch.sin(2*np.pi*x)*torch.sin(3*np.pi*y)
        elif sys.argv[1] == "14":
            lap_u = torch.sin(5*np.pi*x)*torch.sin(5*np.pi*y)
        elif sys.argv[1] == "15":
            lap_u = torch.sin(8*np.pi*x)*torch.sin(10*np.pi*y)
        else:
            raise NotImplementedError()
    
    elif config["dimension"] == 3:
        r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

        if sys.argv[1] == "1":
            lap_u = torch.cos(theta)
        elif sys.argv[1] == "2":
            lap_u = torch.sin(theta)*torch.cos(phi)
        elif sys.argv[1] == "3":
            lap_u = torch.sin(theta)*torch.sin(phi)
        elif sys.argv[1] == "4":
            lap_u = -12 * torch.sin(theta)**2 * torch.cos(theta) * torch.cos(2*phi)
        else:
            raise NotImplementedError()
    
    return lap_u.squeeze()

# Numpy version of f for FEM
def u_numpy(v):
    if config["dimension"] == 2:
        x, y = v[:, 0], v[:, 1]
        if sys.argv[1] == "1":
            f = np.sin(x)*np.cos(y) / (-2)
        elif sys.argv[1] == "2":
            f = np.sin(np.pi*x)*np.cos(np.pi*y) / (-2*np.pi**2)
        elif sys.argv[1] == "3":
            f = np.sin(2*x)*np.cos(3*y) / (-13)
        elif sys.argv[1] == "4":
            f = np.sin(10*x)*np.cos(20*y) / (-500)
        elif sys.argv[1] == "5":
            f = np.sin(10*np.pi*x)*np.cos(20*np.pi*y) / (-500*np.pi**2) 
        elif sys.argv[1] == "6":
            f = (23*x**3 + 43*y) / 3090.0
        elif sys.argv[1] == "7":
            f = (23*x**3 - 43*y**5) / 137250.0
        elif sys.argv[1] == "8":
            f = (105*x**2 + 324*y**3) / 43125.0
        elif sys.argv[1] == "9":
            f = (-765*x**5 + 78*y**3) / 2400375.0
        elif sys.argv[1] == "10":
            f = (-685*x**8 - 546*y**9) / 1333984375.0
        elif sys.argv[1] == "11":
            f = np.sin(np.pi*x)*np.sin(np.pi*y) / (-2*np.pi**2)
        elif sys.argv[1] == "12":
            f = np.sin(2*np.pi*x)*np.sin(np.pi*y) / (-5*np.pi**2)
        elif sys.argv[1] == "13":
            f = np.sin(2*np.pi*x)*np.sin(3*np.pi*y) / (-13*np.pi**2)
        elif sys.argv[1] == "14":
            f = np.sin(5*np.pi*x)*np.sin(5*np.pi*y) / (-50*np.pi**2)
        elif sys.argv[1] == "15":
            f = np.sin(8*np.pi*x)*np.sin(10*np.pi*y) / (-164*np.pi**2)
        else:
            raise NotImplementedError()
    
    elif config["dimension"] == 3:
        r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

        if sys.argv[1] == "1":
            f = r**2 * (-0.5*np.cos(theta))
        elif sys.argv[1] == "2":
            f = r**2 * (-0.5*np.sin(theta)*np.cos(phi))
        elif sys.argv[1] == "3":
            f = r**2 * (-0.5*np.sin(theta)*np.sin(phi))
        elif sys.argv[1] == "4":
            f = r**2 * np.sin(theta)**2 * np.cos(theta) * np.cos(2*phi)
        else:
            raise NotImplementedError()

    return f.squeeze()

def test_FEM(nx, ny, return_memory=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l2_loss = 0.0

    if config["dimension"] == 2:
        V, F = gpy.regular_square_mesh(nx, ny)

        BV = gpy.boundary_vertices(F)

        L = gpy.cotangent_laplacian(V, F)
        M = gpy.massmatrix(V, F)

        lap_u = -M@rhs(torch.Tensor(V).to(device=device)).detach().cpu().numpy()

        # Apply Dirichlet Boundary
        # Section 4.3 - https://web.stanford.edu/class/energy281/FiniteElementMethod.pdf
        # https://finite-element.github.io/7_boundary_conditions.html
        if config["bc"] == "dirichlet":
            L[BV, :] = 0.0
            L[BV, BV] = 1.0
            lap_u[BV] = u_numpy(V[BV])

        u = sp.sparse.linalg.spsolve(L, lap_u)
        l2_loss = relative_l2_loss(u, u_numpy(V))

    print(f"FEM Loss: {l2_loss}")

    return l2_loss

def get_spherical_coordinates(v, c, r_):
    r = np.ones(v.shape[0]) * r_
    theta = np.arccos((v[:, 2] - c[2]) / r)
    phi = np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0])

    return np.stack([r, theta, phi], axis=1)

def get_spherical_coordinates_torch(v, c, r_):
    r = torch.ones(v.shape[0], device=v.device) * r_
    theta = torch.arccos((v[:, 2] - c[2]) / r)
    phi = torch.arctan2(v[:, 1] - c[1], v[:, 0] - c[0])

    r = r.requires_grad_(True)
    theta = theta.requires_grad_(True)
    phi = phi.requires_grad_(True)

    return torch.stack([r, theta, phi], dim=1)

def sample_in_sphere(n, r):
    # Sample uniformly on the unit sphere
    vec = np.random.uniform(config["domain"]["min"], config["domain"]["max"], (n, 3))
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)  # Normalize to unit length
    vec = vec * r  # Scale to radius r

    return vec

def train_strong_form_sphere(dim, max_iter, size_layer, n_layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(n=size_layer, n_layers=n_layers, in_dim=dim)
    model.to(device=device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["architecture"]["lr"])
    # create a scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=config["architecture"]["scheduler_patience"], verbose=True)

    center = np.array([0.0, 0.0, 0.0])
    center = torch.tensor(center, dtype=torch.float32).to(device=device).requires_grad_(True)
    r = 1.0
    n = config["architecture"]["num_samples"]

    losses = []
    for i in (pbar:= tqdm(range(max_iter))):
        optimizer.zero_grad() # Zero the gradients

        # Sample points on the sphere
        v_mesh_rnd_cart = sample_in_sphere(n, r)
        v_mesh_rnd_cart = torch.tensor(v_mesh_rnd_cart, dtype=torch.float32).to(device=device).requires_grad_(True)

        v_mesh_rnd = get_spherical_coordinates_torch(v_mesh_rnd_cart, center, r)
        
        normal = (v_mesh_rnd_cart - center) / r  # unit normals
        laplacian_pred = get_surface_laplacian(model, v_mesh_rnd_cart, normal)

        f_v = rhs(v_mesh_rnd)

        loss = torch.linalg.norm(laplacian_pred - f_v, 2)**2
        
        loss.backward(retain_graph=True)
        if not torch.isnan(loss):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["architecture"]["grad_clip"])  # Gradient clipping for stability
            optimizer.step()
            scheduler.step(loss)

            loss_value = loss.item()
            losses.append(loss_value)
            pbar.set_description(f"Loss: {loss_value}")

            if loss_value < config["threshold"] and i > 1000:
                break
    
    v_mesh_rnd_cart = sample_in_sphere(100000, r)
    v_mesh_rnd_cart = torch.tensor(v_mesh_rnd_cart, dtype=torch.float32).to(device=device)
    v_mesh_rnd_sp = get_spherical_coordinates_torch(v_mesh_rnd_cart, center, r)

    true = u_torch(v_mesh_rnd_sp).detach().cpu().numpy()
    pred = model(v_mesh_rnd_sp).squeeze().detach().cpu().numpy()

    # Plotting loss curve
    plt.figure()
    plt.plot(np.log(losses))
    plt.title("Loss")
    plt.xlabel("Iteration")
    plt.ylabel("log(Loss)")
    plt.title(f"dim = {dim}, order = 2, {size_layer} x {n_layers}")
    plt.savefig(f"{save_dir}/example{sys.argv[1]}/loss_{size_layer}_{n_layers}.png")

    torch.save(model.state_dict(), f"{save_dir}/example{sys.argv[1]}/model_{size_layer}_{n_layers}.pth")

    l2_loss = relative_l2_loss(pred, true)
    return l2_loss

def train_strong_form(dim, max_iter, size_layer, n_layers): # for square domain
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = MLP(n=size_layer, n_layers=n_layers, in_dim=dim)
    model.to(device=device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["architecture"]["lr"])
    # create a scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=config["architecture"]["scheduler_patience"], verbose=True)

    losses = []
    for i in (pbar:= tqdm(range(int(max_iter)))):
        optimizer.zero_grad()

        # Interior points
        x = torch.tensor(np.random.uniform(config["domain"]["min"], config["domain"]["max"], (config["architecture"]["num_samples"], 2))).to(device=device).requires_grad_(True).float()
        
        # Boundary points
        b = torch.tensor(np.random.uniform(config["domain"]["min"], config["domain"]["max"], (config["architecture"]["num_samples"]))).to(device=device).requires_grad_(True).float()
        b1 = torch.stack([b, torch.ones_like(b)*config["domain"]["max"]], dim=1)
        b2 = torch.stack([b, torch.ones_like(b)*config["domain"]["min"]], dim=1)
        b3 = torch.stack([torch.ones_like(b)*config["domain"]["max"], b], dim=1)
        b4 = torch.stack([torch.ones_like(b)*config["domain"]["min"], b], dim=1)

        true = rhs(x)
        pred = u_torch(x).squeeze()

        grad_pred = torch.autograd.grad(pred, x, torch.ones_like(pred), create_graph=True)[0]
        
        def compute_divergence(grad_, v_):
            # Compute derivatives of each component of grad_f_surf
            div_x = torch.autograd.grad(grad_[:, 0], v_, torch.ones_like(grad_[:, 0]), create_graph=True, retain_graph=True)[0]
            div_y = torch.autograd.grad(grad_[:, 1], v_, torch.ones_like(grad_[:, 1]), create_graph=True, retain_graph=True)[0]
            
            # Build Hessian matrix
            hessian = torch.zeros(v_.shape[0], 2, 2).to(v_.device)
            hessian[:, 0, :] = div_x
            hessian[:, 1, :] = div_y

            # Sum diagonal terms for divergence
            divF = hessian[:, 0, 0] + hessian[:, 1, 1]

            return divF

        laplacian_pred = compute_divergence(grad_pred, x)

        loss = (torch.linalg.norm(laplacian_pred - true, ord=2)**2) / x.shape[0]

        ## Handling Boundary Conditions ##
        pred_b1 = model(b1).squeeze()
        pred_b2 = model(b2).squeeze()
        pred_b3 = model(b3).squeeze()
        pred_b4 = model(b4).squeeze()

        if config["bc"] == "dirichlet":
            temp = (torch.linalg.norm(pred_b1 - u_torch(b1), 2)**2)/b1.shape[0] 
            temp += (torch.linalg.norm(pred_b2 - u_torch(b2), 2)**2)/b2.shape[0]
            temp += (torch.linalg.norm(pred_b3 - u_torch(b3), 2)**2)/b3.shape[0]
            temp += (torch.linalg.norm(pred_b4 - u_torch(b4), 2)**2)/b4.shape[0]
            loss += 100*temp
        
        elif config["bc"] == "neumann":
            grad_pred_b1 = torch.autograd.grad(pred_b1, b1, torch.ones_like(pred_b1), create_graph=True)[0]
            grad_pred_b2 = torch.autograd.grad(pred_b2, b2, torch.ones_like(pred_b2), create_graph=True)[0]
            grad_pred_b3 = torch.autograd.grad(pred_b3, b3, torch.ones_like(pred_b3), create_graph=True)[0]
            grad_pred_b4 = torch.autograd.grad(pred_b4, b4, torch.ones_like(pred_b4), create_graph=True)[0]
            
            true_b1 = u_torch(b1)
            true_b2 = u_torch(b2)
            true_b3 = u_torch(b3)
            true_b4 = u_torch(b4)
            grad_true_b1 = torch.autograd.grad(true_b1, b1, torch.ones_like(true_b1), create_graph=True)[0]
            grad_true_b2 = torch.autograd.grad(true_b2, b2, torch.ones_like(true_b2), create_graph=True)[0]
            grad_true_b3 = torch.autograd.grad(true_b3, b3, torch.ones_like(true_b3), create_graph=True)[0]
            grad_true_b4 = torch.autograd.grad(true_b4, b4, torch.ones_like(true_b4), create_graph=True)[0]
            
            # Boundary normals
            n1 = torch.ones_like(b1)
            n1[:, 0] = 0.0
            n2 = -torch.ones_like(b2)
            n2[:, 0] = 0.0
            n3 = torch.ones_like(b3)
            n3[:, 1] = 0.0
            n4 = -torch.ones_like(b4)
            n4[:, 1] = 0.0

            # Compute Neumann loss
            temp = (torch.linalg.norm(torch.sum(grad_pred_b1 * n1, dim=1) - torch.sum(grad_true_b1 * n1, dim=1), 2)**2)/b1.shape[0]
            temp += (torch.linalg.norm(torch.sum(grad_pred_b2 * n2, dim=1) - torch.sum(grad_true_b2 * n2, dim=1), 2)**2)/b2.shape[0]
            temp += (torch.linalg.norm(torch.sum(grad_pred_b3 * n3, dim=1) - torch.sum(grad_true_b3 * n3, dim=1), 2)**2)/b3.shape[0]
            temp += (torch.linalg.norm(torch.sum(grad_pred_b4 * n4, dim=1) - torch.sum(grad_true_b4 * n4, dim=1), 2)**2)/b4.shape[0]
            loss += 100*temp

        else:
            print("Mention bdry condition")
            raise NotImplementedError

        pbar.set_description(f"Loss: {loss.item()}")
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["architecture"]["grad_clip"])  # Gradient clipping
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < config["threshold"] and i > 1000:
            break
    

    # Plotting loss curve
    plt.figure()
    plt.plot(np.log(losses))
    plt.title("Loss")
    plt.xlabel("Iteration")
    plt.ylabel("log(Loss)")
    plt.title(f"dim = {dim}, order = 2, {size_layer} x {n_layers}")
    plt.savefig(f"{save_dir}/example{sys.argv[1]}/loss_{size_layer}_{n_layers}.png")

    # Plotting predictions
    x = torch.tensor(np.random.uniform(config["domain"]["min"], config["domain"]["max"], (10000, 2))).to(device=device).float()
    true = u_torch(x).detach().cpu().numpy()
    pred = model(x).squeeze().detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], true, label="True", color='blue')
    plt.legend()
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], pred, label="Predicted", color='orange')
    plt.legend()
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], true, label="True", color='blue')
    ax.scatter(x[:, 0], x[:, 1], pred, label="Predicted", color='orange')
    plt.legend()
    plt.savefig(f"{save_dir}/example{sys.argv[1]}/pred_{size_layer}_{n_layers}.png")

    l2_loss = relative_l2_loss(pred, true)

    torch.save(model.state_dict(), f"{save_dir}/example{sys.argv[1]}/model_{size_layer}_{n_layers}.pth")

    return l2_loss

def train():
    n_layers = [config["architecture"]["num_layers"]]
    size_layer = np.arange(30, 160, 20)

    for i in n_layers:
        for j in size_layer:
            torch.cuda.reset_peak_memory_stats()
            if config["dimension"] == 2:
                loss = train_strong_form(dim=2, max_iter=config["architecture"]["max_iter"], size_layer=j, n_layers=i)
            elif config["dimension"] == 3:
                loss = train_strong_form_sphere(dim=3, max_iter=config["architecture"]["max_iter"], size_layer=j, n_layers=i)
            else:
                raise NotImplementedError()

def plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filenames = glob.glob(f"{save_dir}/example{sys.argv[1]}/model*.pth")

    dof = []
    fem_dof = []
    losses = []
    fem_losses = []
    ref_losses = []
    ref_dof = []

    if config["dimension"] == 2:
        x = torch.tensor(np.random.uniform(config["domain"]["min"], config["domain"]["max"], (100000, 2))).to(device=device).requires_grad_(True).float()
        true = u_torch(x).detach().cpu().numpy()
    elif config["dimension"] == 3:
        r = 1.0
        center = torch.zeros(3).to(device=device).requires_grad_(True)
        x_cart = sample_in_sphere(100000, r)
        x = torch.tensor(x_cart, dtype=torch.float32).to(device=device).requires_grad_(True)
        x_sp = get_spherical_coordinates_torch(x, center, r)
        true = u_torch(x_sp).detach().cpu().numpy()

    for file in filenames:
        # get layer size and n_layers from filename
        n_layers = int(file.split("_")[-1][:-4])
        size_layer = int(file.split("_")[-2])

        model = MLP(n=size_layer, n_layers=n_layers, in_dim=config["dimension"])
        model.to(device=device)
        model.load_state_dict(torch.load(file, weights_only=True, map_location=device))

        pred = model(x).squeeze().detach().cpu().numpy()

        # Adjust the constant factor
        const = np.mean(true - pred)
        pred = pred + const

        loss_l2 = relative_l2_loss(pred, true)

        total_trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        dof.append(total_trainable_params)

        losses.append(loss_l2)

        print("Depth and Width:", n_layers, size_layer)
        print("PINN Loss:", loss_l2)

        # Plot the error plots using polyscope
        ps.init()
        ps.set_screenshot_extension(".png")
        ps.register_point_cloud("Surface Points", x.detach().cpu().numpy())
        ps.get_point_cloud("Surface Points").add_scalar_quantity("True Solution", true, enabled=True)
        ps.screenshot(f"{save_dir}/example{sys.argv[1]}/true_{size_layer}_{n_layers}.png")

        ps.get_point_cloud("Surface Points").add_scalar_quantity("Predicted Solution", pred, enabled=True)
        ps.screenshot(f"{save_dir}/example{sys.argv[1]}/pred_{size_layer}_{n_layers}.png")

        ps.get_point_cloud("Surface Points").add_scalar_quantity("Error", np.abs(true - pred), enabled=True)
        ps.screenshot(f"{save_dir}/example{sys.argv[1]}/error_{size_layer}_{n_layers}.png")

        # FEM for 2D case
        if config["dimension"] == 2:
            fem_loss = test_FEM(int(np.sqrt(total_trainable_params)), int(np.sqrt(total_trainable_params)))
            fem_losses.append(fem_loss)
            fem_dof.append(total_trainable_params)
    
    # FEM for 3D case
    if config["dimension"] == 3:
        for i in range(2, 7):
            v_mesh, f_mesh = gpy.icosphere(i)
            M_mesh = gpy.massmatrix(v_mesh, f_mesh)
            M_mesh = sp.sparse.csc_matrix(M_mesh)
            v_mesh_sp = get_spherical_coordinates(v_mesh, center.detach().cpu().numpy(), r)
            f_ = rhs(torch.tensor(v_mesh_sp, dtype=torch.float32).to(device=device)).squeeze().detach().cpu().numpy()
            
            lap_mesh = gpy.cotangent_laplacian(v_mesh, f_mesh)
            u = sp.sparse.linalg.spsolve(lap_mesh, -M_mesh@f_)

            # Adjust the constant factor
            const = np.mean(u_numpy(v_mesh_sp) - u)
            u = u + const

            l2_loss = relative_l2_loss(u, u_numpy(v_mesh_sp))
            fem_losses.append(l2_loss)
            fem_dof.append(v_mesh.shape[0])


    # Convert to numpy arrays
    dof = np.array(dof)
    fem_dof = np.array(fem_dof)
    losses = np.array(losses)
    fem_losses = np.array(fem_losses)
    ref_losses = np.array(ref_losses)
    ref_dof = np.array(ref_dof)

    # Sort the arrays
    dof_sorted = np.sort(dof)
    losses_sorted = losses[np.argsort(dof)]
    fem_dof_sorted = np.sort(fem_dof)
    fem_losses_sorted = fem_losses[np.argsort(fem_dof)]

    x = dof_sorted
    y = losses_sorted

    # Fit a line to log-log data to find the slope
    coeffs = np.polyfit(np.log(x), np.log(y), 1)

    
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


    # Plotting the convergence plot
    plt.figure()
    plt.loglog(x, y, label="PINN", marker='o')
    plt.loglog(x, 0.9*x[0]*y[0]*1/x, color='red', label="1/x", linestyle=':')
    plt.loglog(fem_dof_sorted, fem_losses_sorted, label="FEM", marker='o')
    plt.xlabel("Total Trainable Weights")
    plt.ylabel("Relative L2 Loss")
    plt.title(f"Slope: {coeffs[0]}")
    plt.legend()
    plt.savefig(f"{save_dir}/convergence_example_{sys.argv[1]}.png")

if __name__ == "__main__":
    save_dir = f"poisson_results/{config['dimension']}d/{config['bc']}/domain_{config['domain']['min']}to{config['domain']['max']}"

    if config["operation"] == "train":
        # If directory doesn't exist, create it
        if not os.path.exists(f"{save_dir}/example{sys.argv[1]}"):
            os.makedirs(f"{save_dir}/example{sys.argv[1]}")
        
        # Save config
        json.dump(config, open(f"{save_dir}/example{sys.argv[1]}/config.json", 'w'), indent=4)
        
        train()
    elif config["operation"] == "plot":
        plot()
    else:
        print("Mention operation")
        raise NotImplementedError()