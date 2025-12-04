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
from scipy.sparse.linalg import spsolve
import json
import sys
import pandas as pd

sys.path.append('../src')
from siren import MLP, MLP_normals

with open("config.json", "r") as f:
    config = json.load(f)

usage_msg = "Usage: python test_parameterization.py [1|2|3|...]"

# Define the target function f on the surface
def f(v):
    if config["surface"] == "heightfield":
        x, y = v[:, 0], v[:, 1]
        if sys.argv[1] == "1":
            f = (-1/2)*np.sin(x)*np.cos(y)
        elif sys.argv[1] == "2":
            f = np.sin(np.pi*x)*np.cos(np.pi*y)
        elif sys.argv[1] == "3":
            f = np.sin(2*np.pi*x)*np.cos(3*np.pi*y)
        elif sys.argv[1] == "4":
            f = np.sin(10*np.pi*x)*np.cos(20*np.pi*y)
        elif sys.argv[1] == "5":
            f = np.sin(50*np.pi*x)*np.cos(60*np.pi*y)
        elif sys.argv[1] == "6":
            f = 20*x - 33*y
        elif sys.argv[1] == "7":
            f = 15*x - 76*y
        elif sys.argv[1] == "8":
            f = 101*x - 234*y
        elif sys.argv[1] == "9":
            f = -587*x - 230*y
        elif sys.argv[1] == "10":
            f = 4906*x - 4375*y
        elif sys.argv[1] == "11":
            f = 7689*x + 6759*y
        elif sys.argv[1] == "12":
            f = 23*x**3 + 43*y
        elif sys.argv[1] == "13":
            f = 23*x**3 - 43*y**5
        elif sys.argv[1] == "14":
            f = 105*x + 324*y**3
        elif sys.argv[1] == "15":
            f = -765*x**5 + 78*y**3
        else:
            raise NotImplementedError()
        
    elif config["surface"] == "ellipsoid":
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        if sys.argv[1] == "1":
            f = x - y + z
        elif sys.argv[1] == "2":
            f = 20*x - 33*y + 73*z
        elif sys.argv[1] == "3":
            f = 15*x - 76*y - 91*z
        elif sys.argv[1] == "4":
            f = -9*x - 53*y - 28*z
        elif sys.argv[1] == "5":
            f = -76*x + 27*y + 56*z
        elif sys.argv[1] == "6":
            f = 15*x - 76*y - 91*z
        elif sys.argv[1] == "7":
            f = 101*x - 234*y - 478*z
        elif sys.argv[1] == "8":
            f = -587*x - 230*y - 948*z
        elif sys.argv[1] == "9":
            f = -894*x + 34*y - 430*z
        elif sys.argv[1] == "10":
            f = 324*x - 100*y + 345*z
        elif sys.argv[1] == "11":
            f = -3246*x + 374*y - 9856*z
        elif sys.argv[1] == "12":
            f = 8697*x + 5960*y + 1235*z
        elif sys.argv[1] == "13":
            f = 4906*x - 4375*y +870*z
        elif sys.argv[1] == "14":
            f = -23*x + 9460*y - 8753*z
        elif sys.argv[1] == "15":
            f = 7689*x + 6759*y + 4432*z
        else:
            raise NotImplementedError

    else:
        print(usage_msg)
        exit()

    return f.squeeze()

# Define the target function f on the surface (PyTorch version)
def f_torch(v):
    if config["surface"] == "heightfield":
        x, y = v[:, 0], v[:, 1]
        if sys.argv[1] == "1":
            f = (-1/2)*torch.sin(x)*torch.cos(y)
        elif sys.argv[1] == "2":
            f = torch.sin(np.pi*x)*torch.cos(np.pi*y)
        elif sys.argv[1] == "3":
            f = torch.sin(2*np.pi*x)*torch.cos(3*np.pi*y)
        elif sys.argv[1] == "4":
            f = torch.sin(10*np.pi*x)*torch.cos(20*np.pi*y)
        elif sys.argv[1] == "5":
            f = torch.sin(50*np.pi*x)*torch.cos(60*np.pi*y)
        elif sys.argv[1] == "6":
            f = 20*x - 33*y
        elif sys.argv[1] == "7":
            f = 15*x - 76*y
        elif sys.argv[1] == "8":
            f = 101*x - 234*y
        elif sys.argv[1] == "9":
            f = -587*x - 230*y
        elif sys.argv[1] == "10":
            f = 4906*x - 4375*y
        elif sys.argv[1] == "11":
            f = 7689*x + 6759*y
        elif sys.argv[1] == "12":
            f = 23*x**3 + 43*y
        elif sys.argv[1] == "13":
            f = 23*x**3 - 43*y**5
        elif sys.argv[1] == "14":
            f = 105*x + 324*y**3
        elif sys.argv[1] == "15":
            f = -765*x**5 + 78*y**3
        else:
            raise NotImplementedError()
    
    elif config["surface"] == "ellipsoid":
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        if sys.argv[1] == "1":
            f = x - y + z
        elif sys.argv[1] == "2":
            f = 20*x - 33*y + 73*z
        elif sys.argv[1] == "3":
            f = 15*x - 76*y - 91*z
        elif sys.argv[1] == "4":
            f = -9*x - 53*y - 28*z
        elif sys.argv[1] == "5":
            f = -76*x + 27*y + 56*z
        elif sys.argv[1] == "6":
            f = 15*x - 76*y - 91*z
        elif sys.argv[1] == "7":
            f = 101*x - 234*y - 478*z
        elif sys.argv[1] == "8":
            f = -587*x - 230*y - 948*z
        elif sys.argv[1] == "9":
            f = -894*x + 34*y - 430*z
        elif sys.argv[1] == "10":
            f = 324*x - 100*y + 345*z
        elif sys.argv[1] == "11":
            f = -3246*x + 374*y - 9856*z
        elif sys.argv[1] == "12":
            f = 8697*x + 5960*y + 1235*z
        elif sys.argv[1] == "13":
            f = 4906*x - 4375*y +870*z
        elif sys.argv[1] == "14":
            f = -23*x + 9460*y - 8753*z
        elif sys.argv[1] == "15":
            f = 7689*x + 6759*y + 4432*z
        else:
            raise NotImplementedError()
        
    else:
        print(usage_msg)
        exit()

    return f.squeeze()

# Define the Laplacian of the target function f on the surface (PyTorch version)
def laplacian_f(v):
    if config["surface"] == "heightfield":
        x, y = v[:, 0], v[:, 1]
        if sys.argv[1] == "1":
            lap_f = (0.5*x*(x*y*torch.sin(x)*torch.sin(y) + (y**2 + 1)*torch.cos(x)*torch.cos(y)) - 0.5*y*(x*y*torch.cos(x)*torch.cos(y) + (x**2 + 1)*torch.sin(x)*torch.sin(y)) + (x**2 + y**2 + 1)*(-1.0*x*y*torch.sin(y)*torch.cos(x) + 0.5*x*torch.cos(x)*torch.cos(y) - 0.5*y*torch.sin(x)*torch.sin(y) + (0.5*x**2 + 0.5)*torch.sin(x)*torch.cos(y) + (0.5*y**2 + 0.5)*torch.sin(x)*torch.cos(y)))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "2":
            lap_f = -np.pi*(x*(x*y*torch.sin(np.pi*x)*torch.sin(np.pi*y) + (y**2 + 1)*torch.cos(np.pi*x)*torch.cos(np.pi*y)) - y*(x*y*torch.cos(np.pi*x)*torch.cos(np.pi*y) + (x**2 + 1)*torch.sin(np.pi*x)*torch.sin(np.pi*y)) + (x**2 + y**2 + 1)*(-2*np.pi*x*y*torch.sin(np.pi*y)*torch.cos(np.pi*x) + x*torch.cos(np.pi*x)*torch.cos(np.pi*y) - y*torch.sin(np.pi*x)*torch.sin(np.pi*y) + np.pi*(x**2 + 1)*torch.sin(np.pi*x)*torch.cos(np.pi*y) + np.pi*(y**2 + 1)*torch.sin(np.pi*x)*torch.cos(np.pi*y)))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "3":
            lap_f = -np.pi*(x*(3*x*y*torch.sin(2*np.pi*x)*torch.sin(3*np.pi*y) + (2*y**2 + 2)*torch.cos(2*np.pi*x)*torch.cos(3*np.pi*y)) - y*(2*x*y*torch.cos(2*np.pi*x)*torch.cos(3*np.pi*y) + (3*x**2 + 3)*torch.sin(2*np.pi*x)*torch.sin(3*np.pi*y)) + (x**2 + y**2 + 1)*(-12*np.pi*x*y*torch.sin(3*np.pi*y)*torch.cos(2*np.pi*x) + 2*x*torch.cos(2*np.pi*x)*torch.cos(3*np.pi*y) - 3*y*torch.sin(2*np.pi*x)*torch.sin(3*np.pi*y) + 9*np.pi*(x**2 + 1)*torch.sin(2*np.pi*x)*torch.cos(3*np.pi*y) + 4*np.pi*(y**2 + 1)*torch.sin(2*np.pi*x)*torch.cos(3*np.pi*y)))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "4":
            lap_f = -10*np.pi*(x*(2*x*y*torch.sin(10*np.pi*x)*torch.sin(20*np.pi*y) + (y**2 + 1)*torch.cos(10*np.pi*x)*torch.cos(20*np.pi*y)) - y*(x*y*torch.cos(10*np.pi*x)*torch.cos(20*np.pi*y) + (2*x**2 + 2)*torch.sin(10*np.pi*x)*torch.sin(20*np.pi*y)) + (x**2 + y**2 + 1)*(-40*np.pi*x*y*torch.sin(20*np.pi*y)*torch.cos(10*np.pi*x) + x*torch.cos(10*np.pi*x)*torch.cos(20*np.pi*y) - 2*y*torch.sin(10*np.pi*x)*torch.sin(20*np.pi*y) + 40*np.pi*(x**2 + 1)*torch.sin(10*np.pi*x)*torch.cos(20*np.pi*y) + 10*np.pi*(y**2 + 1)*torch.sin(10*np.pi*x)*torch.cos(20*np.pi*y)))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "5":
            lap_f = -10*np.pi*(x*(6*x*y*torch.sin(50*np.pi*x)*torch.sin(60*np.pi*y) + (5*y**2 + 5)*torch.cos(50*np.pi*x)*torch.cos(60*np.pi*y)) - y*(5*x*y*torch.cos(50*np.pi*x)*torch.cos(60*np.pi*y) + (6*x**2 + 6)*torch.sin(50*np.pi*x)*torch.sin(60*np.pi*y)) + (x**2 + y**2 + 1)*(-600*np.pi*x*y*torch.sin(60*np.pi*y)*torch.cos(50*np.pi*x) + 5*x*torch.cos(50*np.pi*x)*torch.cos(60*np.pi*y) - 6*y*torch.sin(50*np.pi*x)*torch.sin(60*np.pi*y) + 360*np.pi*(x**2 + 1)*torch.sin(50*np.pi*x)*torch.cos(60*np.pi*y) + 250*np.pi*(y**2 + 1)*torch.sin(50*np.pi*x)*torch.cos(60*np.pi*y)))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "6":
            lap_f = (-x*(33*x*y + 20*y**2 + 20) + y*(33*x**2 + 20*x*y + 33) - (20*x - 33*y)*(x**2 + y**2 + 1))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "7":
            lap_f = (-x*(76*x*y + 15*y**2 + 15) + y*(76*x**2 + 15*x*y + 76) - (15*x - 76*y)*(x**2 + y**2 + 1))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "8":
            lap_f = (-x*(234*x*y + 101*y**2 + 101) + y*(234*x**2 + 101*x*y + 234) - (101*x - 234*y)*(x**2 + y**2 + 1))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "9":
            lap_f = (x*(-230*x*y + 587*y**2 + 587) + y*(230*x**2 - 587*x*y + 230) + (587*x + 230*y)*(x**2 + y**2 + 1))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "10":
            lap_f = (-x*(4375*x*y + 4906*y**2 + 4906) + y*(4375*x**2 + 4906*x*y + 4375) - (4906*x - 4375*y)*(x**2 + y**2 + 1))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "11":
            lap_f = 3*(-x*(-2253*x*y + 2563*y**2 + 2563) - y*(2253*x**2 - 2563*x*y + 2253) - (2563*x + 2253*y)*(x**2 + y**2 + 1))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "12":
            lap_f = (x**2*(-69*x*(y**2 + 1) + 43*y) - y*(-69*x**3*y + 43*x**2 + 43) - (x**2 + y**2 + 1)*(69*x**3 - 138*x*(y**2 + 1) + 43*y))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "13":
            lap_f = (-x**2*(69*x*(y**2 + 1) + 215*y**5) + y**2*(69*x**3 + 215*y**3*(x**2 + 1)) - (x**2 + y**2 + 1)*(69*x**3 - 138*x*(y**2 + 1) - 215*y**5 + 860*y**3*(x**2 + 1)))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "14":
            lap_f = 3*(-x*(-324*x*y**3 + 35*y**2 + 35) + y**2*(35*x - 324*y*(x**2 + 1)) - (35*x + 324*y**3 - 648*y*(x**2 + 1))*(x**2 + y**2 + 1))/(x**2 + y**2 + 1)**2
        elif sys.argv[1] == "15":
            lap_f = 9*(x**2*(425*x**3*(y**2 + 1) + 26*y**3) - y**2*(425*x**5 + 26*y*(x**2 + 1)) + (x**2 + y**2 + 1)*(425*x**5 - 1700*x**3*(y**2 + 1) - 26*y**3 + 52*y*(x**2 + 1)))/(x**2 + y**2 + 1)**2
        else:
            raise NotImplementedError()
        
    elif config["surface"] == "ellipsoid":
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        if sys.argv[1] == "1":
            lap_f = 1296*(4*x - 9*y + 36*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "2":
            lap_f = 1296*(80*x - 297*y + 2628*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "3":
            lap_f = 15552*(5*x - 57*y - 273*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "4":
            lap_f = 11664*(-4*x - 53*y - 112*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "5":
            lap_f = 1296*(-304*x + 243*y + 2016*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "6":
            lap_f = 15552*(5*x - 57*y - 273*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "7":
            lap_f = 2592*(202*x - 1053*y - 8604*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "8":
            lap_f = 2592*(-1174*x - 1035*y - 17064*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "9":
            lap_f = 7776*(-596*x + 51*y - 2580*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "10":
            lap_f = 46656*(36*x - 25*y + 345*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "11":
            lap_f = 7776*(-2164*x + 561*y - 59136*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "12":
            lap_f = 15552*(2899*x + 4470*y + 3705*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "13":
            lap_f = 1296*(19624*x - 39375*y + 31320*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "14":
            lap_f = 5184*(-23*x + 21285*y - 78777*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        elif sys.argv[1] == "15":
            lap_f = 3888*(10252*x + 20277*y + 53184*z)*(x**2 + y**2 + z**2 - 14)/(16*x**2 + 81*y**2 + 1296*z**2)**2
        else:
            raise NotImplementedError()
    
    else:
        print(usage_msg)
        exit()

    return lap_f.squeeze()

# Sample n points uniformly on the surface
def sample_in_domain(n):
    if config["surface"] == "heightfield":
        """Uniformly sample points on z = 0.5(x^2 + y^2) by surface area."""
        xmin, xmax = config["domain"]["min"], config["domain"]["max"]
        ymin, ymax = config["domain"]["min"], config["domain"]["max"]

        def height(x, y):
            return 0.5 * (x**2 + y**2)

        def grad_norm(x, y):
            # |∇h| = sqrt((∂h/∂x)^2 + (∂h/∂y)^2)
            return np.sqrt(x**2 + y**2)

        # Compute max area scaling factor for rejection sampling
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        W = np.sqrt(1 + grad_norm(X, Y)**2)
        w_max = W.max()

        vec = []
        while len(vec) < n:
            xy = np.random.uniform(xmin, xmax, (n*2, 2))  # oversample
            x, y = xy[:, 0], xy[:, 1]
            w = np.sqrt(1 + grad_norm(x, y)**2)
            keep = np.random.rand(len(x)) < w / w_max
            x, y = x[keep], y[keep]
            z = height(x, y)
            vec.extend(np.stack([x, y, z], axis=1))
        vec = np.array(vec[:n])
        return vec

    elif config["surface"] == "ellipsoid":
        a, b, c = 3.0, 2.0, 1.0
        # Sample uniformly on the unit sphere
        vec = np.random.uniform(config["domain"]["min"], config["domain"]["max"], (n, 3))
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)  # Normalize to unit length

        # Scale to ellipsoid
        vec[:, 0] *= a
        vec[:, 1] *= b
        vec[:, 2] *= c

        return vec

    else:
        print(usage_msg)
        exit()

def get_normals(v, S_theta=None):
    if config["NN"] == "withNN" or config["NN"] == "withNN_mesh":
        n = S_theta(v)
        n = n / torch.linalg.norm(n, dim=1, keepdim=True)
    
    elif config["NN"] == "noNN":
        if config["surface"] == "heightfield":
            n = torch.zeros_like(v)
            n[:, 0] = -v[:, 0]
            n[:, 1] = -v[:, 1]
            n[:, 2] = 1.0
            n = n / torch.linalg.norm(n, dim=1, keepdim=True)

        elif config["surface"] == "ellipsoid":
            n = v.clone()
            n[:, 0] = 2*n[:, 0]/9.0
            n[:, 1] = 2*n[:, 1]/4.0
            n[:, 2] = 2*n[:, 2]/1.0
            n = n / torch.linalg.norm(n, dim=1, keepdim=True)

    else:
        print(usage_msg)
        exit()

    return n

def get_surface_laplacian(model, v_cart, S_theta=None):

    def model_scalar(v_):
        return model(v_).view(-1).requires_grad_(True)  # Ensure scalar output
        
    u_pred = model_scalar(v_cart)

    n = get_normals(v_cart, S_theta)

    grad_f = torch.autograd.grad(u_pred, v_cart, torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
    
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
    if config["surface"] == "heightfield":
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

    if config["NN"] == "withNN":
        S_theta = MLP_normals(n=64, n_layers=5, in_dim=3, out_dim=3)
        S_theta.to(device=device)
        if config["surface"] == "heightfield":
            S_theta.load_state_dict(torch.load("../data/model_hf_normal.pth", weights_only=True, map_location=device))
            
        elif config["surface"] == "ellipsoid":
            S_theta.load_state_dict(torch.load("../data/model_ellipsoid_normal.pth", weights_only=True, map_location=device))
        S_theta.requires_grad_(True)

    elif config["NN"] == "withNN_mesh":
        S_theta = MLP_normals(n=64, n_layers=5, in_dim=3, out_dim=3)
        S_theta.to(device=device)
        if config["surface"] == "heightfield":
            S_theta.load_state_dict(torch.load("../data/model_hf_normal_mesh.pth", weights_only=True, map_location=device))
        elif config["surface"] == "ellipsoid":
            S_theta.load_state_dict(torch.load("../data/model_ellipsoid_normal_mesh.pth", weights_only=True, map_location=device))
        S_theta.requires_grad_(True)

    else:
        S_theta = None

    losses = []
    for i in (pbar:= tqdm(range(config["architecture"]["max_iter"]))):
        try:
            optimizer.zero_grad() # Zero gradients

            v_cart = sample_in_domain(n)
            v_cart = torch.tensor(v_cart, dtype=torch.float32, device=device).requires_grad_(True)

            laplacian_pred = get_surface_laplacian(l_model, v_cart, S_theta)

            true_lap = laplacian_f(v_cart)

            loss = torch.linalg.norm(laplacian_pred - true_lap, 2)**2
            
            ## DIRICHLET CONDITION ##
            if config["bc"] == "dirichlet" and config["surface"] != "ellipsoid":
                bdry_points = get_bdry_points(n, device)
                loss = loss + 100*(torch.linalg.norm(l_model(bdry_points).squeeze() - f_torch(bdry_points), 2)**2)  # Dirichlet boundary condition

            ## NEUMANN CONDITION ##
            elif config["bc"] == "neumann" and config["surface"] != "ellipsoid":
                bdry_points = get_bdry_points(n, device)

                grad_bdry = torch.autograd.grad(l_model(bdry_points).squeeze(), bdry_points, torch.ones_like(bdry_points[:, 0]), create_graph=True, retain_graph=True)[0]
                true_grad_bdry = torch.autograd.grad(f_torch(bdry_points).squeeze(), bdry_points, torch.ones_like(bdry_points[:, 0]), create_graph=True, retain_graph=True)[0]

                if config["surface"] == "heightfield":
                    n_surf = get_normals(bdry_points, S_theta)

                grad_bdry_surface = grad_bdry - torch.sum(grad_bdry * n_surf, dim=1, keepdim=True) * n_surf
                true_grad_bdry_surface = true_grad_bdry - torch.sum(true_grad_bdry * n_surf, dim=1, keepdim=True) * n_surf

                n_boundary = torch.ones_like(bdry_points)
                if config["surface"] == "heightfield":
                    num = bdry_points.shape[0] // 4
                    # left boundary
                    n_boundary[0:num] = torch.tensor([-1,0,0], device=bdry_points.device)
                    # right boundary
                    n_boundary[num:2*num] = torch.tensor([1,0,0], device=bdry_points.device)
                    # top boundary
                    n_boundary[2*num:3*num] = torch.tensor([0,1,0], device=bdry_points.device)
                    # bottom boundary
                    n_boundary[3*num:4*num] = torch.tensor([0,-1,0], device=bdry_points.device)

                n_boundary_tan = n_boundary - torch.sum(n_boundary * n_surf, dim=1, keepdim=True) * n_surf
                n_boundary_tan = n_boundary_tan / torch.linalg.norm(n_boundary_tan, dim=1, keepdim=True)

                flux_pred = torch.sum(grad_bdry_surface * n_boundary_tan, dim=1)
                flux_true = torch.sum(true_grad_bdry_surface * n_boundary_tan, dim=1)

                loss = loss + 100*(torch.linalg.norm(flux_pred - flux_true, 2)**2)  # Neumann boundary condition

                # --- enforce zero-mean solution on heightfield ---
                u_pred = l_model(v_cart).squeeze()
                metric = torch.sqrt(1 + v_cart[:, 0]**2 + v_cart[:, 1]**2)      # surface metric factor
                mean_u = torch.sum(u_pred * metric) / torch.sum(metric)
                loss = loss + 100*mean_u**2
            
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

            # Adjust for constant shift
            const = np.mean(true - pred)
            pred = pred + const
            
            l2_loss = np.linalg.norm(true - pred, 2) / np.linalg.norm(true, 2)
            print(l2_loss)

            # Save the model
            torch.save(lap_model.state_dict(), f"{save_dir}/example{sys.argv[1]}/model_{j}_{i}_0.pth")

def plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        filenames = glob.glob(f"{save_dir}/example{sys.argv[1]}/*.pth")
    except:
        print(usage_msg)
        exit()
    dof = []
    losses = []

    V = sample_in_domain(100000)

    for file in filenames:
        print(file)
        n_layers = int(file.split("_")[-2])
        size_layer = int(file.split("_")[-3])

        model = MLP(n=size_layer, n_layers=n_layers, in_dim=3, out_dim=1)
        model.to(device=device)
        model.load_state_dict(torch.load(file, weights_only=True, map_location=device))

        pred = model(torch.Tensor(V).to(device=device)).squeeze().detach().cpu().numpy()
        true = f(V)

        if config["bc"] != "dirichlet":
            const = np.mean(true - pred)
            pred = pred + const

        l2_loss = np.linalg.norm(pred - true, 2) / np.linalg.norm(true, 2)
        print("Depth and Width:", n_layers, size_layer)
        print("PINN Loss:", l2_loss)
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
    for iter in range(5):
        print(iter)
        if config["surface"] == "heightfield":
            v_mesh, f_mesh = gpy.read_mesh(f"../data/heightfield_{iter}.obj")
            f_ = laplacian_f(torch.Tensor(v_mesh).to(device=device)).detach().cpu().numpy()
        elif config["surface"] == "ellipsoid":
            v_mesh, f_mesh = gpy.read_mesh(f"../data/ellipsoid_{iter}.obj")
            f_ = laplacian_f(torch.Tensor(v_mesh).to(device=device)).detach().cpu().numpy()

        M_mesh = gpy.massmatrix(v_mesh, f_mesh)
        M_mesh = sp.sparse.csc_matrix(M_mesh)
        f_ = -M_mesh @ f_

        BV = gpy.boundary_vertices(f_mesh)
        
        lap_mesh = gpy.cotangent_laplacian(v_mesh, f_mesh)

        # Apply Dirichlet Boundary
        # Section 4.3 - https://web.stanford.edu/class/energy281/FiniteElementMethod.pdf
        if config["surface"] == "heightfield":
            for i in BV:
                lap_mesh[i, :] = 0
                lap_mesh[i, i] = 1
            f_[BV] = f(v_mesh[BV])

        u = sp.sparse.linalg.spsolve(lap_mesh, f_)

        if config["surface"] == "ellipsoid":
            const = np.mean(f(v_mesh) - u)
            u = u + const
        
        l2_loss = np.linalg.norm(f(v_mesh) - u, 2) / np.linalg.norm(f(v_mesh), 2)
        print("FEM Loss:", l2_loss)
        fem_loss.append(l2_loss)
        fem_dof.append(v_mesh.shape[0])

    dof = np.array(dof)
    fem_dof = np.array(fem_dof)
    losses = np.array(losses)
    fem_loss = np.array(fem_loss)

    dof_sorted = np.sort(dof)
    losses_sorted = losses[np.argsort(dof)]
    fem_dof_sorted = np.sort(fem_dof)
    fem_loss_sorted = fem_loss[np.argsort(fem_dof_sorted)]

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

    # fit a line to the data
    coeffs = np.polyfit(np.log(x), np.log(y), 1)

    plt.loglog(x, y, label="PINN", marker='o')
    plt.loglog(x, 1/x, label="1/x", color="red", linestyle='--')
    plt.loglog(x, (1/x)**2, label="1/x^2", color="green", linestyle='--')
    plt.loglog(fem_dof_sorted, fem_loss_sorted, label="FEM", marker='o')

    plt.xlabel("Total Trainable Parameters")
    plt.ylabel("Relative L2 loss")
    plt.title(f"Slope: {coeffs[0]}")
    plt.legend()
    plt.savefig(f"{save_dir}/convergence_example_{sys.argv[1]}.png")

if __name__ == "__main__":
    save_dir = f"poisson_results/{config['surface']}/{config['NN']}/{config['bc']}/domain_{config['domain']['min']}to{config['domain']['max']}"

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