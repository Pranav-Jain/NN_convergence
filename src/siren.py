import torch
import torch.nn as nn
import numpy as np

# SIREN implementation
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
    
class MLP_normals(nn.Module):
    def __init__(self, n=512, n_layers=3, in_dim=3, out_dim=1, w0=30.0):
        super().__init__()
        layers = []

        # First SIREN layer (high-frequency)
        layers.append(SIRENLayer(in_dim, n, is_first=True, w0=1.0))

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

        output = output / torch.linalg.norm(output, dim=1, ord=2, keepdim=True) # Normalize to unit length

        return output.squeeze()

class MLP_tanh(nn.Module):
    def __init__(self, n=512, n_layers=3, in_dim=3, out_dim=1):
        super().__init__()
        layers = []

        layers.append(nn.Linear(in_dim, n))
        layers.append(nn.Tanh())

        for _ in range(n_layers):
            layers.append(nn.Linear(n, n))
            layers.append(nn.Tanh())

        self.trunk = nn.Sequential(*layers)
        self.final_layer = nn.Linear(n, out_dim, bias=True)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        features = self.trunk(x)
        output = self.final_layer(features)

        return output.squeeze()
    
class MLP_sigmoid(nn.Module):
    def __init__(self, n=512, n_layers=3, in_dim=3, out_dim=1):
        super().__init__()
        layers = []

        layers.append(nn.Linear(in_dim, n))
        layers.append(nn.Sigmoid())

        for _ in range(n_layers):
            layers.append(nn.Linear(n, n))
            layers.append(nn.Sigmoid())
            
        self.trunk = nn.Sequential(*layers)
        self.final_layer = nn.Linear(n, out_dim, bias=True)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        features = self.trunk(x)
        output = self.final_layer(features)

        return output.squeeze()