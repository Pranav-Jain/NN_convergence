import sympy as sp
import polyscope as ps
import numpy as np

# symbols
x, y, z = sp.symbols('x y z', real=True)

# level set (Dziuk surface)
phi = (x - z**2)**2 + y**2 + z**2 - 1

# function u
u = x*y
sigma = 1

# gradient and normal
grad_phi = sp.Matrix([sp.diff(phi, v) for v in (x, y, z)])
n = grad_phi / sp.sqrt(grad_phi.dot(grad_phi))

# gradient of u
grad_u = sp.Matrix([sp.diff(u, v) for v in (x, y, z)])

# identity matrix
I = sp.eye(3)

# tangential gradient
P = I - n*n.T
grad_u_tan = P * grad_u

# surface Laplacian
# laplace_surface = sum(
#     sp.diff(grad_u_tan[i], v)
#     for i, v in enumerate((x, y, z))
# )
H_x = sp.diff(grad_u_tan, x)
H_x = P * H_x
H_y = sp.diff(grad_u_tan, y)
H_y = P * H_y
H_z = sp.diff(grad_u_tan, z)
H_z = P * H_z
laplace_surface = H_x[0] + H_y[1] + H_z[2]


# screened Poisson RHS
rhs = laplace_surface - sigma*u

# simplify
rhs_simplified = sp.simplify(rhs)

print("Surface Laplacian Δ_Γ u =")
print(sp.simplify(laplace_surface))
print("\nScreened Poisson RHS f =")
print(rhs_simplified)
