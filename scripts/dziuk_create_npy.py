import numpy as np
import pandas as pd
import polyscope as ps
import gpytoolbox as gpy

v_mesh, f_mesh = gpy.read_mesh("./data/dziuk.obj")
# ps.init()
# ps.register_surface_mesh("dziuk", v_mesh, f_mesh)
# ps.get_surface_mesh("dziuk").add_scalar_quantity("analytical", analytical, defined_on='vertices')
# ps.get_surface_mesh("dziuk").add_scalar_quantity("estimate", estimate, defined_on='vertices')
# ps.get_surface_mesh("dziuk").add_scalar_quantity("error", np.abs(analytical - estimate), defined_on='vertices')

# load csv
df = pd.read_csv("dziuk.csv")

# extract column
# analytical = v_mesh[:, 0] * v_mesh[:, 1]
# analytical = df["analytical"].to_numpy()
analytical = df["P [X]"].to_numpy() * df["P [Y]"].to_numpy()
estimate = df["estimate [X]"].to_numpy()

const = np.mean(analytical - estimate)
estimate += const

# save as npy
# np.save("pwos_analytical.npy", analytical)
np.save("pwos_estimate.npy", estimate)
np.save("pwos_error.npy", np.abs(analytical - estimate))
print(np.sqrt(np.mean((analytical - estimate)**2)))
