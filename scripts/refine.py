import igl
import numpy as np
import gpytoolbox as gpy
import sys

if __name__ == "__main__":
    v_mesh, f_mesh = gpy.read_mesh(f"data/{sys.argv[1]}_0.obj")

    v_ref, f_ref = igl.loop(v_mesh, f_mesh, sys.argv[2])
    # k = igl.gaussian_curvature(v_mesh, f_mesh)
    # v_mesh_refined = S @ v_mesh
    gpy.write_mesh(f"data/{sys.argv[1]}_{sys.argv[2]}.obj", v_ref, f_ref)