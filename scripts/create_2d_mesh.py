import numpy as np
import gpytoolbox as gpy

# create 2d mesh
def create_2d_mesh(xmin, xmax, ymin, ymax, nx, ny):
    # create grid points
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xv, yv = np.meshgrid(x, y)
    zv = np.zeros_like(xv)

    # flatten points
    vertices = np.vstack([xv.flatten(), yv.flatten(), zv.flatten()]).T

    # create faces
    faces = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            v0 = i * nx + j
            v1 = v0 + 1
            v2 = v0 + nx
            v3 = v2 + 1
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    faces = np.array(faces)

    return vertices, faces

if __name__ == "__main__":
    v_mesh, f_mesh = create_2d_mesh(-1, 1, -1, 1, 256, 256)
    gpy.write_mesh("2d_mesh.obj", v_mesh, f_mesh)