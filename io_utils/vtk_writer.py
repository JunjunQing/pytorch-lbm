"""VTK writer for LBM output (structured grid).

Writes density, velocity, and solid mask as VTK STRUCTURED_POINTS
files readable by ParaView.
"""
import numpy as np
import os


def write_vtk(filename, rho, u, solid=None, dx=1.0, step=0, t=0.0):
    """Write fields to VTK file.

    Parameters
    ----------
    filename : str
        Output path (should end in .vtk).
    rho : ndarray, shape (nx, ny[, nz])
    u : ndarray, shape (dim, nx, ny[, nz])
    solid : ndarray, shape (nx, ny[, nz]), optional
    dx : float
    step : int
    t : float
    """
    dim = u.shape[0]
    if dim == 2:
        nx, ny = rho.shape
        nz = 1
        _write_vtk_2d(filename, rho, u, solid, dx, nx, ny, step, t)
    else:
        nx, ny, nz = rho.shape
        _write_vtk_3d(filename, rho, u, solid, dx, nx, ny, nz, step, t)


def _write_vtk_2d(filename, rho, u, solid, dx, nx, ny, step, t):
    """Write 2D data as 3D VTK with nz=1."""
    rho_out = rho.reshape(nx, ny, 1)
    ux = u[0].reshape(nx, ny, 1)
    uy = u[1].reshape(nx, ny, 1)
    uz = np.zeros_like(ux)
    solid_out = solid.reshape(nx, ny, 1) if solid is not None else None
    _write_vtk_structured(filename, rho_out, ux, uy, uz, solid_out, dx, step, t)


def _write_vtk_3d(filename, rho, u, solid, dx, nx, ny, nz, step, t):
    rho_out = rho
    ux, uy, uz = u[0], u[1], u[2]
    solid_out = solid
    _write_vtk_structured(filename, rho_out, ux, uy, uz, solid_out, dx, step, t)


def _write_vtk_structured(filename, rho, ux, uy, uz, solid, dx, step, t):
    """Write VTK STRUCTURED_POINTS file."""
    nx, ny, nz = rho.shape
    n_points = nx * ny * nz

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"LBM output step={step} t={t:.6e}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN 0 0 0\n")
        f.write(f"SPACING {dx} {dx} {dx}\n")
        f.write(f"POINT_DATA {n_points}\n")

        # Density
        f.write("SCALARS density float\n")
        f.write("LOOKUP_TABLE default\n")
        for val in rho.ravel():
            f.write(f"{val:.6e}\n")

        # Velocity
        f.write("VECTORS velocity float\n")
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    f.write(f"{ux[i,j,k]:.6e} {uy[i,j,k]:.6e} {uz[i,j,k]:.6e}\n")

        # Solid mask
        if solid is not None:
            f.write("SCALARS solid int\n")
            f.write("LOOKUP_TABLE default\n")
            for val in solid.ravel():
                f.write(f"{int(val)}\n")
