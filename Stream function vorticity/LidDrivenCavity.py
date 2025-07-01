#%%
# === Algorithm Outline ===
# 1) Initialize grid, physical parameters, and field arrays for ψ (streamfunction) and ω (vorticity).
# 2) Time-marching loop (for each time step):
#    a) Solve Poisson equation ∇²ψ = -ω for ψ via Successive Over-Relaxation (SOR).
#    b) Compute vorticity at the solid boundaries directly from ψ (using ω = -∇²ψ) and lid velocity.
#    c) Advect and diffuse interior ω using the vorticity transport equation (explicit FTCS).
# 3) After convergence or final time, post-process and plot vorticity and streamfunction.

# Import necessary libraries
from mpl_toolkits.mplot3d import Axes3D  # enables 3D plotting (not directly used here)
import numpy as np                      # numerical arrays and operations
import matplotlib.pyplot as plt         # plotting library
import pandas as pd

from qiskit.visualization import array_to_latex
from qiskit.quantum_info import Operator, SparsePauliOp

from scipy.sparse import diags, eye, kron

# Function to build the Poisson system A ψ = b for given grid and vorticity (dense)
def build_poisson_system(nx, ny, dx, omega):
    N = nx * ny
    A = np.zeros((N, N))
    b = np.zeros(N)
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            # right-hand side b
            b[k] = -omega[i, j] * dx**2
            # diagonal
            A[k, k] = -4.0
            # neighbors: up, down, left, right
            if i > 0:
                A[k, (i-1)*ny + j] = 1.0
            if i < nx-1:
                A[k, (i+1)*ny + j] = 1.0
            if j > 0:
                A[k, i*ny + (j-1)] = 1.0
            if j < ny-1:
                A[k, i*ny + (j+1)] = 1.0
    return A, b

# Function to build a sparse Poisson system using Kronecker sums
# Suitable for Pauli-string decomposition in quantum simulations
def build_sparse_poisson_system(nx, ny, dx, omega):
    """
    Build the sparse Poisson system A ψ = b for a 2D grid.

    Parameters:
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        dx (float): Grid spacing (assumed equal in x and y).
        omega (array_like): 2D array of vorticity values, shape (nx, ny).

    Returns:
        A (scipy.sparse.csr_matrix): Sparse matrix of shape (N, N),
            representing the 5-point Laplacian (∇²) scaled by 1/dx²,
            where N = nx * ny.  It enforces ∇²ψ = -ω.
        b (numpy.ndarray): 1D array of length N, the right-hand side
            vector with entries -ω[i,j] * dx^2, flattened in row-major order.
    """
    # Create 1D Laplacian matrices in x and y directions
    main_x = -2.0 * np.ones(nx)
    off_x  = 1.0  * np.ones(nx-1)
    Tx = diags([off_x, main_x, off_x], [-1, 0, 1], format='csr')
    main_y = -2.0 * np.ones(ny)
    off_y  = 1.0  * np.ones(ny-1)
    Ty = diags([off_y, main_y, off_y], [-1, 0, 1], format='csr')
    # Identity matrices
    Ix = eye(nx, format='csr')
    Iy = eye(ny, format='csr')
    # 2D Laplacian via Kronecker sum: A = I_y ⊗ T_x + T_y ⊗ I_x
    A = (kron(Iy, Tx, format='csr') + kron(Ty, Ix, format='csr')) / (dx*dx)
    # Right-hand side vector
    b = -omega.flatten() * dx**2
    return A, b


# Example: printing the sparse matrix in dense form
# After building A as a CSR matrix you can use:
#    dense_A = A.toarray()        # convert to NumPy array
# or use qiskit for better visualization
# from qiskit.visualization import array_to_latex
# display(array_to_latex(A.toarray()))
# 
# Qiskit: to map A into a quantum operator (SparsePauliOp), you can decompose A
# into a sum of Pauli strings. Here's a sketch:
#    from qiskit.quantum_info import SparsePauliOp, Pauli
#    import numpy as np
#    # Suppose dense_A is your N×N NumPy matrix
#    dense_A = A.toarray()
#    # Generate Pauli basis for N=2^n; here N must be power of 2
#    # Flatten matrix and match to Pauli basis decomposition
#    # This can be done by vectorizing A in the Pauli basis:
#    pauli_labels, pauli_coeffs = [], []
#    for i, pauli in enumerate(Pauli.pauli_basis(num_qubits=n_qubits)):
#        M = SparsePauliOp(pauli).to_matrix()
#        coeff = np.trace(M.conj().T @ dense_A) / (2**n_qubits)
#        if abs(coeff) > 1e-12:
#            pauli_labels.append(pauli.to_label())
#            pauli_coeffs.append(coeff)
#    sparse_pauli_op = SparsePauliOp(pauli_labels, pauli_coeffs)
#    # Now sparse_pauli_op represents A in the Pauli basis


#%%

# === Problem parameters ===
nx = 2       # number of grid points in x-direction
ny = 2       # number of grid points in y-direction
max_step = 200  # total number of time steps to march
visc = 0.1      # kinematic viscosity nu
u_wall = 1.0    # lid (top boundary) horizontal velocity

dt = 0.005      # time step size
dx = 1.0/(nx-1) # grid spacing (uniform) in both x and y

# === SOR (Successive Over-Relaxation) parameters for Poisson solver ===
max_iter = 100   # max iterations per time step for solving psi
beta = 1.5       # relaxation factor (>1 accelerates convergence)
max_err = 1e-3   # convergence tolerance for psi

# === Field arrays initialization ===
psi   = np.zeros((nx, ny))  # streamfunction array
omega = np.zeros((nx, ny))  # vorticity array
omega0 = np.zeros_like(omega)  # temporary array for previous vorticity
# coordinate arrays for plotting
x = np.zeros((nx, ny))
y = np.zeros((nx, ny))

# Fill coordinate arrays assuming domain [0,1]x[0,1]
for i in range(nx):
    for j in range(ny):
        x[i,j] = dx * i  # x-coordinate at grid index i
        y[i,j] = dx * j  # y-coordinate at grid index j

#%%
# Time-marching loop
t_current = 0.0
for tstep in range(max_step):
    # ----- 1) Solve Poisson equation for streamfunction psi -----
    for it in range(max_iter):
        psi_old = psi.copy()
        # Update interior points using SOR
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # 5-point Laplacian + source term (vorticity)
                psi[i,j] = (1 - beta) * psi[i,j] + (beta * 0.25) * (
                    psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1] + dx*dx * omega[i,j]
                )
        # check convergence (L1 norm)
        err = np.sum(np.abs(psi - psi_old))
        if err < max_err:
            break

    # ----- 2) Apply and calculate boundary vorticity at walls -----
    # Vorticity ω is computed at the walls directly from ψ (streamfunction) via ω = -∇²ψ,
    # with an extra term on the moving lid: ω = -2ψ/dx² - 2u_wall/dx for y = 1
    # This sets ω on the bottom, top, left and right boundaries.
    # ----- 2) Apply boundary conditions for vorticity at walls -----
    # bottom wall (y=0): u=0 => psi=0 => vorticity = -2*psi[...]/dx^2
    omega[1:nx-1, 0] = -2.0 * psi[1:nx-1, 1] / (dx*dx)
    # top lid (y=1): u=u_wall, psi=0 => includes lid velocity contribution
    omega[1:nx-1, ny-1] = -2.0 * psi[1:nx-1, ny-2] / (dx*dx) - 2.0 * u_wall / dx
    # left wall (x=0): psi=0
    omega[0, 1:ny-1] = -2.0 * psi[1, 1:ny-1] / (dx*dx)
    # right wall (x=1): psi=0
    omega[nx-1, 1:ny-1] = -2.0 * psi[nx-2, 1:ny-1] / (dx*dx)

    # ----- 3) Compute interior vorticity transport: advect + diffuse -----
    # Interior vorticity ω at each grid point is updated using the vorticity transport equation:
    # ∂ω/∂t + u ∂ω/∂x + v ∂ω/∂y = ν ∇²ω, discretized explicitly (FTCS).
    # Here ω0 holds the previous time step's vorticity.
    # ----- 3) Vorticity transport: advect + diffuse -----
    omega0[:] = omega  # store old vorticity
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # compute convective term: u * d(omega)/dx + v * d(omega)/dy
            adv =  -0.25 * (
                    (psi[i,j+1] - psi[i,j-1]) * (omega0[i+1,j] - omega0[i-1,j])
                  - (psi[i+1,j] - psi[i-1,j]) * (omega0[i,j+1] - omega0[i,j-1])
                ) / (dx*dx)
            # diffusion term: nu * Laplacian(omega)
            diff = visc * (
                    omega0[i+1,j] + omega0[i-1,j]
                    + omega0[i,j+1] + omega0[i,j-1]
                  - 4.0 * omega0[i,j]
                ) / (dx*dx)
            # update vorticity explicitly
            omega[i,j] = omega0[i,j] + dt * (adv + diff)

    t_current += dt  # advance physical time

# === 4) Post-processing: plot results ===
print(f"Converged psi error: {err:.2e}")
# Plot vorticity and streamfunction side-by-side
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.contour(x, y, omega, 40)
plt.title(r"Vorticity $\omega$")
plt.axis('square')

plt.subplot(1,2,2)
plt.contour(x, y, psi, 20)
plt.title(r"Streamfunction $\psi$")
plt.axis('square')

plt.tight_layout()
plt.show()

# %%
