import numpy as np
import matplotlib.pyplot as plt
import qrisp
from qrisp import QuantumFloat, jrange, cx, control, h, QFT, invert, ry, auto_uncompute, QuantumSession
from qrisp.operators.qubit import QubitOperator
from qrisp.alg_primitives.prepare import prepare


# ==== PROBLEM PARAMETERS ====
# Define grid size, timestep info, physical parameters, solver tolerances, and quantum precision
nx, ny = 4, 4                    # Grid dimensions (x,y)
max_step = 5                    # Number of simulation timesteps
visc = 0.1                     # Fluid viscosity
u_wall = 1.0                   # Velocity at moving lid boundary
dt = 0.005                    # Temporal step size
dx = 1.0 / (nx - 1)            # Spatial grid spacing (uniform)
beta = 1.5                    # Relaxation parameter for classical SOR solver
max_iter = 100                # Maximum iterations for SOR solver
max_err = 1e-4                # Convergence criterion for classical solver
precision = 6                 # Number of qubits for Quantum Phase Estimation

# ==== MATRIX & HELPER FUNCTIONS ====

def get_A(nx, ny):
    """
    Construct the discretized Laplacian matrix A for the interior points of the grid.
    This matrix represents the Poisson equation for the streamfunction.
    """
    N_x, N_y = nx - 2, ny - 2
    N = N_x * N_y
    main = 4 * np.ones(N)
    off1 = -np.ones(N - 1)
    idx = np.arange(N - 1)
    wrap = (idx + 1) % N_x == 0
    off1[wrap] = 0.0
    offNx = -np.ones(N - N_x)
    data = [main, off1, off1, offNx, offNx]
    offsets = [0, -1, 1, -N_x, N_x]
    from scipy.sparse import diags
    return diags(data, offsets, shape=(N, N)).toarray()

def get_b(omega, dx):
    """
    Produce the right-hand side vector b for the linear system A * psi = b
    by flattening the scaled vorticity interior.
    """
    return omega[1:-1, 1:-1].flatten() * dx ** 2

def eigenrange_laplacian(nx, ny):
    """
    Estimate the spectral range (minimum and maximum eigenvalues) of matrix A,
    useful for scaling prior to quantum phase estimation.
    """
    Nx, Ny = nx - 2, ny - 2
    min_val = 4 - 2 * np.cos(np.pi / (Nx + 1)) - 2 * np.cos(np.pi / (Ny + 1))
    max_val = 4 + 2 * np.cos(np.pi / (Nx + 1)) + 2 * np.cos(np.pi / (Ny + 1))
    return min_val, max_val

def scale_A(A, lambda_min, lambda_max, precision):
    """
    Linearly scale and shift matrix A to map its eigenvalues to a more
    favorable range for QPE.
    """
    dim = A.shape[0]
    scale_factor = (1 - 1 / 2 ** precision) / (lambda_max - lambda_min)
    shift = 1 / 2 ** precision
    return (A - lambda_min * np.identity(dim)) * scale_factor + shift * np.identity(dim)

def scale_b(b):
    """
    Normalize vector b to unit norm required for quantum amplitude encoding.
    """
    b_norm = np.linalg.norm(b)
    return b / b_norm if b_norm else b

def pad_to_power_of_two(A, b):
    """
    Zero-pad matrix A and vector b to next power-of-two dimension,
    required by quantum hardware constraints.
    """
    N = A.shape[0]
    n_bits = (N - 1).bit_length()
    N2 = 1 << n_bits
    M = N2 - N
    if M == 0:
        return A.copy(), b.copy()
    Apad = np.pad(A, ((0, M), (0, M)), mode='constant')
    bpad = np.pad(b, (0, M), mode='constant')
    return Apad, bpad

def recover_full_psi_from_flat(psi_vec, nx, ny, boundary=0.0):
    """
    Reshape a flattened interior vector into the full grid layout with boundaries.
    """
    interior = psi_vec.reshape((nx - 2, ny - 2))
    psi_full = np.full((nx, ny), boundary, dtype=psi_vec.dtype)
    psi_full[1:-1, 1:-1] = interior
    return psi_full

# ==== CLASSICAL SOR SOLVER ====

def solve_streamfunction_sor(psi_init, omega, dx, beta, max_iter, max_err):
    """
    Solve the Poisson equation for streamfunction psi via classical Successive Over-Relaxation (SOR).
    """
    nx, ny = psi_init.shape
    psi = psi_init.copy()
    for it in range(1, max_iter + 1):
        psi_old = psi.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                psi[i, j] = (
                    (1 - beta) * psi[i, j]
                    + (beta * 0.25)
                    * (
                        psi[i + 1, j]
                        + psi[i - 1, j]
                        + psi[i, j + 1]
                        + psi[i, j - 1]
                        + dx * dx * omega[i, j]
                    )
                )
        err = np.sum(np.abs(psi - psi_old))
        if err < max_err:
            return psi, it, err
    return psi, max_iter, err

# ==== VORTICITY TRANSPORT ====

def vorticity_transport(psi, omega):
    """
    Compute the updated vorticity field using convection-diffusion and boundary conditions.
    """
    omega_next = omega.copy()
    omega0 = omega.copy()
    nx, ny = psi.shape

    # Boundary conditions for vorticity
    omega_next[1 : nx - 1, 0] = -2.0 * psi[1 : nx - 1, 1] / (dx * dx)
    omega_next[1 : nx - 1, ny - 1] = (
        -2.0 * psi[1 : nx - 1, ny - 2] / (dx * dx) - 2.0 * u_wall / dx
    )
    omega_next[0, 1 : ny - 1] = -2.0 * psi[1, 1 : ny - 1] / (dx * dx)
    omega_next[nx - 1, 1 : ny - 1] = -2.0 * psi[nx - 2, 1 : ny - 1] / (dx * dx)

    # Interior vorticity transport (convection + diffusion)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            adv = -0.25 * (
                (psi[i, j + 1] - psi[i, j - 1]) * (omega0[i + 1, j] - omega0[i - 1, j])
                - (psi[i + 1, j] - psi[i - 1, j]) * (omega0[i, j + 1] - omega0[i, j - 1])
            ) / (dx * dx)
            diff = visc * (
                omega0[i + 1, j]
                + omega0[i - 1, j]
                + omega0[i, j + 1]
                + omega0[i, j - 1]
                - 4.0 * omega0[i, j]
            ) / (dx * dx)
            omega_next[i, j] = omega0[i, j] + dt * (adv + diff)

    return omega_next

# ==== QUANTUM HHL SOLVER ====

def get_qrisp_unitary(A, t):
    """
    Convert the classical matrix A into a Qrisp unitary via Trotterization.
    """
    H = QubitOperator.from_matrix(A).to_pauli()

    def U(qf):
        H.trotterization()(qf, t=t, steps=1)

    return U

def QPE(psi, U, precision):
    """
    Perform Quantum Phase Estimation on unitary U.
    Returns a quantum float representing the eigenvalue register.
    """
    res = QuantumFloat(precision, -precision)
    h(res)
    for i in range(precision):
        for _ in range(2 ** i):
            with control(res[i]):
                U(psi)
    return QFT(res, inv=True)

@auto_uncompute
def controlled_rotation_inversion(qpe_res, ancilla, c=1.0):
    """
    Approximate HHL's eigenvalue inversion step with controlled rotations.
    Automatically uncomputes via the decorator.
    """
    n = qpe_res.size
    for i in range(n):
        angle = 2 ** i * c  # This simple scaling should be refined for real cases
        with control(qpe_res[i]):
            ry(angle, ancilla)

def HHL_qrisp(b, A, n, precision):
    """
    Build, compile, and simulate the HHL quantum circuit solving A |x> = |b>.
    Returns the quantum float register, session, and simulated statevector.
    """
    qs = QuantumSession()
    qf = QuantumFloat(n, qs=qs)

    # Normalize and encode input vector |b>
    b_norm = b / np.linalg.norm(b)
    prepare(qf, b_norm, reversed=True)

    U = get_qrisp_unitary(A, t=-np.pi)
    qpe_res = QPE(qf, U, precision)

    anc = QuantumFloat(1, qs=qs)  # Ancilla for inversion
    controlled_rotation_inversion(qpe_res, anc, c=1.0)

    with invert():
        QPE(qf, U, precision)
        controlled_rotation_inversion(qpe_res, anc, c=1.0)  # Uncompute rotations

    # Reorder qubits after algorithm
    for i in jrange(qf.size // 2):
        qrisp.swap(qf[i], qf[n - i - 1])

    qs.compile()
    quantum_state = qs.statevector()

    return qf, qs, quantum_state

# ==== MAIN SIMULATION LOOP ====

psi = np.zeros((nx, ny))
omega = np.zeros((nx, ny))
sor_errors = []
quantum_solutions = []
classical_solutions = []

# Prepare inputs
A = get_A(nx, ny)
print(A)
lambda_min, lambda_max = eigenrange_laplacian(nx, ny)
A_scaled = scale_A(A, lambda_min, lambda_max, precision)
A_padded = pad_to_power_of_two(A_scaled)

for tstep in range(max_step):
    # Update vorticity via transport equation
    omega = vorticity_transport(psi, omega)

    # Classical SOR solution for streamfunction
    psi_classical, sor_iters, sor_err = solve_streamfunction_sor(psi, omega, dx, beta, max_iter, max_err)
    sor_errors.append(sor_err)
    classical_solutions.append(psi_classical.copy())

    # Prepare quantum inputs
    b = get_b(omega, dx)
    b_scaled = scale_b(b)
    b_padded = pad_to_power_of_two(b_scaled)
    n_qubits = int(np.log2(A_padded.shape[0]))

    norm_b = np.linalg.norm(b_padded)

    if norm_b == 0:
        # Fallback to classical solver if zero vector encountered
        psi_quantum = psi_classical.copy()
        print(f"Step {tstep}: Zero vector b encountered, skipping quantum HHL solve.")
    else:
        # Run quantum HHL solver
        qf, qs, quantum_state = HHL_qrisp(b_padded, A_padded, n_qubits, precision)
        psi_quantum = recover_full_psi_from_flat(quantum_state, nx, ny)
        quantum_solutions.append(psi_quantum.copy())

    # Use classical solution as initial guess for next step
    psi = psi_classical.copy()

# ==== POSTPROCESSING & PLOTTING ====

plt.figure(figsize=(10, 4))

# Contour plot of classical SOR solution
plt.subplot(1, 2, 1)
plt.contour(classical_solutions[-1], 20)
plt.title('Streamfunction (SOR Solver)')

# Sorted value plot of classical solution
plt.subplot(1, 2, 2)
plt.plot(np.sort(classical_solutions[-1].flatten()), 'bo', label='SOR Classical')
# Uncomment to plot quantum solution comparison if available
# plt.plot(np.sort(quantum_solutions[-1].flatten()), 'ro', label='Quantum HHL')
plt.legend()
plt.tight_layout()
plt.show()
