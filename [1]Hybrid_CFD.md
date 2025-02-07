# Notes

## Article [1]

[[1]](https://github.com/Weidsn/Quantum_Computing_Collaboration/blob/main/A%20hybrid%20quantum-classical%20CFD%20methodology%20with%20benchmark%20HHL%20solutions.pdf) *A hybrid quantum-classical CFD methodology with benchmark HHL solutions* (June 2022)

## Abstract

Decomposing CFD problems into a linear combination of unitariary matrices (LCU), whose coefficients can be computed using HHL.

## Navier-Stokes equations

### For incompressible, laminar flow

This is a fluid that cannot be compressed.

As the flow is laminar, as opposed turbulent, the fluid flows in layers with minimal mixing. 

As a result, the `Renauld number` of the fluid, $Re$, is low.

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0
$$

$$
\rho \frac{\partial \mathbf{u}}{\partial t} + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) = -\nabla p + \mu \nabla^2 \mathbf{u}
$$

Where $\rho$ is the fluid density, $\mathbf{u}$ is the velocity, $p$ is the pressure, $\mu$ is the viscosity.

## Takeaway

In CFD, we first create a mesh (or grid) of points in space.
The flow of the fluid is then measured (or calculated) at each point in the mesh.

For instance, a 2D 5 x 5 mesh consists of 25 points in total.

Due to the limitation of NISQ computers, many modern CFD solvers first linearize and decompose the equations (or matrices) on classical computers, then solve the linear system of equations (or matrices) on quantum computers.

The problem is divided into `momentum equations` and `pressure correction` equations.
The latter is much more costly to solve.

## Process

The pressure correction equations can be written as a linear system of equations.

$$Ap'=b$$

To prepare for quantum states, we need to

### 1. Transform $A$ into a Hermitian matrix

$$
H =
\begin{pmatrix}
0 & A \\
A^\dagger & 0
\end{pmatrix}$$

We need to solve

$$H 
\begin{pmatrix*} 
0 \\
x
\end{pmatrix*}=
\begin{pmatrix} 
b \\
0
\end{pmatrix}$$


To prepare for quantum computing, we need to

### 3. Convert $H$ into a linear combination of unitary matrices (LCU)

$$
H = \sum_{i=1}^{n} \alpha_i U_i
$$

where $U_i$ is a unitary matrix, and $\alpha_i$ is the corresponding coefficient.


### 4. Implement a quantum algorithm, such as the HHL, to solve for $x$.

## Famous test case - lid driven cavity

### Approaches to decomposition

An approach based on the orthogonality of grand sums of Hadamard products of Pauli stings.
Trace orthogonality of products of Pauli matrices, commonly termed Pauli strings.
The former is a lot faster. 

Exponentiation of the sum of unitaries using Trotter product formula creates a single
unitary matrix. 

#### Note: Partial unitary decomposition

Unitaries in the LCU with small coefficient, $\alpha_i$, may be ignored
for a price in convergence or convergence speed.

In the 5 x 5 mesh test case, the program still converges after ignoring 50% of unitaries.[1]

If full LCU decomposition is still required, as it is in the paper, doing this only reduces quantum computing workloads.

### State preparation

#### Amplitudes loader

#### Binary tree loader

[45] I. F. Araujo, D. K. Park, F. Petruccione, and A. J. da Silva, “A divide-and-conquer algorithm for quantum state
preparation,” Scientific Reports, vol. 11, no. 1, pp. 1–12, 2021.

### Phase Estimation

### Eigenvalue Inversion

Polynomial State Preparation (PSP) is used.

### Ancilla measurement

Ancilla qubit.

### Re-dimensionalization

The problem of quantum state measurement is ignored.

The output is a unitary classical vector, which needs to be re-dimensionalized
to uncover the parameters we need.

## Article [2]

[[2]](https://github.com/Weidsn/Quantum_Computing_Collaboration/blob/main/A%20hybrid%20quantum-classical%20CFD%20methodology%20with%20benchmark%20HHL%20solutions.pdf) *A Performance Study of Variational Quantum Algorithms for Solving the Poisson Equation on a Quantum Computer* (May 2023)

## Abstract

Abstract: Solving Poisson equations using a variational quantum algorithm (VQA), i.e., Variational Quantum Linear Solver (VQLS) on noisy intermediate scale quantum (NISQ) computers. Results were not promising. VQLS is further discussed in [[5]](https://github.com/Weidsn/Quantum_Computing_Collaboration/blob/main/Variational%20quantum%20linear%20solver.pdf).

## Notes

Decomposition and HHL methods are less NISQ friendly. 

Use quantum linear solvers (VQLS) to solve PDEs, e.g., Poisson equations.

In VQLS, quantum computers is only used to computer
simpler tasks, such as to estimate "cost functionals."

### Error cancellation techniques (IBM)

Probabilitic error cancellation (PEC) was added to IBM's Qiskit. PEC attempts to invert physical noise of the quantum device using a pre-trained Pauli noise model. 

### Poisson equations

Consider a very special case of a Poisson equation descretized by the finite element method (FEM).

$$
Au = f
$$
where
$$
\mathbf{A} = \frac{1}{h}
\begin{pmatrix}
2 & -1 & 0 & \cdots \\
-1 & 2 & -1 & \cdots \\
0 & -1 & 2 & \cdots \\
\vdots & \vdots & \vdots & \ddots & -1 \\
& & & -1 & 2
\end{pmatrix}
$$

$$A = 2I^{⊗n} − I^{⊗{n−1}} ⊗ X + R$$

Consider two special cases of $f$. 

1. $f$ is constant and is represented by quantum state

$$
|f_C \rangle := H^{\otimes n} |0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{\{0,1\}^n} |0...010...\rangle
$$

2. $f$ contains a continuous jump.

$$
|f_C \rangle := H^{\otimes {n-1}} \otimes X |0\rangle^{\otimes n} = \frac{1}{\sqrt{2^{n-1}}} \sum_{\{0,1\}^{n-1}} |0...010...\rangle \otimes |1\rangle
$$

where $H$ and $X$ are Hadamard and Pauli-X gates, respectively.


### 
