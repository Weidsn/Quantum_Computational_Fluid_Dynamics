# Article [1]

[[1]](https://github.com/Weidsn/Quantum_Computing_Collaboration/blob/main/A%20hybrid%20quantum-classical%20CFD%20methodology%20with%20benchmark%20HHL%20solutions.pdf) *A hybrid quantum-classical CFD methodology with benchmark HHL solutions* (June 2022)

## Abstract

Decomposing CFD problems into a linear combination of unitariary matrices (LCU), whose coefficients can be computed using HHL.

This paper demonstrates the potential advantage of hybrid quantum-classical algorithms for CFD problems on simulated error-free quantum computers, which we currently do not have access to.

In practice, the HHL algorithm is not suitable for large-scale problems on NISQ computers. Instead, variational hybrid algorithms (VQAs) such as VQLS should be considered.

In either case, LCU demposition may pose a significant bottleneck.

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

### Discretisation

First, for each of u-velocity, v-velocity and pressure, a mesh $\mathcal{M}_u$, $\mathcal{M}_v$ and $\mathcal{M}_p$ is created.

Here, `u-velocity` and `v-velocity` are the components of the velocity vector in the x (say, horizontal) and y (say, vertical) directions, respectively.

The three meshes are usually `staggered`, meaning that u, v and p values can be located on the nodes, edges and centers of a mesh. For example, on the $\mathcal{M}_u$ mesh, u-velocities are on the nodes, v-velocities are on the edges, and pressures are at the centers.

### Momentum conservation equations

Physics determines that momentum of the fluid is fixed. This means that, from one moment to the next, as fluid flows and disperses, total fluid momentum is fixed.

As fluid flows, we update u, v and p and represent them by $u^\*$, $v^\*$ and $p^\*$.

This gives rise to `discrete u-momentum equation` and `discrete v-momentum equation`.

<p>$$
a^u_P u^*_{i,j} = \sum_{nb} a^u_{nb} u^*_{nb} - (p^*_{e} - p^*_{w}) \Delta y
$$</p>

Similar for $v^*$

where `nb` stands for `neighbor`, `e` stands for `east`, `w` stands for `west`, `i` and `j` are the indices of the point in the mesh.

Discrete momentum equations can be solved using **iterative schemes**.

### Mass conservation equation

Physics also determines that total mass is fixed. This can be used to update $u^*$, $v^*$ and $p^*$ to make sure that they also obey the law of mass conservation.

Hence, we have the following:

$$
\begin{aligned}
    u &= u^* + u' \\
    v &= v^* + v' \\
    p &= p^* + p'
\end{aligned}
$$

where $u'$ is the incremental updates in u-velocity, and so on.

Combining this with the momentum equations, and ignoring the sum over `nb`, we get the following:
<p>
$$
a^u_P u'_{i,j} = (p'_{w} - p'_{e}) \Delta y
$$

<p>
$$
a^v_P v'_{i,j} = (p'_{s} - p'_{n}) \Delta x
$$

Hence, $u'$ and $v'$ are obtained from $p'$.

In the end, we get an equation involving $p'$, plus some residual involving $u^\*$ and $v^\*$. This is the `continuity equation`.

<p>$$
a^p_P p'_{i,j} = \sum_{nb} a^p_{nb} p'_{nb} - \rho \left( (u^*_{i+1,j} - u^*_{i,j}) \Delta y + (v^*_{i,j+1} - v^*_{i,j}) \Delta x \right)
$$

We want to solve for $p'$, which is the `pressure correction`??

## SIMPLE algorithm for Navier-Stokes equations

We respresent the momentum equations and the continuity equation in matrix form.

First, solve for $u^\*$ and $v^\*$.

Next, solve for $p'$, and $u'$ and $v'$.

Finally, update $u^\*$, $v^{\*}$ and $p^\*$ to $u$, $v$ and $p$.

If $u'$, $v'$ and $p'$ are small enough, we are done. Otherwise, we repeat the process.

## Takeaway

In CFD, we first create a mesh (or grid) of points in space.
The flow of the fluid is then measured (or calculated) at each point in the mesh.

For instance, a 2D 5 x 5 mesh consists of 25 points in total.

Due to the limitation of NISQ computers, many modern CFD solvers first linearize and decompose the equations (or matrices) on classical computers, then solve the linear system of equations (or matrices) on quantum computers.

The problem is divided into `momentum equations` and `pressure correction` equations.
The latter is much more costly to solve.

## Quantum Process

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

where $U_i$ are unitary matrices, and $\alpha_i$ are the corresponding coefficients.

For pressure correction matrice, we only need to preform LCU decomposition once since the sparsity pattern of Ppresure correction matrix is fixed. Only the coefficients, $\alpha_i$, are updated for each outer iteration.

### 4. Implement a quantum algorithm, such as the HHL, to solve for $x$.

### Approaches to decomposition

An approach based on the orthogonality of grand sums of Hadamard products of Pauli stings.
Trace orthogonality of products of Pauli matrices, commonly termed Pauli strings.
The former is a lot faster.

Exponentiation of the sum of unitaries using Trotter product formula creates a single
unitary matrix.

#### Note: Partial unitary decomposition

Unitaries in the LCU with small coefficient may be ignored at a price in convergence or convergence speed.

In the 5 x 5 mesh test case, the algorithm still converges after ignoring 50% of unitaries.[1]

Doing this only reduces quantum computing workloads, as LCU still needs to be calculated and its coefficients updated.

### Preparing LCU

This is a significant bottleneck in hybrid CFD algrithms.

The `ancilla` register is used for preparing the coefficients of the unitary decomposition.

<details>
  <summary><i>More papers on this topic </i> [20-25]</summary>

<br>

[20] G. H. Low and I. L. Chuang, “Hamiltonian Simulation by Qubitization,” Quantum, vol. 3, p. 163, July 2019.
[21] A. M. Childs and N. Wiebe, “Hamiltonian simulation using linear combinations of unitary operations,” arXiv
preprint arXiv:1202.5822, 2012.
[22] R. Kothari, Efficient algorithms in quantum query complexity. PhD thesis, University of Waterloo, 2014.
[23] D. W. Berry, A. M. Childs, R. Cleve, R. Kothari, and R. D. Somma, “Simulating hamiltonian dynamics with a
truncated taylor series,” Physical review letters, vol. 114, no. 9, p. 090502, 2015.

[24] D. W. Berry, M. Kieferová, A. Scherer, Y. R. Sanders, G. H. Low, N. Wiebe, C. Gidney, and R. Babbush,
“Improved techniques for preparing eigenstates of fermionic hamiltonians,” npj Quantum Information, vol. 4, no. 1,
pp. 1–7, 2018.

[25] R. Babbush, C. Gidney, D. W. Berry, N. Wiebe, J. McClean, A. Paler, A. Fowler, and H. Neven, “Encoding
electronic spectra in quantum circuits with linear t complexity,” Physical Review X, vol. 8, no. 4, p. 041015, 2018.
</details>

Now that we have prepared LCU, let us solve the linear system of equations using HHL.

### State preparation

There are two state preparation methods:

1. #### Amplitudes loader

2. #### Binary tree loader

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

<br><br>
