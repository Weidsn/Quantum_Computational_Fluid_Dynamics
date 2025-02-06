# Notes

## Article Source

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

The pressure correction eqations can be written as a linear system of equations.

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
