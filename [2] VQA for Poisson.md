# Article [2]

[[2]](https://github.com/Weidsn/Quantum_Computing_Collaboration/blob/main/A%20hybrid%20quantum-classical%20CFD%20methodology%20with%20benchmark%20HHL%20solutions.pdf) *A Performance Study of Variational Quantum Algorithms for Solving the Poisson Equation on a Quantum Computer* (May 2023)

## Abstract

Abstract: Solving Poisson equations using a variational quantum algorithm (VQA), i.e., Variational Quantum Linear Solver (VQLS) on noisy intermediate scale quantum (NISQ) computers. Results were not promising. VQLS is further discussed in [[5]](https://github.com/Weidsn/Quantum_Computing_Collaboration/blob/main/Variational%20quantum%20linear%20solver.pdf).

This paper concludes that direct resolution of the Poisson equation, via amplitude encoding, using VQA is not feasible on NISQ computers.

### Reader's note

As
[[1]](https://github.com/Weidsn/Quantum_Computing_Collaboration/blob/main/A%20hybrid%20quantum-classical%20CFD%20methodology%20with%20benchmark%20HHL%20solutions.pdf)
 pointed out, it might be possible to use binary-tree encoding instead of amplitude encoding, though the possible advantages is yet to be examined.

## Notes

Decomposition and HHL methods are less NISQ friendly.

Use quantum linear solvers (VQLS) to solve PDEs, e.g., Poisson equations.

In VQLS, quantum computers is only used to compute
simpler tasks, such as to estimate "cost functions."

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
