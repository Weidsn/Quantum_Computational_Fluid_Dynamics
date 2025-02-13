# Article [5]

[[5]](https://github.com/Weidsn/Quantum_Computing_Collaboration/blob/main/Variational%20quantum%20linear%20solver.pdf) *Variational quantum linear solver* (2023)

Abstract: As parts of a variational quantum algorithm (VQA), introducing the Variational Quantum Linear Solver (VQLS) to quantify the closeness between $A$ $|x\rangle$ and $|b\rangle$ on NISQ computers.

## Notes

"Further improvements to HHL have reduced the complexity to linear κ scaling [7, 8] and polylogarithmic scaling in 1/ϵ [9, 10], as well as improved the sparsity requirements [11]"

HHL algorithm is greatly constrained by the noise of NISQ computers.
Thus, this paper introduces a variational hybrid quamtum algorithm (VQA), or, in this paper's term, variational hybrid quantum-classical algorithm (VHQCA).

## Overview of VHQCA

First, we prepare a vector $|x(\alpha)\rangle$, using a set of parameters $\alpha$.

Second, using quantum computers, we calculate the cost, $C(\alpha)$, that captures the difference between $| x(\alpha) \rangle$ and the real solution, $|x_0\rangle$, to $A {|x\rangle} = {|b\rangle}$.

Next, using classical algorithms, we find a new parameter, $\alpha$, that reduces $C(\alpha)$.

We repeat this process until $C(\alpha)$ falls within a reasonable limit, at which point we settle with $|x(\alpha)\rangle$ as an approximate solution.

### States preparation

Decompose $A$ into linear combination of unitaries.

We assume $A$ is Hermitian (and Hamiltonian), which is analgous to decomposition into Pauli strings in other QLES.

### Cost function

Several cost functions are proposed:

1. $C_G$

$$C_G = 1 - | \langle b| \frac{|\psi\rangle}{\|\psi\|} \rangle |^2$$

We project $|\psi\rangle = A |x(\alpha)\rangle$ onto the subspace orthogonal to $|b\rangle$.
$C_G(\alpha)$ equals the $ \| \cdot\ |^2 $ of the projection.

Note that $|\psi\rangle$ is not normalized, so we need to normalize it before computing the projection.

As the number of qubits increases, the cost function $C_G$ gradient vanishes so we introduce the local cost function.

<br>

1. $C_L$, a local version of $C_G$.

$$ \hat{C}_L = \langle x | H_L | x \rangle $$

$$C_L = \frac{\hat{C}_L} {\| \psi\|^2}$$

where

$$ H_L = A^\dagger U \left( 1 - \frac{1}{n} \sum_{j=1}^{n} |0_j \rangle \langle 0_{} | \otimes |1_{\bar{j}} \rangle \right) U^\dagger A $$

### Implementing cost functions

The goal is to reduce burden on the quantum computer.

We can use `Hadamard Test` circuit.

The paper introduces a new circuit, `Hadamard-Overlap Test`, to compute $C_L$.

### Classical hardness

Computing cost functions using classical methods are DQC1-hard (Deterministic Quantum Computing with 1 Clean Qubit hard).

### Ansatz and Preparing $|x(\alpha)\rangle$

$|x(\alpha)\rangle$ is prepared using a "trainable" sequence of quantum gates, $V(\alpha)$.

The gates are chosen from a set of quantum gates native to the quantum hardware.

Trainability issue for "fixed gate ansatz" can be resolved by layer-by-layer training and correlating the $\alpha$ parameters, and also

1. Variable structure ansatz
2. Quantum Alternating Operator Ansatz (QAOA)

### Training (2.1.7)

Training the ansatz using classical methods. Gradient-based (classical) method may also be possible.
 (Appendix F)

### Noise resilience (2.1.8)

VQLS exhibits Optimal Parameter Resilience (OPR) phenomenon.

$C_L$ is resilient to global depolarizing noise and measurement noise.

Noise in evaluating cost functions is unavoidable. This can be mitigated by Probabilitic Error Cancellation (PEC) procedure outlined in the paper.

PEC can

## References

[7] A. Ambainis, “Variable time amplitude amplification and a faster quantum algorithm for solving
systems of linear equations,” arXiv:1010.4458
[quant-ph] .

[8] Y. Subaşı, R. D. Somma, and D. Orsucci, “Quantum algorithms for systems of linear equations inspired by adiabatic quantum computing,” Phys.
Rev. Lett. 122, 060504 (2019).

[9] A. Childs, R. Kothari, and R. Somma, “Quantum algorithm for systems of linear equations
with exponentially improved dependence on precision,” SIAM J. Computing 46, 1920–1950
(2017).

[10] S. Chakraborty, A. Gilyén, and S. Jeffery,
“The power of block-encoded matrix powers: improved regression techniques via faster Hamiltonian simulation,” in 46th International Colloquium on Automata, Languages, and Programming (Schloss Dagstuhl-Leibniz-Zentrum fuer Informatik, 2019) pp. 33:1-33:14.

[11] L. Wossnig, Z. Zhao, and A. Prakash, “Quantum linear system algorithm for dense matrices,”
Phys. Rev. Lett. 120, 050502 (2018)
