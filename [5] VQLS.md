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

### Cost functions

Several cost functions are proposed:

1. $C_G$

$$C_G = 1 - | \langle b| \frac{|\psi\rangle}{\|\psi\|} \rangle |^2$$

We project $|\psi\rangle = A |x(\alpha)\rangle$ onto the subspace orthogonal to $|b\rangle$. 
$C_G(\alpha)$ equals the $ \| \cdot\|^2 $ of the projection.

Note that $|\psi\rangle$ is not normalized, so we need to normalize it before computing the projection.

As the number of qubits increases, the cost function $C_G$ gradient vanishes so we introduce the local cost function.

<br>

2. $C_L$, a local version of $C_G$.

$$ \hat{C}_L = \langle x | H_L | x \rangle $$

$$C_L = \frac{\hat{C}_L} {\| \psi\|^2}$$

where

$$ H_L = A^\dagger U \left( 1 - \frac{1}{n} \sum_{j=1}^{n} |0_j \rangle \langle 0_{} | \otimes |1_{\bar{j}} \rangle \right) U^\dagger A $$



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