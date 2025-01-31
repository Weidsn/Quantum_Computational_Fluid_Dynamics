# HHL Algorithm

Source: [Quera Glossary](https://www.quera.com/glossary/hhl)

Harrow, Hassidim and Lloyd (HHL) algorithm is a quantum algorithm for solving linear systems of equations. It is a quantum algorithm that uses the principles of quantum computing to solve linear systems of equations in a more efficient way than classical algorithms.

## Key Concepts

### Quantum Phase Estimation (QPE)

QPE is derived from QFT.

### Quantum Fourier Transform (QFT)

Quantum analgous of the Fast Fourier Transform (FFT).

## Constraints

The matrix needs to be `Hermitian`, `sparse` and `well-conditioned`.

### Hermitian matrices

A matrix $A$ is `Hermitian` if and only if $A$ is equal to its conjugate transpose.

$$A = A^\dagger$$

In that case, $A$ can be written as

$$A = \sum_{i=1}^{n} \lambda_i |u_i\rangle \langle u_i|$$

where the $|u_i\rangle$'s are orthonormal, and the $\lambda_i$'s are the eigenvalues of $A$.

The outter product multiplied to the right by a vector $|v\rangle$  gives

$$|w\rangle = |u\rangle \langle u| v\rangle$$

If $|u\rangle$ is a unit vector, $|w\rangle$ is the projection of $|v\rangle$ onto the 1-dimensional vector space spanned by $|u\rangle$.

### Sparce matrices

A matrix $A$ is `sparse` if and only if the number of non-zero elements in $A$ is much smaller than the total number of elements in $A$.

### Well-conditioned matrices

A matrix $A$ is `well-conditioned` if and only if the ratio of the largest eigenvalue to the smallest eigenvalue is small.

## Processes

HHL uses QPE twice, the second time being the inverse of QPE.

The algorithm is as follows:

We want to solve for $x$ in $Ax=b$. 

We write $A$ as 

$$A = \sum_{i=1}^{n} \lambda_i |u_i\rangle \langle u_i|$$

Write $|b\rangle$ as 

$$b = \sum_{i=1}^{n} b_i |u_i\rangle$$

<!--
1. Initialize a quantum state $|\psi\rangle = \sum_{j=1}^{n} \alpha_j |j\rangle$.
2. Apply the QFT to $|\psi\rangle$ to obtain $|\psi'\rangle = \sum_{j=1}^{n} \alpha_j |j\rangle$.
3. Apply the matrix $A$ to $|\psi'\rangle$ to obtain $A|\psi'\rangle = \sum_{j=1}^{n} \lambda_j \alpha_j |j\rangle$.
-->

## References

[1] Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). Quantum algorithm for solving linear systems of equations. Physical Review Letters, 103(15), 150502.

## Further readings

[HHL: Solving Linear Systems of Equations with Quantum Computing](https://medium.com/mit-6-s089-intro-to-quantum-computing/hhl-solving-linear-systems-of-equations-with-quantum-computing-efb07eb32f74)


A Step-by-Step HHL Algorithm Walkthrough to Enhance Understanding of Critical Quantum Computing Concepts](https://arxiv.org/abs/2108.09004)
