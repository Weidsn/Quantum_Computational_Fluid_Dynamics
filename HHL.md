# HHL Algorithm

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

$$A = \sum_{j=1}^{n} \lambda_j {|u_j\rangle} \langle u_j | $$

where the $|u_j\rangle$'s are orthonormal, and the $\lambda_j$'s are the eigenvalues of $A$. 


#### Note:

1. For any k, $\lambda_k$ is an eigenvalue of the eigenvector $|u_k\rangle$ of $A$. This is because 

$$A {|u_k \rangle}= \sum_{j=1}^{n} \lambda_j {|u_j\rangle} \langle u_j |u_k \rangle $$

$$ = \lambda_k {|u_k\rangle} \langle u_k |u_k \rangle + \sum_{j \neq k} \lambda_j {|u_j\rangle} \langle u_j |u_k \rangle $$ 

$$ = \lambda_k {|u_k\rangle} * 1 + 0 = \lambda_k {|u_k\rangle} $$



2. The outter product multiplied to the right by a vector $|v\rangle$ gives

$$|w\rangle = {|u\rangle} \langle u | v\rangle$$

3. If $|u\rangle$ is a unit vector, $|w\rangle$ is the projection of $|v\rangle$ onto the 1-dimensional vector space spanned by $|u\rangle$.

### Sparce matrices

A matrix $A$ is **`sparse`** if and only if the number of non-zero elements in $A$ is much smaller than the total number of elements in $A$.

A matrix is `s-sparse` if and only if it has at most `s` non-zero elements in each row or column. 

### Well-conditioned matrices

A matrix $A$ is `well-conditioned` if and only if the ratio of the largest eigenvalue (the largest of the $\lambda_i$'s) to the smallest eigenvalue is small. 

The `condition number` **$\kappa$** is the ratio between the largest and the smallest eigenvalue. 

## Processes

HHL applies QPE twice, the second time being the inverse of QPE.

The algorithm is as follows:

We want to solve for $x$ in $Ax=b$. 

We write $A$ as 

$$A = \sum_{j=1}^{n} \lambda_j {|u_j\rangle} \langle u_j|$$

Write $|b\rangle$ as 

$$b = \sum_{j=1}^{n} b_j {|u_j\rangle}$$

The goal is to solve

$$ |x\rangle = A^{-1} {|b\rangle} = \sum_{j=1}^{n} \lambda_j^{-1} b_j {|u_j\rangle}$$


Quantum Phase Estimation (QPE) is used to find the eigenvalues of $A$, which are the complex scalars $\lambda_j$ in the equations above. 

In QPE, however, instead of representing the scalars by $\lambda_j \in \mathbb{C}$, 
the scalars are expressed in the form of $e^{i\lambda_j t}$, for $\lambda_j \in \mathbb{R}$.

$$
U = e^{iAt} := \sum_{j=1}^{n} e^{i\lambda_j t} {|u_j\rangle} \langle u_j|
$$


<!--
1. Initialize a quantum state $|\psi\rangle = \sum_{j=1}^{n} \alpha_j |j\rangle$.
2. Apply the QFT to $|\psi\rangle$ to obtain $|\psi'\rangle = \sum_{j=1}^{n} \alpha_j |j\rangle$.
3. Apply the matrix $A$ to $|\psi'\rangle$ to obtain $A|\psi'\rangle = \sum_{j=1}^{n} \lambda_j \alpha_j |j\rangle$.
-->



## Runtime Analysis<sup>[2]</sup>

The runtime of HHL using quantum computer is **O(log(n)s²𝜅²/ε)**, where s is the sparsity parameter, 𝜅 is the condition number and ε is the error parameter which controls the precision of the result. Hence, the higher the required precision, the lower the ε value and the longer it takes to execute the algorithm. 

The runtime of classical algorithm is polynomial time with respect to n, thus HHL provide exponential speedup compared to classical algorithms. With respect to sparsity, s, and the measure of conditionedness, 𝜅, classical algorithms run in **O(s𝜅)** time, which is faster than HHL, which runs in **O(s²𝜅²)** time. 

However, reading out the full quantum solution would take **O(n)** time. 

## References

[[1]](https://www.quera.com/glossary/hhl)
Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). Quantum algorithm for solving linear systems of equations. Physical Review Letters, 103(15), 150502. Retreived from Quera Glossary.

[[2]](https://medium.com/mit-6-s089-intro-to-quantum-computing/hhl-solving-linear-systems-of-equations-with-quantum-computing-efb07eb32f74) 
Vaughn II, R. (n.d.). HHL: Solving Linear Systems of Equations with Quantum Computing. Medium. Retrieved from Medium.

## Further readings

A Step-by-Step [HHL Algorithm Walkthrough](https://arxiv.org/abs/2108.09004) to Enhance Understanding of Critical Quantum Computing Concepts. 


