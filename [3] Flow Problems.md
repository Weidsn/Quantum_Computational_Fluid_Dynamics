# Article [3]

[[3]](https://github.com/Weidsn/Quantum_Computing_Collaboration/blob/main/Hybrid%20quantum%20algorithms%20for%20flow%20problems.pdf) *Hybrid quantum algorithms for flow problems* (June 2023)

## Abstract

Introducing QFlowS (Quantum Flow Simulator). Solving CFD problems by implementing HHL (referred to as a kind of Quantum Linear Systems Algorithm or in other words Quantum Linear Equation Solver QLES) with a discussion of Quantum Postprocessing Protocol (QPP), which concerns information extraction via quantum state measurements.

## Notes

Quantum machanic is linear.

<details>
<summary><i>Papers on this topic </i> [13-15]</summary>
13. J.-P. Liu et al., Efficient quantum algorithm for dissipative nonlinear differential equations. Proc.
Natl. Acad. Sci. U.S.A. 118, e2026805118 (2021).
14. Y. T. Lin, R. B. Lowrie, D. Aslangil, Y. Suba¸sı, A. T. Sornborger, Koopman von Neumann mechanics
and the Koopman representation: A perspective on solving nonlinear dynamical systems with
quantum computers. arXiv [Preprint] (2022). http://arxiv.org/abs/2202.02188 (Accessed 29 June
2023).
15. D. Giannakis, A. Ourmazd, P. Pfeffer, J. Schumacher, J. Slawinska, Embedding classical dynamics in
a quantum computer. Phys. Rev. A 105, 052404 (2022)
</details>

## QFlowS (Quantum Flow Simulator)

This paper uses 1-dimensional flow as an example. The form of PDE for 1-dimensional flow is given by:

$$
\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial y^2} -\frac{\partial p}{\partial x}
$$

where $u$ is the velocity, $p$ is the pressure, and $\rho$ is the density. The boundary conditions are given by:
$$
u(0, t) = u(D, t) = 0
$$
for the Poiseuille flow.

The initial condition is given by $u(y,0) = 1$.

Three schemes are used to discretize time (page 2-3):

1. Backward Euler, iterative

2. Backward Euler, one-step

2. Forward Euler, one-step

### Improvements in VQA

Papers [36-38] outlines methods for improving HHL algorithm so that error complexity becomes $poly(log(1/ \epsilon))$ instead of $poly(1/ \epsilon)$.

<details>
<summary><i>List of papers </i> [36-38]</summary>
[36] I. M. Georgescu, S. Ashhab, F. Nori, Quantum simulation. Rev. Mod. Phys. 86, 153 (2014).

[37] D. W. Berry, A. M. Childs, R. Cleve, R. Kothari, R. D. Somma, “Exponential improvement in precision
for simulating sparse hamiltonians” in Proceedings of the Forty-Sixth Annual ACM Symposium on
Theory of Computing (2014), pp. 283–292.

[38] D. W. Berry, A. M. Childs, R. Kothari, “Hamiltonian simulation with nearly optimal dependence on
all parameters” in 2015 IEEE 56th Annual Symposium on Foundations of Computer Science (IEEE,
2015), pp. 792–809
</details>

<br>

Paper [27] develops an efficient LCU decomposition strategy. 

[27] A. M. Childs, R. Kothari, R. D. Somma, Quantum algorithm for systems of linear equations
with exponentially improved dependence on precision. SIAM J. Comput. 46, 1920–1950
(2017).

### QLSA

QLSA-1 uses HHL algorithm. 

QLSA-2 uses LCU method. 

### Quantum state preparation

#### QSP-1

Grover-Rudolph state preparation technique. [40, 41, 42]

#### QSP-2

Constructing decision trees to represent quantum states. [45]
