# Qiskit Compilation for IBM devices

## 1. Introduction

Qiskit provides a compilation pipeline that allows you to map a user circuit to a physical device. This pipeline is designed to be flexible and extensible, allowing you to customize the behavior of each step in the process. In this tutorial, we will explore the different components of the compilation pipeline and how to use them to optimize your circuits for execution on IBM devices.

## IBM hardware

### Instruction Set Architecture (ISA)

For IBM Heron Systems, CZ, Rz, sqrt(X), and X gates are natively supported. These gates can be used to construct any Unitary operation.

For different quantum hardware and for each qubit, a different set of Instructions may be available.

### Connectivity between qubits

For IBM `superconducting` quantum processors, qubits have limited connectivity.

Connectivity only exists between qubits that are physically close. We can only run CZ gate, for instance, between connected qubits.

We can use SWAP or other techniques to swap qubits, but we should minimize this. Each SWAP involves three CZ gates.

### Noise and Errors

#### Gate Errors

Gate have error rates

#### Decoherence Times

How long quantum states can be maintained. Shorter decoherence times mean that we need to run circuits faster.

T1: Energy Relaxation: time for a qubit at $|1\rangle$ state to relax to $|0\rangle$ state.

T2: Dephasing of a qubit in superposition state.

#### Measurment error

## Qiskit Transpiler

### Target

The Target contains a list of every instruction and their properties, error rate and etc.

```print(backend.target)```

### DAG representation of circuit

Use Directed Acyclic Graph (DAG) to visualize dependencies of gates.

### Pass manager

Pass managers operates on DAG.

Preset pass managers have 6 stages, which can be customized.

#### Init stage

For example, qiskit-qubit-reuse package can be used for the `init` stage.

```Python
% pip install qiskit-qubit-reuse

transpile(qc, backend, ini_method="qubit_reuse")

# Alternatively

pm = generate_preset_pass_manager(optimization_level = 2, backend, init_method="qubit_reuse")

pm.run(qc)

```

#### Layout stage

VF2Layout creates a graph from the 2 qubit interactions in the circuit and the connectivity graph of the backend. It then finds a mapping between the two graphs.

It is computationally expensive, and `rustworkx` is used to speed up the process.

SabreLayout tries to find a good mapping between the two graphs.
