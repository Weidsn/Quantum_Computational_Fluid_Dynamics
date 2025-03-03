# Qiskit Compilation

## 1. Introduction

Qiskit provides a compilation pipeline to map a quantum circuit to a quantum hardware.

## IBM hardware

### Instruction Set Architecture (ISA)

For IBM Heron superconducting devices, CZ, Rz, sqrt(X), and X gates are natively supported. Any Unitary operation can be built using these gates.

For different quantum hardware and for each qubit, a different set of gates may be available.

### Connectivity between qubits

For IBM (superconducting) quantum devices, qubits have limited connectivity.

We can only run CZ gate, for instance, on connected qubits.

The SWAP gate can be used to get around connectivity issues, but this creates extra computational burden.

In general, we want to minimize the number of gates executed by the quantum hardware.

### Noise and Errors

#### Gate errors

The execution of a gate may introduce error.

#### Decoherence times

This measures how long quantum states can be maintained. Shorter decoherence time means we need to run circuits faster.

T1: Energy Relaxation: time for a qubit at $|1\rangle$ state to relax to $|0\rangle$ state.

T2: Dephasing of a qubit in superposition state.

#### Measurment errors

Errors are introduced when measuring a qubit

## Qiskit Transpiler

### Target

The Target contains all properties of the quantum hardware. The transpiler will optimize the circuit based on these properties.

```print(backend.target)```

### DAG representation of circuit

Use Directed Acyclic Graph (DAG) for us to visualize dependencies of gates.

### Pass manager

Pass managers operates on DAG.

Preset pass managers have 6 stages, which can be customized.

#### 1. Init stage

User created packages, such as `qiskit-qubit-reuse` may be used in the `init` stage.

```Python
% pip install qiskit-qubit-reuse

# Calling the transpiler directly

transpile(quantum_circuit, backend, ini_method="qubit_reuse")

# Or, setting up a pass manager

pm = generate_preset_pass_manager(
    optimization_level = 2, 
    backend = backend, 
    init_method="qubit_reuse"
    )

pm.run(quantum_circuit)
```

#### Layout stage

`VF2Layout` and `SabreLayout` algorithms are excecuted by default.

`VF2Layout` tries to find a perfect mapping from the circuit connectivity graph to the hardware connectivity graphs.

It is computationally expensive, and `rustworkx` is used to speed up the process.

`SabreLayout` tries to find a "good" mapping between the two graphs from an initial setup. Multiple initial setups are used to find the best mapping.
