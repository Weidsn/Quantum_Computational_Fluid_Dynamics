from IPython.display import display

import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit.circuit.library import UGate
from qiskit.exceptions import QiskitError
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, EstimatorOptions
from qiskit.quantum_info import (
    Statevector,
    Operator,
    PauliList,
    Operator,
    SparsePauliOp,
    Pauli,
)
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.visualization import array_to_latex, plot_histogram

import math
import random
import numpy as np
from numpy import pi
from numpy import array, sqrt

from scipy.optimize import minimize
import matplotlib.pyplot as plt

import tqdm

# For high-performance simulation
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
