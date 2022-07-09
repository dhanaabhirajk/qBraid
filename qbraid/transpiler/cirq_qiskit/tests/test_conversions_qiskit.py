# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Unit tests for conversions between Cirq circuits and Qiskit circuits.

"""
import copy

import cirq
import numpy as np
import pytest
import qiskit

from qbraid.interface import circuits_allclose
from qbraid.interface.qbraid_cirq._utils import _equal
from qbraid.transpiler.cirq_qiskit.conversions import (
    _map_bit_index,
    _measurement_order,
    _transform_registers,
    from_qiskit,
    to_qiskit,
)
from qbraid.transpiler.cirq_utils import from_qasm, to_qasm


def test_bell_state_to_from_circuits():
    """Tests cirq.Circuit --> qiskit.QuantumCircuit --> cirq.Circuit
    with a Bell state circuit.
    """
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit([cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1])])
    qiskit_circuit = to_qiskit(cirq_circuit)  # Qiskit from Cirq
    circuit_cirq = from_qiskit(qiskit_circuit)  # Cirq from Qiskit
    assert np.allclose(cirq_circuit.unitary(), circuit_cirq.unitary())


def test_bell_state_to_from_qasm():
    """Tests cirq.Circuit --> QASM string --> cirq.Circuit
    with a Bell state circuit.
    """
    qreg = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit([cirq.ops.H.on(qreg[0]), cirq.ops.CNOT.on(qreg[0], qreg[1])])
    qasm = to_qasm(cirq_circuit)  # Qasm from Cirq
    circuit_cirq = from_qasm(qasm)
    assert np.allclose(cirq_circuit.unitary(), circuit_cirq.unitary())


def test_random_circuit_to_from_circuits():
    """Tests cirq.Circuit --> qiskit.QuantumCircuit --> cirq.Circuit
    with a random two-qubit circuit.
    """
    cirq_circuit = cirq.testing.random_circuit(
        qubits=2, n_moments=10, op_density=0.99, random_state=1
    )
    qiskit_circuit = to_qiskit(cirq_circuit)
    circuit_cirq = from_qiskit(qiskit_circuit)
    assert np.allclose(cirq_circuit.unitary(), circuit_cirq.unitary())


def test_random_circuit_to_from_qasm():
    """Tests cirq.Circuit --> QASM string --> cirq.Circuit
    with a random one-qubit circuit.
    """
    circuit_0 = cirq.testing.random_circuit(qubits=2, n_moments=10, op_density=0.99, random_state=2)
    qasm = to_qasm(circuit_0)
    circuit_1 = from_qasm(qasm)
    u_0 = circuit_0.unitary()
    u_1 = circuit_1.unitary()
    assert cirq.equal_up_to_global_phase(u_0, u_1)


@pytest.mark.parametrize("as_qasm", (True, False))
def test_convert_with_barrier(as_qasm):
    """Tests converting a Qiskit circuit with a barrier to a Cirq circuit."""
    n = 5
    qiskit_circuit = qiskit.QuantumCircuit(qiskit.QuantumRegister(n))
    qiskit_circuit.barrier()

    if as_qasm:
        cirq_circuit = from_qasm(qiskit_circuit.qasm())
    else:
        cirq_circuit = from_qiskit(qiskit_circuit)

    assert _equal(cirq_circuit, cirq.Circuit())


@pytest.mark.parametrize("as_qasm", (True, False))
def test_convert_with_multiple_barriers(as_qasm):
    """Tests converting a Qiskit circuit with barriers to a Cirq circuit."""
    n = 1
    num_ops = 10

    qreg = qiskit.QuantumRegister(n)
    qiskit_circuit = qiskit.QuantumCircuit(qreg)
    for _ in range(num_ops):
        qiskit_circuit.h(qreg)
        qiskit_circuit.barrier()

    if as_qasm:
        cirq_circuit = from_qasm(qiskit_circuit.qasm())
    else:
        cirq_circuit = from_qiskit(qiskit_circuit)

    qbit = cirq.LineQubit(0)
    correct = cirq.Circuit(cirq.ops.H.on(qbit) for _ in range(num_ops))
    assert _equal(cirq_circuit, correct)
    assert circuits_allclose(qiskit_circuit, cirq_circuit, strict_gphase=True)


@pytest.mark.parametrize("reg_sizes", [[2, 4, 1, 6], [5, 4, 2], [6]])
def test_map_bit_index(reg_sizes):
    expected_register_index = 0
    expected_mapped_index = 0
    for bit_index in range(sum(reg_sizes)):
        register_index, mapped_index = _map_bit_index(bit_index, reg_sizes)

        assert register_index == expected_register_index
        assert mapped_index == expected_mapped_index

        expected_mapped_index += 1
        if bit_index == sum(reg_sizes[: expected_register_index + 1]) - 1:
            expected_register_index += 1
            expected_mapped_index = 0


@pytest.mark.parametrize("nqubits", [1, 5])
@pytest.mark.parametrize("with_ops", [True, False])
@pytest.mark.parametrize("measure", [True, False])
def test_transform_qregs_one_qubit_ops(nqubits, with_ops, measure):
    qreg = qiskit.QuantumRegister(nqubits)
    circ = qiskit.QuantumCircuit(qreg)
    if with_ops:
        circ.h(qreg)
    if measure:
        circ.add_register(qiskit.ClassicalRegister(nqubits))
        circ.measure(qreg, circ.cregs[0])

    orig = circ.copy()
    assert circ.qregs == [qreg]

    new_qregs = [qiskit.QuantumRegister(1) for _ in range(nqubits)]
    _transform_registers(circ, new_qregs=new_qregs)

    assert circ.qregs == new_qregs
    assert circ.cregs == orig.cregs
    assert _equal(from_qiskit(circ), from_qiskit(orig))


@pytest.mark.parametrize("new_reg_sizes", [[1], [1, 2], [2, 1], [1, 1, 1]])
def test_transform_qregs_two_qubit_ops(new_reg_sizes):
    nqubits = sum(new_reg_sizes)
    circ = to_qiskit(
        cirq.testing.random_circuit(nqubits, n_moments=5, op_density=1, random_state=1)
    )
    orig = circ.copy()

    new_qregs = [qiskit.QuantumRegister(s) for s in new_reg_sizes]
    _transform_registers(circ, new_qregs=new_qregs)

    cirq1 = from_qiskit(orig)
    cirq2 = from_qiskit(circ)

    assert circ.qregs == new_qregs
    assert circ.cregs == orig.cregs
    assert _equal(cirq2, cirq1)
    assert np.allclose(cirq2.unitary(), cirq1.unitary())


@pytest.mark.parametrize("new_reg_sizes", [[1], [1, 2], [2, 1], [1, 1, 1]])
@pytest.mark.parametrize("measure", [True, False])
def test_transform_qregs_random_circuit(new_reg_sizes, measure):
    nbits = sum(new_reg_sizes)
    circ = to_qiskit(cirq.testing.random_circuit(nbits, n_moments=5, op_density=1, random_state=10))
    creg = qiskit.ClassicalRegister(nbits)
    circ.add_register(creg)
    if measure:
        circ.measure(circ.qregs[0], creg)
    orig = circ.copy()

    new_qregs = [qiskit.QuantumRegister(s) for s in new_reg_sizes]
    _transform_registers(circ, new_qregs=new_qregs)

    assert circ.qregs == new_qregs
    assert _equal(from_qiskit(circ), from_qiskit(orig))


def test_transform_qregs_no_new_qregs():
    qreg = qiskit.QuantumRegister(5)
    circ = qiskit.QuantumCircuit(qreg)
    _transform_registers(circ, new_qregs=None)
    assert circ.qregs == [qreg]


def test_transform_registers_too_few_qubits():
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(2))
    new_qregs = [qiskit.QuantumRegister(1)]

    with pytest.raises(ValueError):
        _transform_registers(circ, new_qregs=new_qregs)


def test_transform_registers_adds_idle_qubits():
    """Tests transforming registers in a circuit with n qubits to a circuit
    with m > n qubits.
    """
    qreg = qiskit.QuantumRegister(1)
    creg = qiskit.ClassicalRegister(1)
    circuit = qiskit.QuantumCircuit(qreg, creg)
    circuit.x(qreg[0])
    circuit.measure(qreg[0], creg[0])

    assert len(circuit.qregs) == 1
    assert circuit.num_qubits == 1
    old_data = copy.deepcopy(circuit.data)

    _transform_registers(circuit, new_qregs=[qreg, qiskit.QuantumRegister(4)])

    assert len(circuit.qregs) == 2
    assert circuit.num_qubits == 5
    assert circuit.data == old_data


def test_transform_registers_wrong_reg_number():
    nqubits = 2
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(nqubits))
    new_qregs = [qiskit.QuantumRegister(1) for _ in range(2 * nqubits)]
    circ.add_register(*new_qregs)

    with pytest.raises(ValueError):
        _transform_registers(circ, new_qregs=new_qregs)


@pytest.mark.parametrize("size", [5])
def test_measurement_order(size):
    q, c = qiskit.QuantumRegister(size), qiskit.ClassicalRegister(size)
    circuit = qiskit.QuantumCircuit(q, c)

    index_order = [int(i) for i in np.random.RandomState(1).permutation(size)]
    for i in index_order:
        circuit.measure(q[i], c[i])

    order = _measurement_order(circuit)
    assert order == [(q[i], c[i]) for i in index_order]
