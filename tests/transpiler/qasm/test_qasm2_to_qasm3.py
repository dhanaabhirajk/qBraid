# Copyright (C) 2024 qBraid
#
# This file is part of the qBraid-SDK
#
# The qBraid-SDK is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the qBraid-SDK, as per Section 15 of the GPL v3.

"""
Unit tests for qasm2 to qasm3 transpilation

"""

import logging

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.qasm3 import loads as qasm3_loads

from qbraid.interface import circuits_allclose
from qbraid.interface.random import random_circuit
from qbraid.programs import load_program
from qbraid.transpiler.conversions.qasm2.qasm2_to_qasm3 import _get_qasm3_gate_defs, qasm2_to_qasm3

logger = logging.getLogger(__name__)

gate_def_qasm3 = _get_qasm3_gate_defs()


def _check_output(output, expected):
    actual_circuit = qasm3_loads(output)
    expected_circuit = qasm3_loads(expected)
    assert actual_circuit == expected_circuit


QASM_TEST_DATA = [
    (
        """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1] ;
qreg qubits  [10]   ;
creg c[1];
creg bits   [12]   ;
        """,
        f"""
OPENQASM 3.0;
include "stdgates.inc";
{gate_def_qasm3}
qubit[1] q;
qubit[10] qubits;
bit[1] c;
bit[12] bits;
        """,
    ),
    (
        """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
measure q->c;
measure q[0] -> c[1];
        """,
        f"""
OPENQASM 3.0;
include "stdgates.inc";
{gate_def_qasm3}
qubit[2] q;
bit[2] c;
c = measure q;
c[1] = measure q[0];
        """,
    ),
    (
        """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
opaque custom_gate (a,b,c) p,q,r;
        """,
        f"""
OPENQASM 3.0;
include "stdgates.inc";
{gate_def_qasm3}
qubit[2] q;
bit[2] c;
// opaque custom_gate (a,b,c) p,q,r;
        """,
    ),
    (
        """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
        """,
        f"""
OPENQASM 3.0;
include "stdgates.inc";
{gate_def_qasm3}
qubit[1] q;
        """,
    ),
    (
        """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
u(1,2,3) q[0];
sxdg q[0];
csx q[0], q[1];
cu1(0.5) q[0], q[1];
cu3(1,2,3) q[0], q[1];
rzz(0.5) q[0], q[1];
rccx q[0], q[1], q[2];
rc3x q[0], q[1], q[2], q[3];
c3x q[0], q[1], q[2], q[3];
c3sqrtx q[0], q[1], q[2], q[3];
c4x q[0], q[1], q[2], q[3], q[4];
        """,
        f"""
OPENQASM 3.0;   
include "stdgates.inc";
{gate_def_qasm3}
qubit[5] q;
U(1,2,3) q[0];
sxdg q[0];
csx q[0], q[1];
cu1(0.5) q[0], q[1];
cu3(1,2,3) q[0], q[1];
rzz(0.5) q[0], q[1];
rccx q[0], q[1], q[2];
rc3x q[0], q[1], q[2], q[3];
c3x q[0], q[1], q[2], q[3];
c3sqrtx q[0], q[1], q[2], q[3];
c4x q[0], q[1], q[2], q[3], q[4];
        """,
    ),
]


@pytest.mark.parametrize("test_input, expected_output", QASM_TEST_DATA)
def test_qasm2_to_qasm3_parametrized(test_input, expected_output):
    """Test the conversion of OpenQASM 2 to 3"""
    _check_output(qasm2_to_qasm3(test_input), expected_output)


def _generate_valid_qasm_strings(seed=42, gates_to_skip=None, num_circuits=100):
    """Returns a list of 100 random qasm2 strings
    which do not contain any of the gates in gates_to_skip

    Current list of invalid gates is ["u", "cu1", "cu2", "cu3", "rxx"]
    For the motivation, see discussion
    - https://github.com/Qiskit/qiskit-qasm3-import/issues/12
    - https://github.com/Qiskit/qiskit-qasm3-import/issues/11#issuecomment-1568505732
    """
    if gates_to_skip is None:
        gates_to_skip = []

    qasm_strings = []
    while len(qasm_strings) < num_circuits:
        try:
            circuit_random = random_circuit("qiskit", seed=seed)
            qasm_str = qasm2_dumps(circuit_random)
            circuit_from_qasm = QuantumCircuit.from_qasm_str(qasm_str)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Invalid QASM generated by random_circuit: %s", e)
            continue

        for gate in gates_to_skip:
            if len(circuit_from_qasm.get_instructions(gate)) > 0:
                break
        else:
            qasm_strings.append(qasm_str)

    return qasm_strings


@pytest.mark.parametrize("qasm2_str", _generate_valid_qasm_strings(gates_to_skip=["r"]))
def test_random_conversion_to_qasm3(qasm2_str):
    """test random gates conversion"""
    qasm3_str = qasm2_to_qasm3(qasm2_str)
    circuit_orig = QuantumCircuit.from_qasm_str(qasm2_str)
    circuit_test = qasm3_loads(qasm3_str)

    # ensure that the conversion is correct
    assert circuits_allclose(circuit_orig, circuit_test)


@pytest.mark.skip(reason="Qiskit terra bug")
def test_u0_gate_conversion():
    """test u0 gate conversion
    Separate test due to bug in terra,
    see https://github.com/Qiskit/qiskit-terra/issues/10184
    """

    test_u0 = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    u0(0.5) q[0];"""

    test_u0_expected = f"""
    OPENQASM 3.0;
    include "stdgates.inc";
    {gate_def_qasm3}
    qubit[1] q;
    u0(0.5) q[0];
    """

    _check_output(qasm2_to_qasm3(test_u0), test_u0_expected)


def test_rxx_gate_conversion():
    """Test rxx gate conversion"""

    test_rxx = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    rxx(0.5) q[0], q[1];"""

    test_rxx_expected = f"""
    OPENQASM 3.0;
    include "stdgates.inc";
    {gate_def_qasm3}
    qubit[3] q;

    // rxx gate
    h q[0];
    h q[1];
    cx q[0],q[1];
    rz(0.5) q[1];
    cx q[0],q[1];
    h q[1];
    h q[0];
    """

    _check_output(qasm2_to_qasm3(test_rxx), test_rxx_expected)


@pytest.mark.skip(reason="Syntax not yet supported")
def test_qasm3_num_qubits_alternate_synatx():
    """Test calculating num qubits for qasm3 syntax edge-case"""
    qasm3_str = """
OPENQASM 3;
include "stdgates.inc";
qubit _qubit0;
qubit _qubit1;
h _qubit0;
cx _qubit0, _qubit1;
"""
    circuit = qasm3_loads(qasm3_str)
    qprogram = load_program(qasm3_str)
    assert qprogram.num_qubits == circuit.num_qubits


def test_reverse_qubit_order():
    """Test the reverse qubit ordering function"""
    qasm_str = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    qubit[4] q2;
    qubit q3;

    cnot q[0], q[1];
    cnot q2[0], q2[1];
    x q2[3];
    cnot q2[0], q2[2];
    x q3[0];
    """

    reverse_qasm = load_program(qasm_str).reverse_qubit_order()
    expected_qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    qubit[4] q2;
    qubit q3;

    cnot q[1], q[0];
    cnot q2[3], q2[2];
    x q2[0];
    cnot q2[3], q2[1];
    x q3[0];
    """
    assert reverse_qasm == expected_qasm


def test_remap_qubit_order():
    """Test the remapping of qubits in qasm string"""
    qubit_mapping = {"q1": {0: 1, 1: 0}, "q2": {0: 2, 1: 0, 2: 1}}
    qasm_str = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q1;
    qubit[3] q2;
    
    cnot q1[1], q1[0];
    cnot q2[2], q2[1];
    x q2[0];
    """

    remapped_qasm = load_program(qasm_str).apply_qubit_mapping(qubit_mapping)

    expected_qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q1;
    qubit[3] q2;
    
    cnot q1[0], q1[1];
    cnot q2[1], q2[0];
    x q2[2];
    """
    assert expected_qasm == remapped_qasm


def test_incorrect_remapping():
    """Test that incorrect remapping raises error"""
    reg_not_there_mapping = {"q2": {0: 2, 1: 0, 2: 1}}
    incomplete_reg_mapping = {"q1": {0: 1, 1: 0}, "q2": {0: 2, 1: 0}}
    out_of_bounds_mapping = {"q1": {0: 1, 1: 2}, "q2": {0: 2, 1: 0, 3: 1}}

    qasm_str = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q1;
    qubit[3] q2;
    
    cnot q1[1], q1[0];
    cnot q2[2], q2[1];
    x q2[0];
    """

    with pytest.raises(ValueError):
        _ = load_program(qasm_str).apply_qubit_mapping(reg_not_there_mapping)

    with pytest.raises(ValueError):
        _ = load_program(qasm_str).apply_qubit_mapping(incomplete_reg_mapping)

    with pytest.raises(ValueError):
        _ = load_program(qasm_str).apply_qubit_mapping(out_of_bounds_mapping)
