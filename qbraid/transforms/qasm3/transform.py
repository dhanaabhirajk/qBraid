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
Module for providing transforamtions with basis gates.
across various other quantum software frameworks.

"""

from typing import List

from openqasm3 import ast
from openqasm3.parser import parse

from qbraid.transpiler.conversions.openqasm3 import openqasm3_to_qasm3


def decompose_crx(gate: ast.QuantumGate) -> List[ast.Statement]:
    """Decompose a crx gate into its basic gate equivalents."""
    theta = gate.arguments[0]
    control = gate.qubits[0]
    target = gate.qubits[1]

    rz_pos_pi_half = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="rz"),
        arguments=[
            ast.BinaryExpression(
                op=ast.BinaryOperator(17),
                lhs=ast.Identifier(name="pi"),
                rhs=ast.FloatLiteral(value=2),
            )
        ],
        qubits=[target],
    )
    ry_pos_theta_half = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="ry"),
        arguments=[
            ast.BinaryExpression(
                op=ast.BinaryOperator(17), lhs=theta, rhs=ast.FloatLiteral(value=2)
            )
        ],
        qubits=[target],
    )
    cx = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="cx"), arguments=[], qubits=[control, target]
    )
    ry_neg_theta_half = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="ry"),
        arguments=[
            ast.BinaryExpression(
                op=ast.BinaryOperator(17),
                lhs=ast.UnaryExpression(ast.UnaryOperator(3), theta),
                rhs=ast.FloatLiteral(value=2),
            )
        ],
        qubits=[target],
    )
    rz_neg_pi_half = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="rz"),
        arguments=[
            ast.BinaryExpression(
                op=ast.BinaryOperator(17),
                lhs=ast.UnaryExpression(ast.UnaryOperator(3), ast.Identifier(name="pi")),
                rhs=ast.FloatLiteral(value=2),
            )
        ],
        qubits=[target],
    )
    return [rz_pos_pi_half, ry_pos_theta_half, cx, ry_neg_theta_half, cx, rz_neg_pi_half]


def decompose_cry(gate: ast.QuantumGate) -> List[ast.Statement]:
    """Decompose a cry gate into its basic gate equivalents."""
    theta = gate.arguments[0]
    control = gate.qubits[0]
    target = gate.qubits[1]

    ry_pos_theta_half = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="ry"),
        arguments=[
            ast.BinaryExpression(
                op=ast.BinaryOperator(17), lhs=theta, rhs=ast.FloatLiteral(value=2)
            )
        ],
        qubits=[target],
    )
    cx = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="cx"), arguments=[], qubits=[control, target]
    )
    ry_neg_theta_half = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="ry"),
        arguments=[
            ast.UnaryExpression(
                ast.UnaryOperator(3),
                ast.BinaryExpression(
                    op=ast.BinaryOperator(17), lhs=theta, rhs=ast.FloatLiteral(value=2)
                ),
            )
        ],
        qubits=[target],
    )
    return [ry_pos_theta_half, cx, ry_neg_theta_half, cx]


def decompose_crz(gate: ast.QuantumGate) -> List[ast.Statement]:
    """Decompose a cry gate into its basic gate equivalents."""
    theta = gate.arguments[0]
    control = gate.qubits[0]
    target = gate.qubits[1]

    rz_pos_theta_half = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="rz"),
        arguments=[
            ast.BinaryExpression(
                op=ast.BinaryOperator(17), lhs=theta, rhs=ast.FloatLiteral(value=2)
            )
        ],
        qubits=[target],
    )
    cx = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="cx"), arguments=[], qubits=[control, target]
    )
    rz_neg_theta_half = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="rz"),
        arguments=[
            ast.UnaryExpression(
                ast.UnaryOperator(3),
                ast.BinaryExpression(
                    op=ast.BinaryOperator(17), lhs=theta, rhs=ast.FloatLiteral(value=2)
                ),
            )
        ],
        qubits=[target],
    )
    return [rz_pos_theta_half, cx, rz_neg_theta_half, cx]


def decompose_cy(gate: ast.QuantumGate) -> List[ast.Statement]:
    """Decompose a cy gate into its basic gate equivalents."""
    control = gate.qubits[0]
    target = gate.qubits[1]

    cry_pi = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="cry"),
        arguments=[ast.Identifier(name="pi")],
        qubits=[control, target],
    )
    s = ast.QuantumGate(modifiers=[], name=ast.Identifier(name="s"), arguments=[], qubits=[control])
    return transform_program(ast.Program(statements=[cry_pi, s])).statements


def decompose_cz(gate: ast.QuantumGate) -> List[ast.Statement]:
    """Decompose a cz gate into its basic gate equivalents."""
    control = gate.qubits[0]
    target = gate.qubits[1]

    crz_pi = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="crz"),
        arguments=[ast.Identifier(name="pi")],
        qubits=[control, target],
    )
    s = ast.QuantumGate(modifiers=[], name=ast.Identifier(name="s"), arguments=[], qubits=[control])
    return transform_program(ast.Program(statements=[crz_pi, s])).statements


def decompose_cp(gate: ast.QuantumGate) -> List[ast.Statement]:
    """Decompose a cp gate into its basic gate equivalents."""
    theta = gate.arguments[0]
    control = gate.qubits[0]
    target = gate.qubits[1]

    rz_pos_theta_half_target = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="rz"),
        arguments=[
            ast.BinaryExpression(
                op=ast.BinaryOperator(17), lhs=theta, rhs=ast.FloatLiteral(value=2)
            )
        ],
        qubits=[target],
    )
    rz_pos_theta_half_control = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="rz"),
        arguments=[
            ast.BinaryExpression(
                op=ast.BinaryOperator(17), lhs=theta, rhs=ast.FloatLiteral(value=2)
            )
        ],
        qubits=[control],
    )
    cz = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="cz"), arguments=[], qubits=[control, target]
    )
    rz_neg_theta_half_control = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="rz"),
        arguments=[
            ast.UnaryExpression(
                ast.UnaryOperator(3),
                ast.BinaryExpression(
                    op=ast.BinaryOperator(17), lhs=theta, rhs=ast.FloatLiteral(value=2)
                ),
            )
        ],
        qubits=[control],
    )
    rz_neg_theta_half_target = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="rz"),
        arguments=[
            ast.UnaryExpression(
                ast.UnaryOperator(3),
                ast.BinaryExpression(
                    op=ast.BinaryOperator(17), lhs=theta, rhs=ast.FloatLiteral(value=2)
                ),
            )
        ],
        qubits=[target],
    )
    statements = [
        rz_pos_theta_half_target,
        rz_pos_theta_half_control,
        cz,
        rz_neg_theta_half_control,
        rz_neg_theta_half_target,
    ]

    return transform_program(ast.Program(statements=statements)).statements


def decompose_ch(gate: ast.QuantumGate) -> List[ast.Statement]:
    """Decompose a ch gate into its basic gate equivalents."""
    control = gate.qubits[0]
    target = gate.qubits[1]

    ry_neg_pi_quarter = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="ry"),
        arguments=[
            ast.UnaryExpression(
                ast.UnaryOperator(3),
                ast.BinaryExpression(
                    op=ast.BinaryOperator(17),
                    lhs=ast.Identifier(name="pi"),
                    rhs=ast.FloatLiteral(value=4),
                ),
            )
        ],
        qubits=[target],
    )
    cz = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="cz"), arguments=[], qubits=[control, target]
    )
    rz_pos_pi_quarter = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="ry"),
        arguments=[
            ast.BinaryExpression(
                op=ast.BinaryOperator(17),
                lhs=ast.Identifier(name="pi"),
                rhs=ast.FloatLiteral(value=4),
            )
        ],
        qubits=[target],
    )

    statements = [ry_neg_pi_quarter, cz, rz_pos_pi_quarter]

    return transform_program(ast.Program(statements=statements)).statements


def decompose_ccx(gate: ast.QuantumGate) -> List[ast.Statement]:
    """Decompose a ccx gate into its basic gate equivalents."""
    control_qubit1 = gate.qubits[0]
    control_qubit2 = gate.qubits[1]
    target = gate.qubits[2]

    h_target = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="h"), arguments=[], qubits=[target]
    )
    cx_control_qubit2_target = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="cx"), arguments=[], qubits=[control_qubit2, target]
    )
    tdg_target = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="tdg"), arguments=[], qubits=[target]
    )
    cx_control_qubit1_target = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="cx"), arguments=[], qubits=[control_qubit1, target]
    )
    t_target = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="t"), arguments=[], qubits=[target]
    )
    t_control_qubit2 = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="t"), arguments=[], qubits=[control_qubit2]
    )
    cx_control_qubit1_control_qubit2 = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="cx"),
        arguments=[],
        qubits=[control_qubit1, control_qubit2],
    )
    t_control_qubit1 = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="t"), arguments=[], qubits=[control_qubit1]
    )
    tdg_control_qubit2 = ast.QuantumGate(
        modifiers=[], name=ast.Identifier(name="tdg"), arguments=[], qubits=[control_qubit2]
    )

    return [
        h_target,
        cx_control_qubit2_target,
        tdg_target,
        cx_control_qubit1_target,
        t_target,
        cx_control_qubit2_target,
        tdg_target,
        cx_control_qubit1_target,
        t_target,
        t_control_qubit2,
        cx_control_qubit1_control_qubit2,
        h_target,
        t_control_qubit1,
        tdg_control_qubit2,
        cx_control_qubit1_control_qubit2,
    ]


def decompose_cswap(gate: ast.QuantumGate) -> List[ast.Statement]:
    """Decompose a cswap gate into its basic gate equivalents."""
    control = gate.qubits[0]
    target1 = gate.qubits[1]
    target2 = gate.qubits[2]

    ccx_control_target2_target1 = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="ccx"),
        arguments=[],
        qubits=[control, target2, target1],
    )
    ccx_control_target1_target2 = ast.QuantumGate(
        modifiers=[],
        name=ast.Identifier(name="ccx"),
        arguments=[],
        qubits=[control, target1, target2],
    )

    statements = [
        ccx_control_target2_target1,
        ccx_control_target1_target2,
        ccx_control_target2_target1,
    ]
    return transform_program(ast.Program(statements=statements)).statements


def transform_program(program: ast.Program) -> ast.Program:
    """Transform a QASM program, decomposing crx gates."""
    transformed_statements = []
    for statement in program.statements:
        if isinstance(statement, ast.QuantumGate):
            if statement.name.name == "crx":
                transformed_statements.extend(decompose_crx(statement))
            elif statement.name.name == "cry":
                transformed_statements.extend(decompose_cry(statement))
            elif statement.name.name == "crz":
                transformed_statements.extend(decompose_crz(statement))
            elif statement.name.name == "cy":
                transformed_statements.extend(decompose_cy(statement))
            elif statement.name.name == "cz":
                transformed_statements.extend(decompose_cz(statement))
            elif statement.name.name == "cp":
                transformed_statements.extend(decompose_cp(statement))
            elif statement.name.name == "ch":
                transformed_statements.extend(decompose_ch(statement))
            elif statement.name.name == "ccx":
                transformed_statements.extend(decompose_ccx(statement))
            elif statement.name.name == "cswap":
                transformed_statements.extend(decompose_cswap(statement))
            else:
                transformed_statements.append(statement)
        else:
            transformed_statements.append(statement)

    return ast.Program(statements=transformed_statements, version=program.version)


def convert_to_basis_gates(qasm: str, basis_gates: list[str]) -> str:
    """
    Converts an OpenQASM 3 program to an equivalent program
    Only uses the specified basis gates.

    Args:
        openqasm_program (str): The original OpenQASM 3 program as a string.
        basis_gates (list[str]): A list of gate names allowed in the basis set.

    Returns:
        str: The converted OpenQASM 3 program.

    Raises:
        ValueError: if the decomposition is not possible
    """
    print(basis_gates)

    program = parse(qasm)

    converted_program = transform_program(program)

    return openqasm3_to_qasm3(converted_program)
