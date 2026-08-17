#  RLIonS - Reinforcement Learning Ion Shuttling Compiler
#  Copyright (C) 2026 Maximilian Schier, Lea Richtmann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os

import numpy as np

os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from rlions.Util import make_gate_depth, find_first_k_gates, make_sub_gate_depth
from rlions.Program import make_program, make_random_program, Program
from rlions.ObservationsImproved import make_fixed_lookahead


def test_gate_depth():
    prog = make_program(jnp.array([(0, 2), (1, 3), (0, 4), (0, 2)]))

    actual = make_gate_depth(prog, 5)

    expected = jnp.array([0, 0, 1, 2])

    assert jnp.allclose(expected, actual)


def test_sub_gate_depth_agrees():
    program = make_program(jnp.array([(0, 2), (1, 3), (0, 4), (0, 2)]))

    max_qubits = 5
    lookahead = 1

    lookahead_buffer = find_first_k_gates(program, max_qubits, lookahead)

    cells = jnp.arange(max_qubits)

    idxs = jnp.where(
        cells[..., None] >= 0,
        lookahead_buffer[cells],
        -1,
    )

    gt_idxs = make_fixed_lookahead(make_gate_depth(program, max_qubits), idxs)
    sub_idxs = make_fixed_lookahead(make_sub_gate_depth(program, max_qubits, lookahead_buffer), idxs)

    assert jnp.allclose(sub_idxs, gt_idxs)

    # Check other order also valid
    sub_lookahead_buffer = make_fixed_lookahead(make_sub_gate_depth(program, max_qubits, lookahead_buffer), lookahead_buffer)
    sub_idxs2 = jnp.where(
        cells[..., None] >= 0,
        sub_lookahead_buffer[cells],
        -1,
    )

    assert jnp.allclose(sub_idxs2, gt_idxs)


def test_sub_gate_depth_agrees2():
    # A simple failing case found by random testing
    program = Program(
        completed=jnp.array([False, False, False, False, False, False, True, False, False, True, False, False]),
        operations=jnp.array([[3, 1],
              [3, 1],
              [3, 0],
              [0, 3],
              [0, 1],
              [1, 0],
              [2, 3],
              [3, 2],
              [3, 1],
              [1, 2],
              [1, 0],
              [0, 3]])
    )

    max_qubits = 5
    lookahead = 1

    lookahead_buffer = find_first_k_gates(program, max_qubits, lookahead)
    print(lookahead_buffer)

    cells = jnp.arange(max_qubits)

    idxs = jnp.where(
        cells[..., None] >= 0,
        lookahead_buffer[cells],
        -1,
    )

    gt_idxs = make_fixed_lookahead(make_gate_depth(program, max_qubits), idxs)
    sub_idxs = make_fixed_lookahead(make_sub_gate_depth(program, max_qubits, lookahead_buffer), idxs)

    assert jnp.allclose(sub_idxs, gt_idxs)


def test_sub_gate_depth_agrees_random():
    lookahead = 2
    max_qubits = 4

    rng = jax.random.key(0)
    for i in range(100):
        rng, program_rng, shuffle_rng = jax.random.split(rng, 3)
        program = make_random_program(12, max_qubits, 0.75, program_rng)

        cells = jnp.concatenate([jnp.arange(max_qubits), jnp.full(max_qubits, -1, dtype=jnp.int32)])
        cells = jax.random.permutation(shuffle_rng, cells, independent=True)

        n_cells, = cells.shape

        lookahead_buffer = find_first_k_gates(program, max_qubits, lookahead)

        idxs = jnp.where(
            cells[..., None] >= 0,
            lookahead_buffer[cells],
            -1,
        )

        expected_lookahead = make_fixed_lookahead(make_gate_depth(program, max_qubits), idxs)
        actual_lookahead = make_fixed_lookahead(make_sub_gate_depth(program, max_qubits, lookahead_buffer), idxs)

        assert jnp.allclose(expected_lookahead, actual_lookahead), f"Iteration {i}, Program {np.array2string(program.completed, separator=', ')}, {np.array2string(program.operations, separator=', ')}"


def test_gate_depth_with_completed():
    prog = make_program(jnp.array([(0, 2), (1, 4), (1, 3), (0, 4), (0, 2)]))
    prog = prog.replace(completed=prog.completed.at[1].set(True))

    actual = make_gate_depth(prog, 5)

    expected = jnp.array([0, -1, 0, 1, 2])

    assert jnp.allclose(expected, actual)

