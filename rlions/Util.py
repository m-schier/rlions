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

import jax
import jax.numpy as jnp
import chex
from rlions.Program import Program


def compatible_load(path):
    import sys
    import pickle
    has_patched_rlions = False
    has_patched_stack = False

    # Required for backward compat due to refactoring
    if 'JaxIonShuttle' not in sys.modules:
        import rlions
        sys.modules['JaxIonShuttle'] = rlions
        has_patched_rlions = True
    if 'JaxIonShuttle.Stack' not in sys.modules:
        from rlions.Elements import Stack
        sys.modules['JaxIonShuttle.Stack'] = Stack
        has_patched_stack = True

    try:
        with open(path, "rb") as fp:
            return pickle.load(fp)
    finally:
        if has_patched_stack:
            del sys.modules['JaxIonShuttle.Stack']
        if has_patched_rlions:
            del sys.modules['JaxIonShuttle']


def find_first_k(arr: jax.Array, k: int, axis: int = -1, fill_value: jax.Array = -1) -> jax.Array:
    # Not implemented for anything else yet
    assert axis == -1 or axis == len(arr.shape) - 1
    assert jnp.dtype(arr) == jnp.bool

    # topk on boolean arrays seems to be stable regarding sort order,
    # but since the documentation does not mention it, we should probably not
    # rely on it. Therefore, introduce auxiliary scores to enforce returning
    # first top values.
    dummy_indices = jnp.arange(arr.shape[-1])

    scores = jnp.where(arr, arr.shape[-1] - dummy_indices, -1)

    values, indices = jax.lax.top_k(scores, k)

    return jnp.where(values >= 0, indices, jnp.full_like(indices, fill_value))


def broadcast_left(arr, shape):
    assert len(arr.shape) <= len(shape)
    chex.assert_equal(arr.shape, shape[:len(arr.shape)])
    indexer = (Ellipsis,) + (None,) * (len(shape) - len(arr.shape))
    return jnp.broadcast_to(arr[indexer], shape)


def find_first_k_gates(program: Program, max_qubits: int, lookahead: int) -> jax.Array:
    chex.assert_shape(max_qubits, ())
    program_capacity, = program.completed.shape
    # Stack-capacity x Program-capacity
    qubits = jnp.arange(max_qubits)

    matches = ((qubits[..., None] == program.operations[..., None, :, 0]) | (qubits[..., None] == program.operations[..., None, :, 1])) & ~program.completed[..., None, :]
    chex.assert_shape(matches, (max_qubits, program_capacity))

    idxs = find_first_k(matches, lookahead)
    chex.assert_shape(idxs, (max_qubits, lookahead))

    return idxs


def make_gate_depth(program: Program, max_qubits: int) -> jax.Array:
    def _scan(_carry, x):
        operands, completed = x

        gate_depth = jax.lax.select(
            completed,
            -1,
            jnp.max(_carry[operands]),
        )

        _carry = _carry.at[operands].set(
            jnp.where(
                completed,
                _carry[operands],
                gate_depth + 1,
            )
        )

        return _carry, gate_depth

    _, result = jax.lax.scan(_scan, jnp.zeros(max_qubits, dtype=jnp.int32), (program.operations, program.completed), unroll=10)
    return result


def make_sub_gate_depth(program: Program, max_qubits: int, sub_indices) -> jax.Array:
    """
    Calculate gates based on a subset of program indices.
    :param program: The program
    :param max_qubits: Maximum number of qubits, all valid operands in programs must be in range [0, max_qubits).
    :param sub_indices: The subset of the program. Indices may be duplicate two times at most.
    :return: Depths in original program shape. Depths of operation not in sub_indices are not calculated, and it is
    the caller's responsibility to ensure that such a subset is selected that the depths the caller is interested in
    are correct.
    """

    # Caution: sub_indices will have duplicates due to two operands per operation

    program_capacity, = program.completed.shape

    sub_indices_sorted = jnp.sort(sub_indices, axis=None)

    # Now we have adjacent duplicates in some places
    ignore_mask = jnp.zeros(sub_indices_sorted.shape, jnp.bool).at[1:].set(sub_indices_sorted[:-1] == sub_indices_sorted[1:])

    # We also likely have a bunch of leading -1 entries, these are not valid, so "completed"
    sub_completed = program.completed[sub_indices_sorted] | ignore_mask | (sub_indices_sorted < 0)
    sub_operations = program.operations[sub_indices_sorted]

    depths_sorted = make_gate_depth(Program(operations=sub_operations, completed=sub_completed), max_qubits)

    return jnp.full(program_capacity, -1, jnp.int32).at[jnp.where(sub_completed, program_capacity, sub_indices_sorted)].set(depths_sorted, mode='drop')
