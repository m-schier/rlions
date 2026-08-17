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


@chex.dataclass(frozen=True)
class Program:
    """
    A quantum circuit as a sequence of two-qubit quantum gates, with completion flags.
    """

    """
    Nx2 list of binary operations to be carried out
    """
    operations: jax.Array

    """
    List of shape N, marks whether an operation has been completed.
    """
    completed: jax.Array


def is_fully_processed(program: Program):
    return jnp.all(program.completed)


def mark_elidable_fast(program: Program, max_qubits: int):
    def _scan_fn(carry, x):
        (a, b), completed = x

        mark_completed = completed | ((carry[a] == b) & (carry[b] == a))

        # Update carry
        carry = carry.at[jax.lax.select(completed, max_qubits + 1, a)].set(b, mode='drop')
        carry = carry.at[jax.lax.select(completed, max_qubits + 1, b)].set(a, mode='drop')

        return carry, mark_completed

    def _scan_unbatched(p: Program):
        _, new_completed = jax.lax.scan(_scan_fn, jnp.full(max_qubits, -1, dtype=jnp.int32), (p.operations, p.completed))
        return p.replace(completed=new_completed)

    fn = _scan_unbatched

    for _ in program.completed.shape[:-1]:
        fn = jax.vmap(fn)

    return fn(program)


def can_mark(program: Program, q_first, q_second, return_idx=False):
    matches = jnp.all(program.operations == jnp.asarray((q_first, q_second)), axis=-1) | jnp.all(program.operations == jnp.asarray((q_second, q_first)), axis=-1)
    matches_mask = matches & ~program.completed
    first_exact_idx = jnp.argmax(matches_mask)
    first_exact_valid = matches_mask[first_exact_idx]

    usage = jnp.any(program.operations == q_first, axis=-1) | jnp.any(program.operations == q_second, axis=-1)
    first_usage_idx = jnp.argmax(usage & ~program.completed)

    valid = first_exact_valid & (first_usage_idx == first_exact_idx)

    if return_idx:
        return valid, first_exact_idx
    else:
        return valid


def maybe_mark(program: Program, q_first, q_second):
    valid, idx = can_mark(program, q_first, q_second, return_idx=True)

    new_completed = program.completed.at[idx].set(valid | program.completed[idx])

    return program.replace(completed=new_completed)


def make_program(unnormalized_program: jax.Array) -> Program:
    """
    This method is for testing only.
    """
    chex.assert_rank(unnormalized_program, 2)
    chex.assert_equal(unnormalized_program.shape[-1], 2)

    operations = jnp.stack([
        jnp.min(unnormalized_program, axis=-1),
        jnp.max(unnormalized_program, axis=-1),
        ], axis=-1)
    completed = jnp.zeros(operations.shape[:-1], dtype=jnp.bool)
    return Program(operations=operations, completed=completed)


def make_random_program(max_op_count: int, ion_count: int, op_chance: float, rng) -> Program:
    assert max_op_count > 0

    rng1, rng2, rng3 = jax.random.split(rng, 3)

    left_operands = jax.random.randint(rng1, (max_op_count,), 0, ion_count)
    right_operands = jax.random.randint(rng2, (max_op_count,), 0, ion_count - 1)

    # Prevent same operand
    right_operands = right_operands + jnp.astype(left_operands <= right_operands, jnp.int32)

    completed = jax.random.uniform(rng3, (max_op_count,)) >= op_chance

    # Ensure program never empty
    completed = completed.at[..., 0].set(False)

    return Program(operations=jnp.stack([left_operands, right_operands], axis=-1), completed=completed)
