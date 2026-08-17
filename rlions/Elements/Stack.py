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

import chex
import jax
import jax.numpy as jnp
from typing import Tuple
from functools import partial


@chex.dataclass(frozen=True)
class Stack:
    buffer: jax.Array
    count: jax.Array


def stack_init(capacity, dtype=jnp.int32) -> Stack:
    return Stack(buffer=jnp.zeros(shape=(capacity,), dtype=dtype), count=0)


def stack_pop(stack: Stack) -> Tuple[Stack, jax.Array]:
    # Can't throw if empty on GPU so indicate with large value
    valid = stack.count > 0
    el = jax.lax.select(valid, stack.buffer[stack.count - 1], jnp.iinfo(stack.buffer.dtype).max)
    new_count = jax.lax.select(valid, stack.count - 1, 0)
    return stack.replace(count=new_count), el


def stack_pop_conditional(stack: Stack, do_pop) -> Tuple[Stack, jax.Array]:
    new_stack, el = stack_pop(stack)

    return jax.tree.map(lambda a, b: jax.lax.select(do_pop, a, b), new_stack, stack), el


def stack_push(stack: Stack, el: jax.Array) -> Stack:
    valid = stack.count < stack.buffer.shape[-1]
    new_count = jax.lax.select(valid, stack.count + 1, stack.count)
    new_buffer = stack.buffer.at[new_count - 1].set(jax.lax.select(valid, el, stack.buffer[new_count - 1]))
    return stack.replace(buffer=new_buffer, count=new_count)


def stack_push_conditional(stack: Stack, el: jax.Array, do_push) -> Stack:
    new_stack = stack_push(stack, el)
    return jax.tree.map(lambda a, b: jax.lax.select(do_push, a, b), new_stack, stack)


def stack_peek(stack: Stack) -> jax.Array:
    return stack.buffer[stack.count - 1]


def stack_reverse(stack: Stack) -> Stack:
    """
    Reverse a stack (or multiple stacks). This will leave the stack count(s) unchanged.
    """

    *batch_shape, capacity = stack.buffer.shape
    chex.assert_shape(stack.count, batch_shape)

    # Only required if we are not jitted
    count = jnp.asarray(stack.count)

    idxs = capacity - 1 - jnp.arange(capacity)
    # 0-th idxs-entry points to theoretical top of stack if count == capacity

    idxs = idxs - (capacity - count[..., None])

    new_buffer = stack.buffer[tuple(jnp.indices(tuple(batch_shape) + (1,)))[:-1] + (idxs,)]

    return stack.replace(buffer=new_buffer)


def stack_resize(stack: Stack, target_capacity: int):
    """
    Either trims (from bottom) or pads with empty (upon top) a stack (or batched stacks) to a desired capacity
    """

    *batch_shape, current_capacity = stack.buffer.shape

    if current_capacity == target_capacity:
        return stack
    elif current_capacity < target_capacity:
        # Pad
        new_buffer = jnp.zeros((*batch_shape, target_capacity), stack.buffer.dtype).at[..., :current_capacity].set(stack.buffer)
        return stack.replace(buffer=new_buffer)
    else:  # current_capacity > target_capacity:
        # Require a batched version of dynamic_slice_in_dim
        dyn_slice = partial(jax.lax.dynamic_slice_in_dim, slice_size=target_capacity, axis=-1)

        for _ in range(len(jnp.shape(stack.count))):
            dyn_slice = jax.vmap(dyn_slice)

        # Slice away bottom of stack. Obviously the agent may perform poorly.
        # Case 1: Stack fits
        # Case 2: Stack must be cut (keep top)
        new_count = jnp.where(stack.count <= target_capacity, stack.count, target_capacity)
        new_buffer = jnp.where(
            (stack.count <= target_capacity)[..., None],
            stack.buffer[..., :target_capacity],
            dyn_slice(operand=stack.buffer, start_index=stack.count - target_capacity)
        )
        return stack.replace(count=new_count, buffer=new_buffer)
