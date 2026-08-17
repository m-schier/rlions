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

from typing import Tuple

import jax
import jax.numpy as jnp
import chex


@chex.dataclass(frozen=True)
class DoubleEndedBuffer:
    """
    A buffer which can be append to from both sides, shifting elements inward. Like a double shift register.
    """

    """
    Underlying buffer shared by both stacks
    """
    buffer: jax.Array

    """
    The count of elements on the front (or left)
    """
    front_count: jax.Array

    """
    The count of elements on the back (or right)
    """
    back_count: jax.Array


def de_buffer_init(capacity: int) -> DoubleEndedBuffer:
    return DoubleEndedBuffer(
        buffer=jnp.empty(capacity, jnp.int32),
        front_count=jnp.zeros((), jnp.int32),
        back_count=jnp.zeros((), jnp.int32),
    )


def de_buffer_can_push(s: DoubleEndedBuffer) -> jax.Array:
    *_, capacity = s.buffer.shape

    return capacity > s.front_count + s.back_count


def de_buffer_can_pop_front(s: DoubleEndedBuffer) -> jax.Array:
    return s.front_count > 0


def de_buffer_can_pop_back(s: DoubleEndedBuffer) -> jax.Array:
    return s.back_count > 0


def de_buffer_push_front(s: DoubleEndedBuffer, elem: jax.Array) -> DoubleEndedBuffer:
    # No effect if cannot push
    capacity, = s.buffer.shape

    do_push = de_buffer_can_push(s)

    new_front_count = jnp.where(do_push, s.front_count + 1, s.front_count)
    # Shift right
    updated_buffer = jnp.where(
        jnp.arange(capacity) < s.front_count + 1,
        jnp.roll(s.buffer, 1),
        s.buffer,
    ).at[0].set(elem)

    new_buffer = jnp.where(do_push, updated_buffer, s.buffer)

    return s.replace(buffer=new_buffer, front_count=new_front_count)


def de_buffer_push_back(s: DoubleEndedBuffer, elem: jax.Array) -> DoubleEndedBuffer:
    # No effect if cannot push
    capacity, = s.buffer.shape

    do_push = de_buffer_can_push(s)

    new_back_count = jnp.where(do_push, s.back_count + 1, s.back_count)
    # Shift left
    updated_buffer = jnp.where(
        jnp.arange(capacity) >= capacity - s.back_count - 1,
        jnp.roll(s.buffer, -1),
        s.buffer,
    ).at[-1].set(elem)

    new_buffer = jnp.where(do_push, updated_buffer, s.buffer)

    return s.replace(buffer=new_buffer, back_count=new_back_count)


def de_buffer_pop_front(s: DoubleEndedBuffer) -> Tuple[DoubleEndedBuffer, jax.Array]:
    capacity, = s.buffer.shape
    elem = s.buffer[0]

    can_pop = de_buffer_can_pop_front(s)

    new_front_count = jnp.where(can_pop, s.front_count - 1, s.front_count)
    # Shift front left
    updated_buffer = jnp.where(
        jnp.arange(capacity) < s.front_count,
        jnp.roll(s.buffer, -1),
        s.buffer,
    )

    new_buffer = jnp.where(can_pop, updated_buffer, s.buffer)

    return s.replace(buffer=new_buffer, front_count=new_front_count), elem


def de_buffer_pop_back(s: DoubleEndedBuffer) -> Tuple[DoubleEndedBuffer, jax.Array]:
    capacity, = s.buffer.shape
    elem = s.buffer[-1]

    can_pop = de_buffer_can_pop_back(s)

    new_back_count = jnp.where(can_pop, s.back_count - 1, s.back_count)
    # Shift back right
    updated_buffer = jnp.where(
        jnp.arange(capacity) >= capacity - s.back_count,
        jnp.roll(s.buffer, 1),
        s.buffer,
    )

    new_buffer = jnp.where(can_pop, updated_buffer, s.buffer)

    return s.replace(buffer=new_buffer, back_count=new_back_count), elem


def de_buffer_rotate_front(s: DoubleEndedBuffer) -> DoubleEndedBuffer:
    # Rotates all elements into the front
    capacity, = s.buffer.shape

    new_buffer = jnp.where(
        jnp.arange(capacity) < s.front_count,
        # Keep original front unchanged
        s.buffer,
        # Roll left by empty space
        jnp.roll(s.buffer, -(capacity - s.front_count - s.back_count)),
    )

    return DoubleEndedBuffer(
        buffer=new_buffer,
        front_count=s.front_count + s.back_count,
        back_count=jnp.zeros_like(s.back_count),
    )


def de_buffer_rotate_back(s: DoubleEndedBuffer) -> DoubleEndedBuffer:
    # Rotates all elements into the back
    capacity, = s.buffer.shape

    new_buffer = jnp.where(
        jnp.arange(capacity) >= capacity - s.back_count,
        # Keep original back unchanged
        s.buffer,
        # Roll right by empty space
        jnp.roll(s.buffer, capacity - s.front_count - s.back_count),
    )

    return DoubleEndedBuffer(
        buffer=new_buffer,
        front_count=jnp.zeros_like(s.front_count),
        back_count=s.front_count + s.back_count,
    )
