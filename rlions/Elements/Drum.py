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
from typing import Tuple


DRUM_INVALID = -1


@chex.dataclass(frozen=True)
class DrumState:
    buffer: jax.Array


def drum_init(capacity: int) -> DrumState:
    return DrumState(
        buffer=jnp.full(capacity, DRUM_INVALID, dtype=jnp.int32)
    )


def drum_peek_front(ds: DrumState) -> jax.Array:
    return ds.buffer[..., 0]


def drum_peek_back(ds: DrumState) -> jax.Array:
    return ds.buffer[..., -1]


def drum_can_push_front(ds: DrumState) -> jax.Array:
    return drum_peek_back(ds) == DRUM_INVALID


def drum_can_push_back(ds: DrumState) -> jax.Array:
    return drum_peek_front(ds) == DRUM_INVALID


# Inverse conditions
drum_can_pop_front = drum_can_push_back
drum_can_pop_back = drum_can_push_front


def drum_pop_front(ds: DrumState) -> Tuple[DrumState, jax.Array]:
    ret_val = drum_peek_front(ds)
    new_buffer = jnp.roll(ds.buffer, shift=-1, axis=-1).at[..., -1].set(DRUM_INVALID)
    return ds.replace(buffer=new_buffer), ret_val


def drum_pop_back(ds: DrumState) -> Tuple[DrumState, jax.Array]:
    ret_val = drum_peek_back(ds)
    new_buffer = jnp.roll(ds.buffer, shift=1, axis=-1).at[..., 0].set(DRUM_INVALID)
    return ds.replace(buffer=new_buffer), ret_val


def drum_rotate_front_to_back(ds: DrumState) -> DrumState:
    return ds.replace(buffer=jnp.roll(ds.buffer, shift=-1, axis=-1))


def drum_rotate_back_to_front(ds: DrumState) -> DrumState:
    return ds.replace(buffer=jnp.roll(ds.buffer, shift=1, axis=-1))


def drum_push_front(ds: DrumState, el: jax.Array) -> DrumState:
    return ds.replace(buffer=jnp.roll(ds.buffer, shift=1, axis=-1).at[..., 0].set(el))


def drum_push_back(ds: DrumState, el: jax.Array) -> DrumState:
    return ds.replace(buffer=jnp.roll(ds.buffer, shift=-1, axis=-1).at[..., -1].set(el))


def drum_set_front(ds: DrumState, el: jax.Array) -> DrumState:
    return ds.replace(buffer=ds.buffer.at[..., 0].set(el))


def drum_set_back(ds: DrumState, el: jax.Array) -> DrumState:
    return ds.replace(buffer=ds.buffer.at[..., -1].set(el))
