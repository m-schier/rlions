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

import jax.numpy as jnp
from rlions.Elements.Stack import Stack, stack_reverse, stack_resize
from rlions.Elements.DoubleEndedBuffer import DoubleEndedBuffer


def stack_to_cells(s: Stack, cap: int):
    s = stack_reverse(stack_resize(s, cap))
    return jnp.where(jnp.arange(cap) < s.count, s.buffer, -1)


def stacks_as_double_to_cells(s_bottom: Stack, s_top: Stack, cap: int):
    # Given two stacks with normal order, encode into a shared cell memory such that
    # the bottom stack is reversed (top-of-stack at bottom-most element) and the top stack is not reversed
    # (top-of-stack at top-most element). Capacity is increased by 1 such that there is always one empty element.
    # It is illegal for the combined count of s_bottom and s_top to exceed cap.
    s_bottom = stack_reverse(stack_resize(s_bottom, cap))
    s_top = stack_resize(s_top, cap)

    idxs = jnp.arange(cap + 1)

    return jnp.where(
        idxs < s_bottom.count,
        s_bottom.buffer[idxs],  # Indexing required for same shape
        jnp.where(
            idxs > cap - s_top.count,
            s_top.buffer[jnp.roll(idxs, -s_top.count)],
            -1,
        )
    )


def de_buffer_to_cells(de: DoubleEndedBuffer):
    cap, = de.buffer.shape
    idxs = jnp.arange(cap)
    valid = (idxs < de.front_count) | (idxs >= cap - de.back_count)
    return jnp.where(valid, de.buffer, -1)
