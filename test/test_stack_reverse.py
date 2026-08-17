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

from rlions.Elements.Stack import stack_init, stack_push, stack_reverse, stack_pop


def test_empty():
    s = stack_init(5)

    s_rev = stack_reverse(s)

    assert s_rev.count == 0


def test_single():
    s = stack_init(5)
    s = stack_push(s, 1)

    s_rev = stack_reverse(s)

    assert s_rev.count == 1
    assert jnp.allclose(s_rev.buffer[:1], jnp.asarray([1]))


def test_multiple():
    s = stack_init(5)
    s = stack_push(s, 1)
    s = stack_push(s, 2)
    s = stack_push(s, 3)

    s_rev = stack_reverse(s)

    assert s_rev.count == 3
    assert jnp.allclose(s_rev.buffer[:3], jnp.asarray([3, 2, 1]))


def test_full():
    s = stack_init(5)
    s = stack_push(s, 1)
    s = stack_push(s, 2)
    s = stack_push(s, 3)
    s = stack_push(s, 4)
    s = stack_push(s, 5)

    s_rev = stack_reverse(s)

    assert s_rev.count == 5
    assert jnp.allclose(s_rev.buffer, jnp.asarray([5, 4, 3, 2, 1]))


def test_batched():
    s1 = stack_init(5)
    s1 = stack_push(s1, 1)
    s1 = stack_push(s1, 2)
    s1 = stack_push(s1, 3)
    s1 = stack_push(s1, 4)
    s1 = stack_push(s1, 5)

    s2 = stack_init(5)
    s2 = stack_push(s2, 1)
    s2 = stack_push(s2, 2)
    s2 = stack_push(s2, 3)

    s = jax.tree.map(lambda x, y: jnp.stack([x, y]), s1, s2)

    s_rev = stack_reverse(s)

    assert jnp.allclose(s_rev.count, jnp.asarray([5, 3]))
    assert jnp.allclose(s_rev.buffer[0], jnp.asarray([5, 4, 3, 2, 1]))
    assert jnp.allclose(s_rev.buffer[1, :3], jnp.asarray([3, 2, 1]))
