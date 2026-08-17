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

from functools import partial

from rlions.Util import find_first_k


def test1():
    arr = jnp.array([
        [False, True, True],
        [True, False, False],
    ])

    expected = jnp.array([
        [1, 2],
        [0, -1],
    ])

    actual = find_first_k(arr, 2, -1)

    chex.assert_equal_shape([actual, expected])

    assert jnp.allclose(expected, actual)


def test_returns_first():
    arr = jnp.array([
        [False, True, True, True],
        [True, False, False, False],
    ])

    expected = jnp.array([
        [1, 2],
        [0, -1],
    ])

    actual = find_first_k(arr, 2, -1)

    chex.assert_equal_shape([actual, expected])

    assert jnp.allclose(expected, actual)


def test_returns_compiled_gpu():
    arr = jnp.array([
        [False, True, True, True],
        [True, False, False, False],
    ])

    expected = jnp.array([
        [1, 2],
        [0, -1],
    ])

    fn = jax.jit(partial(find_first_k, k=2, axis=-1))

    actual = fn(jax.device_put(arr, jax.devices('gpu')[0]))

    chex.assert_equal_shape([actual, expected])

    assert jnp.allclose(expected, actual)
