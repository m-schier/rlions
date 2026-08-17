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

from rlions.Elements.Stack import stack_init, stack_push, stack_resize, stack_pop


def test_same_capacity():
    s = stack_init(5)
    s = stack_push(s, 1)
    s = stack_push(s, 2)

    s_res = stack_resize(s, 3)

    assert s_res.count == 2
    s_res, el = stack_pop(s_res)
    assert el == 2
    s_res, el = stack_pop(s_res)
    assert el == 1


def test_pad():
    s = stack_init(2)
    s = stack_push(s, 1)
    s = stack_push(s, 2)

    s_res = stack_resize(s, 3)

    assert s_res.count == 2
    s_res, el = stack_pop(s_res)
    assert el == 2
    s_res, el = stack_pop(s_res)
    assert el == 1


def test_shrink_fit():
    s = stack_init(3)
    s = stack_push(s, 1)
    s = stack_push(s, 2)

    s_res = stack_resize(s, 2)

    assert s_res.count == 2
    s_res, el = stack_pop(s_res)
    assert el == 2
    s_res, el = stack_pop(s_res)
    assert el == 1


def test_shrink_no_fit():
    s = stack_init(3)
    s = stack_push(s, 1)
    s = stack_push(s, 2)
    s = stack_push(s, 3)

    s_res = stack_resize(s, 2)

    assert s_res.count == 2
    s_res, el = stack_pop(s_res)
    assert el == 3
    s_res, el = stack_pop(s_res)
    assert el == 2


def test_shrink_no_fit_batched():
    s1 = stack_init(3)
    s1 = stack_push(s1, 1)
    s1 = stack_push(s1, 2)
    s1 = stack_push(s1, 3)

    s2 = stack_init(3)
    s2 = stack_push(s2, 4)
    s2 = stack_push(s2, 5)
    s2 = stack_push(s2, 6)

    s = jax.tree.map(lambda x, y: jnp.stack([x, y]), s1, s2)

    s_res = stack_resize(s, 2)
