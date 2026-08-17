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
from rlions.Elements.Stack import stack_init, stack_push
from rlions.Adapters import stacks_as_double_to_cells


def test_stacks_as_double_to_cells():
    s_top = stack_init(5)
    s_top = stack_push(s_top, 0)
    s_top = stack_push(s_top, 1)
    s_bottom = stack_init(5)
    s_bottom = stack_push(s_bottom, 2)
    s_bottom = stack_push(s_bottom, 3)
    s_bottom = stack_push(s_bottom, 4)

    cells = stacks_as_double_to_cells(s_bottom, s_top, 5)
    expected = jnp.array([4, 3, 2, -1, 0, 1])
    assert jnp.allclose(cells, expected)
