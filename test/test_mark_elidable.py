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
from rlions.Program import make_program, mark_elidable_fast


def test1_fast():
    p = make_program(jnp.array([(0, 1), (2, 3), (0, 1), (1, 3)]))
    new_program = mark_elidable_fast(p, 4)

    expected_completed = jnp.array([False, False, True, False])

    assert jnp.allclose(expected_completed, new_program.completed)


def test2_fast():
    p = make_program(jnp.array([(0, 1), (2, 3), (0, 1), (1, 3)]))
    p = p.replace(completed=p.completed.at[0].set(True))
    new_program = mark_elidable_fast(p, 4)

    expected_completed = jnp.array([True, False, False, False])

    assert jnp.allclose(expected_completed, new_program.completed)


def test3_fast():
    p = make_program(jnp.array([(0, 1), (0, 1), (0, 1), (0, 1)]))
    p = p.replace(completed=p.completed.at[-1].set(True))
    new_program = mark_elidable_fast(p, 4)

    expected_completed = jnp.array([False, True, True, True])

    assert jnp.allclose(expected_completed, new_program.completed)
