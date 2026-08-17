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

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from rlions.Program import make_program
from rlions.Env import EnvState, env_pretty_print
from rlions.ObservationsImproved import make_qubit_positions_from_cells


def make_test_state() -> EnvState:
    from rlions.Elements.Stack import stack_init, stack_push

    left = stack_init(5)
    left = stack_push(left, 0)
    left = stack_push(left, 3)
    right = stack_init(5)
    right = stack_push(right, 4)
    right = stack_push(right, 2)
    compute = stack_init(2)
    compute = stack_push(compute, 1)
    spam = stack_init(1)

    return EnvState(
        left=left,
        right=right,
        compute=compute,
        spam=spam,
        program=make_program(jnp.array([(0, 2), (1, 3), (0, 4), (0, 2)])),
        steps=0,
        last_action=-1,
    )


def test_make_x_junction_adapter():
    from rlions.ChipImplementations.QVLSXChip import QVLSXJunctionChip
    state = make_test_state()
    max_qubits = 5
    env_pretty_print(state)

    adapter = QVLSXJunctionChip(storage_capacity=max_qubits).make_observation_adapter()

    cells = adapter.encode(state)

    positions = make_qubit_positions_from_cells(cells, max_qubits)
    expected = jnp.array([4, 0, 8, 3, 9])

    assert jnp.allclose(positions, expected)
