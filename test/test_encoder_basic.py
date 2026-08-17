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

import jax.random

from rlions.ObservationsImproved import make_encoder_basic
from rlions.Chips import get_chip_by_name


def test1():
    return
    chip = get_chip_by_name('qvls_x_10')

    encoder = make_encoder_basic(chip.make_observation_adapter(), 2, number_encoder='linear')
    env = chip.make_train_env(3, 1.0, -1.0, min_ion_count=3, max_ion_count=3)
    state = env.reset(jax.random.key(0))
    print(state.program.operations)

    actual = encoder(state)
    raise ValueError(actual)
