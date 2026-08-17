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

import chex
from typing import Callable


@chex.dataclass(frozen=True)
class Adapter:
    n_cells: int
    max_qubits: int
    encode: Callable


class AbstractTestFactory:
    def preflight(self, cases):
        raise NotImplementedError

    def produce(self, case):
        raise NotImplementedError


class AbstractChip:
    def make_eval_env(self):
        """
        Create an evaluation environment. Generally, an evaluation environment must not have a reset method and
        can assume that given states are pre-processed, especially that the program of any given state is fully
        deduplicated.
        :return: Evaluation environment
        """
        raise NotImplementedError

    def make_train_env(self, max_op_count, smdp_gamma, step_reward, shaped_gamma=None, timeout='truncate',
                       shape_factor_gates=1.0, min_ion_count=None, max_ion_count=None, shape_dynamic_factor=False):
        raise NotImplementedError

    def make_observation_adapter(self) -> Adapter:
        raise NotImplementedError

    @staticmethod
    def is_done(env_state):
        """
        Check whether the given state is immediately acceptable as a terminal state. The primary use case is to
        check whether a given start state is immediately terminal, which happens with some test cases.
        :param env_state: Environment state
        :return: True if terminal, otherwise false
        """
        raise NotImplementedError

    @staticmethod
    def count_ions(env_state):
        raise NotImplementedError

    @property
    def n_actions(self):
        raise NotImplementedError

    def make_test_factory(self):
        raise NotImplementedError


def make_shaped_reward(ion_count, factor_completion: float, dynamic_factor: bool = False):
    def shaped_reward(state):
        import jax.numpy as jnp
        if dynamic_factor:
            fc = factor_completion * jnp.maximum(1., ion_count(state) / 20.)
        else:
            fc = factor_completion

        # A potential reward where the potential is 0 in the goal states, but then isn't because of bias.
        # Still, this formulation makes more sense, because it is only based on actually observable factors.
        # The bias really helps with training stability
        # bias = 2 * env_ion_count(state)
        return (-jnp.sum(~state.program.completed, axis=-1)) * fc

    return shaped_reward


def get_chip_by_name(name: str) -> AbstractChip:
    import re

    def handle_x():
        match = re.match("^qvls_x_([0-9]+)$", name)

        if match:
            from rlions.ChipImplementations.QVLSXChip import QVLSXJunctionChip
            return QVLSXJunctionChip(storage_capacity=int(match.groups()[0]), allow_bad_compute=True)

        match = re.match("^qvls_x_([0-9]+)_s([0-9]+)$", name)

        if match:
            from rlions.ChipImplementations.QVLSXChip import QVLSXJunctionChip
            return QVLSXJunctionChip(storage_capacity=int(match.groups()[0]), allow_bad_compute=True,
                                     spam_capacity=int(match.groups()[1]))

    def handle_xd():
        match = re.match("^qvls_x_([0-9]+)_d$", name)

        if match:
            from rlions.ChipImplementations.QVLSXChip import QVLSXJunctionChip
            return QVLSXJunctionChip(storage_capacity=int(match.groups()[0]), allow_bad_compute=True, double_encode=True)

    def handle_q():
        match = re.match("^qvls_q_([0-9]+)$", name)

        if match:
            from rlions.ChipImplementations.QVLSQChip import QVLSQChip
            return QVLSQChip(storage_capacity=int(match.groups()[0]))

        match = re.match("^qvls_q_([0-9]+)_s([0-9]+)$", name)

        if match:
            from rlions.ChipImplementations.QVLSQChip import QVLSQChip
            return QVLSQChip(storage_capacity=int(match.groups()[0]), spam_capacity=int(match.groups()[1]))

    def handle_qs():
        match = re.match("^qvls_qs_([0-9]+)$", name)

        if match:
            from rlions.ChipImplementations.QVLSQChip import QVLSQChip
            return QVLSQChip(storage_capacity=int(match.groups()[0]), swap_capable=True)

    handlers = [handle_x, handle_xd, handle_q, handle_qs]

    for h in handlers:
        res = h()

        if res:
            return res

    raise ValueError(f"Failed to interpret chip name: {name}")
