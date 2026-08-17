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

from rlions.Chips import AbstractChip, AbstractTestFactory, Adapter, make_shaped_reward


class XJunctionFactory(AbstractTestFactory):
    def __init__(self, allow_bad_compute, spam_capacity: int = 1):
        self.max_program_length = None
        self.storage_capacity = None
        self.allow_bad_compute = allow_bad_compute
        self.spam_capacity = spam_capacity

    def preflight(self, json_objs):
        from rlions.TestSets import get_tmp_storage

        max_program_length = 0
        storage_capacity = 0

        for obj in json_objs:
            max_program_length = max(max_program_length, len(obj['circuit']))
            cs = obj['chip_state']
            storage_capacity = max(storage_capacity,
                                   len(cs['storage']) + len(get_tmp_storage(cs)) + len(cs['compute']) + len(cs['spam']))

        self.max_program_length = max_program_length
        self.storage_capacity = storage_capacity

    def produce(self, json_obj):
        from rlions.TestSets import program_from_json, stack_from_json, get_tmp_storage
        from rlions.Env import EnvState, env_ion_count
        import jax.numpy as jnp

        cs = json_obj['chip_state']

        program = program_from_json(json_obj, self.max_program_length)

        env = EnvState(
            spam=stack_from_json(cs['spam'], self.spam_capacity),
            left=stack_from_json(cs['storage'], self.storage_capacity),
            right=stack_from_json(get_tmp_storage(cs), self.storage_capacity),
            compute=stack_from_json(cs['compute'], 2),
            program=program,
            steps=0,
            last_action=-1,
        )

        return env.replace(steps=2 * env_ion_count(env) * jnp.sum(~program.completed))  # TODO: Increase here?


class QVLSXJunctionChip(AbstractChip):
    def __init__(self, storage_capacity: int = 50, allow_bad_compute: bool = True, double_encode: bool = False,
                 spam_capacity: int = 1):
        self.storage_capacity = storage_capacity
        self.spam_capacity = spam_capacity
        self.allow_bad_compute = allow_bad_compute
        self.double_encode = double_encode

    @property
    def n_actions(self):
        return 12

    @staticmethod
    def is_done(env_state):
        from rlions.Env import env_program_completable
        return env_program_completable(env_state)

    def make_eval_env(self):
        from rlions.Env import make_env

        # Configure such that negative reward is exactly positive number of steps

        return make_env(
            reset=None,
            shaped_reward_fn=None,
            smdp_gamma=1.,
            timeout='truncate',
            max_mark_steps=1,
            step_reward=-1,
            allow_bad_compute=self.allow_bad_compute,
        )

    def make_observation_adapter(self) -> Adapter:
        """
        Make a state-to-cells adapter for the QVLS Q1 X-Junction chip
        :return: Adapter
        """
        import jax.numpy as jnp
        from rlions.Env import EnvState
        from rlions.Adapters import stack_to_cells, stacks_as_double_to_cells

        stack_capacity = self.storage_capacity
        spam_capacity = self.spam_capacity
        double_encode = self.double_encode

        def _adapter(s: EnvState):
            assert s.compute.buffer.shape == (2,)
            assert s.spam.buffer.shape == (spam_capacity,)

            left_capacity, = s.left.buffer.shape
            right_capacity, = s.right.buffer.shape

            # While stack_resize can shrink, we do not allow it because it is pointless and likely a misconfiguration
            if left_capacity > stack_capacity:
                raise ValueError(
                    f"This adapter only supports storage stacks up to {stack_capacity} qubits, have {left_capacity}")

            if right_capacity > stack_capacity:
                raise ValueError(
                    f"This adapter only supports storage stacks up to {stack_capacity} qubits, have {right_capacity}")

            if double_encode:
                return jnp.concatenate([
                    stack_to_cells(s.compute, 2),
                    stack_to_cells(s.spam, spam_capacity),
                    stacks_as_double_to_cells(s.left, s.right, stack_capacity),
                ], axis=-1)
            else:
                return jnp.concatenate([
                    stack_to_cells(s.compute, 2),
                    stack_to_cells(s.spam, spam_capacity),
                    stack_to_cells(s.left, stack_capacity),
                    stack_to_cells(s.right, stack_capacity),
                ], axis=-1)

        return Adapter(
            n_cells=2 + spam_capacity + 1 + stack_capacity if double_encode else 2 + spam_capacity + 2 * stack_capacity,
            max_qubits=stack_capacity,
            encode=_adapter,
        )

    @staticmethod
    def count_ions(env_state):
        from rlions.Env import env_ion_count
        return env_ion_count(env_state)

    def make_train_env(self, max_op_count, smdp_gamma, step_reward, shaped_gamma=None, timeout='truncate',
                       shape_factor_gates=1.0, max_ion_count=None, min_ion_count=None, shape_dynamic_factor=False):
        from rlions.Env import make_env, make_random_reset

        if max_ion_count is None:
            max_ion_count = self.storage_capacity
        elif max_ion_count > self.storage_capacity:
            raise ValueError

        if min_ion_count is None:
            min_ion_count = 2

        return make_env(
            smdp_gamma=smdp_gamma,
            timeout=timeout,
            reset=make_random_reset(
                max_op_count=max_op_count,
                min_ion_count=min_ion_count,
                max_ion_count=max_ion_count,
                spam_capacity=self.spam_capacity,
            ),
            shaped_gamma=shaped_gamma,
            step_reward=step_reward,
            shaped_reward_fn=make_shaped_reward(self.count_ions, shape_factor_gates, shape_dynamic_factor),
            allow_bad_compute=self.allow_bad_compute,
        )

    def make_test_factory(self):
        return XJunctionFactory(self.allow_bad_compute, spam_capacity=self.spam_capacity)
