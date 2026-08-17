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

from rlions.Chips import AbstractChip, Adapter, AbstractTestFactory


class QTestFactory(AbstractTestFactory):
    def __init__(self, storage_capacity: int, spam_capacity: int = 1):
        # Ribbon has fixed storage capacity
        self.storage_capacity = storage_capacity
        self.spam_capacity = spam_capacity
        self.max_program_length = None

    def preflight(self, json_objs):
        from rlions.TestSets import get_tmp_storage
        max_program_length = 0

        for obj in json_objs:
            max_program_length = max(max_program_length, len(obj['circuit']))
            cs = obj['chip_state']
            storage_capacity = len(cs['storage']) + len(get_tmp_storage(cs)) + len(cs['compute']) + len(cs['spam'])

            if storage_capacity > self.storage_capacity:
                raise ValueError

        self.max_program_length = max_program_length

    def produce(self, json_obj):
        import jax.numpy as jnp
        from rlions.EnvQ import EnvStateQ, q_env_ion_count
        from rlions.TestSets import program_from_json, stack_from_json, compatible_drum_from_json
        cs = json_obj['chip_state']

        program = program_from_json(json_obj, self.max_program_length)

        env = EnvStateQ(
            spam=stack_from_json(cs['spam'], self.spam_capacity),
            storage=compatible_drum_from_json(cs, self.storage_capacity),
            compute=stack_from_json(cs['compute'], 2),
            program=program,
            steps=0,
            last_action=-1,
        )

        return env.replace(steps=2 * q_env_ion_count(env) * jnp.sum(~program.completed))  # TODO: Increase here?


class QVLSQChip(AbstractChip):
    def __init__(self, storage_capacity: int, swap_capable: bool = False, spam_capacity: int = 1):
        self.storage_capacity = storage_capacity
        self.swap_capable = swap_capable
        self.spam_capacity = spam_capacity

    @property
    def n_actions(self):
        return 13 if self.swap_capable else 12

    @staticmethod
    def is_done(env_state):
        from rlions.Env import env_program_completable
        return env_program_completable(env_state)

    def make_train_env(self, max_op_count, smdp_gamma, step_reward, shaped_gamma=None, timeout='truncate',
                       shape_factor_gates=1.0, min_ion_count=None, max_ion_count=None, shape_dynamic_factor=False):
        from rlions.Chips import make_shaped_reward
        from rlions.EnvQ import make_q_env, q_env_make_random_reset

        if max_ion_count is None:
            max_ion_count = self.storage_capacity
        elif max_ion_count > self.storage_capacity:
            raise ValueError

        if min_ion_count is None:
            min_ion_count = 2

        return make_q_env(
            smdp_gamma=smdp_gamma,
            timeout=timeout,
            reset=q_env_make_random_reset(self.storage_capacity, max_op_count=max_op_count, min_ion_count=min_ion_count,
                                          max_ion_count=max_ion_count, spam_capacity=self.spam_capacity),
            shaped_gamma=shaped_gamma,
            step_reward=step_reward,
            shaped_reward_fn=make_shaped_reward(self.count_ions, shape_factor_gates, shape_dynamic_factor),
            swap_capable=self.swap_capable,
        )

    def make_eval_env(self):
        from rlions.EnvQ import make_q_env
        # Configure such that negative reward is exactly positive number of steps
        return make_q_env(
            reset=None,
            shaped_reward_fn=None,
            smdp_gamma=1.,
            timeout='truncate',
            max_mark_steps=1,
            step_reward=-1,
            swap_capable=self.swap_capable,
        )

    @staticmethod
    def count_ions(env_state):
        from rlions.EnvQ import q_env_ion_count
        return q_env_ion_count(env_state)

    def make_observation_adapter(self) -> Adapter:
        import jax
        import jax.numpy as jnp

        from rlions.Adapters import stack_to_cells
        from rlions.EnvQ import EnvStateQ

        n_cells = 2 + self.storage_capacity + self.spam_capacity
        max_qubits = self.storage_capacity
        n_spam_max = self.spam_capacity

        def _adapter(s: EnvStateQ):
            assert s.compute.buffer.shape == (2,)
            assert s.spam.buffer.shape == (n_spam_max,)
            assert s.storage.buffer.shape == (max_qubits,)

            return jnp.concatenate([
                stack_to_cells(s.compute, 2),
                stack_to_cells(s.spam, n_spam_max),
                s.storage.buffer,
            ], axis=-1)

        return Adapter(
            n_cells=n_cells,
            max_qubits=max_qubits,
            encode=_adapter,
        )

    def make_test_factory(self):
        return QTestFactory(
            storage_capacity=self.storage_capacity,
            spam_capacity=self.spam_capacity,
        )
