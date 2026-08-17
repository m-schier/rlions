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
import numpy as np
from functools import partial
from typing import Optional

from rlions.Chips import AbstractChip


@partial(jax.jit, static_argnums=(0, 1))
def run_search(policy, env_step, val_init, use_max_len):
    def _loop_body(_val):
        _rng, _s_vec, _step_durations, _terminated, _truncated, _act, _steps = _val
        _rng, step_rng = jax.random.split(_rng)
        parallelism, max_len = _act.shape

        vectorized_rng = jax.random.split(step_rng, parallelism)

        a = jax.vmap(policy)(_s_vec, vectorized_rng)

        _done = _truncated | _terminated
        _act = _act.at[jnp.arange(parallelism), _steps].set(jnp.where(_done, _act[jnp.arange(parallelism), _steps], a))

        _s_vec, _, terminated, truncated, duration = jax.vmap(env_step)(_s_vec, a)
        # Deliberately using old done here as the duration should be recorded if freshly done
        _step_durations = _step_durations.at[jnp.arange(parallelism), jnp.where(_done, max_len, _steps)].set(duration, mode='drop')

        # Increment steps after using for indexing
        _steps = _steps + jnp.astype(~_done, jnp.int32)

        # Terminated has precedence
        _terminated = _terminated | (~_truncated & terminated)
        _truncated = _truncated | (~_terminated & truncated)

        return _rng, _s_vec, _step_durations, _terminated, _truncated, _act, _steps

    def _loop_cond(_val):
        _, _, _step_durations, _terminated, _truncated, _, _steps = _val

        _, n_max = _step_durations.shape

        total_durations = jnp.sum(_step_durations * (jnp.arange(n_max) < _steps[..., None]), axis=-1)
        done = _terminated | _truncated

        min_done = jnp.min(total_durations * done)
        min_not_done = jnp.min(total_durations * (~done))

        # Terminate if
        #  1. Any trajectory truncated (max steps reached for problem)
        #  2. Any step number has reached use_max_len (rollout limited)
        #  3. All trajectories terminated
        #  4. Any episodes have terminated and the ratio of lowest terminated to lowest non-terminated is lower.
        return ~(
                jnp.any(_truncated) |
                jnp.any(_steps >= use_max_len) |
                jnp.all(_terminated) |
                (jnp.any(_terminated) & (min_done < min_not_done))
        )

    return jax.lax.while_loop(_loop_cond, _loop_body, val_init)


@partial(jax.jit, static_argnums=(1,))
def _preprocess_state(env_state, max_qubits):
    from rlions.Program import mark_elidable_fast
    return env_state.replace(program=mark_elidable_fast(env_state.program, max_qubits))


@partial(jax.jit, static_argnums=(0,))
def _parallelize(parallelism, env_state):
    return jax.tree.map(lambda x: jnp.repeat(jnp.asarray(x)[None], parallelism, 0), env_state)


@chex.dataclass
class FoundTrajectory:
    steps: int
    duration: float
    actions: np.ndarray

    def __repr__(self):
        return f"FoundTrajectory(steps={self.steps}, duration={self.duration})"


@chex.dataclass
class OptimizationResult:
    solved: bool
    best: Optional[FoundTrajectory]

    def __repr__(self):
        return f"OptimizationResult(solved={self.solved}, best={self.best})"


def _make_trajectory_from_search(steps, durations, terminated, trajectories) -> Optional[FoundTrajectory]:
    parallelism, max_len = durations.shape
    costs = jnp.sum(durations * (jnp.arange(max_len) < steps[..., None]), axis=-1)
    chex.assert_equal_shape([steps, costs, terminated])
    arg_min = jnp.argmin(jnp.where(terminated, costs, jnp.inf))

    if not terminated[arg_min]:
        return None

    return FoundTrajectory(
        steps=steps[arg_min].item(),
        duration=costs[arg_min].item(),
        actions=np.asarray(trajectories[arg_min, :steps[arg_min]]),
    )


class AOTSearch:
    def __init__(self, policy, env_step, parallelism, max_len, chip: AbstractChip):
        self.policy = policy
        self.env_step = env_step
        self.parallelism = parallelism
        self.max_len = max_len
        self.max_qubits = chip.make_observation_adapter().max_qubits
        self.rng_state = None
        self._chip_is_done = jax.jit(chip.is_done)

    def parallelize(self, state):
        return _parallelize(self.parallelism, state)

    def raw_search(self, s_vec):
        if self.rng_state is None:
            self.rng_state = jax.random.key(0)

        val_init = (
            self.rng_state,  # RNG
            s_vec,  # States
            # jnp.zeros(self.parallelism),  # Returns
            jnp.zeros((self.parallelism, self.max_len)),  # Step durations
            jnp.zeros(self.parallelism, dtype=jnp.bool),  # Terminated
            jnp.zeros(self.parallelism, dtype=jnp.bool),  # Truncated
            jnp.zeros((self.parallelism, self.max_len), dtype=jnp.int32),  # Actions
            jnp.zeros(self.parallelism, dtype=jnp.int32)  # Steps
        )
        self.rng_state, *result = run_search(self.policy, self.env_step, val_init, self.max_len)
        return result

    def do_optimization(self, env_state, time_budget=1.) -> OptimizationResult:
        # In the future, this should be an AOT compilation which automatically compiles all relevant functions
        # for a dataset, currently it is not
        from time import time

        result = OptimizationResult(
            solved=False,
            best=None,
        )

        t_start = now = time()

        # Optimize and vectorize
        env_state = _preprocess_state(env_state, self.max_qubits)

        if self._chip_is_done(env_state):
            return OptimizationResult(
                solved=True,
                best=FoundTrajectory(steps=0, duration=0., actions=np.empty((0,), dtype=np.int32)),
            )

        s_vec = self.parallelize(env_state)

        # The condition is written this way such that time_budget=0.0 does not fail immediately
        while t_start + time_budget >= now:
            print('.', end='', flush=True)
            _, durations, terminated, truncated, trajectories, steps = self.raw_search(s_vec)

            best = _make_trajectory_from_search(steps, durations, terminated, trajectories)

            if best is not None:
                result.solved = True

                if result.best is None or result.best.duration > best.duration:
                    result.best = best

            now = time()

        return result
