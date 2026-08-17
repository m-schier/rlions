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

from typing import Optional

import jax
import jax.numpy as jnp
import chex

# Taken from JARL


@chex.dataclass(frozen=True)
class RolloutMetricState:
    collected_returns: jax.Array
    collected_lengths: jax.Array
    collected_durations: jax.Array
    collected_valid: jax.Array
    current_returns: jax.Array
    current_durations: jax.Array
    current_lengths: jax.Array
    indices: jax.Array


def rm_init(n_envs: int, window_size: int = 20) -> RolloutMetricState:
    return RolloutMetricState(
        collected_returns=jnp.zeros((n_envs, window_size)),
        collected_durations=jnp.zeros((n_envs, window_size)),
        collected_lengths=jnp.zeros((n_envs, window_size), dtype=jnp.int32),
        collected_valid=jnp.zeros((n_envs, window_size), dtype=bool),
        current_returns=jnp.zeros(n_envs),
        current_durations=jnp.zeros(n_envs),
        current_lengths=jnp.zeros(n_envs, dtype=jnp.int32),
        indices=jnp.zeros(n_envs, dtype=jnp.int32)
    )


def rm_update(state: RolloutMetricState, reward: jax.Array, done: jax.Array, valid: jax.Array, duration: Optional[jax.Array] = None) -> RolloutMetricState:
    if duration is None:
        duration = jnp.ones_like(reward)

    n_envs, window_size = state.collected_returns.shape

    chex.assert_shape(reward, (n_envs,))
    chex.assert_equal_shape([reward, duration, done, valid])

    new_current_returns = jax.lax.select(
        done,
        jnp.zeros_like(state.current_returns),
        state.current_returns + reward * valid
    )

    new_current_durations = jax.lax.select(
        done,
        jnp.zeros_like(state.current_durations),
        state.current_durations + duration * valid,
    )

    new_current_lengths = jax.lax.select(
        done,
        jnp.zeros_like(state.current_lengths),
        state.current_lengths + 1 * valid,
    )

    do_collect = done & valid

    new_indices = jax.lax.select(do_collect, (state.indices + 1) % window_size, state.indices)

    write_returns = jax.lax.select(
        do_collect,
        state.current_returns + reward,
        state.collected_returns[jnp.arange(n_envs), state.indices]
    )

    write_durations = jax.lax.select(
        do_collect,
        state.current_durations + duration,
        state.collected_durations[jnp.arange(n_envs), state.indices]
    )

    write_lengths = jax.lax.select(
        do_collect,
        state.current_lengths + 1,
        state.collected_lengths[jnp.arange(n_envs), state.indices]
    )

    write_valid = state.collected_valid[jnp.arange(n_envs), state.indices] | do_collect

    return state.replace(
        indices=new_indices,
        current_returns=new_current_returns,
        current_lengths=new_current_lengths,
        current_durations=new_current_durations,
        collected_returns=state.collected_returns.at[jnp.arange(n_envs), state.indices].set(write_returns),
        collected_lengths=state.collected_lengths.at[jnp.arange(n_envs), state.indices].set(write_lengths),
        collected_durations=state.collected_durations.at[jnp.arange(n_envs), state.indices].set(write_durations),
        collected_valid=state.collected_valid.at[jnp.arange(n_envs), state.indices].set(write_valid),
    )


def rm_finish_episodes(state: RolloutMetricState) -> RolloutMetricState:
    # TODO: If an episode hasn't received any step this will incorrectly collect it with return of 0, but the use-case
    # TODO: for this function where this would often happen is rare
    n_envs, window_size = state.collected_returns.shape

    new_indices = (state.indices + 1) % window_size
    new_current_returns = jnp.zeros_like(state.current_returns)
    new_current_durations = jnp.zeros_like(state.current_durations)
    new_current_lengths = jnp.zeros_like(state.current_lengths)
    new_collected_returns = state.collected_returns.at[jnp.arange(n_envs), state.indices].set(state.current_returns)
    new_collected_durations = state.collected_durations.at[jnp.arange(n_envs), state.indices].set(state.current_durations)
    new_collected_lengths = state.collected_lengths.at[jnp.arange(n_envs), state.indices].set(state.current_lengths)

    return state.replace(
        indices=new_indices,
        current_returns=new_current_returns,
        current_durations=new_current_durations,
        current_lengths=new_current_lengths,
        collected_returns=new_collected_returns,
        collected_durations=new_collected_durations,
        collected_lengths=new_collected_lengths,
    )


def rm_get_episodic_return(state: RolloutMetricState) -> jax.Array:
    return jax.lax.cond(
        jnp.any(state.collected_valid),
        lambda: jnp.mean(state.collected_returns, where=state.collected_valid),
        lambda: 0.
    )


def rm_get_episode_duration(state: RolloutMetricState) -> jax.Array:
    return jax.lax.cond(
        jnp.any(state.collected_valid),
        lambda: jnp.mean(state.collected_durations, where=state.collected_valid),
        lambda: 0.
    )


def rm_get_episode_length(state: RolloutMetricState) -> jax.Array:
    return jax.lax.cond(
        jnp.any(state.collected_valid),
        lambda: jnp.mean(state.collected_lengths, where=state.collected_valid),
        lambda: 0.
    )
