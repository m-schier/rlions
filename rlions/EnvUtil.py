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

from rlions.Program import is_fully_processed


def update_generic_smdp(old_state, new_state, duration, shaped_reward_fn, timeout, step_reward, smdp_gamma, shaped_gamma=None):
    """
    :param old_state: Previous state
    :param new_state: New state
    :param duration: The duration this step took.
    :param shaped_reward_fn: Shaped reward function
    :param timeout: Timeout handling
    :param step_reward: The step reward *rate* or a function returning the step reward *rate*.
    :param smdp_gamma: Discount factor of the SMDP for a  duration of tau = 1, so discount rate beta = -ln gamma.
    :param shaped_gamma: Discount factor for the potential shaped reward.
    :return:
    """
    new_state = new_state.replace(steps=new_state.steps - 1)

    terminated = is_fully_processed(new_state.program)
    truncated = jnp.zeros_like(terminated)

    if callable(step_reward):
        use_step_reward = step_reward(new_state)
    else:
        use_step_reward = step_reward

    beta = -jnp.log(smdp_gamma)
    reward = use_step_reward * jnp.where(
        smdp_gamma < 1.,
        (1 - jnp.exp(-beta * duration)) / beta,
        duration,
    )

    is_timeout = new_state.steps <= 0

    if timeout == 'truncate':
        truncated = truncated | is_timeout
    else:
        raise ValueError(f"{timeout = }")

    if shaped_reward_fn is not None:
        old_shaped_reward = shaped_reward_fn(old_state)
        new_shaped_reward = shaped_reward_fn(new_state)

        if shaped_gamma is None:
            shaped_gamma = smdp_gamma

        reward = reward + (shaped_gamma ** duration) * new_shaped_reward * (~terminated) - old_shaped_reward

    return new_state, reward, terminated, truncated, duration
