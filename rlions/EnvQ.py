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

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import chex

from rlions.Env import EnvImpl, try_mark
from rlions.Elements.Stack import Stack, stack_peek, stack_pop_conditional, stack_push_conditional, stack_init
from rlions.Elements.Drum import DrumState, DRUM_INVALID, drum_peek_front, drum_peek_back
from rlions.Elements.Drum import drum_pop_front, drum_pop_back, drum_set_front, drum_set_back
from rlions.Program import Program, make_random_program


@chex.dataclass(frozen=True)
class EnvStateQ:
    compute: Stack
    spam: Stack
    storage: DrumState
    program: Program
    steps: jax.Array
    last_action: jax.Array


def q_env_make_random_reset(storage_capacity, max_op_count=25, min_ion_count=10, max_ion_count=10, op_chance=.75,
                            spam_capacity: int = 1):
    assert max_ion_count <= storage_capacity

    def _stub(rng):
        count_rng, program_rng, start_rng = jax.random.split(rng, 3)
        ion_count = jax.random.randint(count_rng, (), min_ion_count, max_ion_count + 1)
        use_op_chance = op_chance * (ion_count / max_ion_count)

        program = make_random_program(max_op_count, ion_count, use_op_chance, rng)

        op_count = jnp.sum(~program.completed, axis=-1)

        buffer_one = jnp.where(
            jnp.arange(storage_capacity) < ion_count,
            jnp.arange(storage_capacity),
            jnp.full(storage_capacity, -1, dtype=jnp.int32),
        )

        buffer_two = buffer_one[::-1]

        start_one = jax.random.uniform(start_rng) < 0.5

        return EnvStateQ(
            spam=stack_init(spam_capacity),
            storage=DrumState(buffer=jax.lax.select(start_one, buffer_one, buffer_two)),
            compute=stack_init(2),
            program=program,
            steps=4 * op_count * ion_count,  # Twice the anticipated step count of a naive optimizer should be enough
            last_action=-1,
        )

    return _stub


def q_env_ion_count(state: EnvStateQ):
    return jnp.sum(state.storage.buffer >= 0, axis=-1) + state.compute.count + state.spam.count


def q_env_action_valid(state: EnvStateQ, action):
    dir_from, dir_to = jnp.divmod(action, 3)
    dir_to = dir_to + jnp.astype(dir_from <= dir_to, jnp.int32)

    # Parts definition is spam, storage_front, storage_back, compute (so similiar to X-Junction env)

    *_, spam_capacity = state.spam.buffer.shape

    source_is_drum = (dir_from == 1) | (dir_from == 2)
    dest_is_drum = (dir_to == 1) | (dir_to == 2)

    source_valid = jnp.array([
        state.spam.count > 0,
        (drum_peek_front(state.storage) != DRUM_INVALID) | dest_is_drum,  # Empty rotation allowed
        (drum_peek_back(state.storage) != DRUM_INVALID) | dest_is_drum,   # Empty rotation allowed
        state.compute.count > 0,
    ])[dir_from]

    dest_valid = jnp.array([
        state.spam.count < spam_capacity,
        # Opposite element must be empty only when inserting from outside drum
        (drum_peek_back(state.storage) == DRUM_INVALID) | source_is_drum,
        (drum_peek_front(state.storage) == DRUM_INVALID) | source_is_drum,
        state.compute.count < 2,
    ])[dir_to]

    return source_valid & dest_valid


def q_env_could_swap(state: EnvStateQ):
    return state.compute.count == 2


def make_q_env(smdp_gamma: float, reset, timeout='truncate', shaped_reward_fn=None, step_reward=-.1,
               shaped_gamma: Optional[float] = None, max_mark_steps: int = 10, swap_capable: bool = False):
    if shaped_gamma is None:
        shaped_gamma = smdp_gamma

    def valid_mask(state: EnvStateQ):
        base_mask = [q_env_action_valid(state, i) for i in range(12)]

        if swap_capable:
            base_mask = base_mask + [q_env_could_swap(state)]

        return jnp.stack(base_mask, axis=-1)

    def _execute_swap(state: EnvStateQ) -> EnvStateQ:
        # Swap has no effect if less than two ions present in compute
        new_compute_buffer = jax.lax.select(
            state.compute.count == 2,
            state.compute.buffer[..., ::-1],
            state.compute.buffer,
        )

        return state.replace(
            compute=state.compute.replace(buffer=new_compute_buffer),
        )

    def _handle_transfer_action(state: EnvStateQ, act) -> Tuple[EnvStateQ, jax.Array]:
        # The result of this function is only valid if 0 <= act < 12
        dir_from, dir_to = jnp.divmod(act, 3)
        dir_to = dir_to + jnp.astype(dir_from <= dir_to, jnp.int32)

        moved_ion = jnp.array([
            stack_peek(state.spam),
            drum_peek_front(state.storage),
            drum_peek_back(state.storage),
            stack_peek(state.compute),
        ])[dir_from]

        both_valid = q_env_action_valid(state, act)

        new_state = state.replace(last_action=act)

        new_state = new_state.replace(spam=stack_pop_conditional(new_state.spam, both_valid & (dir_from == 0))[0])
        new_state = new_state.replace(compute=stack_pop_conditional(new_state.compute, both_valid & (dir_from == 3))[0])
        # Drum rotation logic, note that both conditions per statement are met for either rotate command
        turn_drum_front_to_back = (dir_from == 1) | (dir_to == 2)
        turn_drum_back_to_front = (dir_from == 2) | (dir_to == 1)
        new_state = new_state.replace(
            storage=jax.tree.map(lambda a, b: jax.lax.select(both_valid & turn_drum_front_to_back, a, b),
                                 drum_pop_front(new_state.storage)[0], new_state.storage))
        new_state = new_state.replace(
            storage=jax.tree.map(lambda a, b: jax.lax.select(both_valid & turn_drum_back_to_front, a, b),
                                 drum_pop_back(new_state.storage)[0], new_state.storage))

        # Pushing
        new_state = new_state.replace(
            spam=stack_push_conditional(new_state.spam, moved_ion, both_valid & (dir_to == 0)))
        new_state = new_state.replace(
            storage=jax.tree.map(lambda a, b: jax.lax.select(both_valid & (dir_to == 1), a, b),
                                 drum_set_front(new_state.storage, moved_ion), new_state.storage))
        new_state = new_state.replace(
            storage=jax.tree.map(lambda a, b: jax.lax.select(both_valid & (dir_to == 2), a, b),
                                 drum_set_back(new_state.storage, moved_ion), new_state.storage))
        new_state = new_state.replace(
            compute=stack_push_conditional(new_state.compute, moved_ion, both_valid & (dir_to == 3)))

        # Deliberately using old state here
        is_fast_drum_rotate = both_valid & (
            ((dir_from == 1) & (drum_peek_front(state.storage) == DRUM_INVALID)) |
            ((dir_from == 2) & (drum_peek_back(state.storage) == DRUM_INVALID))
        )

        duration = jax.lax.select(is_fast_drum_rotate, 0.25, 1.0)

        return new_state, duration

    def step(state: EnvStateQ, act) -> Tuple[EnvStateQ, jax.Array, jax.Array, jax.Array, jax.Array]:
        from rlions.Env import try_mark
        from rlions.EnvUtil import update_generic_smdp

        new_state_transfer, duration_transfer = _handle_transfer_action(state, act)

        if swap_capable:
            is_swap_action = act >= 12

            new_state_swap = _execute_swap(state)

            new_state = jax.tree.map(lambda a, b: jax.lax.select(is_swap_action, a, b), new_state_swap, new_state_transfer)
            duration = jax.lax.select(is_swap_action, 1.0, duration_transfer)
        else:
            new_state = new_state_transfer
            duration = duration_transfer

        # Program and the rest
        # Handle program
        new_state = try_mark(new_state, max_mark_steps=max_mark_steps)

        # Detect when rotating the drum without moving ions through the junction

        return update_generic_smdp(state, new_state, duration, shaped_reward_fn, timeout, step_reward, smdp_gamma, shaped_gamma)

    return EnvImpl(
        reset=reset,
        step=step,
        try_mark=partial(try_mark, max_mark_steps=max_mark_steps),
        valid_mask=valid_mask,
    )
