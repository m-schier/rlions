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

from rlions.EnvQ import EnvStateQ, make_q_env
from rlions.Elements.Drum import drum_init, drum_set_front, drum_set_back, DrumState, DRUM_INVALID
from rlions.Elements.Stack import stack_init, stack_push, Stack
from rlions.Program import make_program


def assert_stacks_close(actual: Stack, expected: Stack):
    assert actual.count == expected.count
    assert jnp.allclose(actual.buffer[:actual.count], expected.buffer[:expected.count])


def test_illegal_insert_front_blocked():
    state = EnvStateQ(
        spam=stack_init(1),
        compute=stack_push(stack_init(2), 0),
        storage=drum_set_back(drum_init(5), 1),
        program=make_program(jnp.asarray([[0, 1]])),
        steps=1,
        last_action=-1,
    )

    env = make_q_env(1.0, reset=None)
    actual_mask = env.valid_mask(state)
    expected_mask = jnp.asarray([
        False, False, False,
        False, True, False,
        True, True, True,
        True, False, True,
    ])

    assert jnp.allclose(actual_mask, expected_mask)


def test_illegal_insert_back_blocked():
    state = EnvStateQ(
        spam=stack_push(stack_init(1), 0),
        compute=stack_init(2),
        storage=drum_set_front(drum_init(5), 1),
        program=make_program(jnp.asarray([[0, 1]])),
        steps=1,
        last_action=-1,
    )

    env = make_q_env(1.0, reset=None)
    actual_mask = env.valid_mask(state)
    expected_mask = jnp.asarray([
        True, False, True,
        False, True, True,
        False, True, False,
        False, False, False,
    ])

    assert jnp.allclose(actual_mask, expected_mask)


def test_illegal_insert_both_blocked():
    state = EnvStateQ(
        spam=stack_push(stack_init(1), 0),
        compute=stack_push(stack_init(2), 3),
        storage=drum_set_back(drum_set_front(drum_init(5), 1), 2),
        program=make_program(jnp.asarray([[0, 1]])),
        steps=1,
        last_action=-1,
    )

    env = make_q_env(1.0, reset=None)
    actual_mask = env.valid_mask(state)
    expected_mask = jnp.asarray([
        False, False, True,
        False, True, True,
        False, True, True,
        False, False, False,
    ])

    assert jnp.allclose(actual_mask, expected_mask)


def test_compute_to_spam():
    state = EnvStateQ(
        spam=stack_init(1),
        compute=stack_push(stack_init(2), 0),
        storage=drum_set_back(drum_set_front(drum_init(5), 1), 2),
        program=make_program(jnp.asarray([[0, 1]])),
        steps=1,
        last_action=-1,
    )

    env = make_q_env(1.0, reset=None)

    new_state, reward, terminated, truncated, duration = env.step(state, 9)

    expected_state = EnvStateQ(
        spam=stack_push(stack_init(1), 0),
        compute=stack_init(2),
        storage=drum_set_back(drum_set_front(drum_init(5), 1), 2),
        program=make_program(jnp.asarray([[0, 1]])),
        steps=1,
        last_action=-1,
    )

    assert_stacks_close(new_state.compute, expected_state.compute)
    assert_stacks_close(new_state.spam, expected_state.spam)
    assert jnp.allclose(new_state.storage.buffer, expected_state.storage.buffer)


def test_storage_to_tmp_storage():
    state = EnvStateQ(
        spam=stack_init(1),
        compute=stack_push(stack_init(2), 0),
        storage=drum_set_back(drum_set_front(drum_init(5), 1), 2),
        program=make_program(jnp.asarray([[0, 1]])),
        steps=1,
        last_action=-1,
    )

    env = make_q_env(1.0, reset=None)

    new_state, reward, terminated, truncated, duration = env.step(state, 4)

    expected_state = EnvStateQ(
        spam=stack_init(1),
        compute=stack_push(stack_init(2), 0),
        storage=DrumState(buffer=jnp.asarray([DRUM_INVALID, DRUM_INVALID, DRUM_INVALID, 2, 1])),
        program=make_program(jnp.asarray([[0, 1]])),
        steps=1,
        last_action=-1,
    )

    assert_stacks_close(new_state.compute, expected_state.compute)
    assert_stacks_close(new_state.spam, expected_state.spam)
    assert jnp.allclose(new_state.storage.buffer, expected_state.storage.buffer)


def test_tmp_storage_to_storage():
    state = EnvStateQ(
        spam=stack_init(1),
        compute=stack_push(stack_init(2), 0),
        storage=drum_set_back(drum_set_front(drum_init(5), 1), 2),
        program=make_program(jnp.asarray([[0, 1]])),
        steps=1,
        last_action=-1,
    )

    env = make_q_env(1.0, reset=None)

    new_state, reward, terminated, truncated, duration = env.step(state, 7)

    expected_state = EnvStateQ(
        spam=stack_init(1),
        compute=stack_push(stack_init(2), 0),
        storage=DrumState(buffer=jnp.asarray([2, 1, DRUM_INVALID, DRUM_INVALID, DRUM_INVALID])),
        program=make_program(jnp.asarray([[0, 1]])),
        steps=1,
        last_action=-1,
    )

    assert_stacks_close(new_state.compute, expected_state.compute)
    assert_stacks_close(new_state.spam, expected_state.spam)
    assert jnp.allclose(new_state.storage.buffer, expected_state.storage.buffer)
