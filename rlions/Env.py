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

import sys
from functools import partial
from typing import Tuple, Optional, Literal

import chex
from chex import dataclass
import jax
import jax.numpy as jnp
from .Program import Program, can_mark, maybe_mark, is_fully_processed
from rlions.Elements.Stack import Stack, stack_peek, stack_pop_conditional, stack_push_conditional, stack_init


@dataclass(frozen=True)
class EnvState:
    spam: Stack
    left: Stack
    right: Stack
    compute: Stack
    program: Program
    """
    The remaining steps before terminating/truncating
    """
    steps: jax.Array
    last_action: jax.Array


def env_pretty_print(state: EnvState, file=sys.stdout):
    def print_stack(s):
        for i in range(s.count):
            print(str(s.buffer[i]).ljust(3), end="", file=file)
        print(file=file)

    print("Spam        ", end="", file=file)
    print_stack(state.spam)
    print("Storage     ", end="", file=file)
    print_stack(state.left)
    print("TempStorage ", end="", file=file)
    print_stack(state.right)
    print("Compute     ", end="", file=file)
    print_stack(state.compute)

    # Print plan
    for (i, j), done in zip(state.program.operations, state.program.completed):
        if not done:
            print(f"({i}, {j}) ", end="", file=file)

    print(file=file)
    print(file=file)


def env_ion_count(state: EnvState):
    return state.left.count + state.right.count + state.compute.count + state.spam.count


@dataclass(frozen=True)
class EnvImpl:
    """
    Generic interface of all environments
    """

    """
    Function to reset to a training state, may be None for evaluation
    """
    reset: None

    """
    Function transforming an environment state, marking immediately completable gates, e.g. qubits in compute
    """
    try_mark: None

    """
    Environment step function
    """
    step: None

    """
    Function calculating the valid operation mask per state
    """
    valid_mask: None


def can_move_to_spam(state: EnvState, q):
    *_, spam_capacity = state.spam.buffer.shape
    return state.spam.count < spam_capacity


def _can_move_to_compute(state: EnvState, q, allow_bad_compute: bool = None):
    # Force explicitly giving the argument
    assert allow_bad_compute is True or allow_bad_compute is False

    if allow_bad_compute:
        return state.compute.count < 2
    else:
        empty = state.compute.count == 0
        valid_combination = (state.compute.count == 1) & can_mark(state.program, stack_peek(state.compute), q)
        return empty | valid_combination


def make_valid_mask(state: EnvState, allow_bad_compute: bool = None, have_swap: bool = None):
    act = jnp.arange(12)

    dir_from, dir_to = jnp.divmod(act, 3)
    dir_to = dir_to + jnp.astype(dir_from <= dir_to, jnp.int32)

    source_valid = jnp.array([
        state.spam.count > 0,
        state.left.count > 0,
        state.right.count > 0,
        state.compute.count > 0,
    ])[dir_from]

    chex.assert_shape(source_valid, (12,))

    moved_ion = jnp.array([
        stack_peek(state.spam),
        stack_peek(state.left),
        stack_peek(state.right),
        stack_peek(state.compute),
    ])[dir_from]

    chex.assert_shape(moved_ion, (12,))

    dest_valid = jnp.array([
        jax.vmap(partial(can_move_to_spam, state))(moved_ion),
        jnp.ones(12, dtype=jnp.bool),
        jnp.ones(12, dtype=jnp.bool),
        jax.vmap(partial(_can_move_to_compute, state, allow_bad_compute=allow_bad_compute))(moved_ion),
    ])

    dest_valid = dest_valid[dir_to, jnp.arange(12)]

    result = source_valid & dest_valid

    if have_swap:
        result = jnp.concatenate([result, jnp.asarray([state.compute.count == 2])], axis=-1)

    return result


def env_program_completable(state: EnvState, max_mark_steps: int = 3) -> bool:
    try_mark = state.compute.count == 2

    for _ in range(max_mark_steps):  # 3 is arbitrary but this number cannot be dynamic, obviously
        marked_program = maybe_mark(state.program, state.compute.buffer[0], state.compute.buffer[1])
        state = state.replace(
            program=jax.tree.map(lambda a, b: jax.lax.select(try_mark, a, b), marked_program, state.program))

    return is_fully_processed(state.program)


def make_example_reset():
    max_ions = 4

    def _stub(_rng):
        from .Program import make_program

        init_left = stack_init(max_ions)
        init_left = init_left.replace(count=max_ions, buffer=max_ions - 1 - jnp.arange(max_ions))

        state = EnvState(
            spam=stack_init(1),
            left=init_left,
            right=stack_init(max_ions),
            compute=stack_init(2),
            program=make_program(jnp.array([(0, 2), (1, 3), (2, 3), (0, 1), (1, 3), (0, 3), (1, 3), (0, 2)])),
            steps=2 * 5 * max_ions,  # Should be enough time to complete
            last_action=-1,
        )

        return state

    return _stub


def make_random_reset(max_op_count=25, min_ion_count=10, max_ion_count=10, op_chance=.75, spam_capacity: int = 1,
                      random_start_stack: bool = True):
    from .Program import make_random_program

    def _stub(rng):
        count_rng, program_rng, start_rng = jax.random.split(rng, 3)
        # TODO: Use different ions rather than lowest number
        ion_count = jax.random.randint(count_rng, (), min_ion_count, max_ion_count + 1)
        use_op_chance = op_chance * (ion_count / max_ion_count)

        program = make_random_program(max_op_count, ion_count, use_op_chance, rng)

        op_count = jnp.sum(~program.completed, axis=-1)

        init_start_stack = stack_init(max_ion_count)
        init_buffer = jnp.arange(max_ion_count) * (jnp.arange(max_ion_count) < ion_count)
        init_start_stack = init_start_stack.replace(count=ion_count, buffer=init_buffer)
        init_nonstart_stack = stack_init(max_ion_count)

        if random_start_stack:
            start_left = jax.random.uniform(start_rng) < 0.5
            init_left = jax.tree.map(lambda ss, nss: jax.lax.select(start_left, ss, nss), init_start_stack, init_nonstart_stack)
            init_right = jax.tree.map(lambda ss, nss: jax.lax.select(~start_left, ss, nss), init_start_stack, init_nonstart_stack)
        else:
            init_left = init_start_stack
            init_right = init_nonstart_stack

        return EnvState(
            spam=stack_init(spam_capacity),
            left=init_left,
            right=init_right,
            compute=stack_init(2),
            program=program,
            steps=4 * op_count * ion_count,  # Twice the anticipated step count of a naive optimizer should be enough
            last_action=-1,
        )

    return _stub


def try_mark(state: EnvState, max_mark_steps: int = 5) -> EnvState:
    """
    Mark any completable circuit gates for the present chip state. Return new simulator state, may be unchanged.
    :param state: Simulator state to check
    :return: New simulator state with possibly marked gates
    """
    # This is a bit hacky, but we can have multiple of the same binary operations in the plan and it may be legal
    # to execute them at once, e.g.: (0, 2), (1, 3), (0, 2) can resolve both (0, 2) at once.
    # This does not require moving ions, but we can only check a limited amount, so this is hardcoded for now.
    try_mark = state.compute.count == 2

    for _ in range(max_mark_steps):
        marked_program = maybe_mark(state.program, state.compute.buffer[0], state.compute.buffer[1])
        state = state.replace(program=jax.tree.map(lambda a, b: jax.lax.select(try_mark, a, b), marked_program, state.program))

    return state


def update_generic(old_state, new_state, shaped_reward_fn, timeout, step_reward, goal_reward, gamma):
    new_state = new_state.replace(steps=new_state.steps - 1)

    terminated = is_fully_processed(new_state.program)
    truncated = jnp.zeros_like(terminated)

    if callable(step_reward):
        use_step_reward = step_reward(new_state)
    else:
        use_step_reward = step_reward

    use_goal_reward = use_step_reward if goal_reward is None else goal_reward

    reward = jnp.where(terminated, use_goal_reward, use_step_reward)

    is_timeout = new_state.steps <= 0

    if timeout == 'truncate':
        truncated = truncated | is_timeout
    elif timeout == 'terminate_penalty':
        assert gamma is not None

        terminated = terminated | is_timeout

        # A pessimistic estimate of the remaining operations
        expected_rem_steps = jnp.sum(~new_state.program.completed, axis=-1) * env_ion_count(new_state)

        # The expected return under the geometric series
        expected_return = use_step_reward * (1 - (gamma ** expected_rem_steps)) / (1 - gamma)

        reward = reward - is_timeout * expected_return
    else:
        raise ValueError(f"{timeout = }")

    if shaped_reward_fn is not None:
        old_shaped_reward = shaped_reward_fn(old_state)
        new_shaped_reward = shaped_reward_fn(new_state)

        reward = reward + gamma * new_shaped_reward * (~terminated) - old_shaped_reward

    return new_state, reward, terminated, truncated


def make_env(smdp_gamma: float, reset=None, shaped_reward_fn=None, shaped_gamma: Optional[float] = None,
             timeout: Literal['truncate', 'terminate_penalty'] = 'truncate',
             max_mark_steps: int = 3, step_reward: float = -.1, allow_bad_compute: bool = False) -> EnvImpl:
    """
    Makes a QVLS-Q1 chip environment
    :param reset: The reset function, which initializes the environment. Default is `make_example_reset()`
    :param shaped_reward_fn: The function for shaped reward, if given must specify gamma.
    :param smdp_gamma: Discount factor of the SMDP.
    :param shaped_gamma: Discount factor to use for shaped reward
    :param timeout: Handling on timeout. Either 'truncate', which truncates, or 'terminate_penalty', which terminates
    with reward set to the expected return if a naive agent where to take over at this step, which requires gamma to
    be specified.
    :param max_mark_steps: The maximum number of gates which can be marked as completed after one movement. For programs
    with a lot of duplicates gates, setting this too low may require unnecessary movements.
    :param step_reward: Constant reward per step.
    :param allow_bad_compute: If True, allow moving two ions into the compute zone even if no gate can be applied.
    Otherwise, still allow one ion in compute even if this has no possible gates.
    :return: Implementation of the environment
    """
    if shaped_gamma is None:
        shaped_gamma = smdp_gamma

    if reset is None:
        reset = make_example_reset()

    def step(state: EnvState, act) -> Tuple[EnvState, jax.Array, jax.Array, jax.Array]:
        """
        Action must be in {0 ... 11}. Directions are (0 input, 1 left, 2 right, 3 processing), and the action is
        constructed as 3 * dir_from + dir_to
        :return: (new_state, reward, terminated)
        """

        from rlions.EnvUtil import update_generic_smdp

        old_state = state

        dir_from, dir_to = jnp.divmod(act, 3)
        dir_to = dir_to + jnp.astype(dir_from <= dir_to, jnp.int32)

        # print(f"{dir_from = }, {dir_to = }")

        source_valid = jnp.array([
            state.spam.count > 0,
            state.left.count > 0,
            state.right.count > 0,
            state.compute.count > 0,
        ])[dir_from]

        moved_ion = jnp.array([
            stack_peek(state.spam),
            stack_peek(state.left),
            stack_peek(state.right),
            stack_peek(state.compute),
        ])[dir_from]

        dest_valid = jnp.array([
            can_move_to_spam(state, moved_ion),
            True,
            True,
            _can_move_to_compute(state, moved_ion, allow_bad_compute=allow_bad_compute),
        ])[dir_to]

        # print(f"{source_valid = }, {dest_valid = }")

        both_valid = source_valid & dest_valid

        # Pop source
        new_state = state.replace(last_action=act)
        new_state = new_state.replace(spam=stack_pop_conditional(new_state.spam, both_valid & (dir_from == 0))[0])
        new_state = new_state.replace(left=stack_pop_conditional(new_state.left, both_valid & (dir_from == 1))[0])
        new_state = new_state.replace(right=stack_pop_conditional(new_state.right, both_valid & (dir_from == 2))[0])
        new_state = new_state.replace(compute=stack_pop_conditional(new_state.compute, both_valid & (dir_from == 3))[0])

        # Push target
        new_state = new_state.replace(spam=stack_push_conditional(new_state.spam, moved_ion, both_valid & (dir_to == 0)))
        new_state = new_state.replace(left=stack_push_conditional(new_state.left, moved_ion, both_valid & (dir_to == 1)))
        new_state = new_state.replace(right=stack_push_conditional(new_state.right, moved_ion, both_valid & (dir_to == 2)))
        new_state = new_state.replace(compute=stack_push_conditional(new_state.compute, moved_ion, both_valid & (dir_to == 3)))

        # Maybe complete gates with qubits in compute zone
        new_state = try_mark(new_state, max_mark_steps=max_mark_steps)

        return update_generic_smdp(old_state, new_state, 1.0, shaped_reward_fn, timeout, step_reward, smdp_gamma, shaped_gamma)

    return EnvImpl(
        reset=reset,
        step=step,
        try_mark=try_mark,
        valid_mask=partial(make_valid_mask, allow_bad_compute=allow_bad_compute),
    )
