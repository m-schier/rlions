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

from rlions.Program import Program
from rlions.Elements.Drum import DrumState
from rlions.Elements.Stack import stack_init, stack_push, Stack
from rlions.Elements.DoubleEndedBuffer import DoubleEndedBuffer, de_buffer_init, de_buffer_push_back, de_buffer_push_front


# Some JIT
stack_init = jax.jit(stack_init, static_argnums=(0, 1))
stack_push = jax.jit(stack_push)
de_buffer_init = jax.jit(de_buffer_init, static_argnums=(0,))
de_buffer_push_front = jax.jit(de_buffer_push_front)
de_buffer_push_back = jax.jit(de_buffer_push_back)


def deduplicate_program_object(prog_obj):
    # Removes duplicate gates from a list of tuples (or lists) as found in the JSON problem descriptions
    result = []

    last = {}

    for a, b in prog_obj:
        if last.get(a) == b and last.get(b) == a:
            continue

        result.append([a, b])
        last[a] = b
        last[b] = a

    return result


def program_from_json(json_obj, max_program_length) -> Program:
    return Program(
        operations=jnp.zeros((max_program_length, 2), jnp.int32).at[:len(json_obj['circuit'])].set(json_obj['circuit']),
        completed=jnp.ones(max_program_length, jnp.bool).at[:len(json_obj['circuit'])].set(False),
    )


def stack_from_json(arr, cap) -> Stack:
    s = stack_init(cap)

    for el in arr:
        s = stack_push(s, el)

    return s


def compatible_drum_from_json(json_chip_state, capacity) -> DrumState:
    front = json_chip_state['storage']
    back = get_tmp_storage(json_chip_state)

    total_length = len(front) + len(back)
    assert total_length <= capacity

    buffer = jnp.full(capacity, -1, dtype=jnp.int32).at[:len(front)].set(front).at[len(front):total_length].set(back)

    return DrumState(buffer=buffer)


def compatible_de_buffer_from_json(json_chip_state, capacity) -> DoubleEndedBuffer:
    front = json_chip_state['storage']
    back = get_tmp_storage(json_chip_state)

    assert len(front) + len(back) <= capacity

    de = de_buffer_init(capacity)

    for el in front:
        de = de_buffer_push_front(de, el)

    for el in back:
        de = de_buffer_push_back(de, el)

    return de


def get_tmp_storage(obj):
    # Was for some reason renamed at a point in time
    try:
        return obj['tempstorage']
    except KeyError:
        return obj['temp_storage']


def load_tests(path, chip, batched=True):
    """
    Load company test set
    :param path: Path to JSON file to load or List of multiple paths.
    :param batched: If true return batched tensors, otherwise split into list of states
    :return: Loaded tests
    """
    import json
    import jax
    import jax.numpy as jnp

    if not isinstance(path, list):
        path = [path]

    obj = []

    for p in path:
        with open(p, 'r') as fp:
            o = json.load(fp)

        if isinstance(o, list):
            obj += o
        elif isinstance(o, dict):
            obj.append(o)
        else:
            raise TypeError

    # Deduplicate all programs
    for o in obj:
        o['circuit'] = deduplicate_program_object(o['circuit'])

    factory = chip.make_test_factory()
    factory.preflight(obj)

    states = [factory.produce(el) for el in obj]

    mark_fn = jax.jit(chip.make_eval_env().try_mark)
    states = [mark_fn(s) for s in states]

    if not batched:
        return states
    else:
        return jax.tree.map(lambda *x: jnp.stack(x), *states)
