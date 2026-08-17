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

import os.path
import sys

import jax
import jax.numpy as jnp
from time import time
from rlions.Env import EnvState


def make_ppo_guided(path, env, chip, temperature=1.0):
    from rlions.Util import compatible_load
    from rlions_train import make_configured_ppo

    data = compatible_load(path)

    ppo = make_configured_ppo(data['args'], env, chip)

    @jax.jit
    def policy(s: EnvState, rng):
        return ppo.act_stochastic(data['state'], rng, s, temperature=temperature)

    return policy


@jax.jit
def check_no_outstanding_gates(s):
    from rlions.Program import can_mark
    return (s.compute.count < 2) | ~can_mark(s.program, s.compute.buffer[0], s.compute.buffer[1])


def make_qubit_check(adapter):
    @jax.jit
    def _stub(state):
        cells = adapter.encode(state)
        return jnp.zeros(adapter.max_qubits + 1, dtype=jnp.int32).at[cells + 1].add(1)

    return _stub


def main():
    from rlions.StandardDatasets import load_qvs, load_mqt
    from rlions.Util import compatible_load
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--path', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--force', default=None)
    args = parser.parse_args()

    all_qvs = [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]

    path, name = args.path, args.name

    if args.data == 'qv':
        use_qvs = all_qvs[::-1]
    elif args.data.startswith('qv'):
        # Only used for fast evaluation of a specific qv
        use_qvs = [q for q in all_qvs if q == int(args.data[len('qv'):])]
        assert use_qvs
    elif args.data == 'mqt':
        use_qvs = None
    else:
        raise ValueError(f"{args.data = }")

    budgets = {
        18: 2.0,
        22: 4.5,
        26: 8.5,
        30: 14,
        34: 20.5,
        38: 32.0,
        42: 44.0,
        46: 61.0,
        50: 78,
    }

    # Pre-load args for env_kwargs
    agent_args = compatible_load(path)['args']

    if hasattr(agent_args, 'chip'):
        from rlions.Chips import get_chip_by_name
        policy_chip = get_chip_by_name(agent_args.chip)
    else:
        # Legacy
        from rlions.ChipImplementations.QVLSXChip import QVLSXJunctionChip
        policy_chip = QVLSXJunctionChip(storage_capacity=agent_args.ion_count, allow_bad_compute=agent_args.allow_bad_compute)
        agent_args.chip = f"qvls_x_{agent_args.ion_count}"

    if args.force is not None:
        # This should really only be used to ensure that policies run on a newer chip which is backward-compatible
        from rlions.Chips import get_chip_by_name
        print(f"WARNING: Forcing simulation inside an environment of chip type {args.force} which is different from the policy chip type {agent_args.chip}", file=sys.stderr)
        env_chip = get_chip_by_name(args.force)
    else:
        env_chip = policy_chip

    policy = make_ppo_guided(path, policy_chip.make_eval_env(), policy_chip)

    if use_qvs is None:
        # MQT
        ds = load_mqt(env_chip)
        do_work(policy, name, ds, 1.0, env_chip)
    else:
        for qv in use_qvs:
            ds = load_qvs(qv, env_chip)
            budget = budgets.get(qv, 1.0)
            do_work(policy, name, ds, budget, env_chip)


def get_parallelism_max_len(chip, initial_states):
    # Set a reasonable max step size based on expected step count
    max_len = 0

    for c in initial_states:
        max_len = max(chip.count_ions(c) * jnp.sum(~c.program.completed) * 2, max_len)

    # Set parallelism such that max env steps in cycle is 1024 * 1024
    parallelism = max(64, int(1024 * 1024 / max_len))

    return parallelism, max_len


def do_work(policy, name, ds, time_budget, chip):
    import pandas as pd
    from rlions.AOTSearch import AOTSearch
    from rlions.Env import env_program_completable
    from functools import partial

    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp', 'searches')
    print(f"{output_folder = }", file=sys.stderr)
    os.makedirs(output_folder, exist_ok=True)

    env_program_completable = jax.jit(partial(env_program_completable, max_mark_steps=20))

    env = ds.env
    configurations = ds.initial_states
    problem_names = ds.problem_names
    test_name = ds.name

    env_step = jax.jit(env.step)

    parallelism, max_len = get_parallelism_max_len(chip, configurations)

    print(f"{max_len = }", file=sys.stderr)
    print(f"{parallelism = }", file=sys.stderr)

    search = AOTSearch(policy, ds.env.step, parallelism, max_len, chip)

    # Warm up compiler
    print("Warming up JIT...", file=sys.stderr, end='', flush=True)
    # For some reason if we do this once the first output is slower
    for _ in range(3):
        search.do_optimization(configurations[0], time_budget=1.)
    print("Done", file=sys.stderr, flush=True)

    records = []
    df = None

    qubit_checker = make_qubit_check(chip.make_observation_adapter())

    # Benchmark
    for i, (c, p_name) in enumerate(zip(configurations, problem_names)):
        ion_count = chip.count_ions(c)
        op_count = jnp.sum(~c.program.completed)

        t_start = time()
        opt_result = search.do_optimization(c, time_budget)
        t_stop = time()

        if not opt_result.solved:
            record = (name, i, p_name, ion_count, op_count, False, None, t_stop - t_start, "")
        else:
            # Confirm valid on original env after optimizations
            if not env_program_completable(c):
                state = c
                initial_check = qubit_checker(state)
                for j, act in enumerate(opt_result.best.actions):
                    assert check_no_outstanding_gates(state)
                    state, _, terminated, truncated, _ = env_step(state, act)
                    new_check = qubit_checker(state)

                    if not jnp.allclose(initial_check, new_check):
                        raise ValueError(f"Simulation error, qubits changed: {p_name}")

                    if truncated:
                        raise ValueError(f"Unexpected truncation: {p_name}")

                    if terminated != (j == len(opt_result.best.actions) - 1):
                        raise ValueError(f"Replay mismatch: {opt_result.best.actions}, {p_name}")

            record = (name, i, p_name, ion_count, op_count, True, opt_result.best.duration, t_stop - t_start, " ".join([str(a) for a in opt_result.best.actions]) or "")

        records.append(record)
        print(*record)

        df = pd.DataFrame.from_records(records, columns=["agent", "env", "problem_name", "ion_count", "op_count", "solved", "duration", "compile_time", "actions"])
        df.to_csv(os.path.join(output_folder, f"{test_name}-{name}.csv"), index=False)

    return df


if __name__ == '__main__':
    main()
