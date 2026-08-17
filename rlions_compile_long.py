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

import pandas as pd


def main():
    from rlions.StandardDatasets import load_qvs
    from rlions.Chips import get_chip_by_name
    from rlions.AOTSearch import AOTSearch
    from rlions.Logging.Util import make_agent_name
    from rlions_compile import get_parallelism_max_len, make_ppo_guided
    from rlions.Util import compatible_load
    from time import time
    from argparse import ArgumentParser

    problems = {
        'qv6': lambda c: load_qvs(6, c, root="Data/qv6_long/heuristic"),
    }

    parser = ArgumentParser()
    parser.add_argument('--problem', choices=list(problems.keys()), default='qv6')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('PATH')
    cmd_args = parser.parse_args()

    args = compatible_load(cmd_args.PATH)['args']
    print(f"{args = }", file=sys.stderr)

    name = cmd_args.name or make_agent_name(args)

    output_folder = f"tmp/searches-long"
    os.makedirs(output_folder, exist_ok=True)

    # Determine output path
    for i in range(1000):
        output_path = os.path.join(output_folder, f"long-{cmd_args.problem}-{name}-{i:03}.csv")

        if not os.path.isfile(output_path):
            break
    else:  # no break
        raise IOError("No file")

    print(f"{output_path = }", file=sys.stderr)

    chip = get_chip_by_name(args.chip)

    ds = problems[cmd_args.problem](chip)

    policy = make_ppo_guided(cmd_args.PATH, ds.env, chip)

    parallelism, max_len = get_parallelism_max_len(chip, ds.initial_states)
    search = AOTSearch(policy, ds.env.step, parallelism, max_len, chip)

    print(f"{len(ds.initial_states) = }", file=sys.stderr)

    records = []

    stop = time() + 100 * 1800

    while time() < stop:
        for pn, s in zip(ds.problem_names, ds.initial_states):
            t_start = time()
            opt_result = search.do_optimization(s, 0.0)
            t_stop = time()

            records.append((name, pn, parallelism, t_stop - t_start, opt_result.solved, opt_result.best.steps if opt_result.solved else ''))
        pd.DataFrame.from_records(records, columns=["agent", "problem_name", "rollouts", "time", "solved", "steps"]).to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
