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

import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import os
import sys
from collections import defaultdict


def best_solution_for_budget(df: pd.DataFrame, time_budget: float):
    # Shuffle data
    df = df.sample(frac=1, replace=False)

    take = df['time'].cumsum() < time_budget
    take.iloc[0] = True  # Always take at least one

    best_steps = df.loc[take, 'steps'].min()
    return best_steps


def main():
    from itertools import product

    parser = ArgumentParser()
    parser.add_argument('--csv', nargs='+', default='Data/qv6_long/rlions/long-qv6-qvls_x_50_b_fd_l4.csv')
    args = parser.parse_args()

    print(f"{args = }", file=sys.stderr)

    sat_folder = "Data/qv6_long/sat"
    ref_folder = "Data/qv6_long/heuristic"

    df = []

    for path in args.csv:
        df.append(pd.read_csv(path))

    df = pd.concat(df, ignore_index=True)

    df.info()

    # Group the solutions
    solutions = dict(iter(df.groupby('problem_name')))
    problem_names = list(solutions.keys())

    if len(problem_names) != 100:
        raise ValueError

    useful_budget = min((s['time'].sum() for s in solutions.values()))
    print(f"{useful_budget = }")

    # problem_names = df['problem_name'].unique()

    sat_compile_times = []
    ref_compile_times = []

    # Find optimal solutions
    optimality = {}
    references = {}

    for pn in problem_names:
        sat_path = os.path.join(sat_folder, f"{pn}.json")

        with open(sat_path, "r") as fp:
            obj = json.load(fp)

        sat_steps = len(obj['compiler_movements'])
        sat_compile_times.append(obj['compile_time'])
        optimality[pn] = sat_steps

        ref_path = os.path.join(ref_folder, f"{pn}.json")

        with open(ref_path, "r") as fp:
            obj = json.load(fp)

        ref_steps = len(obj['compiler_movements'])
        ref_compile_times.append(obj['compile_time'])
        references[pn] = ref_steps

    print(f"{np.mean(list(optimality.values())) = }")

    # Do the search
    records = []

    budgets = (0.1, 1.0, 10.0, 100.0)
    samples = 50

    for budget, pn, _ in product(budgets, problem_names, range(samples)):
        optimal = optimality[pn]
        actual = best_solution_for_budget(solutions[pn], budget)

        if actual < optimal:
            raise ValueError(f"Problem {pn} solved faster than optimal, actual steps: {actual}, optimal: {optimal}")

        records.append((budget, pn, actual - optimal, (actual - optimal) / optimal))

    eval_df = pd.DataFrame.from_records(records, columns=["budget", "problem", "optimality_gap", "optimality_gap_rel"])

    opt_gaps = [0, 1, 2]
    print(f"{opt_gaps = }")

    records = []

    # Create the table rows for RLIonS
    for budget, budget_df in eval_df.groupby('budget'):
        values = tuple([(budget_df['optimality_gap'] == opt_gap).mean() for opt_gap in opt_gaps])

        # Find larger than opt_gaps[-1]
        larger_ratio = (eval_df.loc[eval_df['budget'] == budget, 'optimality_gap'] > opt_gaps[-1]).mean()

        avg_opt_gap = eval_df.loc[eval_df['budget'] == budget, 'optimality_gap'].mean()
        records.append(("RLIonS (ours)", budget) + values + (larger_ratio, avg_opt_gap,))

    # Create the table row for Reference
    ref_counters = defaultdict(lambda: 0)
    for pn, sat_steps in optimality.items():
        gap = references[pn] - sat_steps
        ref_counters[gap] = ref_counters[gap] + 1

    avg_opt_gap = sum((k * v for k, v in ref_counters.items())) / len(problem_names)
    larger_ratio = 1. - sum([ref_counters[opt_gap] for opt_gap in opt_gaps]) / len(problem_names)
    records.append(("Heuristic", np.mean(ref_compile_times)) + tuple([ref_counters[opt_gap] / len(problem_names) for opt_gap in opt_gaps]) + (larger_ratio, avg_opt_gap,))

    # Create the table row for SAT
    records.append(("SAT", np.mean(sat_compile_times)) + tuple([1.0] + [0.0] * (len(opt_gaps) + 1)))

    table_df = pd.DataFrame.from_records(records, columns=["Method", "Time"] + [f"Opt. Gap {i}" for i in opt_gaps] + [f"Opt. Gap > {opt_gaps[-1]}", "Avg. Opt. Gap"])
    table_df["Avg. Opt. Gap"] = table_df["Avg. Opt. Gap"].apply(lambda x: f"\\SI{{{x:.02f}}}{{steps}}")
    table_df["Time"] = table_df["Time"].apply(lambda x: f"\\SI{{{x:.01f}}}{{\\second}}")
    print(table_df.to_latex(index=False, float_format=lambda x: f"\\SI{{{x * 100:.02f}}}{{\\percent}}", column_format='l|r|rrrr|r'))


if __name__ == '__main__':
    main()
