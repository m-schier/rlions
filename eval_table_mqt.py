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
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


def pretty_print(df: pd.DataFrame, file=sys.stdout):
    def _sn(x):
        def f(a):
            try:
                return f"{int(a):04d}"
            except ValueError:
                return a

        return '_'.join([f(x) for x in x.split('_')])

    df = df.copy()
    df['sort_name'] = df['problem_name'].apply(_sn)
    df = df.sort_values('sort_name')
    df = df[['problem_name', 'opt_steps', 'steps', 'reference_steps']]
    df['steps'] = df['steps'].astype(int)
    df['opt_steps'] = df['opt_steps'].apply(lambda x: str(int(x)) if x is not None and not pd.isna(x) else '-')
    df.loc[:, 'problem_name'] = "\\texttt{" + df['problem_name'].str.replace('_', '\\_') + "}"
    df = df.rename(columns={'problem_name': 'Problem', 'opt_steps': 'Optimal steps', 'reference_steps': 'Reference steps', 'steps': 'RLIonS steps'})
    buffer = df.to_latex(index=False)
    print("%", *sys.argv, file=file)
    print(buffer, file=file)


def scatter_plot_difference_to_opt(df: pd.DataFrame):
    fig, ax = plt.subplots(layout="constrained")

    df = df[~df['opt_steps'].isna()].copy()
    df.loc[:, 'diff_duration_rl'] = df['steps'] - df['opt_steps']
    df.loc[:, 'diff_duration_ref'] = df['reference_steps'] - df['opt_steps']

    sns.scatterplot(df, x='diff_duration_ref', y='diff_duration_rl',  ax=ax)
    plt.axline((0, 0), slope=1, color='gray', zorder=-20)
    plt.xlabel("Optimality Gap of Reference (Duration)")
    plt.ylabel("Optimality Gap of RLIonS (Duration)")
    ax.set_axisbelow(True)
    ax.grid()
    plt.savefig("tmp/mqt_difference_scatter.pdf")
    plt.close(fig)


def scatter_plot_full(df: pd.DataFrame):
    fig, ax = plt.subplots(layout="constrained")
    ax.scatter(df['reference_steps'], df['steps'], marker='.', alpha=0.6, edgecolors='k')
    # sns.scatterplot(df, x='reference_steps', y='steps', ax=ax)
    plt.axline((0, 0), slope=1, color='gray', zorder=-20)
    ax.set_axisbelow(True)
    ax.grid()
    # ax.loglog()
    ax.set_xlabel("Reference Shuttling Duration [Steps]")
    ax.set_ylabel("RLIonS ($\\mathbf{ours}$) Shuttling Duration [Steps]")
    plt.savefig("tmp/mqt_scatter_full.pdf")
    plt.close(fig)


def scatter_plot_reference_improvement(df: pd.DataFrame):
    fig, ax = plt.subplots(layout="constrained")
    ax.scatter(df['reference_steps'], df['rel_improvement'], marker='.', alpha=0.6, edgecolors='k', label="RLIonS ($\\mathbf{ours}$)")
    ax.set_axisbelow(True)
    ax.grid()
    ax.semilogx()
    ax.set_xlabel("Reference Shuttling Durations [Steps]")
    plt.ylabel("Relative Duration of Reference")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend()
    plt.savefig("tmp/mqt_scatter_rel_improvement.pdf")
    plt.close(fig)


def fix_ion_counts(df):
    indicated_ion_count = df['problem_name'].str.split('_').apply(lambda x: int(x[-1]))

    mismatches = df['ion_count'] != indicated_ion_count

    if not mismatches.any():
        return

    print(f"WARN: {mismatches.sum()} mismatches in ion count in problem name: {df['problem_name'][mismatches].values}", file=sys.stderr)

    stripped_problem_names = df['problem_name'].str.split('_').apply(lambda x: '_'.join(x[:-1]))

    df['problem_name'] = stripped_problem_names + '_' + df['ion_count'].astype(str)


def strip_opt_level(df):
    df['problem_name'] = df['problem_name'].apply(lambda x: x.replace('_indep_opt3', ''))


def scatter_best_ref_with_opt(df: pd.DataFrame, big: bool = False):
    df = df[~df['opt_steps'].isna()]
    df = df.copy()
    df['family'] = df['problem_name'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    df = df.loc[df.groupby('family')['ion_count'].idxmax()].reset_index(drop=True).copy()

    strip_opt_level(df)
    fix_ion_counts(df)

    if big:
        print(f"{df['opt_compile_time'].mean() = }")
        print(f"{df['reference_compile_time'].mean() = }")
        print(f"{df['compile_time'].mean() = }")

        print(f"{df['opt_steps'].mean() = }")
        print(f"{df['reference_steps'].mean() = }")
        print(f"{df['steps'].mean() = }")

    # Print statistics
    with open("tmp/mqt_scatter_largest_opt.txt", "w") as fp:
        print(f"{(df['steps'] / df['reference_steps']).mean().item() = }", file=fp)
        print(f"{(df['steps'] / df['opt_steps']).mean().item() = }", file=fp)
        print(f"{df['compile_time'].mean().item() = }", file=fp)
        print(f"{df['opt_compile_time'].mean().item() = }", file=fp)

    figsize = (6.4, 6.4) if not big else (6.4, 6.4 / 1.5)

    fig, (ax1, ax2) = plt.subplots(2, 1, layout="constrained", sharex=True, figsize=figsize)
    ax1.scatter(np.arange(len(df)) - .1, df['opt_compile_time'], marker='P', label="SAT", color='tab:purple')
    ax1.scatter(np.arange(len(df)), df['reference_compile_time'], marker='D', label="Heuristic", color='tab:orange')
    ax1.scatter(np.arange(len(df)) + .1, df['compile_time'], marker='o', label="RLIonS ($\\mathbf{ours}$)", color='tab:green')
    ax1.set_axisbelow(True)
    ax1.grid()
    ax1.semilogy()
    ax1.set_ylabel("Compile Time [s]")
    ax2.scatter(np.arange(len(df)) - .1, df['opt_steps'], marker='P', color='tab:purple')
    ax2.scatter(np.arange(len(df)), df['reference_steps'], marker='D', color='tab:orange')
    ax2.scatter(np.arange(len(df)) + .1, df['steps'], marker='o', color='tab:green')
    ax2.set_axisbelow(True)
    ax2.grid()
    ax2.set_ylim(ymin=0)
    ax2.set_xticks(np.arange(len(df)), df['problem_name'], rotation=90)
    if big:
        ax2.set_ylabel("Shuttling\nDuration\n[Steps]")
    else:
        ax2.set_ylabel("Shuttling Duration [Steps]")
    ax2.tick_params(axis='x', which='major', labelsize=8)
    fig.legend(loc='outside upper center', ncols=3)
    plt.savefig("tmp/mqt_scatter_largest_opt.pdf")
    plt.close(fig)


def scatter_best_ref(df: pd.DataFrame):
    df = df.copy()
    df['family'] = df['problem_name'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    df = df.loc[df.groupby('family')['ion_count'].idxmax()].reset_index(drop=True)

    strip_opt_level(df)
    fix_ion_counts(df)

    # Print table
    table_df: pd.DataFrame = df[['problem_name', 'steps', 'reference_steps']].copy()
    table_df['steps'] = table_df['steps'].astype(int)
    table_df['problem_name'] = "\\texttt{" + table_df['problem_name'].str.replace('_', '\\_') + "}"
    buffer = table_df.to_latex(index=False)
    with open("tmp/mqt_ref.tex", "w") as fp:
        print("%", *sys.argv, file=fp)
        print(buffer, file=fp)

    # Print statistics
    with open("tmp/mqt_ref.txt", "w") as fp:
        print(f"{(table_df['reference_steps'] - table_df['steps']).mean().item() = }", file=fp)
        print(f"{(table_df['steps'] / table_df['reference_steps']).mean().item() = }", file=fp)

    fig, (ax1, ax2) = plt.subplots(2, 1, layout="constrained", sharex=True, figsize=(6.4, 6.4))
    ax1.scatter(np.arange(len(df)) - 0.05, df['reference_compile_time'], marker='o', edgecolors='k', label="Reference")
    ax1.scatter(np.arange(len(df)) + 0.05, df['compile_time'], marker='o', edgecolors='k', label="RLIonS ($\\mathbf{ours}$)")
    ax1.set_axisbelow(True)
    ax1.grid()
    ax1.semilogy()
    ax1.set_ylabel("Compile Time [s]")
    ax2.scatter(np.arange(len(df)) - 0.05, df['reference_steps'], marker='o', edgecolors='k')
    ax2.scatter(np.arange(len(df)) + 0.05, df['steps'], marker='o', edgecolors='k')
    ax2.set_axisbelow(True)
    ax2.grid()
    ax2.semilogy()
    ax2.set_xticks(np.arange(len(df)), df['problem_name'], rotation=90)
    ax2.set_ylabel("Shuttling Duration [Steps]")
    ax2.tick_params(axis='x', which='major', labelsize=6)
    fig.legend(loc='outside upper center', ncols=3)
    plt.savefig("tmp/mqt_scatter_much_largest_opt.pdf")
    plt.close(fig)


def make_long_table(df):
    strip_opt_level(df)
    fix_ion_counts(df)

    # Remove duplicates caused by wrong ion number in problem name
    df_sorted = df.sort_values(by=['problem_name', 'opt_steps'], ascending=True)
    df = df_sorted.drop_duplicates(subset='problem_name', keep='first')

    print(f"{(df['steps'] > df['reference_steps']).sum() = }")

    # Statistics
    with open("tmp/mqt_long.txt", "w") as fp:
        print(f"{df['opt_steps'].isna().mean() = }", file=fp)

    with open("tmp/mqt_long.tex", "w") as fp:
        pretty_print(df, file=fp)

    df.to_csv("tmp/mqt_long.csv", index=False)


def main():
    parser = ArgumentParser()
    parser.add_argument('--csv', default='Data/mqt/rlions/mqt-ppo_x50_b_fd_l4.csv', help="Path to RLIonS MQT results, usually in Data/mqt/rlions")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    records = []

    ref_folder = "Data/mqt/heuristic"
    sat_folder = "Data/mqt/sat"

    os.makedirs("tmp/", exist_ok=True)

    for env, steps, pn, oc, ic, compile_time in zip(df['env'], df['duration'], df['problem_name'], df['op_count'], df['ion_count'], df['compile_time']):
        ref_path = os.path.join(ref_folder, f"{pn}.json")
        sat_path = os.path.join(sat_folder, f"{pn}.json")

        try:
            with open(sat_path, "r") as fp:
                sat_obj = json.load(fp)
            sat_steps = len(sat_obj['compiler_movements'])
            sat_compile_time = sat_obj['compile_time']
        except (TypeError, FileNotFoundError):
            sat_steps = None
            sat_compile_time = None

        try:
            with open(ref_path, "r") as fp:
                obj = json.load(fp)
        except FileNotFoundError:
            continue

        # For some reason some reference data does not have a list but just the number of steps
        ref_steps = obj['compiler_movements'] if isinstance(obj['compiler_movements'], int) else len(obj['compiler_movements'])
        ref_compile_time = obj['compile_time']
        records.append((pn, oc, ic, sat_steps, ref_steps, steps, ref_steps - steps, steps / max(1, ref_steps), compile_time, ref_compile_time, sat_compile_time))

    result_df = pd.DataFrame.from_records(records, columns=["problem_name", "op_count", "ion_count", "opt_steps", "reference_steps", "steps", "improvement", "rel_improvement", "compile_time", "reference_compile_time", "opt_compile_time"])
    result_df = result_df[result_df['ion_count'] > 2]

    scatter_plot_difference_to_opt(result_df)
    scatter_plot_full(result_df)
    scatter_plot_reference_improvement(result_df)
    scatter_best_ref_with_opt(result_df)
    scatter_best_ref(result_df)
    make_long_table(result_df.copy())


if __name__ == '__main__':
    main()
