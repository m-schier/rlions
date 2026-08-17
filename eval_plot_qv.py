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
from typing import Optional

import pandas as pd


def load_ppo_data(skip_unsolved: bool = False, root: Optional[str] = None):
    from glob import glob, escape

    if root is None:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "qv", "rlions")

    paths = list(glob(escape(root) + "/qv*-ppo*.csv"))

    dfs = []

    for p in paths:
        df = pd.read_csv(p)
        df['file_path'] = p
        if 'problem_name' not in df.columns:
            continue

        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)

    # Skip agents with unsolved
    if skip_unsolved:
        result = result[result['agent'].isin([g for g, s in result.groupby('agent')['duration'] if not s.isna().any()])]

    return result


def read_reference():
    from glob import glob
    import json
    import os

    records = []

    ref_path = "Data/qv/heuristic"

    for p in glob(f"{ref_path}/qv*.json"):
        with open(p, "r") as fp:
            obj = json.load(fp)

            name = os.path.basename(os.path.splitext(p)[0])
            ion_count = len(obj['chip_state']['temp_storage'])
            steps = len(obj['compiler_movements'])  # Reference has no duration, but 1 step = 1 step duration
            compile_time = obj['compile_time']

            records.append(("reference", name, ion_count, steps, compile_time))

    return pd.DataFrame.from_records(records, columns=['agent', 'problem_name', 'ion_count', 'duration', 'compile_time'])


def table_largest_solved(df):
    # 1. Group by both to see if EVERY entry for that size/method is solved
    all_solved = df.groupby(['Method', 'ion_count'])['is_solved'].all()

    # 2. Filter for only the True cases, then find the max ion_count per method
    result = (all_solved[all_solved]
              .reset_index()
              .groupby('Method')['ion_count']
              .max())

    # 3. Ensure methods with no solved sizes are included (defaulting to 0)
    final_output = result.reindex(df['Method'].unique(), fill_value=0)

    table_df = pd.DataFrame.from_dict({'Method': final_output.index, 'Largest solvable QV': final_output.values})

    print(table_df.to_latex(index=False))


def make_palette(df: pd.DataFrame, hue: str, palette):
    if palette is None:
        return 'tab10'

    return [palette[k] for k in df[hue].unique()]


def main():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from functools import partial
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--mode', default='main', choices=['main', 'architectures', 'ablation_x'], help="Select which experiment to plot")
    parser.add_argument('--big', action='store_true', help="Make big plot elements more suited for presentations")
    args = parser.parse_args()

    ppo_df_unsolved = None
    normalizer = None
    title = None
    ylabel = "Relative Duration of Ref. Mean"
    ypercent = True
    fixed_palette = None
    ncols = 2

    def _set_steps_normalizer(_df):
        nonlocal normalizer, ylabel, ypercent
        normalizer = _df.groupby('ion_count')['op_count'].mean()
        ylabel = "Steps per Two-Qubit-Gate"
        ypercent = False

    if args.mode == "main":
        # title = "Comparison X-Chip"
        title = None
        ppo_df = load_ppo_data()
        ppo_df = ppo_df[ppo_df['agent'].isin((
            'ppo_x50_b_fd_l4',
        ))]
        renames = {
            'ppo_x50_b_fd_l4': 'RLIonS ($\\mathbf{ours}$)',
        }
        for k, v in renames.items():
            ppo_df.loc[ppo_df['agent'] == k, 'agent'] = v

        fixed_palette = {
            'RLIonS ($\\mathbf{ours}$)': 'tab:green',
            'Heuristic': 'tab:orange',
        }
    elif args.mode == "architectures":
        ppo_df = load_ppo_data()
        ppo_df = ppo_df[ppo_df['agent'].isin((
            'ppo_q50_b_fd_l4',
            'ppo_q50_s3_b_fd_l4',
            'ppo_x50_b_fd_l4',
        ))]

        _set_steps_normalizer(ppo_df)
        title = None
        ncols = 3

        renames = {
            'ppo_x50_b_fd_l4': 'X-Chip',
            'ppo_q50_b_fd_l4': 'Q-Chip',
            'ppo_q50_s3_b_fd_l4': 'Q-Chip (SPAM Cap. 3)',
        }
        for k, v in renames.items():
            ppo_df.loc[ppo_df['agent'] == k, 'agent'] = v
    elif args.mode == "ablation_x":
        ppo_df_unsolved = load_ppo_data(skip_unsolved=False)
        ppo_df_unsolved = ppo_df_unsolved[ppo_df_unsolved['agent'].isin((
            'ppo_x50_b_fd_l4_same_gamma',
            'ppo_x50_b_fd_l4_linear',
            'ppo_x50_b_fd_l4_noshaped',
            'ppo_x50_b_fd_l4_basic_enc',
            'ppo_x50_b_fd_l4',
        ))]
        renames = {
            'ppo_x50_b_fd_l4': 'RLIonS ($\\mathbf{ours}$)',
            'ppo_x50_b_fd_l4_linear': 'RLIonS (Linear numeric encoding)',
            'ppo_x50_b_fd_l4_noshaped': 'RLIonS / QVLS-X-50 (No Shaped)',
            'ppo_x50_b_fd_l4_same_gamma': 'RLIonS / QVLS-X-50 (Same Gamma)',
            'ppo_x50_b_fd_l4_basic_enc': 'RLIonS / QVLS-X-50 (Basic Encoder)',
        }
        ppo_df_unsolved['agent_index'] = -1
        for i, (k, v) in enumerate(renames.items()):
            where = ppo_df_unsolved['agent'] == k
            ppo_df_unsolved.loc[where, 'agent'] = v
            ppo_df_unsolved.loc[where, 'agent_index'] = i

        ppo_df = ppo_df_unsolved[ppo_df_unsolved['agent'].isin([g for g, s in ppo_df_unsolved.groupby('agent')['duration'] if not s.isna().any()])]
        ppo_df = ppo_df.sort_values('agent_index')

        _set_steps_normalizer(ppo_df)
    else:
        raise ValueError(f"{args.mode = }")

    if normalizer is None:
        ref_df = read_reference()
        ref_df['agent'] = 'Heuristic'
        normalizer = ref_df.groupby('ion_count')['duration'].mean()

        df = pd.concat([
            ppo_df[['agent', 'problem_name', 'ion_count', 'duration', 'compile_time']],
            ref_df[['agent', 'problem_name', 'ion_count', 'duration', 'compile_time']],
        ])

        print(df.groupby(['agent', 'ion_count'])['compile_time'].mean())
    else:
        df = ppo_df
        ref_df = None

    os.makedirs("tmp/", exist_ok=True)

    df['duration_rel'] = df[['ion_count', 'duration']].apply(lambda x: x.duration / normalizer.loc[x.ion_count], axis=1)
    print(df.groupby(['agent', 'ion_count'])['duration_rel'].mean())

    # Human formatting
    df = df.rename(columns={'agent': 'Method'})

    if ppo_df_unsolved is not None:
        if ref_df:
            df_unsolved = pd.concat([
                ppo_df_unsolved[['agent', 'problem_name', 'ion_count', 'duration']],
                ref_df[['agent', 'problem_name', 'ion_count', 'duration']],
            ]).rename(columns={'agent': 'Method'})
        else:
            df_unsolved = ppo_df_unsolved[['agent', 'problem_name', 'ion_count', 'duration']].rename(columns={'agent': 'Method'})

    def _plot(_df, _path):
        fig, ax = plt.subplots(layout="constrained")
        sns.lineplot(_df, x='ion_count', y='duration', hue='Method', ax=ax, marker='o')
        plt.xlabel("QV($n$)")
        plt.ylabel("Steps")
        plt.savefig(_path)
        plt.close(fig)

        if ppo_df_unsolved is not None:
            df_unsolved['is_solved'] = ~df_unsolved['duration'].isna()
            table_largest_solved(df_unsolved)

            fig, ax = plt.subplots(layout="constrained")
            sns.lineplot(df_unsolved, x='ion_count', y='is_solved', hue='Method', ax=ax, palette='tab10')
            plt.xlabel("QV($n$)")
            plt.ylabel("Solution rate")
            ax.grid()
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            ax.get_legend().remove()
            fig.legend(loc='outside lower center', ncols=2)
            plt.savefig(os.path.splitext(_path)[0] + "-solved.pdf")
            plt.close(fig)

        fig, ax = plt.subplots(layout="constrained")
        ax.axhline(1.0, color='gray', zorder=-1000)
        sns.boxplot(_df, x='ion_count', y='duration_rel', hue='Method', ax=ax, palette='tab10')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        plt.xlabel("QV($n$)")
        plt.ylabel("Relative Duration of Ref. Mean")
        plt.savefig(os.path.splitext(_path)[0] + "-rel.pdf")
        plt.close(fig)

        kwargs = {}

        if args.big:
            kwargs['figsize'] = (6.4 / 1.5, 4.8 / 1.5)
        elif args.mode in ("ablation_x", "main", "architectures"):
            kwargs['figsize'] = (6.4, 3.0)

        fig, ax = plt.subplots(layout="constrained", **kwargs)
        sns.lineplot(_df, x='ion_count', y='duration_rel', hue='Method', ax=ax, palette=make_palette(_df, 'Method', fixed_palette), marker='.', legend="full", style='Method')
        ax.grid()
        if ypercent:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.get_legend().remove()

        fig_kwargs = {
            'loc': 'outside lower center',
        }

        if args.mode == "architectures":
            fig_kwargs['loc'] = 'upper left'
            fig_kwargs['title'] = 'All RLIonS ($\\mathbf{ours}$)'
            ax.legend(**fig_kwargs, ncols=1)
        elif args.mode == "main":
            ax.legend(title="All on X-Chip", ncols=1, loc="center right")
        elif args.mode == "ablation_x":
            ax.legend(title="All on X-Chip", ncols=1, loc="upper left")
        else:
            fig.legend(**fig_kwargs, ncols=ncols)

        if title is not None:
            fig.suptitle(title)
        plt.xlabel("Qubits in Quantum Volume")
        plt.ylabel(ylabel=ylabel)
        plt.savefig(os.path.splitext(_path)[0] + "-rel-line.pdf")
        plt.close(fig)

    _plot(df, f"tmp/qv-full-comparison-{args.mode}.pdf")


if __name__ == '__main__':
    main()
