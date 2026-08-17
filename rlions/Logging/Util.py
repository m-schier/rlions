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
import yaml


def make_agent_name(args):
    if args.obs_encoder.startswith('encoder_'):
        encoder_token = args.obs_encoder[len('encoder_'):]
    else:
        raise ValueError(f"{args.obs_encoder = }")

    return f"{args.chip}_{encoder_token}_l{args.lookahead}"


def search_fast(params):
    from glob import glob

    track_uri = 'tmp/mlruns'

    result = []

    for path in glob(track_uri + "/**/artifacts/agent.pckl", recursive=True):
        run_root = os.path.dirname(os.path.dirname(os.path.abspath(path)))

        with open(os.path.join(run_root, "meta.yaml"), "r") as fp:
            meta = yaml.safe_load(fp)

        if meta['lifecycle_stage'] == 'deleted':
            continue

        for k, v in params.items():
            param_path = os.path.join(run_root, 'params', k)

            try:
                with open(param_path, 'r') as fp:
                    if fp.read().strip() != str(v):
                        break
            except FileNotFoundError:
                break
        else:  # no break, thus all match
            result.append(run_root)

    return result
