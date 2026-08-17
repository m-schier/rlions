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

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class DataSet:
    env: Any
    problem_names: Optional[List[str]]
    initial_states: Any
    name: str


def load_qvs(size, chip, root="Data/qv_json_new"):
    from glob import glob
    from rlions.TestSets import load_tests
    import os

    paths = list(glob(root + f"/qv{size}_*.json"))

    names = [os.path.basename(os.path.splitext(p)[0]) for p in paths]

    return DataSet(env=chip.make_eval_env(), name=f"qv{size}", problem_names=names, initial_states=load_tests(paths, chip, batched=False))


def load_mqt(chip, problems=None):
    from glob import glob
    import os
    from rlions.TestSets import load_tests, get_tmp_storage
    import json

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = list(glob(f"{root}/Data/mqt_json_new/*.json"))

    max_qubits = chip.make_observation_adapter().max_qubits

    def _conditions_ok(p):
        problem_name = os.path.basename(os.path.splitext(p)[0])

        # Hybrid problems were partially solved problems that have been removed from the data
        if '_hybrid_' in problem_name:
            return False

        if problems is not None:
            if problem_name not in problems:
                return False

        try:
            with open(p, 'r') as fp:
                obj = json.load(fp)
                length_ok = len(obj['circuit']) <= 2000

                all_qubits = (
                    obj['chip_state']['compute'] +
                    obj['chip_state']['spam'] +
                    obj['chip_state']['storage'] +
                    get_tmp_storage(obj['chip_state'])
                )

                # For some reason in data
                if len(all_qubits) < 2:
                    return False

                if max_qubits is None:
                    qubits_ok = True
                else:
                    highest_qubit = max(all_qubits)
                    qubits_ok = highest_qubit < max_qubits

                return length_ok and qubits_ok
        except Exception as ex:
            raise IOError(f"Failed to load {p}") from ex

    paths = [p for p in paths if _conditions_ok(p)]

    names = [os.path.basename(os.path.splitext(p)[0]) for p in paths]

    return DataSet(env=chip.make_eval_env(), name=f"mqt", problem_names=names,
                   initial_states=load_tests(paths, chip, batched=False))
