def _build_action_translation() -> list[int]:
    our_map = {
        0: 'Spam',
        1: 'Storage',
        2: 'TmpStorage',
        3: 'Compute',
    }

    tiasm_reverse_map = {
        'Spam': 3,
        'Storage': 0,
        'TmpStorage': 2,
        'Compute': 1,
    }

    result = []

    for our_action in range(12):
        our_from, our_to = divmod(our_action, 3)
        if our_from <= our_to:
            our_to += 1

        src = our_map[our_from]
        dst = our_map[our_to]

        tiasm_from = tiasm_reverse_map[src]
        tiasm_to = tiasm_reverse_map[dst]

        if tiasm_to >= tiasm_from:
            tiasm_to -= 1

        result.append(tiasm_from * 3 + tiasm_to)

    return result


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

_TRANSLATE_ACTION_OURS_TO_TIASM = _build_action_translation()
_TRANSLATE_ACTION_TIASM_TO_OURS = [_TRANSLATE_ACTION_OURS_TO_TIASM.index(i) for i in range(len(_TRANSLATE_ACTION_OURS_TO_TIASM))]


def translate_action_to_tiasm_int(our_action: int) -> int:
    return _TRANSLATE_ACTION_OURS_TO_TIASM[our_action]


def translate_action_from_tiasm_int(our_action: int) -> int:
    return _TRANSLATE_ACTION_TIASM_TO_OURS[our_action]
