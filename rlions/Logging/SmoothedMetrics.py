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

import numpy as np
from collections.abc import Mapping


def _ag_is_leaf(x):
    return isinstance(x, tuple)


def tree_map(fn, arg, *args, is_leaf=None):
    """
    Something similar to jax' tree-map outside jax
    """

    if is_leaf is not None and is_leaf(arg):
        return fn(arg, *args)

    if isinstance(arg, dict):
        keys = list(arg.keys())

        if args:
            kset = set(keys)
            for a in args:
                if not isinstance(a, dict):
                    raise TypeError(f"Unexpected type {type(a)} in remaining arguments, first argument {type(arg)}")

                assert kset == set(a.keys())

        return {k: tree_map(fn, *([arg[k]] + [a[k] for a in args]), is_leaf=is_leaf) for k in keys}
    elif isinstance(arg, list):
        for a in args:
            if not isinstance(a, list):
                raise TypeError(f"Unexpected type {type(a)} in remaining arguments, first argument {type(arg)}")

            if len(arg) != len(a):
                raise ValueError(f"Length of all lists should agree: {len(arg)} != {len(a)}")

        return [tree_map(fn, *els, is_leaf=is_leaf) for els in zip(arg, *args)]
    elif isinstance(arg, tuple):
        for a in args:
            assert isinstance(a, tuple)
            assert len(arg) == len(a)

        return tuple([tree_map(fn, *els, is_leaf=is_leaf) for els in zip(arg, *args)])
    else:
        return fn(arg, *args)


class SmoothedMetrics:
    """
    A finite response rectangle filter for metrics. Useful if metrics are continuously collected but only logged every
    X time steps.
    """

    def __init__(self):
        self._values = {}

    def update(self, metrics):
        queue = [(metrics, self._values)]

        # TODO: Probably more optimized if we use lists instead of tuples which are not constantly reallocated

        while queue:
            data, ip = queue.pop(0)

            for k, v in data.items():
                if v is None:
                    continue
                elif isinstance(v, Mapping):
                    try:
                        child_ip = ip[k]
                    except KeyError:
                        child_ip = ip[k] = {}
                    queue.append((v, child_ip))
                else:
                    ip[k] = ip.get(k, ()) + (v,)

    def report(self):
        result = tree_map(np.mean, self._values, is_leaf=_ag_is_leaf)
        self._values = {}
        return result


def main():
    a = SmoothedMetrics()
    a.update({'loss_test': 0.3, 'loss': 0.2})
    a.update({'loss': 0.1})
    print(a.report())
    a.update({'loss': 0.5})
    print(a.report())


if __name__ == '__main__':
    main()
