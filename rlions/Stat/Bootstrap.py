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

import numpy as np


@dataclass
class BootstrapResult:
    confidence_interval: np.ndarray
    bootstrap_distribution: np.ndarray


def bootstrap(arr, samples=100_000, estimator=np.mean, alpha=0.05, ax=None):
    arr = np.asarray(arr)
    n, = arr.shape

    idxs = np.random.randint(0, n, (samples, n))

    estimates = estimator(arr[idxs], axis=-1)

    ci_low, ci_high = np.quantile(estimates, alpha / 2), np.quantile(estimates, 1 - alpha / 2)

    if ax is not None:
        ax.set_title(f"$n = {n}$", loc="right")
        ax.hist(estimates, bins=100, color="tab:blue", label="Bootstrap Estimates")
        ax.axvline(ci_low, color='tab:red', label=f"CI ($\\alpha = {alpha}$)")
        ax.axvline(ci_high, color='tab:red')
        ax.axvline(np.mean(estimates), color='k')
        ax.axvline(estimator(arr), color='tab:orange', label="Estimate")
        ax.legend()

    return ci_low, ci_high


@dataclass
class BootstrapHypothesisResult:
    p_val: float
    bootstrap_distribution: np.ndarray


def bootstrap_hypothesis_test_mean(arr1, arr2, samples=100_000, ax=None, alternative='two-sided', estimator=np.mean) -> BootstrapHypothesisResult:
    """
    Bootstrap hypothesis test for difference of means
    :param arr1: Samples of first group
    :param arr2: Samples of second group
    :param samples: Number of bootstrap samples
    :param ax: pyplot axis to visualize result on
    :param alternative: If 'two-sided', alternative is mean(arr1) != mean(arr2). If 'one-sided', alternative is
    mean(arr1) > mean(arr2)
    :return: p-value
    """

    arr1, arr2 = np.asarray(arr1), np.asarray(arr2)
    # Our test statistic is difference of means, calculate on original split
    test_statistic = estimator(arr1) - estimator(arr2)

    n, = arr1.shape
    m, = arr2.shape

    # Null hypothesis is both groups come from a population with equal mean, thus we resample from the combined
    # and observe our sample test statistics
    combined = np.concatenate([arr1, arr2])

    idxs1 = np.random.randint(0, n + m, (samples, n))
    idxs2 = np.random.randint(0, n + m, (samples, m))

    estimates = estimator(combined[idxs1], axis=-1) - estimator(combined[idxs2], axis=-1)

    # p-val is ratio of more extreme samples in bootstrap than original test statistic
    if alternative == 'two-sided':
        p_val = np.mean(np.abs(estimates) >= np.abs(test_statistic))
    elif alternative == 'one-sided':
        p_val = np.mean(estimates >= test_statistic)
    else:
        raise ValueError(f"{alternative = }")

    if ax is not None:
        ax.set_title(f"$p = {p_val}$", loc="left")
        if alternative == 'two-sided':
            ax.set_title("$H_0: \\overline{x_1} = \\overline{x_2}, H_A: \\overline{x_1} \\neq \\overline{x_2}$",
                         loc="center")
        elif alternative == 'one-sided':
            ax.set_title("$H_0: \\overline{x_1} \\leq \\overline{x_2}, H_A: \\overline{x_1} > \\overline{x_2}$",
                         loc="center")
        ax.set_title(f"$|x_1| = {n}, |x_2| = {m}$", loc="right")
        ax.hist(estimates, bins=100, color='tab:blue', label="Bootstrap under $H_0$")
        ax.axvline(test_statistic, color='tab:orange', label="Test Statistic")
        ax.set_xlabel("$\\overline{x_1} - \\overline{x_2}$")
        ax.legend()

    return BootstrapHypothesisResult(
        p_val=p_val.item(),
        bootstrap_distribution=estimates,
    )


def bootstrap_difference(arr1, arr2, samples=100_000, alpha=0.05, estimator=np.mean, ax=None) -> BootstrapResult:
    arr1, arr2 = np.asarray(arr1), np.asarray(arr2)

    n, = arr1.shape
    m, = arr2.shape

    idxs1 = np.random.randint(0, n, (samples, n))
    idxs2 = np.random.randint(0, m, (samples, m))

    estimates = estimator(arr1[idxs1], axis=-1) - estimator(arr2[idxs2], axis=-1)

    ci_low, ci_high = np.quantile(estimates, alpha / 2), np.quantile(estimates, 1 - alpha / 2)

    p_val = np.mean(np.abs(estimates) >= np.abs(estimator(arr1) - estimator(arr2)))

    if ax is not None:
        ax.set_title(f"$n = {n}, m = {m}$", loc="right")
        ax.hist(estimates, bins=100, color='tab:blue', label="Bootstrap")
        ax.axvline(ci_low, color='tab:red', label=f"CI ($\\alpha = {alpha}$)")
        ax.axvline(ci_high, color='tab:red')
        # ax.axvline(np.mean(estimates), color='k')
        ax.axvline(estimator(arr1) - estimator(arr2), color='tab:orange', label="Test statistic")
        ax.legend()

    return BootstrapResult(
        confidence_interval=np.asarray([ci_low, ci_high]),
        bootstrap_distribution=estimates,
    )
