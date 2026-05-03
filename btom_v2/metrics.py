from __future__ import annotations

import math
import random
import statistics
from typing import Iterable, Sequence

from .schemas import EpisodeSummary


def count_successes(summaries: Iterable[EpisodeSummary]) -> int:
    return sum(1 for s in summaries if s.success)


def bootstrap_mean_ci(values: Sequence[float], n_resamples: int = 5000, seed: int = 123) -> tuple[float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    if len(set(values)) == 1:
        v = float(values[0])
        return v, v
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(math.floor(0.025 * (n_resamples - 1)))]
    hi = means[int(math.floor(0.975 * (n_resamples - 1)))]
    return float(lo), float(hi)
