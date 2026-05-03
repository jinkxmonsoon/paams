from __future__ import annotations

from typing import Iterable

from .schemas import EpisodeSummary


def count_successes(summaries: Iterable[EpisodeSummary]) -> int:
    return sum(1 for s in summaries if s.success)
