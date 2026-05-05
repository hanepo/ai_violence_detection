from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Deque, Generic, Iterable, List, Tuple, TypeVar

T = TypeVar("T")


@dataclass
class TimedItem(Generic[T]):
    timestamp: datetime
    data: T


class TimeRingBuffer(Generic[T]):
    """Time-based ring buffer storing timestamped items for quick window queries."""

    def __init__(self, max_seconds: int):
        self.max_seconds = max_seconds
        self._items: Deque[TimedItem[T]] = deque()

    def append(self, data: T, timestamp: datetime | None = None) -> None:
        ts = timestamp or datetime.now(timezone.utc)
        self._items.append(TimedItem(timestamp=ts, data=data))
        self._evict_old(ts)

    def _evict_old(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.max_seconds)
        while self._items and self._items[0].timestamp < cutoff:
            self._items.popleft()

    def window(self, center: datetime, before_seconds: int, after_seconds: int) -> List[TimedItem[T]]:
        start = center - timedelta(seconds=before_seconds)
        end = center + timedelta(seconds=after_seconds)
        return [x for x in self._items if start <= x.timestamp <= end]

    def latest(self) -> TimedItem[T] | None:
        return self._items[-1] if self._items else None

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> Iterable[TimedItem[T]]:
        return list(self._items)
