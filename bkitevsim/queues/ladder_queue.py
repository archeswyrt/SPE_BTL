# Copyright (C) 2026 pdnguyen of HCMC University of Technology VNU-HCM
#
# BkitTEvSim release
# Source Code License Grant: The authors hereby grant to Licensee
# personal permission to use and modify the Licensed Source Code
# for the sole purpose of studying while attending the course CO3007.
#
# Modifications Copyright (C) 2026 SPE Team - Dev 1 (DES Core & Queueing Station)
#
# LadderQueue - parallel discrete event simulation.
#
"""
LadderQueue - Min-heap priority queue with monotonic sequence numbering.

Replaces the original modulo-bucket circular buffer with a heap-based priority
queue to ensure strict timestamp causality without bucket boundary wrap-around bugs
(both for bucket_width=1.0 and bucket_width=0.001) and deterministic FCFS tie-breaking.
"""
from __future__ import annotations
from typing import Generic, TypeVar, List, Optional
import heapq
import threading

from bkitevsim.core.event import Event

T = TypeVar('T', bound=Event)


class LadderQueue(Generic[T]):
    """
    Min-heap based event priority queue for discrete-event simulation.

    Attributes:
        bucket_width: Maintained for backward compatibility with existing configs.
        num_buckets: Maintained for backward compatibility with existing configs.
    """

    def __init__(self, bucket_width: float = 1.0, num_buckets: int = 1024):
        self.bucket_width = bucket_width
        self.num_buckets = num_buckets
        # Stored entries are tuples: (timestamp, schedule_seq, item)
        self._heap: List[tuple[float, int, T]] = []
        self._schedule_seq: int = 0
        self._lock = threading.RLock()

    def push(self, item: T) -> None:
        """Add an event to the priority queue with monotonic sequence tie-breaking."""
        ts = item.timestamp
        if ts != ts:  # NaN check
            raise ValueError("Event timestamp cannot be NaN")

        with self._lock:
            heapq.heappush(self._heap, (ts, self._schedule_seq, item))
            self._schedule_seq += 1

    def pop(self) -> Optional[T]:
        """Pop the event with the earliest simulation timestamp."""
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)[2]

    def peek(self) -> Optional[T]:
        """Peek at the earliest event without removing it."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0][2]

    def peek_timestamp(self) -> Optional[float]:
        """Peek at the timestamp of the earliest event without removing it."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0][0]

    def is_empty(self) -> bool:
        """Return True if the queue contains no events."""
        with self._lock:
            return len(self._heap) == 0

    def size(self) -> int:
        """Return the current number of events in the queue."""
        with self._lock:
            return len(self._heap)

    def clear(self) -> None:
        """Clear all events and reset the scheduling sequence."""
        with self._lock:
            self._heap.clear()
            self._schedule_seq = 0

    def pop_events_after(self, timestamp: float) -> List[T]:
        """Remove and return all events scheduled strictly after the given timestamp."""
        with self._lock:
            retained = []
            removed = []
            for entry in self._heap:
                if entry[0] > timestamp:
                    removed.append(entry[2])
                else:
                    retained.append(entry)
            self._heap = retained
            heapq.heapify(self._heap)
            return removed

    def remove_events_after(self, timestamp: float) -> List[T]:
        """Deprecated alias for pop_events_after."""
        return self.pop_events_after(timestamp)

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    def __bool__(self) -> bool:
        with self._lock:
            return len(self._heap) > 0

    def __repr__(self) -> str:
        with self._lock:
            return f"LadderQueue(size={len(self._heap)})"
