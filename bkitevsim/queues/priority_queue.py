# Copyright (C) 2026 pdnguyen of HCMC University of Technology VNU-HCM
#
# BkitTEvSim release
# Source Code License Grant: The authors hereby grant to Licensee
# personal permission to use and modify the Licensed Source Code
# for the sole purpose of studying while attending the course CO3007.
#
# Modifications Copyright (C) 2026 SPE Team - Dev 1 (DES Core & Queueing Station)
#
# PriorityQueue - A generic priority queue wrapper using LadderQueue implementation.
#
"""
PriorityQueue - Standard priority queue wrapper using heapq.

Guarantees strict priority ordering with monotonic sequence numbers.
Supports Event instances (priority defaults to item.timestamp) and arbitrary items.
Eliminates direct comparison of payloads or Event objects on priority ties.
"""
from __future__ import annotations
from typing import Generic, TypeVar, Optional, List, Any
import heapq
import threading

from bkitevsim.core.event import Event

T = TypeVar('T')


class PriorityQueue(Generic[T]):
    """
    Generic priority queue implementation using heapq.

    Entries are stored as (priority, sequence, item).
    Sequence guarantees deterministic FCFS tie-breaking and prevents
    comparisons between items when priorities are identical.
    """

    def __init__(self, bucket_width: float = 1.0):
        self.bucket_width = bucket_width
        self._heap: List[tuple[float, int, T]] = []
        self._sequence: int = 0
        self._lock = threading.RLock()

    def push(self, item: T, priority: Optional[float] = None) -> None:
        """Push an item with optional explicit priority (defaults to item.timestamp if Event)."""
        if priority is None:
            if isinstance(item, Event):
                priority = item.timestamp
            else:
                priority = 0.0

        if priority != priority:  # NaN check
            raise ValueError("Priority cannot be NaN")

        with self._lock:
            heapq.heappush(self._heap, (float(priority), self._sequence, item))
            self._sequence += 1

    def pop(self) -> Optional[T]:
        """Pop the item with the highest priority (smallest priority value)."""
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)[2]

    def peek(self) -> Optional[T]:
        """Peek at the highest priority item without removing it."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0][2]

    def peek_priority(self) -> Optional[float]:
        """Peek at the smallest priority value currently in the queue."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0][0]

    def is_empty(self) -> bool:
        """Return True if the queue is empty."""
        with self._lock:
            return len(self._heap) == 0

    def size(self) -> int:
        """Return the number of items in the queue."""
        with self._lock:
            return len(self._heap)

    def clear(self) -> None:
        """Clear all items and reset sequence."""
        with self._lock:
            self._heap.clear()
            self._sequence = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    def __bool__(self) -> bool:
        with self._lock:
            return len(self._heap) > 0

    def __repr__(self) -> str:
        with self._lock:
            return f"PriorityQueue(size={len(self._heap)})"
