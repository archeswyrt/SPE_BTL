# Copyright (C) 2026 SPE Team - Dev 1
"""
Test V01: Verification of priority queues (LadderQueue and PriorityQueue).

Tests:
1. Bucket boundary and cycle wrap-around tests at 1024 * bucket_width
   (for bucket_width=1.0 and bucket_width=0.001).
2. Deterministic FCFS tie-breaking for equal timestamps (pop and peek).
3. Payload safety (no direct comparisons between Event or dictionary objects).
"""

import unittest
from itertools import zip_longest
from bkitevsim.core.event import Event
from bkitevsim.queues.ladder_queue import LadderQueue
from bkitevsim.queues.priority_queue import PriorityQueue
from tests.trace_utils import print_table


class TestV01HeapBoundary(unittest.TestCase):
    """Test suite for V01 priority queue boundary and causality verification."""

    def test_ladder_queue_boundary_width_1(self):
        """Verify LadderQueue does not invert timestamps at 1024.0 (width=1.0)."""
        lq = LadderQueue(bucket_width=1.0)
        e1 = Event(timestamp=1023.9, sender_id=1, receiver_id=1, event_type="E1")
        e2 = Event(timestamp=1024.1, sender_id=1, receiver_id=1, event_type="E2")
        e3 = Event(timestamp=1023.95, sender_id=1, receiver_id=1, event_type="E3")
        e4 = Event(timestamp=1024.0, sender_id=1, receiver_id=1, event_type="E4")

        # Push in mixed order
        pushed_order = [
            "#1: E2 (t=1024.1)",
            "#2: E1 (t=1023.9)",
            "#3: E4 (t=1024.0)",
            "#4: E3 (t=1023.95)",
        ]
        lq.push(e2)  # 1024.1
        lq.push(e1)  # 1023.9
        lq.push(e4)  # 1024.0
        lq.push(e3)  # 1023.95

        # Inspect peek
        peek_item = lq.peek()
        peek_t = lq.peek_timestamp()

        # Pop all items safely
        p1 = lq.pop()
        p2 = lq.pop()
        p3 = lq.pop()
        p4 = lq.pop()
        is_empty_after = lq.is_empty()

        # Build verification trace table BEFORE assertions
        popped_events = [p1, p2, p3, p4]
        expected_events = [(1023.9, "E1"), (1023.95, "E3"), (1024.0, "E4"), (1024.1, "E2")]
        trace_rows = []
        for idx, (pushed, act, exp) in enumerate(zip_longest(pushed_order, popped_events, expected_events, fillvalue=None), start=1):
            act_tuple = (act.timestamp, act.event_type) if isinstance(act, Event) else (None, None)
            match = "PASS" if act_tuple == exp else "FAIL"
            trace_rows.append([idx, pushed or "N/A", str(act_tuple), str(exp), match])

        print_table(
            title="V01-1: LadderQueue Bucket Boundary & Cycle Wrap-Around (width=1.0)",
            headers=["Step", "Pushed Order", "Actual Popped (t, type)", "Expected (t, type)", "Match"],
            rows=trace_rows,
            alignments=["center", "left", "left", "left", "center"],
        )

        # Assertions executed after table output
        self.assertIsNotNone(peek_item)
        self.assertEqual(peek_item.timestamp, 1023.9)
        self.assertEqual(peek_t, 1023.9)
        self.assertIsNotNone(p1)
        self.assertEqual(p1.timestamp, 1023.9)
        self.assertIsNotNone(p2)
        self.assertEqual(p2.timestamp, 1023.95)
        self.assertIsNotNone(p3)
        self.assertEqual(p3.timestamp, 1024.0)
        self.assertIsNotNone(p4)
        self.assertEqual(p4.timestamp, 1024.1)
        self.assertTrue(is_empty_after)

    def test_ladder_queue_boundary_width_small(self):
        """Verify LadderQueue does not invert timestamps at 1.024 (width=0.001)."""
        lq = LadderQueue(bucket_width=0.001)
        e1 = Event(timestamp=1.0239, sender_id=1, receiver_id=1, event_type="E1")
        e2 = Event(timestamp=1.0241, sender_id=1, receiver_id=1, event_type="E2")
        e3 = Event(timestamp=1.0240, sender_id=1, receiver_id=1, event_type="E3")

        pushed_order = [
            "#1: E2 (t=1.0241)",
            "#2: E1 (t=1.0239)",
            "#3: E3 (t=1.0240)",
        ]
        lq.push(e2)
        lq.push(e1)
        lq.push(e3)

        peek_item = lq.peek()
        p1 = lq.pop()
        p2 = lq.pop()
        p3 = lq.pop()
        is_empty_after = lq.is_empty()

        popped_events = [p1, p2, p3]
        expected_events = [(1.0239, "E1"), (1.0240, "E3"), (1.0241, "E2")]
        trace_rows = []
        for idx, (pushed, act, exp) in enumerate(zip_longest(pushed_order, popped_events, expected_events, fillvalue=None), start=1):
            act_tuple = (act.timestamp, act.event_type) if isinstance(act, Event) else (None, None)
            match = "PASS" if act_tuple == exp else "FAIL"
            trace_rows.append([idx, pushed or "N/A", str(act_tuple), str(exp), match])

        print_table(
            title="V01-2: LadderQueue Boundary & Wrap-Around (width=0.001)",
            headers=["Step", "Pushed Order", "Actual Popped (t, type)", "Expected (t, type)", "Match"],
            rows=trace_rows,
            alignments=["center", "left", "left", "left", "center"],
        )

        # Assertions executed after table output
        self.assertIsNotNone(peek_item)
        self.assertEqual(peek_item.timestamp, 1.0239)
        self.assertIsNotNone(p1)
        self.assertEqual(p1.timestamp, 1.0239)
        self.assertIsNotNone(p2)
        self.assertEqual(p2.timestamp, 1.0240)
        self.assertIsNotNone(p3)
        self.assertEqual(p3.timestamp, 1.0241)
        self.assertTrue(is_empty_after)

    def test_equal_timestamp_fcfs_order(self):
        """Verify events with identical timestamps are popped in deterministic FCFS order."""
        lq = LadderQueue(bucket_width=1.0)
        e_a = Event(timestamp=5.0, sender_id=1, receiver_id=1, event_type="A", payload={"k": 1})
        e_b = Event(timestamp=5.0, sender_id=1, receiver_id=1, event_type="B", payload={"k": 2})
        e_c = Event(timestamp=5.0, sender_id=1, receiver_id=1, event_type="C", payload={"k": 3})

        pushed_order = ["#1: A (t=5.0)", "#2: B (t=5.0)", "#3: C (t=5.0)"]
        lq.push(e_a)
        lq.push(e_b)
        lq.push(e_c)

        peek_1 = lq.peek()
        p1 = lq.pop()
        peek_2 = lq.peek()
        p2 = lq.pop()
        peek_3 = lq.peek()
        p3 = lq.pop()
        is_empty_after = lq.is_empty()

        popped_events = [p1, p2, p3]
        expected_events = [(5.0, "A"), (5.0, "B"), (5.0, "C")]
        trace_rows = []
        for idx, (pushed, act, exp) in enumerate(zip_longest(pushed_order, popped_events, expected_events, fillvalue=None), start=1):
            act_tuple = (act.timestamp, act.event_type) if isinstance(act, Event) else (None, None)
            match = "PASS" if act_tuple == exp else "FAIL"
            trace_rows.append([idx, pushed or "N/A", str(act_tuple), str(exp), match])

        print_table(
            title="V01-3: Deterministic FCFS Tie-Breaking at Identical Timestamp (t=5.0)",
            headers=["Step", "Pushed Order", "Actual Popped (t, type)", "Expected (t, type)", "Match"],
            rows=trace_rows,
            alignments=["center", "left", "left", "left", "center"],
        )

        # Assertions executed after table output
        self.assertIsNotNone(peek_1)
        self.assertEqual(peek_1.event_type, "A")
        self.assertIsNotNone(p1)
        self.assertEqual(p1.event_type, "A")
        self.assertIsNotNone(peek_2)
        self.assertEqual(peek_2.event_type, "B")
        self.assertIsNotNone(p2)
        self.assertEqual(p2.event_type, "B")
        self.assertIsNotNone(peek_3)
        self.assertEqual(peek_3.event_type, "C")
        self.assertIsNotNone(p3)
        self.assertEqual(p3.event_type, "C")
        self.assertTrue(is_empty_after)

    def test_priority_queue_wrapper(self):
        """Verify generic PriorityQueue wrapper handles both Event and non-Event items."""
        pq = PriorityQueue(bucket_width=1.0)
        pq.push("item_high", priority=1.5)
        pq.push("item_low", priority=0.5)
        pq.push("item_mid", priority=1.0)

        peek_val = pq.peek()
        peek_prio = pq.peek_priority()
        p1 = pq.pop()
        p2 = pq.pop()
        p3 = pq.pop()
        p4 = pq.pop()

        self.assertEqual(peek_val, "item_low")
        self.assertEqual(peek_prio, 0.5)
        self.assertEqual(p1, "item_low")
        self.assertEqual(p2, "item_mid")
        self.assertEqual(p3, "item_high")
        self.assertIsNone(p4)


if __name__ == "__main__":
    unittest.main()
