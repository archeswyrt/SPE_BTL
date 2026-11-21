# Copyright (C) 2026 SPE Team - Dev 1
"""
Test V04: Multi-server finite-capacity queueing station (M/M/c/K with c=2, B=3).

Specification:
- Multi-server c = 2, finite capacity B = 3 (2 in service + 1 in queue).
- Fixed deterministic service time = 2.0 seconds.
- External arrivals:
    - Job A at t = 0.0s -> starts immediately on Server 1 (ns=1).
    - Job B at t = 0.1s -> starts immediately on Server 2 (ns=2).
    - Job C at t = 0.2s -> all servers busy (ns=2), admitted into FCFS queue (nq=1, ns=2, n=3).
    - Job D at t = 0.3s -> total occupancy n = nq + ns = 1 + 2 = 3 == B -> Rejected (Dropped).
- Service completions:
    - Job A completes at t = 2.0s. Server freed, Job C immediately dequeued and begins service at t = 2.0s.
    - Job B completes at t = 2.1s. Server freed, queue empty (ns=1, nq=0).
    - Job C completes at t = 4.0s (start 2.0 + service 2.0). All servers idle (ns=0).
- Invariant check:
    - ns <= 2 (number of busy servers never exceeds c=2 at any timestamp).
    - nq + ns <= 3 (total system occupancy never exceeds buffer capacity B=3).
"""

import unittest
from bkitevsim.core.event import Event
from bkitevsim.core.logical_coProcess import BkSysCoEventLoop
from contracts import Request, Visit, EVENT_ARRIVAL
from queue_node import QueueNodeLP
from tests.trace_utils import print_table


class TestV04MultiServer(unittest.TestCase):
    """Test suite for V04 multi-server finite-capacity queue station."""

    def test_multi_server_parallelism_and_capacity(self):
        departures = []
        rejections = []
        state_samples = []

        def on_departure(request, visit, node, t):
            departures.append((request, visit, t))

        def on_rejected(request, visit, node, t):
            rejections.append((request, visit, t))

        def on_state_change(node, t, nq, ns):
            state_samples.append((t, nq, ns))

        node = QueueNodeLP(
            lp_id=2,
            node_name="CPU",
            c=2,
            B=3,
            mean_service_s=2.0,
            service_time_generator=lambda: 2.0,
            on_departure=on_departure,
            on_rejected=on_rejected,
            on_state_change=on_state_change,
        )

        loop = BkSysCoEventLoop(end_time=5.0)
        loop.register_lp(node)

        # 4 arrivals: A=0.0, B=0.1, C=0.2, D=0.3
        req_a = Request(request_id="A", external_arrival_t=0.0)
        vis_a = Visit(request_id="A", visit_id="A-1", node="CPU", arrival_t=0.0)

        req_b = Request(request_id="B", external_arrival_t=0.1)
        vis_b = Visit(request_id="B", visit_id="B-1", node="CPU", arrival_t=0.1)

        req_c = Request(request_id="C", external_arrival_t=0.2)
        vis_c = Visit(request_id="C", visit_id="C-1", node="CPU", arrival_t=0.2)

        req_d = Request(request_id="D", external_arrival_t=0.3)
        vis_d = Visit(request_id="D", visit_id="D-1", node="CPU", arrival_t=0.3)

        loop.schedule_event(Event(timestamp=0.0, sender_id=0, receiver_id=2,
                                  event_type=EVENT_ARRIVAL, payload={"request": req_a, "visit": vis_a}))
        loop.schedule_event(Event(timestamp=0.1, sender_id=0, receiver_id=2,
                                  event_type=EVENT_ARRIVAL, payload={"request": req_b, "visit": vis_b}))
        loop.schedule_event(Event(timestamp=0.2, sender_id=0, receiver_id=2,
                                  event_type=EVENT_ARRIVAL, payload={"request": req_c, "visit": vis_c}))
        loop.schedule_event(Event(timestamp=0.3, sender_id=0, receiver_id=2,
                                  event_type=EVENT_ARRIVAL, payload={"request": req_d, "visit": vis_d}))

        stats = loop.run()

        # -------------------------------------------------------------
        # Safely extract simulation execution results BEFORE assertions
        # -------------------------------------------------------------
        rej_count = len(rejections)
        rej_req, rej_vis, rej_t = rejections[0] if rej_count > 0 else (None, None, None)

        dep_count = len(departures)
        dep_a = departures[0] if dep_count > 0 else (None, None, None)
        dep_b = departures[1] if dep_count > 1 else (None, None, None)
        dep_c = departures[2] if dep_count > 2 else (None, None, None)

        sh = state_samples
        s0 = sh[0] if len(sh) > 0 else None
        s1 = sh[1] if len(sh) > 1 else None
        s2 = sh[2] if len(sh) > 2 else None
        s3 = sh[3] if len(sh) > 3 else None
        s4 = sh[4] if len(sh) > 4 else None
        s5 = sh[5] if len(sh) > 5 else None

        def evaluate_invariant(sample):
            if sample is None:
                return "N/A", False
            t_val, nq_val, ns_val = sample
            valid = (ns_val <= 2 and (nq_val + ns_val) <= 3 and nq_val >= 0 and ns_val >= 0)
            status_text = f"Valid (ns={ns_val}<=2, n={nq_val+ns_val}<=3)" if valid else f"VIOLATED (ns={ns_val}, n={nq_val+ns_val})"
            return status_text, valid

        inv_str0, inv_ok0 = evaluate_invariant(s0)
        inv_str1, inv_ok1 = evaluate_invariant(s1)
        inv_str2, inv_ok2 = evaluate_invariant(s2)
        inv_str3, inv_ok3 = evaluate_invariant(s3)
        inv_str4, inv_ok4 = evaluate_invariant(s4)
        inv_str5, inv_ok5 = evaluate_invariant(s5)

        # Dynamic match evaluation for all timeline rows
        match_row1 = "PASS" if (s0 and (s0[1], s0[2]) == (0, 1) and inv_ok0 and vis_a.start_t == 0.0) else "FAIL"
        match_row2 = "PASS" if (s1 and (s1[1], s1[2]) == (0, 2) and inv_ok1 and vis_b.start_t == 0.1) else "FAIL"
        match_row3 = "PASS" if (s2 and (s2[1], s2[2]) == (1, 2) and inv_ok2) else "FAIL"
        match_row4 = "PASS" if (rej_count == 1 and rej_req and rej_req.request_id == "D" and rej_t == 0.3 and not rej_vis.accepted and inv_ok2) else "FAIL"
        match_row5 = "PASS" if (s3 and (s3[1], s3[2]) == (0, 2) and inv_ok3 and dep_a[0] and dep_a[0].request_id == "A" and dep_a[2] == 2.0 and vis_c.start_t == 2.0) else "FAIL"
        match_row6 = "PASS" if (s4 and (s4[1], s4[2]) == (0, 1) and inv_ok4 and dep_b[0] and dep_b[0].request_id == "B" and dep_b[2] == 2.1) else "FAIL"
        match_row7 = "PASS" if (s5 and (s5[1], s5[2]) == (0, 0) and inv_ok5 and dep_c[0] and dep_c[0].request_id == "C" and dep_c[2] == 4.0) else "FAIL"

        timeline_rows = [
            [
                f"{vis_a.arrival_t:.2f}" if vis_a.arrival_t is not None else "N/A",
                "Arrival: Job A",
                "Server 1 allocated -> Service starts",
                f"(nq={s0[1]}, ns={s0[2]})" if s0 else "N/A",
                inv_str0,
                match_row1,
            ],
            [
                f"{vis_b.arrival_t:.2f}" if vis_b.arrival_t is not None else "N/A",
                "Arrival: Job B",
                "Server 2 allocated -> Service starts",
                f"(nq={s1[1]}, ns={s1[2]})" if s1 else "N/A",
                inv_str1,
                match_row2,
            ],
            [
                f"{vis_c.arrival_t:.2f}" if vis_c.arrival_t is not None else "N/A",
                "Arrival: Job C",
                "All servers busy -> Enqueued in FCFS queue",
                f"(nq={s2[1]}, ns={s2[2]})" if s2 else "N/A",
                inv_str2,
                match_row3,
            ],
            [
                f"{rej_t:.2f}" if rej_t is not None else "N/A",
                f"Arrival: Job {rej_req.request_id if rej_req else '?'}",
                "Buffer capacity full (B=3) -> Dropped",
                f"(nq={s2[1]}, ns={s2[2]})" if s2 else "N/A",
                inv_str2,
                match_row4,
            ],
            [
                f"{dep_a[2]:.2f}" if dep_a[2] is not None else "N/A",
                f"Complete: Job {dep_a[0].request_id if dep_a[0] else '?'}",
                "Depart -> Server 1 freed -> Job C dequeued & starts",
                f"(nq={s3[1]}, ns={s3[2]})" if s3 else "N/A",
                inv_str3,
                match_row5,
            ],
            [
                f"{dep_b[2]:.2f}" if dep_b[2] is not None else "N/A",
                f"Complete: Job {dep_b[0].request_id if dep_b[0] else '?'}",
                "Depart -> Server 2 freed -> Server idle",
                f"(nq={s4[1]}, ns={s4[2]})" if s4 else "N/A",
                inv_str4,
                match_row6,
            ],
            [
                f"{dep_c[2]:.2f}" if dep_c[2] is not None else "N/A",
                f"Complete: Job {dep_c[0].request_id if dep_c[0] else '?'}",
                "Depart -> Server 1 freed -> All servers idle",
                f"(nq={s5[1]}, ns={s5[2]})" if s5 else "N/A",
                inv_str5,
                match_row7,
            ],
        ]

        print_table(
            title="V04-1: M/M/c/K Multi-Server Timeline & Invariant Verification (c=2, B=3, mean_service=2.0s)",
            headers=["t (s)", "Event / Job", "Action", "Actual (nq, ns)", "Invariant (ns<=2, n<=3)", "Match"],
            rows=timeline_rows,
            alignments=["center", "left", "left", "center", "left", "center"],
        )

        # -------------------------------------------------------------
        # Assertions executed after table has been output
        # -------------------------------------------------------------
        self.assertEqual(stats["status"], "COMPLETED")

        # 1. Verify Job A and Job B served immediately
        self.assertEqual(vis_a.start_t, 0.0)
        self.assertEqual(vis_a.complete_t, 2.0)
        self.assertEqual(vis_b.start_t, 0.1)
        self.assertEqual(vis_b.complete_t, 2.1)

        # 2. Verify Job C queued and started when Server 1 freed at 2.0s
        self.assertEqual(vis_c.start_t, 2.0)
        self.assertEqual(vis_c.complete_t, 4.0)
        self.assertAlmostEqual(vis_c.start_t - vis_c.arrival_t, 1.8, places=4)

        # 3. Verify Job D rejected at 0.3s
        self.assertEqual(len(rejections), 1)
        self.assertIsNotNone(rej_req)
        self.assertEqual(rej_req.request_id, "D")
        self.assertEqual(rej_t, 0.3)
        self.assertFalse(rej_vis.accepted)

        # 4. Verify Completed Departures (A at 2.0, B at 2.1, C at 4.0)
        self.assertEqual(len(departures), 3)
        self.assertIsNotNone(dep_a[0])
        self.assertEqual(dep_a[0].request_id, "A")
        self.assertEqual(dep_a[2], 2.0)
        self.assertIsNotNone(dep_b[0])
        self.assertEqual(dep_b[0].request_id, "B")
        self.assertEqual(dep_b[2], 2.1)
        self.assertIsNotNone(dep_c[0])
        self.assertEqual(dep_c[0].request_id, "C")
        self.assertEqual(dep_c[2], 4.0)

        # 5. Invariant Check: ns <= c (2) and nq + ns <= B (3) at all state transition points
        for t, nq, ns in state_samples:
            self.assertLessEqual(ns, 2, f"Server capacity violated at t={t}: ns={ns} > c=2")
            self.assertLessEqual(nq + ns, 3, f"Total capacity violated at t={t}: n={nq + ns} > B=3")
            self.assertGreaterEqual(nq, 0)
            self.assertGreaterEqual(ns, 0)

        # Confirm final state at t=4.0s is (nq=0, ns=0)
        self.assertTrue(len(state_samples) > 0)
        last_t, last_nq, last_ns = state_samples[-1]
        self.assertEqual(last_t, 4.0)
        self.assertEqual(last_nq, 0)
        self.assertEqual(last_ns, 0)


if __name__ == "__main__":
    unittest.main()
