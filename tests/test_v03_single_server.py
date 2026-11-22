# Copyright (C) 2026 SPE Team - Dev 1
"""
Test V03: Single-server finite-capacity queueing station (M/M/1/K with c=1, B=2).

Specification:
- Single server c = 1, finite capacity B = 2 (1 in service + 1 in queue).
- Fixed deterministic service time = 2.0 seconds.
- External arrivals:
    - Job A at t = 0.0s
    - Job B at t = 0.5s
    - Job C at t = 1.0s
- Expected outcomes:
    - Job A starts at 0.0s, completes at 2.0s (Waiting time W(A) = 0.0s).
    - Job B admitted at 0.5s, waits in FCFS queue, starts at 2.0s, completes at 4.0s (W(B) = 1.5s).
    - Job C arrives at 1.0s: station occupancy n = nq + ns = 1 + 1 = 2 == B -> Rejected (Dropped).
- Hand-calculated metric verification over measurement window [0.0, 5.0):
    - Total duration T = 5.0s.
    - Server occupancy ns(t):
        [0.0, 0.5): ns=1 (Job A)
        [0.5, 2.0): ns=1 (Job A)
        [2.0, 4.0): ns=1 (Job B)
        [4.0, 5.0): ns=0 (idle, no events)
        Integral ns = 1*0.5 + 1*1.5 + 1*2.0 + 0*1.0 = 4.0 server-seconds.
        Server Utilization U = 4.0 / (1 * 5.0) = 0.8.
    - Queue occupancy nq(t):
        [0.0, 0.5): nq=0
        [0.5, 2.0): nq=1 (Job B)
        [2.0, 4.0): nq=0
        [4.0, 5.0): nq=0
        Integral nq = 0*0.5 + 1*1.5 + 0*2.0 + 0*1.0 = 1.5 job-seconds.
        Mean Queue Length Lq = 1.5 / 5.0 = 0.3.
    - Completed Jobs Response Time:
        Job A: 2.0 - 0.0 = 2.0s
        Job B: 4.0 - 0.5 = 3.5s
        Mean Response Time R_completed = (2.0 + 3.5) / 2 = 2.75s.
    - Drop Probability P_drop:
        1 dropped out of 3 arrivals = 1/3 = 0.3333...
    - System Throughput X:
        2 completed jobs / 5.0s = 0.4 jobs/s.
"""

import unittest
from bkitevsim.core.event import Event
from bkitevsim.core.logical_coProcess import BkSysCoEventLoop
from contracts import Request, Visit, EVENT_ARRIVAL
from queue_node import QueueNodeLP
from tests.trace_utils import print_table


class MetricCollectorV03:
    """Helper collector to compute time-weighted metrics over window [t0, t1) per Dev 3 contract."""
    def __init__(self, t0: float = 0.0, t1: float = 5.0, c: int = 1):
        self.t0 = t0
        self.t1 = t1
        self.c = c
        self.last_t = t0
        self.saved_nq = 0
        self.saved_ns = 0
        self.integral_ns = 0.0
        self.integral_nq = 0.0
        self.departures: list[tuple[Request, Visit, float]] = []
        self.rejections: list[tuple[Request, Visit, float]] = []
        self.state_history: list[tuple[float, int, int]] = []

    def on_state_change(self, node: str, t: float, new_nq: int, new_ns: int) -> None:
        """Called immediately after station updates its state to (new_nq, new_ns) at timestamp t."""
        self.state_history.append((t, new_nq, new_ns))
        # Accumulate time-weighted integral for the interval [last_t, t) using saved previous state
        if t > self.last_t:
            overlap_start = max(self.last_t, self.t0)
            overlap_end = min(t, self.t1)
            if overlap_end > overlap_start:
                duration = overlap_end - overlap_start
                self.integral_ns += self.saved_ns * duration
                self.integral_nq += self.saved_nq * duration
            self.last_t = t
        self.saved_nq = new_nq
        self.saved_ns = new_ns

    def on_departure(self, request: Request, visit: Visit, node: str, t: float) -> None:
        self.departures.append((request, visit, t))

    def on_rejected(self, request: Request, visit: Visit, node: str, t: float) -> None:
        self.rejections.append((request, visit, t))

    def close_window(self) -> None:
        """Close measurement window at t1, integrating any remaining duration using saved state."""
        if self.t1 > self.last_t:
            duration = self.t1 - max(self.last_t, self.t0)
            self.integral_ns += self.saved_ns * duration
            self.integral_nq += self.saved_nq * duration
            self.last_t = self.t1


class TestV03SingleServer(unittest.TestCase):
    """Test suite for V03 single-server finite-capacity queue station."""

    def test_single_server_fcfs_and_metrics(self):
        collector = MetricCollectorV03(t0=0.0, t1=5.0, c=1)

        # Create QueueNodeLP with c=1, B=2, fixed service=2.0s
        node = QueueNodeLP(
            lp_id=1,
            node_name="Ingress",
            c=1,
            B=2,
            mean_service_s=2.0,
            service_time_generator=lambda: 2.0,  # Fixed 2s
            on_departure=collector.on_departure,
            on_rejected=collector.on_rejected,
            on_state_change=collector.on_state_change,
        )

        loop = BkSysCoEventLoop(end_time=5.0)
        loop.register_lp(node)

        # Prepare 3 requests
        req_a = Request(request_id="A", external_arrival_t=0.0)
        vis_a = Visit(request_id="A", visit_id="A-1", node="Ingress", arrival_t=0.0)

        req_b = Request(request_id="B", external_arrival_t=0.5)
        vis_b = Visit(request_id="B", visit_id="B-1", node="Ingress", arrival_t=0.5)

        req_c = Request(request_id="C", external_arrival_t=1.0)
        vis_c = Visit(request_id="C", visit_id="C-1", node="Ingress", arrival_t=1.0)

        # Schedule arrivals
        loop.schedule_event(Event(timestamp=0.0, sender_id=0, receiver_id=1,
                                  event_type=EVENT_ARRIVAL, payload={"request": req_a, "visit": vis_a}))
        loop.schedule_event(Event(timestamp=0.5, sender_id=0, receiver_id=1,
                                  event_type=EVENT_ARRIVAL, payload={"request": req_b, "visit": vis_b}))
        loop.schedule_event(Event(timestamp=1.0, sender_id=0, receiver_id=1,
                                  event_type=EVENT_ARRIVAL, payload={"request": req_c, "visit": vis_c}))

        stats = loop.run()
        collector.close_window()

        # -------------------------------------------------------------
        # Safely extract simulation metrics and execution results BEFORE assertions
        # -------------------------------------------------------------
        sh = collector.state_history
        sh_0 = sh[0] if len(sh) > 0 else (None, None, None)
        sh_1 = sh[1] if len(sh) > 1 else (None, None, None)
        sh_2 = sh[2] if len(sh) > 2 else (None, None, None)
        sh_3 = sh[3] if len(sh) > 3 else (None, None, None)

        rej_count = len(collector.rejections)
        rej_req, rej_vis, rej_t = collector.rejections[0] if rej_count > 0 else (None, None, None)

        dep_count = len(collector.departures)
        dep_0 = collector.departures[0] if dep_count > 0 else (None, None, None)
        dep_1 = collector.departures[1] if dep_count > 1 else (None, None, None)

        waiting_time_a = (vis_a.start_t - vis_a.arrival_t) if (vis_a.start_t is not None and vis_a.arrival_t is not None) else None
        waiting_time_b = (vis_b.start_t - vis_b.arrival_t) if (vis_b.start_t is not None and vis_b.arrival_t is not None) else None

        T = 5.0
        U = collector.integral_ns / (1 * T)
        Lq = collector.integral_nq / T
        resp_a = (vis_a.complete_t - req_a.external_arrival_t) if (vis_a.complete_t is not None and req_a.external_arrival_t is not None) else None
        resp_b = (vis_b.complete_t - req_b.external_arrival_t) if (vis_b.complete_t is not None and req_b.external_arrival_t is not None) else None
        R_completed = ((resp_a + resp_b) / 2.0) if (resp_a is not None and resp_b is not None) else None
        P_drop = rej_count / 3.0
        X = dep_count / T

        # -------------------------------------------------------------
        # Table 1: Event Timeline Trace (Dynamic evaluation)
        # -------------------------------------------------------------
        act_sh0 = (sh_0[1], sh_0[2]) if sh_0[0] is not None else (None, None)
        act_sh1 = (sh_1[1], sh_1[2]) if sh_1[0] is not None else (None, None)
        act_sh2 = (sh_2[1], sh_2[2]) if sh_2[0] is not None else (None, None)
        act_sh3 = (sh_3[1], sh_3[2]) if sh_3[0] is not None else (None, None)

        match_row1 = "PASS" if (act_sh0 == (0, 1) and vis_a.start_t == 0.0) else "FAIL"
        match_row2 = "PASS" if (act_sh1 == (1, 1) and vis_b.arrival_t == 0.5) else "FAIL"
        match_row3 = "PASS" if (rej_count == 1 and rej_req is not None and rej_req.request_id == "C" and rej_t == 1.0 and not rej_vis.accepted) else "FAIL"
        match_row4 = "PASS" if (act_sh2 == (0, 1) and dep_0[0] is not None and dep_0[0].request_id == "A" and dep_0[2] == 2.0) else "FAIL"
        match_row5 = "PASS" if (act_sh3 == (0, 0) and dep_1[0] is not None and dep_1[0].request_id == "B" and dep_1[2] == 4.0) else "FAIL"
        match_row6 = "PASS" if (collector.saved_nq, collector.saved_ns) == (0, 0) else "FAIL"

        timeline_rows = [
            [
                f"{vis_a.arrival_t:.2f}" if vis_a.arrival_t is not None else "N/A",
                "Arrival: Job A",
                "Service Start (Server busy)",
                f"(nq={act_sh0[0]}, ns={act_sh0[1]})",
                "(nq=0, ns=1)",
                match_row1,
            ],
            [
                f"{vis_b.arrival_t:.2f}" if vis_b.arrival_t is not None else "N/A",
                "Arrival: Job B",
                "Enqueued in FCFS queue",
                f"(nq={act_sh1[0]}, ns={act_sh1[1]})",
                "(nq=1, ns=1)",
                match_row2,
            ],
            [
                f"{rej_t:.2f}" if rej_t is not None else "N/A",
                f"Arrival: Job {rej_req.request_id if rej_req else '?'}",
                "Buffer full (B=2) -> Dropped",
                f"(nq={act_sh1[0]}, ns={act_sh1[1]})",
                "(nq=1, ns=1)",
                match_row3,
            ],
            [
                f"{dep_0[2]:.2f}" if dep_0[2] is not None else "N/A",
                f"Complete: Job {dep_0[0].request_id if dep_0[0] else '?'}",
                "Depart -> Job B starts service",
                f"(nq={act_sh2[0]}, ns={act_sh2[1]})",
                "(nq=0, ns=1)",
                match_row4,
            ],
            [
                f"{dep_1[2]:.2f}" if dep_1[2] is not None else "N/A",
                f"Complete: Job {dep_1[0].request_id if dep_1[0] else '?'}",
                "Depart -> Server idle",
                f"(nq={act_sh3[0]}, ns={act_sh3[1]})",
                "(nq=0, ns=0)",
                match_row5,
            ],
            [
                f"{collector.t1:.2f}",
                "Window Close",
                "Integrate remaining duration [4.0, 5.0)",
                f"(nq={collector.saved_nq}, ns={collector.saved_ns})",
                "(nq=0, ns=0)",
                match_row6,
            ],
        ]

        print_table(
            title="V03-1: M/M/1/K Queueing Station Event Timeline Trace (c=1, B=2, mean_service=2.0s)",
            headers=["t (s)", "Event / Job", "Action", "Actual (nq, ns)", "Expected (nq, ns)", "Match"],
            rows=timeline_rows,
            alignments=["center", "left", "left", "center", "center", "center"],
        )

        # -------------------------------------------------------------
        # Table 2: Theoretical vs. Simulation Performance Metrics (Dynamic evaluation)
        # -------------------------------------------------------------
        metrics_rows = [
            ["Server Utilization (U)", f"{U:.4f}", "0.8000", "PASS" if abs(U - 0.8) < 1e-4 else "FAIL"],
            ["Mean Queue Length (Lq)", f"{Lq:.4f}", "0.3000", "PASS" if abs(Lq - 0.3) < 1e-4 else "FAIL"],
            [
                "Mean Response Time (R_completed)",
                f"{R_completed:.4f} s" if R_completed is not None else "N/A",
                "2.7500 s",
                "PASS" if (R_completed is not None and abs(R_completed - 2.75) < 1e-4) else "FAIL",
            ],
            ["Drop Probability (P_drop)", f"{P_drop:.4f} (1/3)", "0.3333 (1/3)", "PASS" if abs(P_drop - 1.0/3.0) < 1e-4 else "FAIL"],
            ["System Throughput (X)", f"{X:.4f} jobs/s", "0.4000 jobs/s", "PASS" if abs(X - 0.4) < 1e-4 else "FAIL"],
            [
                "Waiting Time W(A)",
                f"{waiting_time_a:.4f} s" if waiting_time_a is not None else "N/A",
                "0.0000 s",
                "PASS" if (waiting_time_a is not None and abs(waiting_time_a - 0.0) < 1e-4) else "FAIL",
            ],
            [
                "Waiting Time W(B)",
                f"{waiting_time_b:.4f} s" if waiting_time_b is not None else "N/A",
                "1.5000 s",
                "PASS" if (waiting_time_b is not None and abs(waiting_time_b - 1.5) < 1e-4) else "FAIL",
            ],
        ]

        print_table(
            title="V03-2: M/M/1/K Theoretical vs. Simulation Performance Metrics over [0.0, 5.0)",
            headers=["Metric", "Actual", "Expected", "Status"],
            rows=metrics_rows,
            alignments=["left", "center", "center", "center"],
        )

        # -------------------------------------------------------------
        # Assertions executed after tables have been output
        # -------------------------------------------------------------
        self.assertEqual(stats["status"], "COMPLETED")

        # 1. Verify Job A
        self.assertEqual(vis_a.start_t, 0.0)
        self.assertEqual(vis_a.complete_t, 2.0)
        self.assertEqual(waiting_time_a, 0.0)

        # 2. Verify Job B
        self.assertEqual(vis_b.start_t, 2.0)
        self.assertEqual(vis_b.complete_t, 4.0)
        self.assertEqual(waiting_time_b, 1.5)

        # 3. Verify Job C (Rejected at 1.0s)
        self.assertEqual(len(collector.rejections), 1)
        self.assertIsNotNone(rej_req)
        self.assertEqual(rej_req.request_id, "C")
        self.assertEqual(rej_t, 1.0)
        self.assertFalse(rej_vis.accepted)

        # 4. Verify Departures
        self.assertEqual(len(collector.departures), 2)
        self.assertIsNotNone(dep_0[0])
        self.assertEqual(dep_0[0].request_id, "A")
        self.assertEqual(dep_0[2], 2.0)
        self.assertIsNotNone(dep_1[0])
        self.assertEqual(dep_1[0].request_id, "B")
        self.assertEqual(dep_1[2], 4.0)

        # 5. Hand-calculated Metrics Verification over [0.0, 5.0)
        self.assertAlmostEqual(U, 0.8, places=4, msg="Server utilization U must be exactly 0.8")
        self.assertAlmostEqual(Lq, 0.3, places=4, msg="Mean queue length Lq must be exactly 0.3")
        self.assertIsNotNone(R_completed)
        self.assertAlmostEqual(R_completed, 2.75, places=4, msg="Mean response time must be 2.75s")
        self.assertAlmostEqual(P_drop, 1.0 / 3.0, places=4, msg="Drop probability must be 1/3")
        self.assertAlmostEqual(X, 0.4, places=4, msg="Throughput X must be 0.4 jobs/s")


if __name__ == "__main__":
    unittest.main()
