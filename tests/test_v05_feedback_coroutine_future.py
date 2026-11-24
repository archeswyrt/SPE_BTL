# Copyright (C) 2026 SPE Team - Dev 2
"""
Test N01: Ingress-CPU-Disk network with feedback, coroutine + Future.

Design under test: every request runs as one ``async def`` flow. At each
station visit it awaits the admission ACK, then sleeps on Future[(request_id, visit_id)].
The station frees its server before calling on_departure, which does ``set_result`` and
wakes the coroutine, which then loops on to the next station (Disk -> CPU feedback).
Awaiting only suspends that coroutine; every other event keeps being processed.

Specification:
- Ingress: c = 1, B = 20, fixed service 1.0 s.
- CPU    : c = 2, B = 40, fixed service 2.0 s (baseline server count).
- Disk   : c = 1, B = 20, fixed service 3.0 s.
- External arrivals (scripted): A (R1) at t = 0.0 s and B (R2) at t = 3.5 s.
- Routing is forced (hook, not random):
    - A goes to Disk after its FIRST CPU visit, then back to CPU, then to Sink.
    - B goes to Sink after its only CPU visit.
- Zero routing delay and zero ACK delay (baseline configuration), no deadline.

Hand-calculated timeline:
    A: Ingress [0, 1) -> CPU [1, 3) -> Disk [3, 6) -> CPU [6, 8) -> Sink at 8.0
    B: Ingress [3.5, 4.5) -> CPU [4.5, 6.5) -> Sink at 6.5
    - t = 3.0: A leaves the CPU, its CPU server is released and A sleeps on the Future of
      visit R1-Disk-1.
    - t = 4.5: B reaches the CPU while A is still sleeping; a CPU server is free, so B is
      served immediately (non-blocking: A's wait does not hold the CPU or the dispatcher).
    - t = 6.0: the Disk finishes, set_result wakes A exactly once, A re-enters the CPU as
      a NEW visit (R1-CPU-2); one CPU server is still busy with B, so ns_CPU = 2.
    - A completes at t = 8.0 (response time 8.0 s), B completes at t = 6.5 s (response 3.0 s).
    - Visits: Ingress 2 (R1-Ingress-1, R2-Ingress-1), CPU 3 (R1-CPU-1, R2-CPU-1, R1-CPU-2,
      two distinct visit_ids for A), Disk 1 (R1-Disk-1).
    - Operations (every transfer): A makes 5 (Ingress, CPU, Disk, CPU, Sink), B makes 3
      (Ingress, CPU, Sink) -> 8 operations, all SUCCEEDED with accepted = True.
    - Future[(req_id, visit_id)]: one per station visit = 6, each resolved exactly once.
- Hand-calculated metrics over window [0.0, 10.0):
    - Ingress: ns = 1 on [0, 1) and [3.5, 4.5) -> integral 2.0 -> U = 2.0 / (1 * 10) = 0.2.
    - CPU    : ns = 1 on [1, 3), 0 on [3, 4.5), 1 on [4.5, 6), 2 on [6, 6.5), 1 on [6.5, 8)
               -> integral 2 + 0 + 1.5 + 1 + 1.5 = 6.0 -> U = 6.0 / (2 * 10) = 0.3.
    - Disk   : ns = 1 on [3, 6) -> integral 3.0 -> U = 3.0 / (1 * 10) = 0.3.
    - No job ever waits: Lq = 0 at every station.
    - Mean response time of completed requests: R = (8.0 + 3.0) / 2 = 5.5 s.
    - P_drop = 0 / 2 = 0; X_net = 2 completions / 10 s = 0.2 request/s.
    - N_external = 2 = N_completed (2) + N_dropped (0) + N_unfinished (0).
"""

import random
import unittest

from contracts import Request, Visit
from network import NetworkStreams, build_network
from tests.trace_utils import print_table


class ScriptedGaps:
    """Stand-in for the arrival random stream: returns pre-defined inter-arrival gaps."""

    def __init__(self, gaps):
        self._it = iter(gaps)

    def expovariate(self, lambd: float) -> float:
        # After the script is exhausted the next arrival lies far beyond t_end.
        return next(self._it, 1e9)


class NetworkMetricCollectorN01:
    """Time-weighted per-station metrics over [t0, t1), following the Dev 3 contract."""

    NODES = ("Ingress", "CPU", "Disk")

    def __init__(self, t0: float = 0.0, t1: float = 10.0):
        self.t0 = t0
        self.t1 = t1
        self.last_t = {n: t0 for n in self.NODES}
        self.saved = {n: (0, 0) for n in self.NODES}          # (nq, ns)
        self.integral_ns = {n: 0.0 for n in self.NODES}
        self.integral_nq = {n: 0.0 for n in self.NODES}
        self.state_history: list[tuple[str, float, int, int]] = []   # (node, t, nq, ns)
        self.externals: list[tuple[str, float]] = []
        self.terminals: list[tuple[str, str, float]] = []     # (request_id, outcome, t)

    def on_state_change(self, node: str, t: float, new_nq: int, new_ns: int) -> None:
        self.state_history.append((node, t, new_nq, new_ns))
        if t > self.last_t[node]:
            start = max(self.last_t[node], self.t0)
            end = min(t, self.t1)
            if end > start:
                nq, ns = self.saved[node]
                self.integral_ns[node] += ns * (end - start)
                self.integral_nq[node] += nq * (end - start)
            self.last_t[node] = t
        self.saved[node] = (new_nq, new_ns)

    def on_external_arrival(self, request: Request, t: float) -> None:
        self.externals.append((request.request_id, t))

    def on_terminal(self, request: Request, t: float) -> None:
        self.terminals.append((request.request_id, request.outcome, t))

    def close_window(self) -> None:
        for node in self.NODES:
            if self.t1 > self.last_t[node]:
                nq, ns = self.saved[node]
                duration = self.t1 - max(self.last_t[node], self.t0)
                self.integral_ns[node] += ns * duration
                self.integral_nq[node] += nq * duration
                self.last_t[node] = self.t1

    def ns_after(self, node: str, t: float):
        """Number of busy servers right after the last state change at or before t."""
        value = None
        for n, ts, _nq, ns in self.state_history:
            if n == node and ts <= t:
                value = ns
        return value


class TestN01FeedbackCoroutineFuture(unittest.TestCase):
    """Network with one coroutine per request, Future[(req_id, visit_id)], feedback via Disk."""

    def test_feedback_with_coroutine_future(self):
        collector = NetworkMetricCollectorN01(t0=0.0, t1=10.0)

        def const(value):
            return lambda: value

        streams = NetworkStreams(
            arrival=ScriptedGaps([0.0, 3.5]),   # A at 0.0, B at 0.0 + 3.5
            routing=random.Random(0),            # unused: routing is forced below
            ingress=random.Random(1), cpu=random.Random(2), disk=random.Random(3),
        )
        net = build_network(
            alpha=1.0,                           # only needs to be > 0; gaps are scripted
            p=0.3,                               # irrelevant: force_route decides
            ingress=(1, 20, 1.0), cpu=(2, 40, 2.0), disk=(1, 20, 3.0),
            end_time=10.0,
            streams=streams,
            metrics=collector,
            service_time_generators={"Ingress": const(1.0), "CPU": const(2.0), "Disk": const(3.0)},
            force_route=lambda req, cpu_visit_index: req.request_id == "R1" and cpu_visit_index == 1,
            record_history=True,
        )
        result = net.run()
        collector.close_window()

        # -------------------------------------------------------------
        # Extract everything BEFORE assertions (so the tables always print)
        # -------------------------------------------------------------
        router = net.router
        visits = {v.visit_id: v for v in router.history_visits}
        v = lambda vid: visits.get(vid, Visit(request_id="?", visit_id=vid, node="?", arrival_t=float("nan")))
        a_ing, a_cpu1, a_disk, a_cpu2 = (v("R1-Ingress-1"), v("R1-CPU-1"), v("R1-Disk-1"), v("R1-CPU-2"))
        b_ing, b_cpu = v("R2-Ingress-1"), v("R2-CPU-1")

        terminal = {rid: (outcome, t) for rid, outcome, t in collector.terminals}
        t_a = terminal.get("R1", (None, None))
        t_b = terminal.get("R2", (None, None))

        flow = router.flow_log
        a_waits = [e for e in flow if e[1] == "R1" and e[2] == "wait" and e[3] == "R1-Disk-1"]
        a_resumes = [e for e in flow if e[1] == "R1" and e[2] == "resume" and e[3] == "R1-Disk-1"]
        wait_t = a_waits[0][0] if a_waits else None
        resume_t = a_resumes[0][0] if a_resumes else None

        cpu_ns_history = [(t, ns) for node, t, _nq, ns in collector.state_history if node == "CPU"]
        node_visit_ids = {
            node: sorted(vid for vid, vis in visits.items() if vis.node == node)
            for node in ("Ingress", "CPU", "Disk")
        }

        T = collector.t1 - collector.t0
        U = {n: collector.integral_ns[n] / (c * T) for n, c in (("Ingress", 1), ("CPU", 2), ("Disk", 1))}
        Lq = {n: collector.integral_nq[n] / T for n in collector.NODES}
        resp = [
            (t_a[1] - 0.0) if t_a[1] is not None else None,    # A: external arrival at 0.0
            (t_b[1] - 3.5) if t_b[1] is not None else None,    # B: external arrival at 3.5
        ]
        R_completed = (sum(resp) / 2.0) if all(r is not None for r in resp) else None
        X = sum(1 for _, o, t in collector.terminals if o == "COMPLETED" and collector.t0 <= t < collector.t1) / T
        P_drop = result.dropped / result.external if result.external else None
        ops = result.operations

        # -------------------------------------------------------------
        # Table 1: Event timeline trace
        # -------------------------------------------------------------
        def row(t_value, event, action, actual_text, expected_text, ok):
            return [f"{t_value:.2f}", event, action, actual_text, expected_text, "PASS" if ok else "FAIL"]

        timeline_rows = [
            row(0.0, "Arrival: A (R1)", "Ingress service starts",
                f"start={a_ing.start_t}", "start=0.0", a_ing.start_t == 0.0),
            row(1.0, "Complete Ingress: A", "A -> CPU, visit R1-CPU-1 starts at once",
                f"arr={a_cpu1.arrival_t}, start={a_cpu1.start_t}", "arr=1.0, start=1.0",
                (a_cpu1.arrival_t, a_cpu1.start_t) == (1.0, 1.0)),
            row(3.0, "Complete CPU: A", "CPU server released; A sleeps on Future (R1-Disk-1)",
                f"CPU ns={collector.ns_after('CPU', 3.0)}, wait@{wait_t}, Disk start={a_disk.start_t}",
                "CPU ns=0, wait@3.0, Disk start=3.0",
                collector.ns_after("CPU", 3.0) == 0 and wait_t == 3.0 and a_disk.start_t == 3.0),
            row(3.5, "Arrival: B (R2)", "Ingress service starts (A still sleeping)",
                f"start={b_ing.start_t}", "start=3.5", b_ing.start_t == 3.5),
            row(4.5, "B reaches CPU", "B served at once while A waits (non-blocking)",
                f"start={b_cpu.start_t}, CPU ns={collector.ns_after('CPU', 4.5)}",
                "start=4.5, CPU ns=1",
                b_cpu.start_t == 4.5 and collector.ns_after("CPU", 4.5) == 1
                and wait_t is not None and resume_t is not None and wait_t < b_cpu.start_t < resume_t),
            row(6.0, "Disk done: set_result()", "A wakes once; feedback R1-CPU-2 starts at once",
                f"Disk done={a_disk.complete_t}, CPU2 start={a_cpu2.start_t}, resumes={len(a_resumes)}, CPU ns={collector.ns_after('CPU', 6.0)}",
                "Disk done=6.0, CPU2 start=6.0, resumes=1, CPU ns=2",
                (a_disk.complete_t, a_cpu2.start_t, len(a_resumes), collector.ns_after("CPU", 6.0)) == (6.0, 6.0, 1, 2)),
            row(6.5, "Complete CPU: B", "B -> Sink, R2 COMPLETED",
                f"{t_b}", "('COMPLETED', 6.5)", t_b == ("COMPLETED", 6.5)),
            row(8.0, "Complete CPU: A", "A -> Sink, R1 COMPLETED",
                f"{t_a}", "('COMPLETED', 8.0)", t_a == ("COMPLETED", 8.0)),
            row(10.0, "Window close", "All stations idle",
                f"saved={ {n: collector.saved[n] for n in collector.NODES} }",
                "all (nq=0, ns=0)", all(collector.saved[n] == (0, 0) for n in collector.NODES)),
        ]
        print_table(
            title="N01-1: network with coroutine + Future, event timeline trace, feedback via Disk",
            headers=["t (s)", "Event", "Action", "Actual", "Expected", "Match"],
            rows=timeline_rows,
            alignments=["center", "left", "left", "left", "left", "center"],
        )

        # -------------------------------------------------------------
        # Table 2: Hand-calculated vs simulated metrics and identifiers
        # -------------------------------------------------------------
        def near(x, y):
            return x is not None and abs(x - y) < 1e-9

        metrics_rows = [
            ["U_Ingress", f"{U['Ingress']:.4f}", "0.2000", "PASS" if near(U["Ingress"], 0.2) else "FAIL"],
            ["U_CPU (per server)", f"{U['CPU']:.4f}", "0.3000", "PASS" if near(U["CPU"], 0.3) else "FAIL"],
            ["U_Disk", f"{U['Disk']:.4f}", "0.3000", "PASS" if near(U["Disk"], 0.3) else "FAIL"],
            ["Lq (all stations)", f"{[Lq[n] for n in collector.NODES]}", "[0.0, 0.0, 0.0]",
             "PASS" if all(near(Lq[n], 0.0) for n in collector.NODES) else "FAIL"],
            ["R_completed", f"{R_completed} s", "5.5 s", "PASS" if near(R_completed, 5.5) else "FAIL"],
            ["X_net", f"{X:.4f} req/s", "0.2000 req/s", "PASS" if near(X, 0.2) else "FAIL"],
            ["P_drop", f"{P_drop}", "0.0", "PASS" if P_drop == 0.0 else "FAIL"],
            ["N_ext = N_comp + N_drop + N_unf",
             f"{result.external} = {result.completed} + {result.dropped} + {result.unfinished}",
             "2 = 2 + 0 + 0",
             "PASS" if (result.external, result.completed, result.dropped, result.unfinished) == (2, 2, 0, 0) and result.conserved else "FAIL"],
            ["CPU visit ids", f"{node_visit_ids['CPU']}", "['R1-CPU-1', 'R1-CPU-2', 'R2-CPU-1']",
             "PASS" if node_visit_ids["CPU"] == ["R1-CPU-1", "R1-CPU-2", "R2-CPU-1"] else "FAIL"],
            ["Disk / Ingress visit ids", f"{node_visit_ids['Disk']} / {node_visit_ids['Ingress']}",
             "['R1-Disk-1'] / ['R1-Ingress-1', 'R2-Ingress-1']",
             "PASS" if (node_visit_ids["Disk"], node_visit_ids["Ingress"]) == (["R1-Disk-1"], ["R1-Ingress-1", "R2-Ingress-1"]) else "FAIL"],
            ["Operations created / succeeded / pending",
             f"{ops['created']} / {ops['succeeded']} / {ops['pending']}", "8 / 8 / 0",
             "PASS" if (ops["created"], ops["succeeded"], ops["pending"]) == (8, 8, 0) else "FAIL"],
            ["ACKs accepted / rejected", f"{ops['acks_accepted']} / {ops['acks_rejected']}", "8 / 0",
             "PASS" if (ops["acks_accepted"], ops["acks_rejected"]) == (8, 0) else "FAIL"],
            ["Station-visit Futures left unresolved", f"{len(router._completions)}", "0",
             "PASS" if len(router._completions) == 0 else "FAIL"],
            ["Request coroutines started / finished", f"{router.flows_started} / {router.flows_finished}", "2 / 2",
             "PASS" if (router.flows_started, router.flows_finished) == (2, 2) else "FAIL"],
        ]
        print_table(
            title="N01-2: Hand-calculated vs simulated metrics over [0.0, 10.0)",
            headers=["Metric", "Actual", "Expected", "Status"],
            rows=metrics_rows,
            alignments=["left", "left", "left", "center"],
        )

        # -------------------------------------------------------------
        # Assertions
        # -------------------------------------------------------------
        self.assertEqual(result.status, "COMPLETED", msg=result.failure_reason)

        # 1. Hand-calculated service intervals of every visit
        self.assertEqual((a_ing.start_t, a_ing.complete_t), (0.0, 1.0))
        self.assertEqual((a_cpu1.start_t, a_cpu1.complete_t), (1.0, 3.0))
        self.assertEqual((a_disk.start_t, a_disk.complete_t), (3.0, 6.0))
        self.assertEqual((a_cpu2.start_t, a_cpu2.complete_t), (6.0, 8.0))
        self.assertEqual((b_ing.start_t, b_ing.complete_t), (3.5, 4.5))
        self.assertEqual((b_cpu.start_t, b_cpu.complete_t), (4.5, 6.5))

        # 2. Outcomes: A completes at 8.0 s (1 + 2 + 3 + 2), B at 6.5 s; request_id preserved
        self.assertEqual(t_a, ("COMPLETED", 8.0))
        self.assertEqual(t_b, ("COMPLETED", 6.5))
        self.assertEqual(router.history_requests["R1"].terminal_t, 8.0)
        self.assertTrue(all(vis.accepted for vis in visits.values()))

        # 3. Visit identifiers: A has two distinct CPU visits, one Disk visit
        self.assertEqual(node_visit_ids["CPU"], ["R1-CPU-1", "R1-CPU-2", "R2-CPU-1"])
        self.assertEqual(node_visit_ids["Disk"], ["R1-Disk-1"])
        self.assertEqual(node_visit_ids["Ingress"], ["R1-Ingress-1", "R2-Ingress-1"])
        self.assertNotEqual(a_cpu1.visit_id, a_cpu2.visit_id)

        # 4. Non-blocking evidence: A sleeps on the Disk Future during [3.0, 6.0],
        #    the CPU server is released at 3.0, B is served on the CPU in between,
        #    and A is resumed exactly once at 6.0.
        self.assertEqual(wait_t, 3.0)
        self.assertEqual(resume_t, 6.0)
        self.assertEqual(len(a_waits), 1)
        self.assertEqual(len(a_resumes), 1)
        self.assertEqual(collector.ns_after("CPU", 3.0), 0)
        self.assertTrue(wait_t < b_cpu.start_t < resume_t)
        self.assertEqual(cpu_ns_history, [(1.0, 1), (3.0, 0), (4.5, 1), (6.0, 2), (6.5, 1), (8.0, 0)])

        # 5. Hand-calculated metrics over [0.0, 10.0)
        self.assertAlmostEqual(U["Ingress"], 0.2, places=9)
        self.assertAlmostEqual(U["CPU"], 0.3, places=9)
        self.assertAlmostEqual(U["Disk"], 0.3, places=9)
        for node in collector.NODES:
            self.assertAlmostEqual(Lq[node], 0.0, places=9)
        self.assertAlmostEqual(R_completed, 5.5, places=9)
        self.assertAlmostEqual(X, 0.2, places=9)
        self.assertEqual(P_drop, 0.0)

        # 6. Request conservation and no business request created by control traffic
        self.assertEqual((result.external, result.completed, result.dropped, result.unfinished), (2, 2, 0, 0))
        self.assertTrue(result.conserved)
        self.assertEqual(collector.externals, [("R1", 0.0), ("R2", 3.5)])

        # 7. Operations and Futures: every transfer is one SUCCEEDED operation, nothing left over
        self.assertEqual((ops["created"], ops["succeeded"], ops["pending"]), (8, 8, 0))
        self.assertEqual((ops["acks_accepted"], ops["acks_rejected"]), (8, 0))
        self.assertEqual((ops["duplicate_acks"], ops["timeouts_scheduled"], ops["failed"]), (0, 0, 0))
        self.assertEqual(len(router._completions), 0)
        self.assertEqual((router.flows_started, router.flows_finished), (2, 2))


if __name__ == "__main__":
    unittest.main()