# Copyright (C) 2026 SPE Team - Dev 2
"""
Test N03: rejection on the Disk -> CPU feedback branch (CPU full), coroutine + Future.

Rule under test (Phase 1, section 1.5): a request that reaches a full station is DROPPED at
admission, leaves the whole network and is never retried. This applies to feedback arrivals
exactly as to external ones. The sending station (Disk) has already released its server
before the router sees the departure, and the decision does not depend on any Future or ACK.

Specification:
- Ingress: c = 1, B = 5, fixed service 1.0 s.
- CPU    : c = 1, B = 1 (no waiting room at all), service 1.0 s for the first job (A) and
           10.0 s for the second job (B).
- Disk   : c = 1, B = 5, fixed service 3.0 s.
- External arrivals (scripted): A (R1) at t = 0.0 s and B (R2) at t = 1.5 s.
- Routing is forced: A goes to Disk after its first CPU visit; B goes to Sink.
- Zero routing / ACK delay, no deadline, window [0.0, 15.0).

Hand-calculated timeline:
    A: Ingress [0, 1) -> CPU [1, 2) -> Disk [2, 5) -> back to CPU at 5.0
    B: Ingress [1.5, 2.5) -> reaches the CPU at 2.5: the CPU is empty since A left at 2.0,
       so B is admitted and served for 10 s: CPU [2.5, 12.5) -> Sink at 12.5.
    - t = 5.0: the Disk finishes A. Its server is released first (Disk ns = 0). A re-enters the
      CPU as a NEW visit (R1-CPU-2); the CPU holds B (n = 1 = B) -> REJECTED.
      R1 is finalised as DROPPED at t = 5.0 with drop_node = 'CPU', exactly once, no retry.
      The CPU state does not change (no state change at t = 5.0).
    - Visits: Ingress 2, CPU 3 (R1-CPU-1, R2-CPU-1 admitted; R1-CPU-2 rejected, never started),
      Disk 1.
    - Operations: A makes 4 (Ingress, CPU, Disk, CPU), B makes 3 (Ingress, CPU, Sink) -> 7,
      all SUCCEEDED (a rejection is a successful communication): 6 accepted, 1 rejected.
      A makes no operation to the Sink and none after the rejection.
    - Future[(req_id, visit_id)]: the rejected visit's Future is released without a service
      result; no Future is left unresolved.
- Hand-calculated metrics over window [0.0, 15.0):
    - Ingress: ns = 1 on [0, 1) and [1.5, 2.5) -> 2.0 -> U = 2 / 15.
    - CPU    : ns = 1 on [1, 2) and [2.5, 12.5) -> 11.0 -> U = 11 / 15; nq = 0 throughout.
    - Disk   : ns = 1 on [2, 5) -> 3.0 -> U = 3 / 15 = 0.2.
    - Two different drop ratios (computed separately, Phase 1 section 1.2):
        per-station Pb_CPU = rejected arrivals / all arrivals at the CPU = 1 / 3
        end-to-end P_drop  = dropped requests / external arrivals       = 1 / 2.
    - Only B completes: R_completed = 12.5 - 1.5 = 11.0 s, X_net = 1 / 15 request/s.
    - N_external = 2 = N_completed (1) + N_dropped (1) + N_unfinished (0).
"""

import random
import unittest

from contracts import Visit
from network import NetworkStreams, build_network
from tests.net_test_utils import NetworkMetricCollector, ScriptedGaps
from tests.trace_utils import print_table


class TestN03FeedbackDropCpuFull(unittest.TestCase):
    """Feedback arrival refused by a full CPU: DROPPED once, Disk freed, no retry."""

    def test_feedback_rejected_when_cpu_full(self):
        collector = NetworkMetricCollector(t0=0.0, t1=15.0)

        cpu_service = iter([1.0, 10.0])          # first CPU job (A) 1 s, second CPU job (B) 10 s

        streams = NetworkStreams(
            arrival=ScriptedGaps([0.0, 1.5]),    # A at 0.0, B at 0.0 + 1.5
            routing=random.Random(0),            # unused: routing is forced below
            ingress=random.Random(1), cpu=random.Random(2), disk=random.Random(3),
        )
        net = build_network(
            alpha=1.0,                           # only needs to be > 0; gaps are scripted
            p=0.3,                               # irrelevant: force_route decides
            ingress=(1, 5, 1.0), cpu=(1, 1, 1.0), disk=(1, 5, 3.0),
            end_time=15.0,
            streams=streams,
            metrics=collector,
            service_time_generators={
                "Ingress": lambda: 1.0,
                "CPU": lambda: next(cpu_service),
                "Disk": lambda: 3.0,
            },
            force_route=lambda req, cpu_visit_index: req.request_id == "R1" and cpu_visit_index == 1,
            record_history=True,
        )
        result = net.run()
        collector.close_window()

        # -------------------------------------------------------------
        # Extract everything BEFORE assertions (so the tables always print)
        # -------------------------------------------------------------
        router = net.router
        visits = {vis.visit_id: vis for vis in router.history_visits}
        v = lambda vid: visits.get(vid, Visit(request_id="?", visit_id=vid, node="?", arrival_t=float("nan")))
        a_ing, a_cpu1, a_disk, a_cpu2 = v("R1-Ingress-1"), v("R1-CPU-1"), v("R1-Disk-1"), v("R1-CPU-2")
        b_ing, b_cpu = v("R2-Ingress-1"), v("R2-CPU-1")

        r1 = router.history_requests.get("R1")
        terminal = {}
        for rid, outcome, t in collector.terminals:
            terminal.setdefault(rid, []).append((outcome, t))
        t_a = terminal.get("R1", [])
        t_b = terminal.get("R2", [])

        node_visit_ids = {
            node: sorted(vid for vid, vis in visits.items() if vis.node == node)
            for node in ("Ingress", "CPU", "Disk")
        }
        cpu_arrivals = [vis for vis in visits.values() if vis.node == "CPU"]
        cpu_rejected = [vis for vis in cpu_arrivals if not vis.accepted]
        Pb_cpu = len(cpu_rejected) / len(cpu_arrivals) if cpu_arrivals else None

        a_log = [e for e in router.flow_log if e[1] == "R1"]
        a_sends_after_drop = [e for e in a_log if e[2] == "send" and e[0] > 5.0]
        a_end = [e for e in a_log if e[2] == "end"]

        T = collector.t1 - collector.t0
        U = {n: collector.integral_ns[n] / (1 * T) for n in collector.NODES}
        Lq = {n: collector.integral_nq[n] / T for n in collector.NODES}
        completed_resp = [t - 1.5 for rid, o, t in collector.terminals if rid == "R2" and o == "COMPLETED"]
        R_completed = (sum(completed_resp) / len(completed_resp)) if completed_resp else None
        X = sum(1 for _, o, t in collector.terminals if o == "COMPLETED" and collector.t0 <= t < collector.t1) / T
        P_drop = result.dropped / result.external if result.external else None
        ops = result.operations
        cpu_changes_at_5 = [h for h in collector.history_of("CPU") if h[0] == 5.0]

        # -------------------------------------------------------------
        # Table 1: Event timeline trace
        # -------------------------------------------------------------
        def row(t_value, event, action, actual_text, expected_text, ok):
            return [f"{t_value:.2f}", event, action, actual_text, expected_text, "PASS" if ok else "FAIL"]

        timeline_rows = [
            row(0.0, "Arrival: A (R1)", "Ingress service starts",
                f"start={a_ing.start_t}", "start=0.0", a_ing.start_t == 0.0),
            row(1.0, "Complete Ingress: A", "A -> CPU, R1-CPU-1 starts (CPU now full, n=1=B)",
                f"start={a_cpu1.start_t}, CPU state={collector.state_after('CPU', 1.0)}",
                "start=1.0, CPU state=(0, 1)",
                a_cpu1.start_t == 1.0 and collector.state_after("CPU", 1.0) == (0, 1)),
            row(1.5, "Arrival: B (R2)", "Ingress service starts",
                f"start={b_ing.start_t}", "start=1.5", b_ing.start_t == 1.5),
            row(2.0, "Complete CPU: A", "CPU freed; A -> Disk and sleeps on its Future",
                f"CPU ns={collector.state_after('CPU', 2.0)[1]}, Disk start={a_disk.start_t}",
                "CPU ns=0, Disk start=2.0",
                collector.state_after("CPU", 2.0)[1] == 0 and a_disk.start_t == 2.0),
            row(2.5, "B reaches CPU", "CPU empty: B admitted, 10 s service",
                f"accepted={b_cpu.accepted}, start={b_cpu.start_t}",
                "accepted=True, start=2.5", b_cpu.accepted is True and b_cpu.start_t == 2.5),
            row(5.0, "Complete Disk: A", "Disk server freed BEFORE the feedback reaches the CPU",
                f"Disk state={collector.state_after('Disk', 5.0)}, done={a_disk.complete_t}",
                "Disk state=(0, 0), done=5.0",
                collector.state_after("Disk", 5.0) == (0, 0) and a_disk.complete_t == 5.0),
            row(5.0, "Feedback R1-CPU-2 arrives", "CPU full (n=1=B): REJECTED, R1 DROPPED at CPU",
                f"accepted={a_cpu2.accepted}, start={a_cpu2.start_t}, "
                f"outcome={getattr(r1, 'outcome', None)}@{getattr(r1, 'terminal_t', None)}, drop_node={getattr(r1, 'drop_node', None)}",
                "accepted=False, start=None, outcome=DROPPED@5.0, drop_node=CPU",
                (a_cpu2.accepted, a_cpu2.start_t) == (False, None) and r1 is not None
                and (r1.outcome, r1.terminal_t, r1.drop_node) == ("DROPPED", 5.0, "CPU")),
            row(5.0, "CPU state at rejection", "No state change at t = 5.0, B keeps its server",
                f"changes at 5.0={cpu_changes_at_5}, state={collector.state_after('CPU', 5.0)}",
                "changes at 5.0=[], state=(0, 1)",
                cpu_changes_at_5 == [] and collector.state_after("CPU", 5.0) == (0, 1)),
            row(12.5, "Complete CPU: B", "B -> Sink (COMPLETED); R1 is NOT retried",
                f"{t_b}, R1 sends after 5.0={len(a_sends_after_drop)}",
                "[('COMPLETED', 12.5)], R1 sends after 5.0=0",
                t_b == [("COMPLETED", 12.5)] and not a_sends_after_drop),
            row(15.0, "Window close", "All stations idle",
                f"saved={ {n: collector.saved[n] for n in collector.NODES} }",
                "all (nq=0, ns=0)", all(collector.saved[n] == (0, 0) for n in collector.NODES)),
        ]
        print_table(
            title="N03-1: feedback Disk -> CPU rejected by a full CPU, event timeline trace",
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
            ["U_Ingress", f"{U['Ingress']:.4f}", f"{2/15:.4f}", "PASS" if near(U["Ingress"], 2 / 15) else "FAIL"],
            ["U_CPU", f"{U['CPU']:.4f}", f"{11/15:.4f}", "PASS" if near(U["CPU"], 11 / 15) else "FAIL"],
            ["U_Disk", f"{U['Disk']:.4f}", "0.2000", "PASS" if near(U["Disk"], 0.2) else "FAIL"],
            ["Lq (all stations)", f"{[Lq[n] for n in collector.NODES]}", "[0.0, 0.0, 0.0]",
             "PASS" if all(near(Lq[n], 0.0) for n in collector.NODES) else "FAIL"],
            ["Pb_CPU = rejected / arrivals at CPU", f"{len(cpu_rejected)} / {len(cpu_arrivals)} = {Pb_cpu:.4f}",
             f"1 / 3 = {1/3:.4f}", "PASS" if near(Pb_cpu, 1 / 3) else "FAIL"],
            ["P_drop = dropped / external", f"{result.dropped} / {result.external} = {P_drop}", "1 / 2 = 0.5",
             "PASS" if near(P_drop, 0.5) else "FAIL"],
            ["R_completed (only B)", f"{R_completed} s", "11.0 s", "PASS" if near(R_completed, 11.0) else "FAIL"],
            ["X_net", f"{X:.4f} req/s", f"{1/15:.4f} req/s", "PASS" if near(X, 1 / 15) else "FAIL"],
            ["N_ext = N_comp + N_drop + N_unf",
             f"{result.external} = {result.completed} + {result.dropped} + {result.unfinished}",
             "2 = 1 + 1 + 0",
             "PASS" if (result.external, result.completed, result.dropped, result.unfinished) == (2, 1, 1, 0) and result.conserved else "FAIL"],
            ["Drops by station", f"{result.drops_by_node}", "{'Ingress': 0, 'CPU': 1, 'Disk': 0}",
             "PASS" if result.drops_by_node == {"Ingress": 0, "CPU": 1, "Disk": 0} else "FAIL"],
            ["CPU visit ids (accepted?)",
             f"{[(vis.visit_id, vis.accepted) for vis in sorted(cpu_arrivals, key=lambda x: x.visit_id)]}",
             "[('R1-CPU-1', True), ('R1-CPU-2', False), ('R2-CPU-1', True)]",
             "PASS" if [(vis.visit_id, vis.accepted) for vis in sorted(cpu_arrivals, key=lambda x: x.visit_id)]
             == [("R1-CPU-1", True), ("R1-CPU-2", False), ("R2-CPU-1", True)] else "FAIL"],
            ["Operations created / succeeded / pending",
             f"{ops['created']} / {ops['succeeded']} / {ops['pending']}", "7 / 7 / 0",
             "PASS" if (ops["created"], ops["succeeded"], ops["pending"]) == (7, 7, 0) else "FAIL"],
            ["ACKs accepted / rejected", f"{ops['acks_accepted']} / {ops['acks_rejected']}", "6 / 1",
             "PASS" if (ops["acks_accepted"], ops["acks_rejected"]) == (6, 1) else "FAIL"],
            ["Station-visit Futures left unresolved", f"{len(router._completions)}", "0",
             "PASS" if len(router._completions) == 0 else "FAIL"],
            ["Request coroutines started / finished", f"{router.flows_started} / {router.flows_finished}", "2 / 2",
             "PASS" if (router.flows_started, router.flows_finished) == (2, 2) else "FAIL"],
        ]
        print_table(
            title="N03-2: Hand-calculated vs simulated metrics over [0.0, 15.0)",
            headers=["Metric", "Actual", "Expected", "Status"],
            rows=metrics_rows,
            alignments=["left", "left", "left", "center"],
        )

        # -------------------------------------------------------------
        # Assertions
        # -------------------------------------------------------------
        self.assertEqual(result.status, "COMPLETED", msg=result.failure_reason)

        # 1. Service intervals of the admitted visits
        self.assertEqual((a_ing.start_t, a_ing.complete_t), (0.0, 1.0))
        self.assertEqual((a_cpu1.start_t, a_cpu1.complete_t), (1.0, 2.0))
        self.assertEqual((a_disk.start_t, a_disk.complete_t), (2.0, 5.0))
        self.assertEqual((b_ing.start_t, b_ing.complete_t), (1.5, 2.5))
        self.assertEqual((b_cpu.start_t, b_cpu.complete_t), (2.5, 12.5))

        # 2. The feedback visit is refused: new visit_id, arrives at 5.0, never started
        self.assertEqual(a_cpu2.visit_id, "R1-CPU-2")
        self.assertNotEqual(a_cpu2.visit_id, a_cpu1.visit_id)
        self.assertEqual(a_cpu2.arrival_t, 5.0)
        self.assertFalse(a_cpu2.accepted)
        self.assertIsNone(a_cpu2.start_t)
        self.assertIsNone(a_cpu2.complete_t)

        # 3. R1 is DROPPED exactly once, at the CPU, at t = 5.0; its identity is unchanged
        self.assertEqual(r1.request_id, "R1")
        self.assertEqual((r1.outcome, r1.terminal_t, r1.drop_node), ("DROPPED", 5.0, "CPU"))
        self.assertEqual(t_a, [("DROPPED", 5.0)])              # one terminal notification only
        self.assertEqual(t_b, [("COMPLETED", 12.5)])
        self.assertEqual(result.drops_by_node, {"Ingress": 0, "CPU": 1, "Disk": 0})
        self.assertEqual(net.sink.terminal_conflicts, 0)

        # 4. The Disk server was released before the rejection; the CPU state did not change
        self.assertEqual(collector.state_after("Disk", 5.0), (0, 0))
        self.assertEqual(collector.history_of("Disk"), [(2.0, 0, 1), (5.0, 0, 0)])
        self.assertEqual(cpu_changes_at_5, [])
        self.assertEqual(collector.history_of("CPU"),
                         [(1.0, 0, 1), (2.0, 0, 0), (2.5, 0, 1), (12.5, 0, 0)])

        # 5. No retry: R1 never makes another operation after the rejection and never reaches the Sink
        self.assertEqual(a_sends_after_drop, [])
        self.assertEqual([e[3] for e in a_log if e[2] == "send"],
                         ["R1-Ingress-1", "R1-CPU-1", "R1-Disk-1", "R1-CPU-2"])
        self.assertEqual(len(a_end), 1)
        self.assertEqual((a_end[0][0], a_end[0][3]), (5.0, "dropped at CPU"))
        self.assertEqual(node_visit_ids["CPU"], ["R1-CPU-1", "R1-CPU-2", "R2-CPU-1"])
        self.assertEqual(node_visit_ids["Disk"], ["R1-Disk-1"])

        # 6. Hand-calculated metrics over [0.0, 15.0)
        self.assertAlmostEqual(U["Ingress"], 2 / 15, places=9)
        self.assertAlmostEqual(U["CPU"], 11 / 15, places=9)
        self.assertAlmostEqual(U["Disk"], 0.2, places=9)
        for node in collector.NODES:
            self.assertAlmostEqual(Lq[node], 0.0, places=9)
        self.assertAlmostEqual(Pb_cpu, 1 / 3, places=9)         # per-station ratio counts every arrival
        self.assertAlmostEqual(P_drop, 0.5, places=9)           # end-to-end ratio counts requests once
        self.assertAlmostEqual(R_completed, 11.0, places=9)
        self.assertAlmostEqual(X, 1 / 15, places=9)

        # 7. Request conservation
        self.assertEqual((result.external, result.completed, result.dropped, result.unfinished), (2, 1, 1, 0))
        self.assertTrue(result.conserved)

        # 8. A rejection is a successful communication: 7 SUCCEEDED operations (6 accepted, 1 rejected)
        self.assertEqual((ops["created"], ops["succeeded"], ops["pending"]), (7, 7, 0))
        self.assertEqual((ops["acks_accepted"], ops["acks_rejected"]), (6, 1))
        self.assertEqual((ops["duplicate_acks"], ops["timeouts_scheduled"], ops["failed"], ops["cancelled"]),
                         (0, 0, 0, 0))
        self.assertEqual(len(router._completions), 0)          # the rejected visit's Future was released
        self.assertEqual((router.flows_started, router.flows_finished), (2, 2))


if __name__ == "__main__":
    unittest.main()
