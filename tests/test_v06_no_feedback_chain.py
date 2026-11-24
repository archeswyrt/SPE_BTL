# Copyright (C) 2026 SPE Team - Dev 2
"""
Test N02: Ingress-CPU network WITHOUT feedback (p = 0), coroutine + Future.

With p = 0 the CPU never routes to the Disk: every request follows
Source -> Ingress -> CPU -> Sink. This is the first milestone of the team plan
("p = 0 runs end to end and produces correct metrics"). The routing decision is NOT
forced here: the real routing stream is used with p = 0.0, so the test also proves that
p = 0 never sends a request to the Disk.

Specification:
- Ingress: c = 1, B = 5, fixed service 1.0 s.
- CPU    : c = 1, B = 5, fixed service 2.0 s (single server, so a queue can build up).
- Disk   : c = 1, B = 5, fixed service 3.0 s (must stay unused).
- External arrivals (scripted): A (R1) at t = 0.0 s and B (R2) at t = 0.5 s.
- p = 0.0, zero routing / ACK delay, no deadline, window [0.0, 6.0).

Hand-calculated timeline:
    A: Ingress [0, 1), wait 0   -> CPU [1, 3), wait 0   -> Sink at 3.0   (response 3.0 s)
    B: arrives 0.5, waits in the Ingress queue until 1.0 (wait 0.5),
       Ingress [1, 2) -> reaches the CPU at 2.0 while A is served, waits until 3.0 (wait 1.0),
       CPU [3, 5) -> Sink at 5.0                                          (response 4.5 s)
    - Visits: Ingress 2 (R1-Ingress-1, R2-Ingress-1), CPU 2 (R1-CPU-1, R2-CPU-1), Disk 0.
    - Operations: 3 per request (Ingress, CPU, Sink) -> 6, all SUCCEEDED, all accepted.
    - Future[(req_id, visit_id)]: one per station visit = 4, each resolved exactly once.
- Hand-calculated metrics over window [0.0, 6.0):
    - Ingress: ns = 1 on [0, 2) -> integral 2.0 -> U = 2.0 / 6 = 1/3;
               nq = 1 on [0.5, 1) -> integral 0.5 -> Lq = 0.5 / 6 = 1/12.
    - CPU    : ns = 1 on [1, 5) -> integral 4.0 -> U = 4.0 / 6 = 2/3;
               nq = 1 on [2, 3) -> integral 1.0 -> Lq = 1.0 / 6 = 1/6.
    - Disk   : never used -> U = 0 and no state change at all.
    - Station times (accepted visits): E[w_Ingress] = (0 + 0.5) / 2 = 0.25 s,
      E[r_Ingress] = (1.0 + 1.5) / 2 = 1.25 s, E[w_CPU] = (0 + 1.0) / 2 = 0.5 s,
      E[r_CPU] = (2.0 + 3.0) / 2 = 2.5 s.
    - End-to-end: R_completed = (3.0 + 4.5) / 2 = 3.75 s, X_net = 2 / 6 = 1/3 request/s,
      P_drop = 0, N_external = 2 = N_completed (2) + N_dropped (0) + N_unfinished (0).
"""

import random
import unittest

from contracts import Visit
from network import NetworkStreams, build_network
from tests.net_test_utils import NetworkMetricCollector, ScriptedGaps
from tests.trace_utils import print_table


class TestN02NoFeedbackChain(unittest.TestCase):
    """p = 0: Source -> Ingress -> CPU -> Sink with queueing, coroutine + Future."""

    def test_chain_without_feedback(self):
        collector = NetworkMetricCollector(t0=0.0, t1=6.0)

        def const(value):
            return lambda: value

        streams = NetworkStreams(
            arrival=ScriptedGaps([0.0, 0.5]),    # A at 0.0, B at 0.0 + 0.5
            routing=random.Random(0),            # REAL routing stream: p = 0 must never pick Disk
            ingress=random.Random(1), cpu=random.Random(2), disk=random.Random(3),
        )
        net = build_network(
            alpha=1.0,                           # only needs to be > 0; gaps are scripted
            p=0.0,
            ingress=(1, 5, 1.0), cpu=(1, 5, 2.0), disk=(1, 5, 3.0),
            end_time=6.0,
            streams=streams,
            metrics=collector,
            service_time_generators={"Ingress": const(1.0), "CPU": const(2.0), "Disk": const(3.0)},
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
        a_ing, a_cpu = v("R1-Ingress-1"), v("R1-CPU-1")
        b_ing, b_cpu = v("R2-Ingress-1"), v("R2-CPU-1")

        terminal = {rid: (outcome, t) for rid, outcome, t in collector.terminals}
        t_a = terminal.get("R1", (None, None))
        t_b = terminal.get("R2", (None, None))

        node_visit_ids = {
            node: sorted(vid for vid, vis in visits.items() if vis.node == node)
            for node in ("Ingress", "CPU", "Disk")
        }
        wait = lambda vis: vis.start_t - vis.arrival_t
        resp_st = lambda vis: vis.complete_t - vis.arrival_t
        E_w_ing, E_r_ing = (wait(a_ing) + wait(b_ing)) / 2, (resp_st(a_ing) + resp_st(b_ing)) / 2
        E_w_cpu, E_r_cpu = (wait(a_cpu) + wait(b_cpu)) / 2, (resp_st(a_cpu) + resp_st(b_cpu)) / 2

        T = collector.t1 - collector.t0
        U = {n: collector.integral_ns[n] / (1 * T) for n in collector.NODES}
        Lq = {n: collector.integral_nq[n] / T for n in collector.NODES}
        resp = [
            (t_a[1] - 0.0) if t_a[1] is not None else None,
            (t_b[1] - 0.5) if t_b[1] is not None else None,
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
            row(0.5, "Arrival: B (R2)", "Ingress busy: B waits in FCFS queue",
                f"arr={b_ing.arrival_t}, state={collector.state_after('Ingress', 0.5)}",
                "arr=0.5, state=(1, 1)",
                b_ing.arrival_t == 0.5 and collector.state_after("Ingress", 0.5) == (1, 1)),
            row(1.0, "Complete Ingress: A", "A -> CPU (starts at once); B starts Ingress",
                f"A CPU start={a_cpu.start_t}, B Ingress start={b_ing.start_t}",
                "A CPU start=1.0, B Ingress start=1.0",
                (a_cpu.start_t, b_ing.start_t) == (1.0, 1.0)),
            row(2.0, "Complete Ingress: B", "B reaches busy CPU: waits in queue",
                f"arr={b_cpu.arrival_t}, CPU state={collector.state_after('CPU', 2.0)}",
                "arr=2.0, CPU state=(1, 1)",
                b_cpu.arrival_t == 2.0 and collector.state_after("CPU", 2.0) == (1, 1)),
            row(3.0, "Complete CPU: A", "A -> Sink (COMPLETED); B starts CPU",
                f"{t_a}, B CPU start={b_cpu.start_t}", "('COMPLETED', 3.0), B CPU start=3.0",
                t_a == ("COMPLETED", 3.0) and b_cpu.start_t == 3.0),
            row(5.0, "Complete CPU: B", "B -> Sink (COMPLETED)",
                f"{t_b}", "('COMPLETED', 5.0)", t_b == ("COMPLETED", 5.0)),
            row(6.0, "Window close", "All stations idle, Disk never touched",
                f"saved={ {n: collector.saved[n] for n in collector.NODES} }, Disk changes={len(collector.history_of('Disk'))}",
                "all (nq=0, ns=0), Disk changes=0",
                all(collector.saved[n] == (0, 0) for n in collector.NODES) and not collector.history_of("Disk")),
        ]
        print_table(
            title="N02-1: p = 0 chain Source -> Ingress -> CPU -> Sink, event timeline trace",
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
            ["U_Ingress", f"{U['Ingress']:.4f}", f"{1/3:.4f}", "PASS" if near(U["Ingress"], 1 / 3) else "FAIL"],
            ["U_CPU", f"{U['CPU']:.4f}", f"{2/3:.4f}", "PASS" if near(U["CPU"], 2 / 3) else "FAIL"],
            ["U_Disk", f"{U['Disk']:.4f}", "0.0000", "PASS" if near(U["Disk"], 0.0) else "FAIL"],
            ["Lq_Ingress", f"{Lq['Ingress']:.4f}", f"{1/12:.4f}", "PASS" if near(Lq["Ingress"], 1 / 12) else "FAIL"],
            ["Lq_CPU", f"{Lq['CPU']:.4f}", f"{1/6:.4f}", "PASS" if near(Lq["CPU"], 1 / 6) else "FAIL"],
            ["E[w_Ingress] / E[r_Ingress]", f"{E_w_ing} / {E_r_ing} s", "0.25 / 1.25 s",
             "PASS" if near(E_w_ing, 0.25) and near(E_r_ing, 1.25) else "FAIL"],
            ["E[w_CPU] / E[r_CPU]", f"{E_w_cpu} / {E_r_cpu} s", "0.5 / 2.5 s",
             "PASS" if near(E_w_cpu, 0.5) and near(E_r_cpu, 2.5) else "FAIL"],
            ["R_completed", f"{R_completed} s", "3.75 s", "PASS" if near(R_completed, 3.75) else "FAIL"],
            ["X_net", f"{X:.4f} req/s", f"{1/3:.4f} req/s", "PASS" if near(X, 1 / 3) else "FAIL"],
            ["P_drop", f"{P_drop}", "0.0", "PASS" if P_drop == 0.0 else "FAIL"],
            ["N_ext = N_comp + N_drop + N_unf",
             f"{result.external} = {result.completed} + {result.dropped} + {result.unfinished}",
             "2 = 2 + 0 + 0",
             "PASS" if (result.external, result.completed, result.dropped, result.unfinished) == (2, 2, 0, 0) and result.conserved else "FAIL"],
            ["Visit ids Ingress / CPU / Disk",
             f"{node_visit_ids['Ingress']} / {node_visit_ids['CPU']} / {node_visit_ids['Disk']}",
             "['R1-Ingress-1', 'R2-Ingress-1'] / ['R1-CPU-1', 'R2-CPU-1'] / []",
             "PASS" if (node_visit_ids["Ingress"], node_visit_ids["CPU"], node_visit_ids["Disk"])
             == (["R1-Ingress-1", "R2-Ingress-1"], ["R1-CPU-1", "R2-CPU-1"], []) else "FAIL"],
            ["Operations created / succeeded / pending",
             f"{ops['created']} / {ops['succeeded']} / {ops['pending']}", "6 / 6 / 0",
             "PASS" if (ops["created"], ops["succeeded"], ops["pending"]) == (6, 6, 0) else "FAIL"],
            ["ACKs accepted / rejected", f"{ops['acks_accepted']} / {ops['acks_rejected']}", "6 / 0",
             "PASS" if (ops["acks_accepted"], ops["acks_rejected"]) == (6, 0) else "FAIL"],
            ["Station-visit Futures left unresolved", f"{len(router._completions)}", "0",
             "PASS" if len(router._completions) == 0 else "FAIL"],
            ["Request coroutines started / finished", f"{router.flows_started} / {router.flows_finished}", "2 / 2",
             "PASS" if (router.flows_started, router.flows_finished) == (2, 2) else "FAIL"],
        ]
        print_table(
            title="N02-2: Hand-calculated vs simulated metrics over [0.0, 6.0)",
            headers=["Metric", "Actual", "Expected", "Status"],
            rows=metrics_rows,
            alignments=["left", "left", "left", "center"],
        )

        # -------------------------------------------------------------
        # Assertions
        # -------------------------------------------------------------
        self.assertEqual(result.status, "COMPLETED", msg=result.failure_reason)

        # 1. Hand-calculated waiting / service intervals of every visit
        self.assertEqual((a_ing.arrival_t, a_ing.start_t, a_ing.complete_t), (0.0, 0.0, 1.0))
        self.assertEqual((a_cpu.arrival_t, a_cpu.start_t, a_cpu.complete_t), (1.0, 1.0, 3.0))
        self.assertEqual((b_ing.arrival_t, b_ing.start_t, b_ing.complete_t), (0.5, 1.0, 2.0))
        self.assertEqual((b_cpu.arrival_t, b_cpu.start_t, b_cpu.complete_t), (2.0, 3.0, 5.0))
        self.assertTrue(all(vis.accepted for vis in visits.values()))

        # 2. Outcomes at the Sink; terminal fields set exactly once on the Request objects
        self.assertEqual(t_a, ("COMPLETED", 3.0))
        self.assertEqual(t_b, ("COMPLETED", 5.0))
        self.assertEqual(router.history_requests["R1"].outcome, "COMPLETED")
        self.assertEqual(router.history_requests["R2"].terminal_t, 5.0)
        self.assertIsNone(router.history_requests["R1"].drop_node)
        self.assertEqual(net.sink.terminal_conflicts, 0)

        # 3. p = 0 never reaches the Disk (real routing stream), request ids preserved
        self.assertEqual(node_visit_ids["Disk"], [])
        self.assertEqual(router.visits_by_node, {"Ingress": 2, "CPU": 2, "Disk": 0})
        self.assertEqual(collector.history_of("Disk"), [])
        self.assertEqual(node_visit_ids["CPU"], ["R1-CPU-1", "R2-CPU-1"])

        # 4. FCFS: B waits behind A at both stations and A is never interrupted
        self.assertEqual(collector.history_of("Ingress"),
                         [(0.0, 0, 1), (0.5, 1, 1), (1.0, 0, 1), (2.0, 0, 0)])
        self.assertEqual(collector.history_of("CPU"),
                         [(1.0, 0, 1), (2.0, 1, 1), (3.0, 0, 1), (5.0, 0, 0)])

        # 5. Hand-calculated metrics over [0.0, 6.0)
        self.assertAlmostEqual(U["Ingress"], 1 / 3, places=9)
        self.assertAlmostEqual(U["CPU"], 2 / 3, places=9)
        self.assertAlmostEqual(U["Disk"], 0.0, places=9)
        self.assertAlmostEqual(Lq["Ingress"], 1 / 12, places=9)
        self.assertAlmostEqual(Lq["CPU"], 1 / 6, places=9)
        self.assertAlmostEqual(Lq["Disk"], 0.0, places=9)
        self.assertAlmostEqual(E_w_ing, 0.25, places=9)
        self.assertAlmostEqual(E_r_ing, 1.25, places=9)
        self.assertAlmostEqual(E_w_cpu, 0.5, places=9)
        self.assertAlmostEqual(E_r_cpu, 2.5, places=9)
        self.assertAlmostEqual(R_completed, 3.75, places=9)
        self.assertAlmostEqual(X, 1 / 3, places=9)
        self.assertEqual(P_drop, 0.0)

        # 6. Request conservation; control traffic created no business request
        self.assertEqual((result.external, result.completed, result.dropped, result.unfinished), (2, 2, 0, 0))
        self.assertTrue(result.conserved)
        self.assertEqual(collector.externals, [("R1", 0.0), ("R2", 0.5)])

        # 7. Operations and Futures: every transfer is one SUCCEEDED operation, nothing left over
        self.assertEqual((ops["created"], ops["succeeded"], ops["pending"]), (6, 6, 0))
        self.assertEqual((ops["acks_accepted"], ops["acks_rejected"]), (6, 0))
        self.assertEqual((ops["duplicate_acks"], ops["timeouts_scheduled"], ops["failed"]), (0, 0, 0))
        self.assertEqual(len(router._completions), 0)
        self.assertEqual((router.flows_started, router.flows_finished), (2, 2))


if __name__ == "__main__":
    unittest.main()