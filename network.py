# Copyright (C) 2026 SPE Team - Dev 2 (Queueing Network & Asynchronous Operations)
# Ingress-CPU-Disk queueing network with feedback, request lifecycle and routing.
"""
Queueing network assembly for the discrete-event simulation.

Topology (fixed, see Phase 1 report section 1.5):

    Source -> Ingress -> CPU --(1-p)--> Sink
                          |  ^
                        p |  | 1.0 (feedback)
                          v  |
                          Disk

Logical processes:

    SourceLP  - Poisson external arrivals (rate alpha); creates Request objects.
    RouterLP  - control LP. It is the ``sender_id`` of every ARRIVAL sent to a
                station, owns the OperationManager, receives ADMISSION_ACK and
                OP_TIMEOUT events, and implements the on_departure / on_rejected
                callbacks of contracts.py (routing + request lifecycle).
    QueueNodeLP (Dev 1) x3 - Ingress (c=1), CPU, Disk: finite M/M/c/K stations.
    SinkLP    - records COMPLETED requests.

Why a RouterLP: QueueNodeLP replies with ADMISSION_ACK to ``event.sender_id`` but can
only *handle* ARRIVAL and SERVICE_COMPLETE (anything else raises ValueError). Every
transfer is therefore sent *by the router*, which knows how to consume the ACK.

Business rules implemented here:

    * request_id is created once at external arrival and never changes.
    * Every arrival at a station gets a fresh visit_id ("R7-CPU-2"), also when the
      arrival is rejected. The Sink is not a station: no Visit is created for it.
    * Every transfer between LPs is an asynchronous *operation* (ARRIVAL ->
      ADMISSION_ACK). Operations are control exchanges, never new requests.
    * A request arriving at a full station is DROPPED at admission, leaves the whole
      network and is never retried. This holds for external arrivals and for
      feedback (Disk -> CPU). The decision is independent of the operation state.
    * The sending station frees its server before the router sees the departure
      (guaranteed by QueueNodeLP), so a rejected feedback never holds the Disk.
    * Routing delay and ACK delay are 0 in the baseline configuration.
    * Each request has exactly one terminal outcome (COMPLETED or DROPPED); the
      others stay "unfinished" at t_end. Conservation:
          N_external = N_completed + N_dropped + N_unfinished.

Metrics integration (Dev 3). ``metrics`` is duck-typed. Required methods (plan 00,
section 3):
    on_external_arrival(request, t)
    on_terminal(request, t)            # request.outcome / terminal_t already set
    on_state_change(node, t, nq, ns)   # forwarded straight to the QueueNodeLPs
Optional methods, called only if the object defines them (to be agreed with Dev 3):
    on_visit_arrival(request, visit, node, t)
    on_visit_departure(request, visit, node, t)
    on_visit_rejected(request, visit, node, t)
close_window(t1) and finalize(t_end) are NOT called here; run.py owns them.

Two execution modes (same business behaviour, see test N14):

    Case 1 "pure"  (use_operations=False): plain event-driven network. ARRIVAL events carry
                    no operation_id, so stations send no ACK; no Future, no coroutine.
                    Routing is done directly inside the on_departure callback.
    Case 2 "async" (use_operations=True, default): the design drawn as "case 2".
        * Each external arrival starts one coroutine ``_request_flow(request)`` that owns
          the route of the request as a ``while`` loop (Ingress -> CPU -> (Disk -> CPU)*
          -> Sink).
        * Every transfer is an *operation* (ARRIVAL -> ADMISSION_ACK) tracked by the
          OperationManager; the coroutine awaits its ACK Future (accepted and rejected
          are both successful communication).
        * For every station visit the router creates Future[(request_id, visit_id)]. The
          sending station has already released its server (QueueNodeLP frees it before
          on_departure), the coroutine just sleeps on the Future. When that visit
          finishes service, on_departure calls ``set_result`` ("Disk set_result()") and
          the coroutine wakes up at the same simulation timestamp and loops on to the
          next station (Disk -> CPU feedback included). Awaiting suspends only the
          coroutine; the dispatcher keeps processing every other event.
        * Admission is decided by the station at the arrival timestamp: a rejected
          arrival finalises the request as DROPPED (on_rejected) and ends its coroutine.
          This never depends on a Future, so a timeout/cancel of the *waiting* cannot
          change a business outcome.
        * BkSysCoEventLoop only inspects LP tasks for exceptions, so the router's own
          ``coroutine_run`` is a supervisor task that re-raises the first failure of any
          request coroutine; the loop then ends the replication with status FAILED.
        * ``track_acks`` is kept for backward compatibility and has no extra effect:
          the request coroutines already await every ACK (counted in ``ack_watch``).
        Admission ACK delay is 0, so business results match Case 1 exactly.

Usage:
    net = build_network(alpha=40, p=0.3, ingress=..., cpu=..., disk=..., end_time=1200,
                        streams=make_streams(replication_seed), metrics=metrics)
    result = net.run()          # one Network = one replication
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Union

from bkitevsim.core import Simulation, SimulationOptions
from bkitevsim.core.event import Event
from bkitevsim.core.logical_coProcess import BkSysCoLogicalProcess

from contracts import (
    EVENT_ADMISSION_ACK,
    EVENT_ARRIVAL,
    EVENT_OP_TIMEOUT,
    OperationPayload,
    Request,
    Visit,
)
from operations import (
    OperationCancelledError,
    OperationFailedError,
    OperationManager,
    OperationRecord,
    OperationTimeoutError,
)
from queue_node import QueueNodeLP


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
EVENT_EXTERNAL_GEN = "EXTERNAL_GEN"  # SourceLP self-event; never creates a business request by itself

NODE_INGRESS = "Ingress"
NODE_CPU = "CPU"
NODE_DISK = "Disk"
NODE_SINK = "Sink"
STATION_NAMES = (NODE_INGRESS, NODE_CPU, NODE_DISK)

OUTCOME_COMPLETED = "COMPLETED"
OUTCOME_DROPPED = "DROPPED"

# Fixed LP ids of the network
LP_ROUTER = 0
LP_SOURCE = 1
LP_INGRESS = 2
LP_CPU = 3
LP_DISK = 4
LP_SINK = 5

DEFAULT_NODE_LP_IDS: dict[str, int] = {
    NODE_INGRESS: LP_INGRESS,
    NODE_CPU: LP_CPU,
    NODE_DISK: LP_DISK,
    NODE_SINK: LP_SINK,
}


# -------------------------------------------------------------------------
# Configuration helpers
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class StationConfig:
    """Parameters of one finite station: c servers, total capacity B, mean service time."""
    c: int
    B: int
    mean_service_s: float

    @staticmethod
    def from_any(value: Union["StationConfig", Mapping[str, Any], tuple]) -> "StationConfig":
        """Accept StationConfig, a dict {c, B, mean_service_s} or a tuple (c, B, mean_service_s)."""
        if isinstance(value, StationConfig):
            return value
        if isinstance(value, Mapping):
            return StationConfig(int(value["c"]), int(value["B"]), float(value["mean_service_s"]))
        c, b, mean = value
        return StationConfig(int(c), int(b), float(mean))


@dataclass
class NetworkStreams:
    """Independent random streams of one replication (arrival, routing, 3 service streams)."""
    arrival: Any
    routing: Any
    ingress: Any
    cpu: Any
    disk: Any


def make_streams(replication_seed: int) -> NetworkStreams:
    """
    Default deterministic stream derivation from a replication seed.

    Each stream is ``random.Random("<seed>/<name>")``. String seeding is hashed with
    SHA-512 by CPython, so streams are independent, stable across runs/platforms and
    never depend on the system clock. Dev 3 may supply a different derivation by
    building NetworkStreams itself.
    """
    def rng(name: str) -> random.Random:
        return random.Random(f"{int(replication_seed)}/{name}")
    return NetworkStreams(
        arrival=rng("arrival"), routing=rng("routing"),
        ingress=rng("service/Ingress"), cpu=rng("service/CPU"), disk=rng("service/Disk"),
    )


# -------------------------------------------------------------------------
# SourceLP
# -------------------------------------------------------------------------
class SourceLP(BkSysCoLogicalProcess):
    """
    Open Poisson source of external arrivals.

    Gaps are drawn with ``arrival_rng.expovariate(alpha)``. The first arrival happens
    at t1 ~ Exp(alpha), not at t = 0. Generation continues with the same law until
    ``end_time`` (also after the measurement window, so the load at the right edge of
    the window is not distorted). alpha == 0 generates nothing.

    Each EXTERNAL_GEN self-event creates exactly one Request (ids R1, R2, ...) and hands
    it to ``on_external_arrival`` (the router), then schedules the next generation.
    """

    def __init__(
        self,
        lp_id: int,
        alpha: float,
        arrival_rng: Any,
        on_external_arrival: Callable[[Request, float], None],
        end_time: Optional[float] = None,
    ):
        super().__init__(lp_id)
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError(f"alpha must be finite and >= 0, got {alpha}")
        self.alpha = float(alpha)
        self.arrival_rng = arrival_rng
        self._on_external_arrival = on_external_arrival
        self.end_time = end_time
        self.generated = 0

    async def coroutine_run(self) -> None:
        """Schedule the first external arrival at t1 ~ Exp(alpha) (none if alpha == 0)."""
        self._schedule_next(0.0)

    def _schedule_next(self, now: float) -> None:
        if self.alpha <= 0:
            return
        t_next = now + self.arrival_rng.expovariate(self.alpha)
        if self.end_time is not None and t_next > self.end_time:
            return  # would never be processed; do not leave dead events in the queue
        self.send_event(Event(
            timestamp=t_next,
            sender_id=self.lp_id,
            receiver_id=self.lp_id,
            event_type=EVENT_EXTERNAL_GEN,
        ))

    def handle_event(self, event: Event) -> None:
        if event.event_type != EVENT_EXTERNAL_GEN:
            raise ValueError(f"SourceLP received unexpected event type {event.event_type!r}")
        t = event.timestamp
        self.generated += 1
        request = Request(request_id=f"R{self.generated}", external_arrival_t=t)
        self._on_external_arrival(request, t)
        self._schedule_next(t)

    def get_statistics(self) -> dict[str, Any]:
        return {"lp_id": self.lp_id, "alpha": self.alpha, "generated": self.generated}


# -------------------------------------------------------------------------
# SinkLP
# -------------------------------------------------------------------------
class SinkLP(BkSysCoLogicalProcess):
    """
    Terminal LP (not a service station: no server, no queue, no Visit).

    On ARRIVAL it fixes ``outcome = COMPLETED`` and ``terminal_t = t`` exactly once,
    notifies ``on_completed`` and, if the payload carries an operation_id, answers with
    ADMISSION_ACK(accepted=True) to ``event.sender_id`` (the router).
    """

    def __init__(self, lp_id: int, on_completed: Callable[[Request, float], None]):
        super().__init__(lp_id)
        self._on_completed = on_completed
        self.completed = 0
        self.terminal_conflicts = 0  # request already had an outcome (must stay 0)

    async def coroutine_run(self) -> None:
        pass

    def handle_event(self, event: Event) -> None:
        if event.event_type != EVENT_ARRIVAL:
            raise ValueError(f"SinkLP received unexpected event type {event.event_type!r}")
        t = event.timestamp
        payload = event.payload
        if isinstance(payload, OperationPayload):
            request, op_id = payload.request, payload.operation_id
        elif isinstance(payload, dict):
            request, op_id = payload.get("request"), payload.get("operation_id")
        else:
            raise ValueError(f"SinkLP: malformed payload {type(payload).__name__}")
        if request is None:
            raise ValueError("SinkLP: ARRIVAL without request")

        if request.outcome is None:
            request.terminal_t = t
            request.outcome = OUTCOME_COMPLETED
            self.completed += 1
            self._on_completed(request, t)
        else:
            self.terminal_conflicts += 1  # never overwrite a final outcome

        if op_id is not None:
            self.send_event(Event(
                timestamp=t,
                sender_id=self.lp_id,
                receiver_id=event.sender_id,
                event_type=EVENT_ADMISSION_ACK,
                payload=OperationPayload(operation_id=op_id, request=request, visit=None, accepted=True),
            ))

    def get_statistics(self) -> dict[str, Any]:
        return {"lp_id": self.lp_id, "completed": self.completed,
                "terminal_conflicts": self.terminal_conflicts}


# -------------------------------------------------------------------------
# RouterLP
# -------------------------------------------------------------------------
class RouterLP(BkSysCoLogicalProcess):
    """
    Control LP: routing, request/visit lifecycle and asynchronous operations.

    Receives (as events): ADMISSION_ACK, OP_TIMEOUT.
    Receives (as callbacks from the stations / source / sink):
        admit_external(request, t)       from SourceLP
        on_departure(request, visit, node, t)   from QueueNodeLP (server already freed)
        on_rejected(request, visit, node, t)    from QueueNodeLP (arrival refused)
        on_completed(request, t)         from SinkLP

    Args:
        node_lp_ids: mapping node name -> LP id ('Ingress', 'CPU', 'Disk', 'Sink').
        p: probability CPU -> Disk (0 <= p < 1); CPU -> Sink with probability 1 - p.
        routing_rng: dedicated stream for the CPU routing decision.
        routing_delay_s: transfer delay (0 in the baseline configuration).
        metrics: optional collector (see module docstring).
        force_route: test hook ``f(request, cpu_visit_index) -> bool`` (True = go to
            Disk) that replaces the random routing decision.
        track_acks: kept for backward compatibility; in Case 2 every request coroutine
            already awaits all of its ACK Futures, so this flag changes nothing.
        record_history: keep every Request, Visit and a flow trace (tests / diagnostics).
        use_operations: False = Case 1 "pure" network (no operation_id, no ACK, no Future,
            no coroutine); True (default) = Case 2, one coroutine per request awaiting
            Future[(request_id, visit_id)] (see module docstring).
    """

    def __init__(
        self,
        lp_id: int,
        node_lp_ids: Mapping[str, int],
        p: float,
        routing_rng: Any,
        routing_delay_s: float = 0.0,
        metrics: Any = None,
        force_route: Optional[Callable[[Request, int], bool]] = None,
        track_acks: bool = False,
        record_history: bool = False,
        use_operations: bool = True,
    ):
        super().__init__(lp_id)
        if track_acks and not use_operations:
            raise ValueError("track_acks requires use_operations=True")
        if not (0.0 <= p < 1.0):
            raise ValueError(f"routing probability p must satisfy 0 <= p < 1, got {p}")
        if not math.isfinite(routing_delay_s) or routing_delay_s < 0:
            raise ValueError(f"routing delay must be finite and >= 0, got {routing_delay_s}")
        self.node_lp_ids: dict[str, int] = dict(node_lp_ids)
        self.p = float(p)
        self.routing_rng = routing_rng
        self.routing_delay_s = float(routing_delay_s)
        self._force_route = force_route
        self._track_acks = track_acks
        self._use_ops = use_operations
        self.ops: Optional[OperationManager] = None

        # Metrics hooks (resolved once; None = not provided)
        self._m_external = getattr(metrics, "on_external_arrival", None)
        self._m_terminal = getattr(metrics, "on_terminal", None)
        self._m_visit_arrival = getattr(metrics, "on_visit_arrival", None)
        self._m_visit_departure = getattr(metrics, "on_visit_departure", None)
        self._m_visit_rejected = getattr(metrics, "on_visit_rejected", None)

        # Request bookkeeping
        self._open: dict[str, Request] = {}                 # requests without outcome
        self._visit_counts: dict[tuple[str, str], int] = {}  # (request_id, node) -> visits so far
        self.n_external = 0
        self.n_completed = 0
        self.n_dropped = 0
        self.drops_by_node: dict[str, int] = {n: 0 for n in STATION_NAMES}
        self.visits_by_node: dict[str, int] = {n: 0 for n in STATION_NAMES}
        # Ack outcomes observed by the request coroutines (Case 2)
        self.ack_watch = {"accepted": 0, "rejected": 0, "timed_out": 0, "cancelled": 0, "failed": 0}

        # Case 2 state: one coroutine per request, Future[(request_id, visit_id)] per visit
        self._completions: dict[tuple[str, str], asyncio.Future] = {}
        self._flows: set[asyncio.Task] = set()   # strong refs: asyncio keeps only weak ones
        self._supervisor: Optional[asyncio.Future] = None
        self._flow_failure: Optional[BaseException] = None
        self.flows_started = 0
        self.flows_finished = 0

        self.history_requests: Optional[dict[str, Request]] = {} if record_history else None
        self.history_visits: Optional[list[Visit]] = [] if record_history else None
        # (t, request_id, kind, detail), kind in send / ack / wait / resume / end
        self.flow_log: Optional[list[tuple[float, str, str, str]]] = [] if record_history else None

    def attach_operations(self, ops: OperationManager) -> None:
        """Bind the OperationManager (built after the router, which it needs for send_event)."""
        self.ops = ops

    async def coroutine_run(self) -> None:
        """
        Supervisor of the request coroutines (Case 2 only).

        BkSysCoEventLoop only inspects LP tasks for exceptions, so request coroutines
        report their failure here: this task re-raises the first one and the loop then
        stops the replication with status FAILED. The loop cancels it at the end.
        """
        if not self._use_ops:
            return
        self._supervisor = self._event_loop.create_future()
        if self._flow_failure is not None:
            self._supervisor.set_exception(self._flow_failure)
        await self._supervisor

    # ------------------------------------------------------------ properties
    @property
    def n_unfinished(self) -> int:
        """Requests that have no outcome yet (final value is read at t_end)."""
        return len(self._open)

    def unfinished_requests(self) -> list[Request]:
        return list(self._open.values())

    def _now(self) -> float:
        return self._event_loop.current_time

    def _log(self, t: float, request_id: str, kind: str, detail: str = "") -> None:
        if self.flow_log is not None:
            self.flow_log.append((t, request_id, kind, detail))

    # --------------------------------------------------------------- events
    def handle_event(self, event: Event) -> None:
        """Consume control events only; business events are handled by the stations."""
        et = event.event_type
        if et == EVENT_ADMISSION_ACK:
            self.ops.on_ack(event)
        elif et == EVENT_OP_TIMEOUT:
            self.ops.on_timeout(event)
        else:
            self.ops.fail(f"RouterLP received unexpected event type {et!r}", t=event.timestamp)

    # ------------------------------------------------------------- external
    def admit_external(self, request: Request, t: float) -> None:
        """New external arrival: register the request and send it to Ingress."""
        if request.request_id in self._open:
            self.ops.fail(f"duplicate request_id {request.request_id}", t=t)
        self._open[request.request_id] = request
        self.n_external += 1
        if self.history_requests is not None:
            self.history_requests[request.request_id] = request
        if self._m_external:
            self._m_external(request, t)
        if self._use_ops:
            task = asyncio.get_running_loop().create_task(self._flow_guard(request))
            self._flows.add(task)
            task.add_done_callback(self._flows.discard)
            self.flows_started += 1
        else:
            self._dispatch(request, NODE_INGRESS, t)

    # ----------------------------------------------------- request coroutine
    async def _flow_guard(self, request: Request) -> None:
        """Run the request coroutine; forward any failure to the supervisor task."""
        try:
            await self._request_flow(request)
            self.flows_finished += 1  # normal end: COMPLETED or DROPPED
        except asyncio.CancelledError:
            raise  # end-of-run cleanup: the request simply stays unfinished
        except BaseException as exc:  # noqa: BLE001 - must reach the event loop
            self._report_flow_failure(exc)

    def _report_flow_failure(self, exc: BaseException) -> None:
        if self._flow_failure is None:
            self._flow_failure = exc
        if self._supervisor is not None and not self._supervisor.done():
            self._supervisor.set_exception(exc)

    async def _await_ack(self, rec: OperationRecord) -> Optional[bool]:
        """Await the admission ACK Future. None = the waiting ended without an answer."""
        try:
            accepted = await self.ops.wait_ack(rec)
        except OperationTimeoutError:
            self.ack_watch["timed_out"] += 1
            return None  # only the waiting ended; the station still decides on its own
        except OperationCancelledError:
            self.ack_watch["cancelled"] += 1
            return None
        except OperationFailedError:
            self.ack_watch["failed"] += 1
            raise
        self.ack_watch["accepted" if accepted else "rejected"] += 1
        return accepted

    async def _request_flow(self, request: Request) -> None:
        """
        The route of one request as a coroutine (the ``while`` loop of the diagram).

        Ingress -> CPU -> (Disk -> CPU)* -> Sink. For every station visit:
          1. send the ARRIVAL operation and await its admission ACK (accepted/rejected);
          2. if admitted, await Future[(request_id, visit_id)], resolved by on_departure
             when that visit finishes service (the server is already free);
          3. choose the next node and loop.
        Suspending here never blocks the dispatcher: other events and coroutines keep
        running at the same and at later simulation times.
        """
        rid = request.request_id
        node = NODE_INGRESS
        while node != NODE_SINK:
            rec, visit, done = self._dispatch(request, node, self._now(), want_completion=True)
            self._log(self._now(), rid, "send", visit.visit_id)
            accepted = await self._await_ack(rec)
            self._log(self._now(), rid, "ack", f"{visit.visit_id} accepted={accepted}")
            if accepted is False:
                self._log(self._now(), rid, "end", f"dropped at {node}")
                return  # DROPPED was finalised by on_rejected at admission

            self._log(self._now(), rid, "wait", visit.visit_id)
            served = await done
            self._log(self._now(), rid, "resume", visit.visit_id)
            if served is None:
                self._log(self._now(), rid, "end", f"dropped at {node}")
                return  # arrival was rejected (only reachable after an ACK timeout/cancel)
            node = self._next_node(request, node)

        rec, _, _ = self._dispatch(request, NODE_SINK, self._now(), want_completion=False)
        self._log(self._now(), rid, "send", NODE_SINK)
        await self._await_ack(rec)
        self._log(self._now(), rid, "end", "completed")

    # -------------------------------------------------------------- routing
    def _next_node(self, request: Request, node: str) -> str:
        if node == NODE_INGRESS:
            return NODE_CPU
        if node == NODE_DISK:
            return NODE_CPU  # feedback
        if node == NODE_CPU:
            cpu_visit_index = self._visit_counts[(request.request_id, NODE_CPU)]
            if self._force_route is not None:
                go_disk = bool(self._force_route(request, cpu_visit_index))
            else:
                go_disk = self.routing_rng.random() < self.p
            return NODE_DISK if go_disk else NODE_SINK
        self.ops.fail(f"departure from unknown node {node!r}")

    def on_departure(self, request: Optional[Request], visit: Visit, node: str, t: float) -> None:
        """Service completed at ``node`` (server already freed): route the request onward."""
        if request is None:
            self.ops.fail(f"departure of visit {visit.visit_id} without request", t=t)
        if self._m_visit_departure:
            self._m_visit_departure(request, visit, node, t)

        if self._use_ops:
            fut = self._completions.pop((visit.request_id, visit.visit_id), None)
            if fut is None:
                self.ops.fail(f"departure of visit {visit.visit_id} has no waiting coroutine", t=t)
            if not fut.done():
                fut.set_result(t)  # wakes the request coroutine (same simulation timestamp)
            return
        self._dispatch(request, self._next_node(request, node), t)

    def on_rejected(self, request: Optional[Request], visit: Visit, node: str, t: float) -> None:
        """
        Arrival refused by a full station: finalise the request as DROPPED.

        Idempotent: a request that already has an outcome is never changed. No retry.
        """
        if request is None:
            self.ops.fail(f"rejection of visit {visit.visit_id} without request", t=t)
        if self._m_visit_rejected:
            self._m_visit_rejected(request, visit, node, t)

        fut = self._completions.pop((visit.request_id, visit.visit_id), None)
        if fut is not None and not fut.done():
            fut.set_result(None)  # this visit will never be served

        if request.outcome is not None:
            return
        request.outcome = OUTCOME_DROPPED
        request.terminal_t = t
        request.drop_node = node
        self.n_dropped += 1
        self.drops_by_node[node] = self.drops_by_node.get(node, 0) + 1
        self._close_request(request, t)

    def on_completed(self, request: Request, t: float) -> None:
        """Sink recorded the request (outcome/terminal_t already set by the Sink)."""
        self.n_completed += 1
        self._close_request(request, t)

    def _close_request(self, request: Request, t: float) -> None:
        self._open.pop(request.request_id, None)
        rid = request.request_id
        for node in STATION_NAMES:
            self._visit_counts.pop((rid, node), None)
        if self._m_terminal:
            self._m_terminal(request, t)

    # ------------------------------------------------------------- dispatch
    def _dispatch(self, request: Request, node: str, t: float, want_completion: bool = False):
        """
        Start one asynchronous transfer of ``request`` to ``node`` at time t.

        Creates a fresh Visit (stations only) and, in Case 2, a fresh operation and (if
        requested) Future[(request_id, visit_id)] for the service completion of that
        visit. The ARRIVAL event is scheduled at t + routing_delay with this router as
        sender.

        Returns:
            (operation record or None, visit or None for the Sink, completion Future or None)
        """
        dest_lp = self.node_lp_ids[node]
        arrival_t = t + self.routing_delay_s

        visit: Optional[Visit] = None
        done: Optional[asyncio.Future] = None
        if node != NODE_SINK:
            key = (request.request_id, node)
            k = self._visit_counts.get(key, 0) + 1
            self._visit_counts[key] = k
            visit = Visit(
                request_id=request.request_id,
                visit_id=f"{request.request_id}-{node}-{k}",
                node=node,
                arrival_t=arrival_t,
            )
            self.visits_by_node[node] += 1
            if self.history_visits is not None:
                self.history_visits.append(visit)
            if want_completion:
                done = self._event_loop.create_future()
                self._completions[(visit.request_id, visit.visit_id)] = done

        if self._use_ops:
            rec = self.ops.create(t, dest_lp, request, visit)  # fails (and cleans) if dest LP is missing
            payload = self.ops.payload_for(rec, request, visit)
        else:
            # Case 1: no operation_id -> the station sends no ACK; a missing destination LP is
            # then caught by the event loop's own fail-fast delivery check.
            rec = None
            payload = OperationPayload(request=request, visit=visit)
        self.send_event(Event(
            timestamp=arrival_t,
            sender_id=self.lp_id,
            receiver_id=dest_lp,
            event_type=EVENT_ARRIVAL,
            payload=payload,
        ))
        if visit is not None and self._m_visit_arrival:
            self._m_visit_arrival(request, visit, node, arrival_t)
        return rec, visit, done

    def get_statistics(self) -> dict[str, Any]:
        return {
            "lp_id": self.lp_id,
            "external": self.n_external,
            "completed": self.n_completed,
            "dropped": self.n_dropped,
            "unfinished": self.n_unfinished,
            "drops_by_node": dict(self.drops_by_node),
            "visits_by_node": dict(self.visits_by_node),
            "ack_watch": dict(self.ack_watch),
            "flows_started": self.flows_started,
            "flows_finished": self.flows_finished,
            "operations": self.ops.stats() if self.ops else None,
        }


# -------------------------------------------------------------------------
# Network facade
# -------------------------------------------------------------------------
@dataclass
class NetworkResult:
    """
    Outcome of one replication.

    Attributes:
        status: 'COMPLETED' or 'FAILED' (communication / handler / task failure).
        failure_reason: Diagnostic text when FAILED. A FAILED run must NOT be merged
            into the performance results; its counts are diagnostic only.
        external / completed / dropped / unfinished: request counts at t_end.
        conserved: external == completed + dropped + unfinished.
        drops_by_node: DROPPED requests by the station that refused them.
        events_processed / wall_time: from the event loop.
        operations: OperationManager counters (duplicate/late ACKs, timeouts, ...).
    """
    status: str
    failure_reason: Optional[str]
    external: int
    completed: int
    dropped: int
    unfinished: int
    conserved: bool
    drops_by_node: dict[str, int]
    events_processed: int
    wall_time: float
    operations: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "COMPLETED"


class Network:
    """One assembled network = one replication (single use)."""

    def __init__(self, sim: Simulation, end_time: float, router: RouterLP, source: SourceLP,
                 sink: SinkLP, stations: dict[str, QueueNodeLP], ops: OperationManager):
        self.sim = sim
        self.end_time = end_time
        self.router = router
        self.source = source
        self.sink = sink
        self.stations = stations
        self.ops = ops
        self._ran = False

    def run(self) -> NetworkResult:
        """
        Run the replication until t_end, then clean up.

        Events with timestamp <= t_end are all processed, including events created at
        exactly t_end. Afterwards the operation table is closed (still-pending operations
        become CANCELLED; this never changes a request outcome). Requests without outcome
        are reported as unfinished.
        """
        if self._ran:
            raise RuntimeError("Network.run() may be called only once; build a new network per replication")
        self._ran = True

        stats = self.sim.run()
        status = stats.get("status", "COMPLETED")
        reason = stats.get("failure_reason")
        if self.ops.failed and status != "FAILED":
            status, reason = "FAILED", self.ops.failure_reason
        if status == "FAILED" and reason is None:
            reason = self.ops.failure_reason or "unknown failure"

        # Remove any waiting registration that is still open (also after a loop-level failure)
        self.ops.close(t=self.end_time, reason=reason or "simulation_end")

        r = self.router
        external, completed, dropped, unfinished = r.n_external, r.n_completed, r.n_dropped, r.n_unfinished
        return NetworkResult(
            status=status,
            failure_reason=reason,
            external=external,
            completed=completed,
            dropped=dropped,
            unfinished=unfinished,
            conserved=(external == completed + dropped + unfinished),
            drops_by_node=dict(r.drops_by_node),
            events_processed=int(stats.get("events_processed", 0)),
            wall_time=float(stats.get("wall_time", 0.0)),
            operations=self.ops.stats(),
        )


def build_network(
    alpha: float,
    p: float,
    ingress: Union[StationConfig, Mapping[str, Any], tuple],
    cpu: Union[StationConfig, Mapping[str, Any], tuple],
    disk: Union[StationConfig, Mapping[str, Any], tuple],
    end_time: float,
    streams: NetworkStreams,
    metrics: Any = None,
    operation_deadline_s: Optional[float] = None,
    routing_delay_s: float = 0.0,
    track_acks: bool = False,
    use_operations: bool = True,
    record_history: bool = False,
    retain_operation_records: bool = False,
    trace_operations: bool = False,
    force_route: Optional[Callable[[Request, int], bool]] = None,
    service_time_generators: Optional[Mapping[str, Callable[[], float]]] = None,
) -> Network:
    """
    Build the Ingress-CPU-Disk network for ONE replication.

    Args:
        alpha: external arrival rate (request/s, >= 0).
        p: CPU -> Disk probability (0 <= p < 1).
        ingress, cpu, disk: station parameters (c, B, mean_service_s). Ingress must
            have a single server (c == 1).
        end_time: t_end; all events with timestamp <= end_time are processed.
        streams: independent random streams (see make_streams()).
        metrics: Dev 3 collector (see module docstring), or None.
        operation_deadline_s: None (baseline: no OP_TIMEOUT events) or a finite
            positive ACK deadline in simulation seconds (timeout test cases only).
        routing_delay_s: transfer delay, 0 in the baseline configuration.
        track_acks: kept for backward compatibility (no extra effect: in Case 2 every
            request coroutine already awaits its ACKs).
        use_operations: False builds the Case 1 pure network (no operations/ACK/Future).
        record_history / retain_operation_records / trace_operations: diagnostics.
        force_route / service_time_generators: deterministic hooks for unit tests
            (service generators are keyed by 'Ingress' / 'CPU' / 'Disk').
    """
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be finite and >= 0, got {alpha}")
    if not (0.0 <= p < 1.0):
        raise ValueError(f"p must satisfy 0 <= p < 1, got {p}")
    if not math.isfinite(end_time) or end_time <= 0:
        raise ValueError(f"end_time must be finite and > 0, got {end_time}")

    cfg = {
        NODE_INGRESS: StationConfig.from_any(ingress),
        NODE_CPU: StationConfig.from_any(cpu),
        NODE_DISK: StationConfig.from_any(disk),
    }
    if track_acks and not use_operations:
        raise ValueError("track_acks requires use_operations=True")
    if cfg[NODE_INGRESS].c != 1:
        raise ValueError("Ingress must have exactly one server (c == 1)")

    if metrics is not None:
        for name in ("on_external_arrival", "on_terminal", "on_state_change"):
            if not callable(getattr(metrics, name, None)):
                raise TypeError(f"metrics object must define {name}(...)")
    on_state_change = getattr(metrics, "on_state_change", None) if metrics is not None else None

    sim = Simulation(SimulationOptions(end_time=end_time, dispatcher="sequential"))

    router = RouterLP(
        LP_ROUTER, DEFAULT_NODE_LP_IDS, p, streams.routing,
        routing_delay_s=routing_delay_s, metrics=metrics, force_route=force_route,
        track_acks=track_acks, record_history=record_history, use_operations=use_operations,
    )
    ops = OperationManager(
        owner_lp_id=LP_ROUTER,
        schedule_event=lambda e: router.send_event(e),  # dispatcher rebinds send_event at registration
        future_factory=(lambda: sim.dispatcher.create_future()) if use_operations else None,
        deadline_s=operation_deadline_s,
        lp_exists=lambda lp_id: lp_id in sim.lps,
        retain_records=retain_operation_records,
        trace_enabled=trace_operations,
    )
    router.attach_operations(ops)

    gens = dict(service_time_generators or {})
    rngs = {NODE_INGRESS: streams.ingress, NODE_CPU: streams.cpu, NODE_DISK: streams.disk}
    stations: dict[str, QueueNodeLP] = {}
    for name in STATION_NAMES:
        c = cfg[name]
        stations[name] = QueueNodeLP(
            lp_id=DEFAULT_NODE_LP_IDS[name],
            node_name=name,
            c=c.c, B=c.B, mean_service_s=c.mean_service_s,
            service_rng=rngs[name],
            service_time_generator=gens.get(name),
            on_departure=router.on_departure,
            on_rejected=router.on_rejected,
            on_state_change=on_state_change,
        )

    source = SourceLP(LP_SOURCE, alpha, streams.arrival, router.admit_external, end_time=end_time)
    sink = SinkLP(LP_SINK, router.on_completed)

    for lp in (router, source, sink, *stations.values()):
        sim.lps[lp.lp_id] = lp

    return Network(sim, end_time, router, source, sink, stations, ops)


__all__ = [
    "EVENT_EXTERNAL_GEN", "NODE_INGRESS", "NODE_CPU", "NODE_DISK", "NODE_SINK",
    "OUTCOME_COMPLETED", "OUTCOME_DROPPED",
    "StationConfig", "NetworkStreams", "make_streams",
    "SourceLP", "SinkLP", "RouterLP", "Network", "NetworkResult", "build_network",
]
