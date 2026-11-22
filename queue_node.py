# Copyright (C) 2026 SPE Team - Dev 1 (DES Core & Queueing Station)
# QueueNodeLP implementation for discrete-event queueing stations.
"""
Generic queueing station logical process (QueueNodeLP) for discrete-event simulation.

Models finite-capacity multi-server queueing stations (M/M/1/K and M/M/c/K) with
First-Come-First-Served (FCFS) queueing discipline and non-preemptive service.
All docstrings and comments strictly adhere to SPE and Queueing Theory terminology.
"""

from __future__ import annotations
from collections import deque
import random
from typing import Optional, Callable, Any

from bkitevsim.core.event import Event
from bkitevsim.core.logical_coProcess import BkSysCoLogicalProcess
from contracts import (
    Request,
    Visit,
    OperationPayload,
    EVENT_ARRIVAL,
    EVENT_SERVICE_COMPLETE,
    EVENT_ADMISSION_ACK,
    DepartureCallback,
    RejectionCallback,
    StateChangeCallback,
)


class QueueNodeLP(BkSysCoLogicalProcess):
    """
    Queueing station logical process representing a finite-capacity queueing node (M/M/c/K).

    Attributes:
        lp_id: Unique integer identifier for this Logical Process.
        node_name: Station identifier (e.g., 'Ingress', 'CPU', 'Disk').
        c: Number of parallel identical servers (c >= 1).
        B: Total system capacity (waiting queue capacity + servers, B >= c).
        mean_service_s: Mean service time in simulation seconds (1 / mu).
        service_rng: Station-specific random number generator for service times.
        service_time_generator: Optional deterministic function returning service times for unit tests.
        on_departure: Callback invoked on completion of service and departure from the station.
        on_rejected: Callback invoked when an arriving job is rejected due to finite buffer capacity B.
        on_state_change: Callback invoked on state changes for time-weighted integral calculation.
    """

    def __init__(
        self,
        lp_id: int,
        node_name: str,
        c: int,
        B: int,
        mean_service_s: float,
        service_rng: Optional[random.Random] = None,
        service_time_generator: Optional[Callable[[], float]] = None,
        on_departure: Optional[DepartureCallback] = None,
        on_rejected: Optional[RejectionCallback] = None,
        on_state_change: Optional[StateChangeCallback] = None,
    ):
        super().__init__(lp_id)
        if c < 1:
            raise ValueError(f"Server count c must be >= 1, got {c}")
        if B < c:
            raise ValueError(f"Total capacity B ({B}) must be >= server count c ({c})")
        if mean_service_s <= 0:
            raise ValueError(f"Mean service time must be positive, got {mean_service_s}")

        self.node_name = node_name
        self.c = c
        self.B = B
        self.mean_service_s = mean_service_s
        self.service_rng = service_rng or random.Random(42)
        self.service_time_generator = service_time_generator

        # Callbacks
        self.on_departure_cb = on_departure
        self.on_rejected_cb = on_rejected
        self.on_state_change_cb = on_state_change

        # Queueing station state
        self.queue: deque[tuple[Optional[Request], Visit]] = deque()
        self.busy_servers: int = 0

    @property
    def nq(self) -> int:
        """Current number of jobs waiting in the FCFS queue."""
        return len(self.queue)

    @property
    def ns(self) -> int:
        """Current number of busy servers actively providing service."""
        return self.busy_servers

    @property
    def n(self) -> int:
        """Total number of jobs in the queueing station (in queue + in service)."""
        return self.nq + self.ns

    def initialize(self) -> None:
        """Initialize station state at simulation startup."""
        self.queue.clear()
        self.busy_servers = 0

    async def coroutine_run(self) -> None:
        """Cooperative coroutine hook for compatibility with BkSysCoEventLoop."""
        pass

    def handle_event(self, event: Event) -> None:
        """
        Process incoming discrete events dispatched to this queueing station.

        Dispatches ARRIVAL and SERVICE_COMPLETE events to their respective handlers.
        """
        if event.event_type == EVENT_ARRIVAL:
            self._handle_arrival(event)
        elif event.event_type == EVENT_SERVICE_COMPLETE:
            self._handle_service_complete(event)
        else:
            raise ValueError(f"Unknown event type '{event.event_type}' for station '{self.node_name}'")

    def _extract_payload(self, event: Event) -> tuple[Optional[Request], Visit, Optional[str]]:
        """Extract request, visit, and operation_id from event payload."""
        payload = event.payload
        if isinstance(payload, OperationPayload):
            return payload.request, payload.visit, payload.operation_id
        if isinstance(payload, dict):
            req = payload.get("request")
            vis = payload.get("visit")
            op_id = payload.get("operation_id")
            return req, vis, op_id
        if isinstance(payload, tuple) and len(payload) == 2:
            return payload[0], payload[1], None
        if isinstance(payload, Visit):
            return None, payload, None
        raise ValueError(f"Invalid payload format in event {event}")

    def _handle_arrival(self, event: Event) -> None:
        """
        Process an arrival event at simulation timestamp t = event.timestamp.

        Rule:
            - Check total station occupancy n = nq + ns against finite capacity B.
            - If n < B (admitted):
                - Call on_state_change(node, t, old_nq, old_ns) after mutating state.
                - If ns < c: start service immediately on an idle server.
                - Else: enqueue visit into the FCFS waiting queue.
                - If operation_id is present: send ADMISSION_ACK(accepted=True) back to sender.
            - If n == B (rejected / buffer full):
                - State nq, ns remains unchanged.
                - Call on_rejected(request, visit, node, t) exactly once.
                - If operation_id is present: send ADMISSION_ACK(accepted=False) back to sender.
        """
        t = event.timestamp
        request, visit, op_id = self._extract_payload(event)
        visit.arrival_t = t

        # Total capacity check (waiting jobs + jobs in service)
        if self.n < self.B:
            visit.accepted = True

            if self.ns < self.c:
                # Server available: begin service immediately
                self._start_service(request, visit, t)
            else:
                # All servers busy: enqueue into FCFS queue
                self.queue.append((request, visit))

            # Notify state change listener of NEW state immediately after transition
            if self.on_state_change_cb:
                self.on_state_change_cb(self.node_name, t, self.nq, self.ns)

            # Acknowledge successful admission if operation_id is present
            if op_id is not None:
                ack_payload = OperationPayload(
                    operation_id=op_id,
                    request=request,
                    visit=visit,
                    accepted=True,
                )
                self.send_event(Event(
                    timestamp=t,
                    sender_id=self.lp_id,
                    receiver_id=event.sender_id,
                    event_type=EVENT_ADMISSION_ACK,
                    payload=ack_payload,
                ))
        else:
            # Buffer full: reject arrival (finite capacity loss)
            visit.accepted = False

            if self.on_rejected_cb:
                self.on_rejected_cb(request, visit, self.node_name, t)

            # Acknowledge rejection if operation_id is present
            if op_id is not None:
                ack_payload = OperationPayload(
                    operation_id=op_id,
                    request=request,
                    visit=visit,
                    accepted=False,
                )
                self.send_event(Event(
                    timestamp=t,
                    sender_id=self.lp_id,
                    receiver_id=event.sender_id,
                    event_type=EVENT_ADMISSION_ACK,
                    payload=ack_payload,
                ))

    def _start_service(self, request: Optional[Request], visit: Visit, t: float) -> None:
        """
        Allocate a server and schedule the SERVICE_COMPLETE event.

        Service time s is drawn from an exponential distribution with mean mean_service_s,
        or from the deterministic service_time_generator if configured.
        Virtual service completion is scheduled at simulation timestamp t + s.
        """
        self.busy_servers += 1
        visit.start_t = t

        if self.service_time_generator is not None:
            service_duration = self.service_time_generator()
        else:
            service_duration = self.service_rng.expovariate(1.0 / self.mean_service_s)

        completion_t = t + service_duration

        completion_event = Event(
            timestamp=completion_t,
            sender_id=self.lp_id,
            receiver_id=self.lp_id,
            event_type=EVENT_SERVICE_COMPLETE,
            payload={"request": request, "visit": visit},
        )
        self.send_event(completion_event)

    def _handle_service_complete(self, event: Event) -> None:
        """
        Process a service completion event at simulation timestamp t = event.timestamp.

        Rule:
            1. Free the server (decrement ns).
            2. If FCFS queue is non-empty: pop head of queue and start service immediately at t.
            3. Notify state change listener of NEW state immediately after transition.
            4. Finally, invoke on_departure callback so routing / downstream transfer can occur.
        """
        t = event.timestamp
        request, visit, _ = self._extract_payload(event)
        visit.complete_t = t

        # (1) Free the server
        self.busy_servers -= 1

        # (2) If jobs are waiting in queue, immediately schedule next job in FCFS order
        if self.queue:
            next_request, next_visit = self.queue.popleft()
            self._start_service(next_request, next_visit, t)

        # (3) Notify state change listener with new state after all transitions at timestamp t
        if self.on_state_change_cb:
            self.on_state_change_cb(self.node_name, t, self.nq, self.ns)

        # (4) Finally, invoke departure callback after server is freed and queue is scheduled
        if self.on_departure_cb:
            self.on_departure_cb(request, visit, self.node_name, t)
