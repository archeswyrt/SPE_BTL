# Copyright (C) 2026 SPE Team - Dev 1 (DES Core & Queueing Station)
# Contracts module for discrete-event simulation of Ingress-CPU-Disk network.
"""
Shared contracts, dataclasses, event types, and callback protocols for Dev 1, Dev 2, and Dev 3.
All docstrings and comments adhere to System Performance Evaluation (SPE) and Queueing Theory terminology.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, Protocol, runtime_checkable


# -------------------------------------------------------------------------
# Event Types
# -------------------------------------------------------------------------
EVENT_ARRIVAL = "ARRIVAL"
EVENT_SERVICE_COMPLETE = "SERVICE_COMPLETE"
EVENT_ADMISSION_ACK = "ADMISSION_ACK"
EVENT_OP_TIMEOUT = "OP_TIMEOUT"


# -------------------------------------------------------------------------
# Core Data Models
# -------------------------------------------------------------------------
@dataclass
class Request:
    """
    Represents an end-to-end user request entering the simulated system.

    Attributes:
        request_id: Globally unique request identifier (e.g., 'R1', 'R2').
        external_arrival_t: Virtual simulation timestamp of external arrival at Ingress.
        terminal_t: Virtual simulation timestamp when request finishes (Sink) or is dropped.
        outcome: Terminal outcome of the request ('COMPLETED' or 'DROPPED').
        drop_node: Identifier of the queueing station where rejection occurred, if dropped.
    """
    request_id: str
    external_arrival_t: float
    terminal_t: Optional[float] = None
    outcome: Optional[str] = None
    drop_node: Optional[str] = None


@dataclass
class Visit:
    """
    Represents a single visit of a request to a specific queueing station.

    Attributes:
        request_id: Identifier of the parent request.
        visit_id: Unique identifier for this visit (e.g., 'R1-CPU-1', 'R1-CPU-2').
        node: Target station identifier ('Ingress', 'CPU', 'Disk').
        arrival_t: Virtual simulation timestamp of arrival at this station.
        start_t: Virtual simulation timestamp when service begins on a server.
        complete_t: Virtual simulation timestamp when service completes on a server.
        accepted: Admission result at this station (True if admitted, False if dropped due to buffer capacity B).
    """
    request_id: str
    visit_id: str
    node: str
    arrival_t: float
    start_t: Optional[float] = None
    complete_t: Optional[float] = None
    accepted: bool = False


@dataclass
class OperationPayload:
    """
    Payload carrier for admission requests and acknowledgments.

    Attributes:
        operation_id: Unique identifier for the asynchronous routing operation / ACK pairing.
        request: Associated Request instance.
        visit: Associated Visit instance.
        accepted: Admission acknowledgment status (True if admitted, False if buffer full).
        details: Optional additional payload data.
    """
    operation_id: Optional[str] = None
    request: Optional[Request] = None
    visit: Optional[Visit] = None
    accepted: Optional[bool] = None
    details: Optional[dict[str, Any]] = None


# -------------------------------------------------------------------------
# Callback Protocols for Inter-Dev Integration
# -------------------------------------------------------------------------
@runtime_checkable
class DepartureCallback(Protocol):
    """Callback invoked when a request completes service and departs a station."""
    def __call__(self, request: Optional[Request], visit: Visit, node: str, t: float) -> None:
        ...


@runtime_checkable
class RejectionCallback(Protocol):
    """Callback invoked when a request arrival is rejected due to finite buffer capacity B."""
    def __call__(self, request: Optional[Request], visit: Visit, node: str, t: float) -> None:
        ...


@runtime_checkable
class StateChangeCallback(Protocol):
    """
    Callback invoked on queue station state changes for time-weighted metric integration.

    Args:
        node: Name of the station ('Ingress', 'CPU', 'Disk').
        t: Simulation virtual timestamp when the state transition occurs.
        nq: Number of jobs in the FCFS waiting queue immediately after the transition.
        ns: Number of busy servers currently in service immediately after the transition.
    """
    def __call__(self, node: str, t: float, nq: int, ns: int) -> None:
        ...
