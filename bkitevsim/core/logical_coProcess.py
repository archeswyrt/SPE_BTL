# Copyright (C) 2026 pdnguyen of HCMC University of Technology VNU-HCM
#
# BkitTEvSim release
# Source Code License Grant: The authors hereby grant to Licensee
# personal permission to use and modify the Licensed Source Code
# for the sole purpose of studying while attending the course CO3007.
#
# Coroutine Logical Process module for BkitTEvSim Simulation Framework
#
"""
BkSys Co-Simulation Logical Process and Event Loop for PDES.

This module provides the BkSysCoLogicalProcess base class and BkSysCoEventLoop
for coroutine-based simulation using cooperative multitasking.
"""
from __future__ import annotations
import time
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import typing as tp

from .event import Event
from .logical_process import BkSysLogicalProcess, BkSysLogicalProcessState
from bkitevsim.queues import PriorityQueue

logger = logging.getLogger(__name__)

T = tp.TypeVar('T')


class BkSysFuture(asyncio.Future):
    """BkSys-specific future implementation using asyncio.Future."""
    pass

# Constants for wait return conditions
FIRST_COMPLETED = asyncio.FIRST_COMPLETED
FIRST_EXCEPTION = asyncio.FIRST_EXCEPTION
ALL_COMPLETED = asyncio.ALL_COMPLETED

# Error types (re-exported from asyncio)
CANCELLED_ERROR = asyncio.CancelledError
TIMEOUT_ERROR = asyncio.TimeoutError
EVENT_TIMEOUT_ERROR = asyncio.TimeoutError


class BkSysCoLPStatus(Enum):
    """Status of a coroutine-based LP.
    
    Attributes:
        READY: LP is ready to run.
        WAITING_EVENT: LP is waiting for an event.
        WAITING_TIME: LP is waiting for virtual time to advance.
        SUSPENDED: LP is suspended.
        TERMINATED: LP has terminated.
    """
    READY = "ready"
    WAITING_EVENT = "waiting_event"
    WAITING_TIME = "waiting_time"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


class BkSysCoEventLoop:
    """
    Single-threaded coroutine-based event loop for PDES.
    
    This event loop runs LogicalProcess coroutines cooperatively.
    Each LP is a coroutine that yields control via await statements.
    The loop manages a priority queue of events and runs ready coroutines.
    Uses LadderQueue internally for parallel-capable priority queue operations.
    
    Attributes:
        end_time: Simulation end time.
        current_time: Current simulation time.
        stats: Dictionary containing simulation statistics.
    """
    
    def __init__(self, end_time: float = float('inf'), bucket_width: float = 1.0):
        """Initialize the BkSysCoEventLoop.
        
        Args:
            end_time: Simulation end time. Defaults to infinity.
            bucket_width: Bucket width for priority queue. Defaults to 1.0.
        """
        self.end_time = end_time
        self._current_time = 0.0
        self._running = False
        
        # Event queue using PDES LadderQueue
        self._event_queue = PriorityQueue[Event](bucket_width=bucket_width)
        self._sequence = 0
        
        # LP coroutines and their state
        self._lp_coroutines: tp.Dict[int, tp.Any] = {}  # Can be callable or Task
        self._lp_instances: tp.Dict[int, BkSysLogicalProcess] = {}  # Store LP instances for handle_event
        self._lp_status: tp.Dict[int, BkSysCoLPStatus] = {}
        self._lp_waiting_for: tp.Dict[int, tp.Any] = {}
        
        # Event waiters: event_type -> list of (lp_id, future)
        self._event_waiters: tp.Dict[str, tp.List[tuple]] = defaultdict(list)
        
        # Time waiters: priority queue of (target_time, lp_id, future)
        self._time_waiters = PriorityQueue[tuple](bucket_width=bucket_width)
        
        # Statistics
        self.stats = {
            "events_processed": 0,
            "coroutines_scheduled": 0,
            "context_switches": 0,
            "wall_time": 0.0,
            "status": "COMPLETED",
            "failure_reason": None,
        }
        
        # Callbacks
        self._on_event: tp.Optional[tp.Callable[[Event], None]] = None
        self._on_lp_complete: tp.Optional[tp.Callable[[int], None]] = None
    
    @property
    def current_time(self) -> float:
        """Get current simulation time.
        
        Returns:
            Current simulation time.
        """
        return self._current_time
    
    def register_lp(self, lp: BkSysLogicalProcess) -> None:
        """Register a LogicalProcess.
        
        Args:
            lp: The LogicalProcess to register.
            
        Raises:
            ValueError: If registered LP is duplicate.
        """
        if lp.lp_id in self._lp_instances:
            raise ValueError(f"LP {lp.lp_id} already registered")
        
        # Store the LP instance for handle_event delivery
        self._lp_instances[lp.lp_id] = lp
        lp.send_event = self.schedule_event
        if hasattr(lp, 'set_event_loop'):
            lp.set_event_loop(self)
        
        # Store the coroutine factory if available
        coro_func = getattr(lp, 'coroutine_run', None)
        if callable(coro_func):
            self._lp_coroutines[lp.lp_id] = coro_func
        self._lp_status[lp.lp_id] = BkSysCoLPStatus.READY
        
        logger.debug(f"Registered LP {lp.lp_id}")
    
    def _create_lp_tasks(self) -> None:
        """Create tasks for all registered LPs (call when loop is running).
        
        Args: None
        """
        for lp_id, coro_factory in list(self._lp_coroutines.items()):
            if callable(coro_factory):
                coro = coro_factory()
                task = asyncio.create_task(coro)
                self._lp_coroutines[lp_id] = task
    
    def schedule_event(self, event: Event) -> None:
        """Schedule an event for delivery at its timestamp.
        
        Args:
            event: The event to schedule.
        """
        self._event_queue.push(event)
        # Waiters are notified in _process_events_at_current_time when the event's timestamp is reached
    
    def schedule_event_at(self, timestamp: float, event: Event) -> None:
        """Schedule an event at a specific timestamp (for LP use).
        
        Args:
            timestamp: The timestamp for the event.
            event: The event to schedule.
        """
        event.timestamp = timestamp
        self.schedule_event(event)
    
    def create_future(self):
        """Create a new PDES future.
        
        Returns:
            A new BkSysFuture instance.
        """
        return BkSysFuture()
    
    async def wait_for_event(self, lp_id: int, event_type: str, timeout: float = None) -> Event:
        """Coroutine wait for a specific event type.
        Elapse time control until an event of the given type arrives.
        
        Args:
            lp_id: ID of the LP waiting for the event.
            event_type: Type of event to wait for.
            timeout: Optional timeout (wall-clock timeout is disabled).
            
        Returns:
            The received event.
            
        Raises:
            NotImplementedError: If wall-clock timeout is specified.
        """
        if timeout is not None:
            raise NotImplementedError("Wall-clock timeout is disabled; schedule an OP_TIMEOUT event in simulation time instead.")

        future = self.create_future()
        self._event_waiters[event_type].append((lp_id, future))
        self._lp_status[lp_id] = BkSysCoLPStatus.WAITING_EVENT
        self._lp_waiting_for[lp_id] = event_type
        
        event = await future
        self._lp_status[lp_id] = BkSysCoLPStatus.READY
        return event
    
    async def wait_until_time(self, lp_id: int, target_time: float) -> float:
        """Coroutine wait until virtual time reaches target_time.
        Elaspse time control until the simulation time advances.
        
        Args:
            lp_id: ID of the LP waiting for time.
            target_time: Target virtual time to wait for.
            
        Returns:
            Current virtual time (>= target_time).
        """
        if target_time <= self._current_time:
            return self._current_time
        
        future = self.create_future()
        self._time_waiters.push((target_time, lp_id, future), priority=target_time)
        self._lp_status[lp_id] = BkSysCoLPStatus.WAITING_TIME
        self._lp_waiting_for[lp_id] = target_time
        
        try:
            await future
            return self._current_time
        finally:
            self._lp_status[lp_id] = BkSysCoLPStatus.READY
    
    async def advance_time(self, lp_id: int, delta: float) -> float:
        """Advance virtual time by delta and yield control.
        
        Args:
            lp_id: ID of the LP advancing time.
            delta: Time delta to advance.
            
        Returns:
            Elapsed new virtual time.
        """
        target = self._current_time + delta
        return await self.wait_until_time(lp_id, target)
    
    def _deliver_event(self, event: Event) -> None:
        """Deliver an event to the target logical process. Fail-fast if LP missing or error occurs."""
        if self._on_event:
            self._on_event(event)
        
        lp_instance = self._lp_instances.get(event.receiver_id)
        if lp_instance is None:
            raise ValueError(f"Target LP {event.receiver_id} does not exist for event {event.event_type}")
        
        if hasattr(lp_instance, 'handle_event'):
            old_lvt = lp_instance.local_virt_time
            lp_instance.set_local_virt_time(event.timestamp)
            logger.debug(f"SYNC: LP {event.receiver_id} LVT {old_lvt:.3f} -> {event.timestamp:.3f} (GVT={self._current_time:.3f})")
            # Fail-fast: do not catch and swallow exceptions raised in handle_event
            lp_instance.handle_event(event)
        
        # After delivering, notify any registered event waiters
        waiters = self._event_waiters.get(event.event_type, [])
        for lp_id, future in waiters:
            if lp_id == event.receiver_id and not future.done():
                future.set_result(event)
                self._lp_status[lp_id] = BkSysCoLPStatus.READY
        
        self._event_waiters[event.event_type] = [
            (lid, f) for lid, f in waiters
            if lid != event.receiver_id or not f.done()
        ]

    def _check_task_exceptions(self) -> None:
        """Check all LP tasks and raise immediately if any failed with an unhandled exception."""
        for lp_id, task in list(self._lp_coroutines.items()):
            if isinstance(task, asyncio.Task) and task.done():
                if self._lp_status.get(lp_id) != BkSysCoLPStatus.TERMINATED:
                    self._lp_status[lp_id] = BkSysCoLPStatus.TERMINATED
                if not task.cancelled():
                    exc = task.exception()
                    if exc is not None:
                        raise exc

    def _advance_time_to_next_event(self) -> bool:
        """Advance simulation time to the next scheduled event or time waiter."""
        next_event_time = self._event_queue.peek_priority()
        
        next_waiter_time = None
        if not self._time_waiters.is_empty():
            next_waiter = self._time_waiters.peek()
            if next_waiter:
                next_waiter_time = next_waiter[0]
        
        next_time = None
        if next_event_time is not None and next_event_time <= self.end_time:
            next_time = next_event_time
        if next_waiter_time is not None and next_waiter_time <= self.end_time:
            if next_time is None or next_waiter_time < next_time:
                next_time = next_waiter_time
        
        if next_time is None:
            return False
        
        self._current_time = next_time
        
        # Resolve all time waiters up to current_time
        while not self._time_waiters.is_empty():
            waiter = self._time_waiters.peek()
            if waiter is None:
                break
            wait_time = waiter[0]
            if wait_time > self._current_time:
                break
            _, lp_id, future = self._time_waiters.pop()
            if not future.done():
                future.set_result(self._current_time)
                self._lp_status[lp_id] = BkSysCoLPStatus.READY
        
        return True

    def _process_events_at_current_time(self) -> int:
        """Process all events at current time."""
        count = 0
        while not self._event_queue.is_empty():
            next_priority = self._event_queue.peek_priority()
            if next_priority is None or next_priority > self._current_time:
                break
            event = self._event_queue.pop()
            if event is None:
                break
            self._deliver_event(event)
            count += 1
            self.stats["events_processed"] += 1
        return count

    def run(self, max_iterations: int = None) -> tp.Dict:
        """Run the coroutine event loop."""
        return asyncio.run(self._run_async(max_iterations))

    async def _run_async(self, max_iterations: int = None) -> tp.Dict:
        """Async implementation of the event loop."""
        self._running = True
        start_wall = time.perf_counter()
        iteration = 0
        
        logger.info(f"Starting coroutine event loop (end_time={self.end_time})")
        
        try:
            self._create_lp_tasks()
            # Let coroutines run once to initialize and schedule initial events
            await asyncio.sleep(0)
            async_loop = asyncio.get_running_loop()
            while hasattr(async_loop, '_ready') and len(async_loop._ready) > 0:
                await asyncio.sleep(0)
            self._check_task_exceptions()
            
            while self._running:
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations})")
                    break
                
                # Check for LP tasks that finished
                self._check_task_exceptions()
                
                # Quiescence loop at current simulation timestamp:
                # Repeatedly process events and allow all coroutine continuations to complete
                # until no new events or runnable coroutines exist at current_time.
                while True:
                    # 1. Process all events at current timestamp
                    self._process_events_at_current_time()
                    self._check_task_exceptions()
                    
                    # 2. Drain all runnable coroutine continuations at current timestamp
                    await asyncio.sleep(0)
                    while hasattr(async_loop, '_ready') and len(async_loop._ready) > 0:
                        await asyncio.sleep(0)
                    self._check_task_exceptions()
                    
                    # 3. Check if any newly generated event was scheduled at current timestamp
                    next_p = self._event_queue.peek_priority()
                    if next_p is not None and next_p <= self._current_time:
                        continue
                    
                    # 4. Check if any coroutine became ready again
                    if hasattr(async_loop, '_ready') and len(async_loop._ready) > 0:
                        continue
                    
                    # Both event queue and coroutines are quiescent at current_time
                    break
                
                # Advance simulation time to the next earliest event or timer
                time_advanced = self._advance_time_to_next_event()
                if not time_advanced:
                    alive_lps = [lp_id for lp_id, status in self._lp_status.items()
                                 if status != BkSysCoLPStatus.TERMINATED]
                    if not alive_lps:
                        logger.info("All LPs terminated - simulation complete")
                    else:
                        logger.info(f"No events or timers left before end_time={self.end_time}; ending simulation at t={self._current_time}")
                    break
                
                logger.debug(f"Advanced time to {self._current_time}")
                iteration += 1

            self.stats["status"] = "COMPLETED"
            self.stats["failure_reason"] = None

        except Exception as e:
            self.stats["status"] = "FAILED"
            self.stats["failure_reason"] = f"{type(e).__name__}: {e}"
            logger.error(f"Event loop failure (transitioning to FAILED): {e}")
        finally:
            self._running = False
            # Clean up pending tasks
            pending = [t for t in self._lp_coroutines.values()
                       if isinstance(t, asyncio.Task) and not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            
            # Clean up time waiters
            while not self._time_waiters.is_empty():
                item = self._time_waiters.pop()
                if item and len(item) >= 3:
                    fut = item[2]
                    if hasattr(fut, 'done') and not fut.done():
                        fut.cancel()
            
            # Clean up event waiters
            for et, waiters in list(self._event_waiters.items()):
                for lp_id, fut in waiters:
                    if hasattr(fut, 'done') and not fut.done():
                        fut.cancel()
            self._event_waiters.clear()
            
            self.stats["wall_time"] = time.perf_counter() - start_wall
            logger.info(f"Coroutine event loop finished. Status: {self.stats['status']}, "
                        f"Events: {self.stats['events_processed']}, "
                        f"Wall time: {self.stats['wall_time']:.4f}s")
        
        return self.get_stats()
    
    def _run_ready_coroutines(self) -> None:
        """Run one step of each ready coroutine.

        Args: None
        """
        ready_lps = [lp_id for lp_id, status in self._lp_status.items()
                    if status == BkSysCoLPStatus.READY]
        
        for lp_id in ready_lps:
            task = self._lp_coroutines.get(lp_id)
            if task and not task.done():
                try:
                    if task.done():
                        self._lp_status[lp_id] = BkSysCoLPStatus.TERMINATED
                        if self._on_lp_complete:
                            self._on_lp_complete(lp_id)
                        self.stats["coroutines_scheduled"] += 1
                except Exception as e:
                    logger.error(f"Coroutine LP {lp_id} error: {e}")
                    self._lp_status[lp_id] = BkSysCoLPStatus.TERMINATED
    
    def get_stats(self) -> tp.Dict:
        """Get simulation statistics.
        
        Returns:
            Dictionary with simulation statistics.
        """
        return dict(self.stats)
    
    def stop(self) -> None:
        """Stop the event loop.
        
        Args: None
        """
        self._running = False


class BkSysCoLogicalProcess(BkSysLogicalProcess):
    """
    Base class for Coroutine BkSysLogicalProcesses.
    Events are delivered via handle_event().
    The default handle_event implementation buffers events.
    
    Attributes:
        _event_loop: Reference to the associated BkSysCoEventLoop.
        _lookahead: Lookahead time for event scheduling.
    """

    def __init__(self, lp_id: int, initial_state: tp.Optional[BkSysLogicalProcessState] = None, **kwargs):
        """Initialize a BkSysCoLogicalProcess.
        
        Args:
            lp_id: Unique identifier for this logical process.
            initial_state: Optional initial state.
            **kwargs: Additional keyword arguments. Supports 'lookahead'.
        """
        super().__init__(lp_id, initial_state)
        self._event_loop: tp.Optional[BkSysCoEventLoop] = None
        self._lookahead = kwargs.get('lookahead', 0.001)
        self._pending_events: tp.List[Event] = []  # Buffer for events received while not waiting
    
    def set_event_loop(self, event_loop: BkSysCoEventLoop) -> None:
        """Set the event loop reference (called by dispatcher).
        
        Args:
            event_loop: The BkSysCoEventLoop to associate with this LP.
        """
        self._event_loop = event_loop
    
    def send_event(self, event: Event) -> None:
        """Send an event through the event loop.
        
        Args:
            event: The event to send.
        """
        if self._event_loop:
            self._event_loop.schedule_event(event)
        else:
            # Fallback to base class behavior
            super().send_event(event)
    
    
    def _create_future(self):
        """Create a new PDES future via the event loop.
        
        Returns:
            A new BkSysFuture instance.
            
        Raises:
            RuntimeError: If event loop is not set.
        """
        if not self._event_loop:
            raise RuntimeError("Event loop not set")
        return self._event_loop.create_future()
    
    async def advance_time(self, delta: float) -> float:
        """Advance virtual time by delta.
        
        Args:
            delta: Time delta to advance.
            
        Returns:
            New current virtual time.
            
        Raises:
            RuntimeError: If event loop is not set.
        """
        if not self._event_loop:
            raise RuntimeError("Event loop not set")
        return await self._event_loop.advance_time(self.lp_id, delta)
    
    def elapse_virt_time(self, delta: float) -> float:
        """Elapse the Local Virtual Time by a delta (local to this LP).
        
        Args:
            delta: Time delta to elapse (must be non-negative).
            
        Returns:
            New local virtual time.
            
        Raises:
            ValueError: If delta is negative.
        """
        if delta < 0:
            raise ValueError("Delta must be non-negative")
        self.local_virt_time += delta
        return self.local_virt_time
    
    def handle_event(self, event: Event) -> None:
        """Handle an incoming event by buffering it for wait_for consumption.
        The event is buffered and will be consumed by wait_for/wait_for_multiple.
        
        Args:
            event: The event to handle.
        """
        self._pending_events.append(event)
    
    async def wait_for(self, event_type: str, timeout: float = None) -> Event:
        """Wait for an event of the given type and wrap the event loop's wait_for_event.
        
        Args:
            event_type: Type of event to wait for.
            timeout: Optional timeout in wall-clock seconds.
            
        Returns:
            The received event.
            
        Raises:
            TIMEOUT_ERROR: If timeout expires before event arrives.
            RuntimeError: If event loop is not set.
        """
        if not self._event_loop:
            raise RuntimeError("Event loop not set")
        return await self._event_loop.wait_for_event(self.lp_id, event_type, timeout)
    
    async def wait_until(self, target_time: float) -> float:
        """Wait until virtual time reaches target_time.
        
        Args:
            target_time: Target virtual time to wait for.
            
        Returns:
            Current virtual time (which will be >= target_time).
            
        Raises:
            RuntimeError: If event loop is not set.
        """
        if not self._event_loop:
            raise RuntimeError("Event loop not set")
        return await self._event_loop.wait_until_time(self.lp_id, target_time)
    
    async def wait_for_multiple(self, event_types: tp.List[str], timeout: float = None) -> Event:
        """Wait for any of multiple event types.
        
        Args:
            event_types: List of event types to wait for.
            timeout: Optional timeout in wall-clock seconds.
            
        Returns:
            The first received event.
            
        Raises:
            NotImplementedError: If wall-clock timeout is specified.
            RuntimeError: If event loop is not set.
        """
        if not self._event_loop:
            raise RuntimeError("Event loop not set")
        if timeout is not None:
            raise NotImplementedError("Wall-clock timeout is disabled; schedule an OP_TIMEOUT event in simulation time instead.")
        
        # Check pending event
        for i, pending_event in enumerate(self._pending_events):
            if pending_event.event_type in event_types:
                self._pending_events.pop(i)
                return pending_event
        
        # Check future 
        futures = []
        for et in event_types:
            future = self._event_loop.create_future()
            self._event_loop._event_waiters[et].append((self.lp_id, future))
            futures.append((et, future))
        
        self._event_loop._lp_status[self.lp_id] = BkSysCoLPStatus.WAITING_EVENT
        self._event_loop._lp_waiting_for[self.lp_id] = event_types
        
        try:
            # Extract future objects
            future_objects = [f for _, f in futures]
            done, pending = await asyncio.wait(future_objects, return_when=FIRST_COMPLETED)
            
            # Cancel pending futures
            for et, future in futures:
                if future not in done:
                    future.cancel()
            
            # Clean up waiters
            for et, future in futures:
                waiters = self._event_loop._event_waiters[et]
                self._event_loop._event_waiters[et] = [(lid, f) for lid, f in waiters if lid != self.lp_id]
            
            self._event_loop._lp_status[self.lp_id] = BkSysCoLPStatus.READY
            
            # Return the completed event
            # Buffer any additional completed events
            completed_events = []
            for future in done:
                if not future.cancelled():
                    completed_events.append(future.result())
            
            if completed_events:
                # Return the first event
                event = completed_events[0]
                # Buffer the rest
                for buffered_event in completed_events[1:]:
                    self._pending_events.append(buffered_event)
                return event
            
            raise RuntimeError("No future completed")
            
        except TIMEOUT_ERROR:
            self._event_loop._lp_status[self.lp_id] = BkSysCoLPStatus.READY
            for et, future in futures:
                waiters = self._event_loop._event_waiters[et]
                self._event_loop._event_waiters[et] = [(lid, f) for lid, f in waiters if lid != self.lp_id]
            raise
    
    # Abstract method implementations
    def initialize(self) -> None:
        """Initialize is handled by coroutine_run."""
        pass
    
    async def coroutine_run(self) -> None:
        """Main coroutine - optional or implemented by subclasses."""
        pass


# Export commonly used types
__all__ = [
    "BkSysCoLPStatus",
    "BkSysCoEventLoop",
    "BkSysCoLogicalProcess",
    "BkSysFuture",
    "CANCELLED_ERROR",
    "TIMEOUT_ERROR",
    "FIRST_COMPLETED",
    "FIRST_EXCEPTION",
    "ALL_COMPLETED",
]