# Copyright (C) 2026 SPE Team - Dev 1
"""
Test Fail-Fast: Verification of robust error handling and fail-fast behavior.

Tests:
1. Intentional unhandled exception in handle_event:
   Verifies loop transitions to FAILED status, records failure_reason,
   and cleans up pending tasks/futures without swallowing errors.
2. Intentional unhandled exception in LP coroutine:
   Verifies loop transitions to FAILED status, records failure_reason,
   and cleans up without swallowing errors.
3. Event addressed to a non-existent LP receiver:
   Verifies fail-fast error detection and transition to FAILED status.
4. Calling wait_for_event or wait_for_multiple with wall-clock timeout:
   Verifies blocking with NotImplementedError.
5. Simulation auto-selection of BkSysCoEventLoop for QueueNodeLP.
"""

import unittest
import asyncio
from bkitevsim.core.event import Event
from bkitevsim.core.logical_coProcess import BkSysCoLogicalProcess, BkSysCoEventLoop
from tests.trace_utils import print_table


class FaultyHandlerLP(BkSysCoLogicalProcess):
    """LP that raises an unhandled exception inside handle_event."""
    def handle_event(self, event: Event) -> None:
        raise RuntimeError("Intentional handler failure for fail-fast test")


class FaultyCoroutineLP(BkSysCoLogicalProcess):
    """LP whose coroutine raises an unhandled exception during execution."""
    async def coroutine_run(self) -> None:
        await asyncio.sleep(0)
        raise ValueError("Intentional coroutine task failure for fail-fast test")


class NormalLP(BkSysCoLogicalProcess):
    """Normal LP to verify pending futures/tasks are cancelled on failure."""
    def __init__(self, lp_id: int):
        super().__init__(lp_id)
        self.fut = None

    async def coroutine_run(self) -> None:
        loop = asyncio.get_running_loop()
        self.fut = loop.create_future()
        await self.fut


class TestFailFast(unittest.TestCase):
    """Test suite for fail-fast error handling in BkSysCoEventLoop."""

    summary_rows = []

    @classmethod
    def tearDownClass(cls):
        """Print summary table of all fail-fast test cases upon completion (even if tests failed)."""
        if cls.summary_rows:
            print_table(
                title="V05: Fail-Fast Error Handling & Robustness Verification Summary",
                headers=["Case", "Expected Behavior", "Actual Status / Result", "Match"],
                rows=cls.summary_rows,
                alignments=["left", "left", "left", "center"],
            )

    def test_handler_exception_fail_fast(self):
        """Loop must transition to FAILED status when handle_event raises an exception."""
        loop = BkSysCoEventLoop(end_time=10.0)
        faulty = FaultyHandlerLP(1)
        normal = NormalLP(2)
        loop.register_lp(faulty)
        loop.register_lp(normal)

        loop.schedule_event(Event(timestamp=1.0, sender_id=1, receiver_id=1, event_type="CRASH"))
        stats = loop.run()

        # Dynamic match evaluation BEFORE assertions
        status = stats.get("status")
        reason = str(stats.get("failure_reason") or "")
        normal_fut_ok = (normal.fut is None or normal.fut.cancelled() or normal.fut.done())
        is_ok = (
            status == "FAILED"
            and "RuntimeError" in reason
            and "Intentional handler failure" in reason
            and normal_fut_ok
        )
        match_str = "PASS" if is_ok else "FAIL"

        TestFailFast.summary_rows.append([
            "Handler Exception",
            "Loop transitions to FAILED, records RuntimeError, cleans up tasks",
            f"Status: {status} ({'RuntimeError caught' if 'RuntimeError' in reason else 'Unexpected error'})",
            match_str,
        ])

        # Assertions executed after recording summary row
        self.assertEqual(stats["status"], "FAILED")
        self.assertIsNotNone(stats["failure_reason"])
        self.assertIn("RuntimeError", stats["failure_reason"])
        self.assertIn("Intentional handler failure", stats["failure_reason"])
        self.assertTrue(normal_fut_ok)

    def test_coroutine_task_exception_fail_fast(self):
        """Loop must transition to FAILED status when a coroutine task fails."""
        loop = BkSysCoEventLoop(end_time=10.0)
        faulty = FaultyCoroutineLP(1)
        normal = NormalLP(2)
        loop.register_lp(faulty)
        loop.register_lp(normal)

        # Trigger event at t=1.0
        loop.schedule_event(Event(timestamp=1.0, sender_id=1, receiver_id=1, event_type="DUMMY"))
        stats = loop.run()

        # Dynamic match evaluation BEFORE assertions
        status = stats.get("status")
        reason = str(stats.get("failure_reason") or "")
        is_ok = (
            status == "FAILED"
            and "ValueError" in reason
            and "Intentional coroutine task failure" in reason
        )
        match_str = "PASS" if is_ok else "FAIL"

        TestFailFast.summary_rows.append([
            "Coroutine Task Exception",
            "Loop transitions to FAILED, records ValueError, cleans up tasks",
            f"Status: {status} ({'ValueError caught' if 'ValueError' in reason else 'Unexpected error'})",
            match_str,
        ])

        # Assertions executed after recording summary row
        self.assertEqual(stats["status"], "FAILED")
        self.assertIsNotNone(stats["failure_reason"])
        self.assertIn("ValueError", stats["failure_reason"])
        self.assertIn("Intentional coroutine task failure", stats["failure_reason"])

    def test_nonexistent_lp_fail_fast(self):
        """Sending an event to a non-existent LP receiver must trigger fail-fast."""
        loop = BkSysCoEventLoop(end_time=10.0)
        normal = NormalLP(1)
        loop.register_lp(normal)

        # Receiver LP 999 does not exist
        loop.schedule_event(Event(timestamp=1.0, sender_id=1, receiver_id=999, event_type="GHOST"))
        stats = loop.run()

        # Dynamic match evaluation BEFORE assertions
        status = stats.get("status")
        reason = str(stats.get("failure_reason") or "")
        is_ok = (status == "FAILED" and "Target LP 999 does not exist" in reason)
        match_str = "PASS" if is_ok else "FAIL"

        TestFailFast.summary_rows.append([
            "Non-existent LP Receiver",
            "Loop transitions to FAILED, records target LP not found",
            f"Status: {status} ({'Target LP 999 not found' if 'Target LP 999' in reason else 'Unexpected error'})",
            match_str,
        ])

        # Assertions executed after recording summary row
        self.assertEqual(stats["status"], "FAILED")
        self.assertIsNotNone(stats["failure_reason"])
        self.assertIn("Target LP 999 does not exist", stats["failure_reason"])

    def test_wall_clock_timeout_blocked(self):
        """Calling wait_for_event or wait_for_multiple with timeout must raise NotImplementedError."""
        loop = BkSysCoEventLoop(end_time=10.0)

        # Test wait_for_event with timeout
        async def run_wait_event():
            await loop.wait_for_event(1, "TEST", timeout=0.01)

        raised_event = False
        try:
            asyncio.run(run_wait_event())
        except NotImplementedError as ctx1:
            if "Wall-clock timeout is disabled" in str(ctx1):
                raised_event = True
        except Exception:
            pass

        # Test wait_for_multiple with timeout
        lp = NormalLP(1)
        lp.set_event_loop(loop)

        async def run_wait_multiple():
            await lp.wait_for_multiple(["EV1", "EV2"], timeout=0.01)

        raised_multiple = False
        try:
            asyncio.run(run_wait_multiple())
        except NotImplementedError as ctx2:
            if "Wall-clock timeout is disabled" in str(ctx2):
                raised_multiple = True
        except Exception:
            pass

        # Dynamic match evaluation BEFORE assertions
        is_ok = (raised_event and raised_multiple)
        match_str = "PASS" if is_ok else "FAIL"

        TestFailFast.summary_rows.append([
            "Wall-Clock Timeout Blocked",
            "wait_for_event / wait_for_multiple with timeout raise NotImplementedError",
            "NotImplementedError successfully raised for both APIs" if is_ok else "Timeout blocking failed",
            match_str,
        ])

        # Assertions executed after recording summary row
        self.assertTrue(raised_event, "wait_for_event with timeout must raise NotImplementedError")
        self.assertTrue(raised_multiple, "wait_for_multiple with timeout must raise NotImplementedError")

    def test_simulation_auto_selects_coroutine_loop(self):
        """Simulation with QueueNodeLP must automatically select BkSysCoEventLoop."""
        from bkitevsim.core.simulation import Simulation, SimulationOptions
        from queue_node import QueueNodeLP

        sim = Simulation(
            options=SimulationOptions(end_time=10.0),
            models={"QueueNodeLP": QueueNodeLP},
        )
        sim.create_lp(lp_id=1, model_name="QueueNodeLP", node_name="Ingress", c=1, B=20, mean_service_s=0.005)
        sim._create_dispatcher()

        # Dynamic match evaluation BEFORE assertions
        selected_name = sim.dispatcher.__class__.__name__
        is_ok = isinstance(sim.dispatcher, BkSysCoEventLoop)
        match_str = "PASS" if is_ok else "FAIL"

        TestFailFast.summary_rows.append([
            "Simulation Loop Auto-Selection",
            "QueueNodeLP inherits BkSysCoLP -> Simulation selects BkSysCoEventLoop",
            f"Selected dispatcher: {selected_name}",
            match_str,
        ])

        # Assertions executed after recording summary row
        self.assertIsInstance(sim.dispatcher, BkSysCoEventLoop)


if __name__ == "__main__":
    unittest.main()
