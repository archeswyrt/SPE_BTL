# Copyright (C) 2026 SPE Team - Dev 2
"""
Shared helpers for the Dev 2 network test cases (test_n02, test_n03, ...).

Each test case stays in its own file, like the Dev 1 cases; only these two small helpers
are shared so that they are not copy-pasted into every file:

    ScriptedGaps            - stand-in for the arrival random stream (fixed inter-arrival gaps)
    NetworkMetricCollector  - per-station time-weighted integrals over a window [t0, t1),
                              following the Dev 3 contract (integrate the OLD saved state up
                              to t, clamp to the window, then store the NEW state).
"""

from contracts import Request


class ScriptedGaps:
    """Stand-in for the arrival random stream: returns pre-defined inter-arrival gaps."""

    def __init__(self, gaps):
        self._it = iter(gaps)

    def expovariate(self, lambd: float) -> float:
        # After the script is exhausted the next arrival lies far beyond t_end.
        return next(self._it, 1e9)


class NetworkMetricCollector:
    """Time-weighted per-station metrics over [t0, t1)."""

    NODES = ("Ingress", "CPU", "Disk")

    def __init__(self, t0: float, t1: float):
        self.t0 = t0
        self.t1 = t1
        self.last_t = {n: t0 for n in self.NODES}
        self.saved = {n: (0, 0) for n in self.NODES}                 # (nq, ns)
        self.integral_ns = {n: 0.0 for n in self.NODES}
        self.integral_nq = {n: 0.0 for n in self.NODES}
        self.state_history: list[tuple[str, float, int, int]] = []   # (node, t, nq, ns)
        self.externals: list[tuple[str, float]] = []
        self.terminals: list[tuple[str, str, float]] = []            # (request_id, outcome, t)

    # --- callbacks required by build_network(metrics=...) -------------------------
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

    # --- helpers -------------------------------------------------------------------
    def close_window(self) -> None:
        """Close the measurement window at t1, integrating the remaining duration."""
        for node in self.NODES:
            if self.t1 > self.last_t[node]:
                nq, ns = self.saved[node]
                duration = self.t1 - max(self.last_t[node], self.t0)
                self.integral_ns[node] += ns * duration
                self.integral_nq[node] += nq * duration
                self.last_t[node] = self.t1

    def state_after(self, node: str, t: float):
        """(nq, ns) right after the last state change of ``node`` at or before t."""
        value = None
        for n, ts, nq, ns in self.state_history:
            if n == node and ts <= t:
                value = (nq, ns)
        return value

    def history_of(self, node: str):
        """[(t, nq, ns), ...] of one station, in order."""
        return [(t, nq, ns) for n, t, nq, ns in self.state_history if n == node]
