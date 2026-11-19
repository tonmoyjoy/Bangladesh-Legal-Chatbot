"""Per-session sliding-window rate limiter (protects the shared Groq key)."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class SlidingWindowLimiter:
    def __init__(
        self,
        max_events: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_events = max_events
        self.window = window_seconds
        self.clock = clock
        self._events: deque[float] = deque()

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0] >= self.window:
            self._events.popleft()

    def try_acquire(self) -> tuple[bool, float]:
        """Return ``(allowed, seconds_to_wait)``; records the event if allowed."""
        now = self.clock()
        self._trim(now)
        if len(self._events) >= self.max_events:
            return False, max(0.0, self.window - (now - self._events[0]))
        self._events.append(now)
        return True, 0.0
