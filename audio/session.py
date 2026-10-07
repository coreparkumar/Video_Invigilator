"""Audio event session: temporal logic with hold timers, cooldowns, and mute.

Pure logic - time passed in as `now`; no I/O, no system clock.
"""
from collections import deque
from dataclasses import dataclass
from typing import Deque, List

from audio.config import HOP_S, EVENT_COVERAGE_S, EVENT_WINDOW_S, EVENT_COOLDOWN_S


@dataclass(frozen=True)
class AudioEvent:
    """An audio event that met the coverage threshold."""
    kind: str              # "whispering" or "talking"
    duration: float        # seconds of coverage within the window
    mean_conf: float       # average confidence of qualifying windows
    start: float           # timestamp of first qualifying window
    end: float             # timestamp when event was emitted (now)


class AudioSession:
    """Manages audio event detection with temporal filtering.

    Raises an event when whisper/speech windows cover at least EVENT_COVERAGE_S
    within a sliding EVENT_WINDOW_S window, with EVENT_COOLDOWN_S between events.
    """

    def __init__(self, hop_s: float = HOP_S):
        self.hop_s = hop_s
        self.muted = False
        # History of (timestamp, label, confidence) for qualifying labels
        self._history: Deque[tuple[float, str, float]] = deque()
        self._last_event_time = float('-inf')  # NOT 0, so first event never blocked

    def update(self, label: str, confidence: float, now: float) -> List[AudioEvent]:
        """Process one classification result.

        Args:
            label: "silence", "speech", "whisper", "noise_burst", "noise"
            confidence: 0..1 from detector
            now: current timestamp (monotonic)

        Returns:
            List of newly emitted AudioEvent (0 or 1 item)
        """
        if self.muted:
            self._history.clear()
            return []

        # Only speech/whisper count toward coverage
        if label in ("speech", "whisper"):
            self._history.append((now, label, confidence))

        # Drop entries older than EVENT_WINDOW_S
        cutoff = now - EVENT_WINDOW_S
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        # Count qualifying windows
        qualifying = [(t, lbl, conf) for t, lbl, conf in self._history
                      if lbl in ("speech", "whisper")]
        count = len(qualifying)
        coverage = count * self.hop_s

        # Check if we should emit an event
        if coverage >= EVENT_COVERAGE_S and (now - self._last_event_time) >= EVENT_COOLDOWN_S:
            # Determine kind: majority whisper vs speech
            whisper_count = sum(1 for _, lbl, _ in qualifying if lbl == "whisper")
            speech_count = count - whisper_count
            kind = "whispering" if whisper_count >= speech_count else "talking"

            mean_conf = sum(conf for _, _, conf in qualifying) / count
            start_time = qualifying[0][0]
            end_time = now

            self._last_event_time = now
            return [AudioEvent(kind, coverage, mean_conf, start_time, end_time)]

        return []

    def clear_history(self) -> None:
        """Clear all history (e.g., on mute)."""
        self._history.clear()

    def get_coverage(self, now: float) -> float:
        """Current coverage in seconds (for debugging/UI)."""
        cutoff = now - EVENT_WINDOW_S
        count = sum(1 for t, lbl, _ in self._history
                    if t >= cutoff and lbl in ("speech", "whisper"))
        return count * self.hop_s


if __name__ == "__main__":
    # Quick smoke test with FakeClock
    import numpy as np
    from audio.config import HOP_S

    class FakeClock:
        def __init__(self, start=1000.0):
            self._now = start
        @property
        def now(self):
            return self._now
        def advance(self, seconds):
            self._now += seconds

    clock = FakeClock(1000.0)
    session = AudioSession()

    print("Testing AudioSession temporal logic...")

    # Single cough - no event
    events = session.update("noise_burst", 0.8, clock.now)
    assert len(events) == 0

    # 8 whisper windows (4 seconds) -> should fire (EVENT_COVERAGE_S=3.0)
    for i in range(8):
        clock.advance(HOP_S)
        events = session.update("whisper", 0.9, clock.now)
    assert len(events) == 1
    assert events[0].kind == "whispering"
    assert events[0].duration >= 3.0
    print(f"  8 whisper windows -> {events[0].kind} ({events[0].duration:.1f}s)")

    # 7 speech windows -> talking
    clock.advance(HOP_S * 5)  # advance past cooldown
    for i in range(7):
        clock.advance(HOP_S)
        events = session.update("speech", 0.85, clock.now)
    assert len(events) == 1
    assert events[0].kind == "talking"
    print(f"  7 speech windows -> {events[0].kind} ({events[0].duration:.1f}s)")

    # Second whisper run 4s later -> blocked by cooldown (8s)
    clock.advance(HOP_S * 8)  # exactly 4s (8 hops)
    for i in range(8):
        clock.advance(HOP_S)
        events = session.update("whisper", 0.9, clock.now)
    assert len(events) == 0, "Should be blocked by cooldown"
    print(f"  2nd whisper run at 4s -> blocked by cooldown")

    # Third run after cooldown -> fires
    clock.advance(HOP_S * 8)  # 4 more seconds = 8s total
    for i in range(8):
        clock.advance(HOP_S)
        events = session.update("whisper", 0.9, clock.now)
    assert len(events) == 1
    assert events[0].kind == "whispering"
    print(f"  3rd whisper run after cooldown -> {events[0].kind}")

    # Alternating whisper/silence for 6s -> 3s coverage -> fires
    session2 = AudioSession()
    clock2 = FakeClock(1000.0)
    for i in range(24):  # 12s at 0.5s hop
        label = "whisper" if i % 2 == 0 else "silence"
        clock2.advance(HOP_S)
        events = session2.update(label, 0.9, clock2.now)
    assert len(events) == 1
    assert events[0].duration >= 3.0
    print(f"  Alternating whisper/silence 12s -> {events[0].kind} ({events[0].duration:.1f}s)")

    # Whisper + 3x silence repeated 4 times (only 4 whisper windows in 16) -> no fire
    session3 = AudioSession()
    clock3 = FakeClock(1000.0)
    for cycle in range(4):
        clock3.advance(HOP_S)
        session3.update("whisper", 0.9, clock3.now)
        for _ in range(3):
            clock3.advance(HOP_S)
            session3.update("silence", 0.0, clock3.now)
    events = session3.update("whisper", 0.9, clock3.now)  # last one
    assert len(events) == 0, f"Expected no event, got {len(events)}"
    print(f"  Whisper + 3x silence x4 -> no event (only 4 whisper windows)")

    # Mute suppresses events
    session4 = AudioSession()
    session4.muted = True
    clock4 = FakeClock(1000.0)
    for i in range(10):
        clock4.advance(HOP_S)
        events = session4.update("whisper", 0.9, clock4.now)
        assert len(events) == 0
    print(f"  Muted -> no events")

    print("All smoke tests passed!")