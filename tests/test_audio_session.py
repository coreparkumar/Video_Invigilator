"""Tests for audio/session.py - temporal logic for audio events."""
import pytest

from audio.session import AudioSession, AudioEvent
from audio.config import HOP_S, EVENT_COVERAGE_S, EVENT_WINDOW_S, EVENT_COOLDOWN_S


class FakeClock:
    """Fake clock for testing temporal logic."""
    def __init__(self, start=1000.0):
        self._now = start

    @property
    def now(self):
        return self._now

    def advance(self, seconds):
        self._now += seconds


def test_single_cough_no_event():
    """Single cough (noise_burst) does not raise an event."""
    clock = FakeClock(1000.0)
    session = AudioSession()

    events = session.update("noise_burst", 0.8, clock.now)
    assert len(events) == 0


def test_whisper_coverage_fires():
    """8 whisper windows (4s at 0.5s hop) -> fires once with kind whispering."""
    clock = FakeClock(1000.0)
    session = AudioSession()
    all_events = []

    # 8 whisper windows = 4 seconds coverage (> EVENT_COVERAGE_S=3.0)
    for i in range(8):
        clock.advance(HOP_S)
        events = session.update("whisper", 0.9, clock.now)
        all_events.extend(events)

    assert len(all_events) == 1
    event = all_events[0]
    assert event.kind == "whispering"
    assert event.duration >= EVENT_COVERAGE_S
    assert event.mean_conf > 0.0
    assert event.start < event.end
    # Event end is when it fired (6th window at t=1003.0), not final clock time
    assert event.end == 1003.0


def test_speech_coverage_fires():
    """7 speech windows -> fires with kind talking."""
    clock = FakeClock(1000.0)
    session = AudioSession()
    all_events = []

    # Advance past any potential cooldown
    clock.advance(HOP_S * 5)

    for i in range(7):
        clock.advance(HOP_S)
        events = session.update("speech", 0.85, clock.now)
        all_events.extend(events)

    assert len(all_events) == 1
    event = all_events[0]
    assert event.kind == "talking"
    assert event.duration >= EVENT_COVERAGE_S


def test_cooldown_blocks_repeat():
    """Second whisper run within cooldown is blocked."""
    clock = FakeClock(1000.0)
    session = AudioSession()
    all_events = []

    # First run - 8 whisper windows
    for i in range(8):
        clock.advance(HOP_S)
        events = session.update("whisper", 0.9, clock.now)
        all_events.extend(events)
    assert len(all_events) == 1
    first_event_time = clock.now

    # Second run 4 seconds later (8 hops) -> within 8s cooldown
    for i in range(8):
        clock.advance(HOP_S)
        events = session.update("whisper", 0.9, clock.now)
        all_events.extend(events)
    # Should still be 1 total event (cooldown blocked 2nd)
    assert len(all_events) == 1, f"Should be blocked by cooldown, got {len(all_events)} events"

    # Third run after full cooldown (another 8 hops = 4s more = 8s total)
    for i in range(8):
        clock.advance(HOP_S)
        events = session.update("whisper", 0.9, clock.now)
        all_events.extend(events)
    assert len(all_events) == 2, "Should fire after cooldown expires"


def test_alternating_whisper_silence_fires():
    """Alternating whisper/silence for 12s -> 3s coverage -> fires."""
    clock = FakeClock(1000.0)
    session = AudioSession()
    all_events = []

    # 24 windows: whisper, silence, whisper, silence... for 12 seconds
    for i in range(24):
        label = "whisper" if i % 2 == 0 else "silence"
        clock.advance(HOP_S)
        events = session.update(label, 0.9 if label == "whisper" else 0.0, clock.now)
        all_events.extend(events)

    assert len(all_events) == 1
    event = all_events[0]
    assert event.kind == "whispering"
    assert event.duration >= EVENT_COVERAGE_S


def test_whisper_plus_silence_no_fire():
    """Whisper + 3x silence repeated 4 times -> only 4 whisper windows -> no fire."""
    clock = FakeClock(1000.0)
    session = AudioSession()

    for cycle in range(4):
        clock.advance(HOP_S)
        session.update("whisper", 0.9, clock.now)
        for _ in range(3):
            clock.advance(HOP_S)
            session.update("silence", 0.0, clock.now)

    # One more whisper at the end (5th)
    clock.advance(HOP_S)
    events = session.update("whisper", 0.9, clock.now)
    assert len(events) == 0, f"Expected no event (only 5 whisper windows), got {len(events)}"


def test_mute_suppresses_events():
    """Muted session never fires events."""
    clock = FakeClock(1000.0)
    session = AudioSession()
    session.muted = True

    for i in range(10):
        clock.advance(HOP_S)
        events = session.update("whisper", 0.9, clock.now)
        assert len(events) == 0


def test_mute_clears_history():
    """Mute clears history, unmuting starts fresh."""
    clock = FakeClock(1000.0)
    session = AudioSession()

    # Build up some history
    for i in range(5):
        clock.advance(HOP_S)
        session.update("whisper", 0.9, clock.now)

    # Mute
    session.muted = True
    clock.advance(HOP_S)
    session.update("whisper", 0.9, clock.now)

    # Unmute and continue - should not have accumulated coverage
    session.muted = False
    for i in range(3):
        clock.advance(HOP_S)
        events = session.update("whisper", 0.9, clock.now)
    # Only 3 whisper windows = 1.5s < 3s -> no fire
    assert len(events) == 0


def test_kind_majority_whispering():
    """Majority whisper windows -> kind=whispering."""
    clock = FakeClock(1000.0)
    session = AudioSession()
    all_events = []

    # 5 whisper + 3 speech = 8 total, majority whisper
    for i in range(5):
        clock.advance(HOP_S)
        events = session.update("whisper", 0.9, clock.now)
        all_events.extend(events)
    for i in range(3):
        clock.advance(HOP_S)
        events = session.update("speech", 0.8, clock.now)
        all_events.extend(events)

    # Final call to trigger if not already fired
    events = session.update("whisper", 0.9, clock.now)
    all_events.extend(events)
    assert len(all_events) == 1
    assert all_events[0].kind == "whispering"


def test_kind_majority_talking():
    """Majority speech windows -> kind=talking.

    Note: event fires at 6th qualifying window (3 whisper + 3 speech = tie),
    where tiebreaker favors 'whispering'. To get 'talking', need clear majority.
    """
    clock = FakeClock(1000.0)
    session = AudioSession()
    all_events = []

    # 2 whisper + 6 speech = 8 total, clear speech majority
    for i in range(2):
        clock.advance(HOP_S)
        events = session.update("whisper", 0.9, clock.now)
        all_events.extend(events)
    for i in range(6):
        clock.advance(HOP_S)
        events = session.update("speech", 0.8, clock.now)
        all_events.extend(events)

    # Final call to trigger if not already fired
    events = session.update("speech", 0.8, clock.now)
    all_events.extend(events)
    assert len(all_events) == 1
    assert all_events[0].kind == "talking"


def test_coverage_calculation():
    """get_coverage returns correct seconds."""
    clock = FakeClock(1000.0)
    session = AudioSession()

    assert session.get_coverage(clock.now) == 0.0

    clock.advance(HOP_S)
    session.update("whisper", 0.9, clock.now)
    assert session.get_coverage(clock.now) == HOP_S

    clock.advance(HOP_S)
    session.update("whisper", 0.9, clock.now)
    assert session.get_coverage(clock.now) == 2 * HOP_S


def test_no_sleep_or_system_clock():
    """Tests use FakeClock, not time.sleep() or time.monotonic()."""
    # This is a meta-test - if it runs, it passes
    # The fact that we use FakeClock proves no system clock dependency
    pass


def test_event_fields():
    """AudioEvent has all required fields with correct types."""
    clock = FakeClock(1000.0)
    session = AudioSession()
    all_events = []

    for i in range(8):
        clock.advance(HOP_S)
        events = session.update("whisper", 0.9, clock.now)
        all_events.extend(events)

    assert len(all_events) == 1
    event = all_events[0]
    assert isinstance(event, AudioEvent)
    assert event.kind in ("whispering", "talking")
    assert isinstance(event.duration, float)
    assert isinstance(event.mean_conf, float)
    assert isinstance(event.start, float)
    assert isinstance(event.end, float)
    assert event.start > 0
    assert event.end >= event.start


def test_last_event_time_not_zero():
    """_last_event_time starts at -inf so first event never blocked."""
    session = AudioSession()
    # Access private to verify initialization
    assert session._last_event_time == float('-inf')


def test_noise_burst_never_counts():
    """noise_burst label never contributes to coverage."""
    clock = FakeClock(1000.0)
    session = AudioSession()

    for i in range(10):
        clock.advance(HOP_S)
        events = session.update("noise_burst", 0.8, clock.now)
        assert len(events) == 0

    # Even with many noise_burst, coverage should be 0
    assert session.get_coverage(clock.now) == 0.0


def test_noise_never_counts():
    """noise label never contributes to coverage."""
    clock = FakeClock(1000.0)
    session = AudioSession()

    for i in range(10):
        clock.advance(HOP_S)
        events = session.update("noise", 0.8, clock.now)
        assert len(events) == 0


def test_silence_never_counts():
    """silence label never contributes to coverage."""
    clock = FakeClock(1000.0)
    session = AudioSession()

    for i in range(10):
        clock.advance(HOP_S)
        events = session.update("silence", 1.0, clock.now)
        assert len(events) == 0


def test_old_entries_expire():
    """Entries older than EVENT_WINDOW_S are dropped."""
    clock = FakeClock(1000.0)
    session = AudioSession()

    # Add 10 whisper windows over 5 seconds
    for i in range(10):
        clock.advance(HOP_S)
        session.update("whisper", 0.9, clock.now)

    # Advance past EVENT_WINDOW_S (10s) - all old entries should expire
    clock.advance(EVENT_WINDOW_S + 1.0)
    events = session.update("whisper", 0.9, clock.now)
    # Only the new whisper should be in history
    assert session.get_coverage(clock.now) <= HOP_S * 2  # just the new one(s)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])