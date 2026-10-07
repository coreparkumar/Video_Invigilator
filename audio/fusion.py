"""Multimodal fusion: combines audio events with video context cues.

Produces risk-scored alerts with evidence breakdown.
"""
from dataclasses import dataclass
from typing import List, Optional
from audio.config import (
    VIDEO_CUE_WEIGHTS, AUDIO_HOLD_S, FUSION_ALERT_RISK,
    FUSION_COOLDOWN_S, AUDIO_ONLY_CAP, FUSION_WEIGHTS,
    LIP_ACTIVITY_THRESHOLD
)


@dataclass(frozen=True)
class Alert:
    """A fused alert with risk score and evidence."""
    t: float                      # timestamp
    risk: float                   # 0..1
    severity: str                 # "alert" or "review"
    reasons: List[str]            # human-readable evidence
    audio_kind: str               # "whispering" or "talking"
    video_cues: List[str]         # active video cues
    audio_conf: float             # audio event confidence


class Fusion:
    """Fuses audio events with video context into risk-scored alerts."""

    def __init__(self):
        self._audio_until: float = float('-inf')
        self._audio_conf: float = 0.0
        self._audio_kind: str = ""
        self._audio_duration: float = 0.0
        self._last_fused: float = float('-inf')

    def update(
        self,
        audio_events: List,
        video_active: set,
        now: float,
        lip_activity: float = 0.0,
    ) -> List:
        """Process new audio events and current video context.

        Args:
            audio_events: list of AudioEvent from AudioSession
            video_active: set of active video gesture keys from InvigilatorSession
            now: current timestamp
            lip_activity: 0..1 lip activity score from LipTracker

        Returns:
            List of Alert (0 or more)
        """
        from audio.config import (
            VIDEO_CUE_WEIGHTS, AUDIO_HOLD_S, FUSION_ALERT_RISK,
            FUSION_COOLDOWN_S, AUDIO_ONLY_CAP, FUSION_WEIGHTS,
            LIP_ACTIVITY_THRESHOLD
        )

        alerts = []

        # Determine active video cues (weighted)
        cues = []
        for cue in video_active:
            if cue in VIDEO_CUE_WEIGHTS:
                cues.append(cue)
        if lip_activity >= LIP_ACTIVITY_THRESHOLD:
            cues.append("lip_activity")

        # Process new audio events
        last_evt = None
        for evt in audio_events:
            self._audio_until = now + AUDIO_HOLD_S
            self._audio_conf = evt.mean_conf
            self._audio_kind = evt.kind
            self._audio_duration = evt.duration
            last_evt = evt

            # Audio-only event -> low-severity "review"
            if not cues:
                risk = min(AUDIO_ONLY_CAP, 0.4 * evt.mean_conf)
                alerts.append(Alert(
                    t=now,
                    risk=risk,
                    severity="review",
                    reasons=[f"{evt.kind} for {evt.duration:.1f}s (audio only, no visual cue)"],
                    audio_kind=evt.kind,
                    video_cues=[],
                    audio_conf=evt.mean_conf,
                ))

        # Fuse: audio context active + visual cues present
        if cues and (now - self._last_fused) >= FUSION_COOLDOWN_S and now < self._audio_until:
            wa, wv, wb = FUSION_WEIGHTS
            max_cue_weight = max(VIDEO_CUE_WEIGHTS.get(c, 0.0) for c in cues)
            risk = min(1.0, wa * self._audio_conf + wv * max_cue_weight + wb)
            severity = "alert" if risk >= FUSION_ALERT_RISK else "review"

            # Build reason list - use stored audio event info
            if last_evt:
                reasons = [f"{self._audio_kind} for {last_evt.duration:.1f}s (conf {self._audio_conf:.2f})"]
            else:
                reasons = [f"{self._audio_kind} (conf {self._audio_conf:.2f})"]
            for c in cues:
                if c == "lip_activity":
                    reasons.append(f"lip activity")
                else:
                    reasons.append(c.replace("_", " "))

            fused_alert = Alert(
                t=now,
                risk=risk,
                severity=severity,
                reasons=reasons,
                audio_kind=self._audio_kind,
                video_cues=cues,
                audio_conf=self._audio_conf,
            )

            # Replace any audio-only review from this call
            if alerts and alerts[-1].severity == "review":
                alerts[-1] = fused_alert
            else:
                alerts.append(fused_alert)

            self._last_fused = now

        return alerts

    def get_audio_context_active(self, now: float) -> bool:
        """Check if audio context is still within hold window."""
        return now < self._audio_until

    def reset(self) -> None:
        """Reset fusion state."""
        self._audio_until = float('-inf')
        self._audio_conf = 0.0
        self._audio_kind = ""
        self._audio_duration = 0.0
        self._last_fused = float('-inf')


if __name__ == "__main__":
    # Quick smoke test
    from audio.session import AudioSession, AudioEvent
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
    fusion = Fusion()

    print("Testing Fusion...")

    # Audio only -> review (risk <= 0.4)
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)
    alerts = fusion.update([audio_event], set(), clock.now)
    assert len(alerts) == 1
    assert alerts[0].severity == "review"
    assert alerts[0].risk <= 0.4
    print(f"  Audio only: {alerts[0].severity} risk={alerts[0].risk:.2f}")

    # Video cue only -> no alert
    alerts = fusion.update([], {"look_left"}, clock.now)
    assert len(alerts) == 0
    print(f"  Video only: {len(alerts)} alerts")

    # Audio + look_left -> alert (risk >= 0.6)
    alerts = fusion.update([audio_event], {"look_left"}, clock.now)
    assert len(alerts) == 1
    assert alerts[0].severity == "alert"
    assert alerts[0].risk >= 0.6
    assert "look_left" in alerts[0].reasons[1]  # second reason is the cue
    print(f"  Audio + look_left: {alerts[0].severity} risk={alerts[0].risk:.2f}")

    # Stale audio context (2s after event, still within AUDIO_HOLD_S=4s) -> still fuses
    clock.advance(2.0)
    alerts = fusion.update([], {"leaning"}, clock.now)
    assert len(alerts) == 1
    assert "leaning" in alerts[0].reasons
    print(f"  Cue 2s after audio: fused, cue=leaning")

    # Cue 10s after audio -> no fuse (audio context expired)
    clock.advance(10.0)
    alerts = fusion.update([], {"look_left"}, clock.now)
    assert len(alerts) == 0
    print(f"  Cue 10s after audio: no fuse")

    # Hand raised (not in VIDEO_CUE_WEIGHTS) -> ignored
    alerts = fusion.update([audio_event], {"hand_raised"}, clock.now)
    assert len(alerts) == 1 and alerts[0].severity == "review"
    print(f"  hand_raised ignored: review only")

    # Lip activity as cue
    clock2 = FakeClock(1000.0)
    fusion2 = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)
    alerts = fusion2.update([audio_event], set(), clock2.now, lip_activity=0.9)
    assert len(alerts) == 1 and alerts[0].severity == "alert"
    assert "lip activity" in alerts[0].reasons[1]
    print(f"  Lip activity 0.9: {alerts[0].severity}")

    # Fused cooldown blocks second alert 1s later
    alerts = fusion2.update([], {"look_left"}, clock2.now)
    assert len(alerts) == 0
    print(f"  Fused cooldown blocks 2nd alert")

    print("All Fusion smoke tests passed!")