"""Tests for audio/fusion.py - multimodal fusion logic."""
import pytest

from audio.fusion import Fusion, Alert
from audio.session import AudioEvent
from audio.config import (
    VIDEO_CUE_WEIGHTS, AUDIO_HOLD_S, FUSION_ALERT_RISK,
    FUSION_COOLDOWN_S, AUDIO_ONLY_CAP, FUSION_WEIGHTS,
    LIP_ACTIVITY_THRESHOLD
)


def test_audio_only_review():
    """Audio event without video cues -> review with risk <= AUDIO_ONLY_CAP."""
    fusion = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)

    alerts = fusion.update([audio_event], set(), 1000.0)

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.severity == "review"
    assert alert.risk <= 0.4
    assert alert.audio_kind == "whispering"
    assert alert.video_cues == []


def test_video_cue_only_no_alert():
    """Video cue alone produces no alert."""
    fusion = Fusion()
    alerts = fusion.update([], {"look_left"}, 1000.0)
    assert len(alerts) == 0


def test_audio_plus_video_cue_alert():
    """Audio event + video cue -> fused alert with severity 'alert'."""
    fusion = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)

    alerts = fusion.update([audio_event], {"look_left"}, 1000.0)

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.severity == "alert"
    assert alert.risk >= 0.6
    # Reason uses space-separated "look left" not "look_left"
    assert "look left" in alert.reasons[1]
    assert alert.audio_kind == "whispering"
    assert alert.video_cues == ["look_left"]


def test_fused_cooldown_blocks_repeat():
    """Fused alert cooldown blocks second alert within FUSION_COOLDOWN_S."""
    fusion = Fusion()
    # Audio event ends at t=1003, processed at t=1000
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)

    # First fused alert at t=1000
    alerts = fusion.update([audio_event], {"look_left"}, 1000.0)
    assert len(alerts) == 1

    # Second fused attempt 1s later -> blocked by FUSION_COOLDOWN_S=8s
    alerts = fusion.update([], {"leaning"}, 1001.0)
    assert len(alerts) == 0

    # After cooldown expires (8s from first fused at t=1000 -> t=1008)
    # Need a new audio event to extend audio context
    audio_event2 = AudioEvent("whispering", 3.5, 0.9, 1007.0, 1008.0)
    alerts = fusion.update([audio_event2], {"look_left"}, 1008.0)
    assert len(alerts) == 1


def test_audio_context_hold():
    """Audio context stays active for AUDIO_HOLD_S after event."""
    fusion = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)

    fusion.update([audio_event], set(), 1000.0)

    # 2 seconds later, cue should still fuse (within AUDIO_HOLD_S=4s)
    alerts = fusion.update([], {"leaning"}, 1002.0)
    assert len(alerts) == 1
    assert "leaning" in alerts[0].reasons[1]

    # 10 seconds later -> audio context expired
    alerts = fusion.update([], {"look_left"}, 1015.0)
    assert len(alerts) == 0


def test_hand_raised_ignored():
    """hand_raised not in VIDEO_CUE_WEIGHTS -> ignored (audio-only review)."""
    fusion = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)

    alerts = fusion.update([audio_event], {"hand_raised"}, 1000.0)
    assert len(alerts) == 1
    assert alerts[0].severity == "review"


def test_lip_activity_cue():
    """Lip activity >= threshold counts as video cue."""
    fusion = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)

    # Below threshold -> no cue
    alerts = fusion.update([audio_event], set(), 1000.0, lip_activity=0.2)
    assert len(alerts) == 1 and alerts[0].severity == "review"

    # Above threshold -> fused alert
    fusion2 = Fusion()
    alerts = fusion2.update([audio_event], set(), 1000.0, lip_activity=0.9)
    assert len(alerts) == 1
    assert alerts[0].severity == "alert"
    assert "lip activity" in alerts[0].reasons[1]


def test_fused_cooldown_blocks_second_alert():
    """Fused cooldown blocks second alert 1s later."""
    fusion = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)

    alerts = fusion.update([audio_event], {"look_left"}, 1000.0)
    assert len(alerts) == 1

    # 1 second later -> blocked
    alerts = fusion.update([], {"look_left"}, 1001.0)
    assert len(alerts) == 0


def test_multiple_cues():
    """Multiple video cues -> uses max weight."""
    fusion = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)

    alerts = fusion.update([audio_event], {"look_left", "hand_to_face"}, 1000.0)
    assert len(alerts) == 1
    assert alerts[0].severity == "alert"
    assert "look_left" in alerts[0].video_cues or "hand_to_face" in alerts[0].video_cues


def test_audio_only_replaces_review_with_fused():
    """Fused alert replaces audio-only review from same call."""
    fusion = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)

    # In single call, fused should replace review
    alerts = fusion.update([audio_event], {"look_left"}, 1000.0)
    assert len(alerts) == 1
    assert alerts[0].severity == "alert"


def test_risk_formula():
    """Risk = wa*conf + wv*max_cue_weight + wb, clipped to [0,1]."""
    fusion = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 1.0, 1002.0, 1003.0)  # max conf

    alerts = fusion.update([audio_event], {"hand_to_face"}, 1000.0)  # weight 0.7
    alert = alerts[0]

    wa, wv, wb = 0.5, 0.3, 0.2
    expected = min(1.0, wa * 1.0 + wv * 0.7 + wb)  # 0.5 + 0.21 + 0.2 = 0.91
    assert abs(alert.risk - expected) < 0.01


def test_severity_threshold():
    """Severity 'alert' if risk >= FUSION_ALERT_RISK, else 'review'."""
    fusion = Fusion()

    # High risk -> alert
    audio_event = AudioEvent("whispering", 3.5, 1.0, 1002.0, 1003.0)
    alerts = fusion.update([audio_event], {"hand_to_face"}, 1000.0)
    assert alerts[0].severity == "alert"

    # Low risk -> review (create new fusion to reset)
    fusion2 = Fusion()
    audio_event2 = AudioEvent("whispering", 3.5, 0.3, 1002.0, 1003.0)  # low conf
    alerts = fusion2.update([audio_event2], {"leaning"}, 1000.0)  # weight 0.5
    # risk = 0.5*0.3 + 0.3*0.5 + 0.2 = 0.15 + 0.15 + 0.2 = 0.5 < 0.6
    assert alerts[0].severity == "review"


def test_alert_fields():
    """Alert has all required fields."""
    fusion = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)

    alerts = fusion.update([audio_event], {"look_left"}, 1000.0)
    alert = alerts[0]

    assert isinstance(alert, Alert)
    assert isinstance(alert.t, float)
    assert isinstance(alert.risk, float)
    assert alert.severity in ("alert", "review")
    assert isinstance(alert.reasons, list)
    assert isinstance(alert.audio_kind, str)
    assert isinstance(alert.video_cues, list)
    assert isinstance(alert.audio_conf, float)


def test_reset_clears_state():
    """reset() clears all fusion state."""
    fusion = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)

    fusion.update([audio_event], {"look_left"}, 1000.0)
    fusion.reset()

    # Should not fuse after reset (no audio context)
    alerts = fusion.update([], {"look_left"}, 1010.0)
    assert len(alerts) == 0


def test_audio_context_active():
    """get_audio_context_active returns True within AUDIO_HOLD_S."""
    fusion = Fusion()
    audio_event = AudioEvent("whispering", 3.5, 0.9, 1002.0, 1003.0)

    fusion.update([audio_event], set(), 1000.0)
    assert fusion.get_audio_context_active(1002.0)  # within 4s
    assert not fusion.get_audio_context_active(1010.0)  # past 4s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])