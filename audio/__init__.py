"""
Audio detection module for Edge Invigilator.

Privacy rules (HARD CONSTRAINTS):
- NO speech-to-text / transcription
- NO speaker identification or voiceprints
- NO raw audio written to disk (except optional evidence clips with explicit consent)
- NO network calls
- Metadata-only by default: timestamps, energy levels, spectral features
- An audio-only result can NEVER exceed low severity
"""