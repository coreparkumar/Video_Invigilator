# Edge Invigilator POC: Product Requirements Document (Simplified)

**Status:** Draft v0.2 | **Date:** 5 October 2026 **Change from v0.1:** decentralized/multi-node architecture removed. Scope reduced to one laptop, one webcam, body-gesture detection.

## 1. Overview

A proof of concept that shows a laptop webcam can be analyzed locally, in real time, to spot body gestures that may indicate exam misconduct, and log them for human review. No cloud, no network, no second device.

**Guiding principles:** the system flags and a human decides; everything runs locally; avoid false accusations by requiring a gesture to persist before flagging.

## 2. Goals and Non-Goals

**Goals**

- G1. Capture live video from the laptop camera and run pose-based gesture detection on the laptop CPU.
- G2. Show live feedback (skeleton, per-gesture timers, flag banners).
- G3. Log each flagged event with a timestamp and a snapshot for review.
- G4. Demonstrate privacy by design: no video recording, no data leaves the laptop.

**Non-goals:** multi-camera or multi-node operation, mesh networking, cloud or remote alerting, dashboards, audio analysis, facial recognition, object detection (phones, watches), custom model training, multi-person tracking, integrations.

## 3. Setup

| Item | Decision |
| --- | --- |
| Hardware | One laptop, built-in or USB webcam, CPU only |
| Video source | Single local camera (index 0), about 640x480 at 30 FPS |
| Runtime | Python 3.9-3.12, OpenCV, MediaPipe Pose (bundled model) |
| Subject | One person at a time, head and shoulders visible |

## 4. Gestures in Scope

| ID | Gesture | Why it matters | Flag after |
| --- | --- | --- | --- |
| GS1 | Hand raised | Seeking attention / signaling | 1.0 s |
| GS2 | Hand at ear | Phone or earpiece use | 1.5 s |
| GS3 | Hand covering face/mouth | Whispering, hiding | 2.0 s |
| GS4 | Looking left / right | Looking at neighbor | 2.5 s |
| GS5 | Looking down | Notes or hidden device | 4.0 s |
| GS6 | Leaning sideways | Peeking, passing items | 2.5 s |
| GS7 | Out of frame | Left seat | 3.0 s |

## 5. Functional Requirements

| ID | Requirement |
| --- | --- |
| FR-1 | Open the webcam and process frames continuously; exit cleanly on `q`. |
| FR-2 | Estimate body pose landmarks per frame. |
| FR-3 | Classify GS1-GS7 from landmark geometry, normalized by shoulder width so results are robust to distance from the camera. |
| FR-4 | Support one-key calibration (`c`) to capture the user's neutral posture as a baseline. |
| FR-5 | Temporal rule: a gesture is flagged only after it persists for its threshold, with a 5 s cooldown between repeat flags. |
| FR-6 | Overlay: skeleton (toggle `s`), per-gesture status with hold timer, FPS, calibration state, flag banners. |
| FR-7 | On each flag, append a row to `events.csv` (timestamp, gesture, held seconds) and save one snapshot JPEG. |
| FR-8 | Thresholds are editable in a single config block in code. |

## 6. Non-Functional Requirements

| Category | Target |
| --- | --- |
| Performance | At least 15 FPS on a typical laptop CPU; flag latency under 500 ms after the hold threshold is met |
| Accuracy (POC) | Each gesture detected correctly in at least 8 of 10 deliberate attempts after calibration; no flags during 2 minutes of normal sitting and typing |
| Privacy | Frames processed in memory; no continuous recording; only flag snapshots written; no network calls |
| Reliability | Runs 30 minutes without crash or memory growth |
| Usability | Setup in under 10 minutes using the README |

## 7. Privacy and Ethics

- Processing is local; the only stored artifacts are the CSV log and flag snapshots in an `evidence/` folder the user controls.
- Users must consent to monitoring; a flag is a prompt for human review, never proof.
- Known limits: pose-based head direction is an estimate, and accuracy may vary with lighting, clothing, and body type. A fairness review is needed before any real deployment.

## 8. Success Criteria

1. Runs end to end on one laptop with a webcam.
2. All seven gestures detected and flagged with visible feedback.
3. Event log and snapshots produced for each flag.
4. Meets the performance and accuracy targets in Section 6.

## 9. Milestones

| Step | Output | Est. |
| --- | --- | --- |
| 1 | Webcam capture + pose overlay | Day 1 |
| 2 | Gesture rules + calibration | Days 2-3 |
| 3 | Temporal flagging + logging | Day 4 |
| 4 | Threshold tuning and testing | Days 5-6 |
| 5 | Demo + findings | Day 7 |

## 10. Risks and Open Questions

| Risk | Mitigation |
| --- | --- |
| False positives from natural movement | Hold-time thresholds, calibration, tuning |
| Poor lighting or framing | Setup guidance (face the screen, head and shoulders in view) |
| Head-direction error without eye tracking | Treat as coarse signal; consider face landmarks later |

**Open questions:** Which gestures matter most to your stakeholders? Is an audible or on-screen alert needed beyond the banner? What would the next phase add (objects such as phones, multiple people, ceiling camera)?