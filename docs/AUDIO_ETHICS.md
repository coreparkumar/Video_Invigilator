# Audio Ethics & Consent Gate

**Status:** DRAFT — must be reviewed by legal counsel before any human audio is recorded.
**Applies to:** Any future audio capability (SD1+). The current POC (v1.0) has **no audio**.

---

## 1. Purpose Limitation

| Allowed | Prohibited |
|---------|------------|
| Detect **acoustic activity level** (RMS/energy) to flag anomalous silence or sustained noise during exam | Record raw audio waveforms |
| Compute **metadata only**: timestamp, duration, energy level, spectral centroid (brightness) | Transcribe speech (ASR) |
| Trigger visual flag: "Unusual acoustic activity detected at HH:MM:SS" | Speaker identification / diarization |
| Store metadata in `events.csv` with same retention as gesture flags | Stream or buffer audio to disk/network |
| Correlate with gesture flags (e.g., hand-to-face + acoustic spike) | Use audio for behavioral profiling beyond exam session |

**Hard rule:** No microphone data leaves the `AudioMonitor` class. No WAV/MP3/Opus files written. No cloud APIs called.

---

## 2. Data Minimization & Retention

| Data Type | Retention | Auto-Delete |
|-----------|-----------|-------------|
| Acoustic metadata rows in `events.csv` | 30 days | Yes |
| Per-session acoustic summary (max/mean energy) | 30 days | Yes |
| Raw audio buffers | **Never stored** | N/A |
| Calibration profiles (noise floor) | Session only | On exit |

**Implementation:** `AudioMonitor` uses ring buffer of 500 ms max. Only aggregated metrics escape the class.

---

## 3. Consent Text (Mock-Exam Participants)

> **CONSENT FOR ACOUSTIC MONITORING DURING MOCK EXAM**
>
> **Purpose:** This mock exam uses a local acoustic activity monitor to detect unusual sound patterns (e.g., sustained talking, phone rings, sustained silence) that may indicate irregular exam conditions. The system **does not record your voice**, **does not transcribe speech**, and **does not identify speakers**.
>
> **What is collected:** Timestamp, acoustic energy level (loudness), and spectral brightness metadata — only when the energy exceeds a calibrated threshold for longer than 3 seconds.
>
> **What is NOT collected:** No audio recordings, no speech content, no voiceprints, no speaker identities.
>
> **Storage:** Metadata stored locally in `evidence/events.csv` for 30 days, then auto-deleted. No cloud upload.
>
> **Your rights:** You may withdraw at any time. Withdrawal stops monitoring and deletes your session metadata immediately.
>
> **Contact:** [RESEARCHER NAME] — [EMAIL] — [INSTITUTION IRB #]
>
> ☐ I have read and understand the above.
> ☐ I consent to acoustic activity monitoring for this mock exam session.
> ☐ I understand I can withdraw at any time.
>
> **Signature:** _________________________  **Date:** _______________

---

## 4. Legal Checklist — [JURISDICTION: ___________]

*Leave blanks for your legal counsel to verify.*

| Requirement | Status | Notes / Statute |
|-------------|--------|-----------------|
| **One-party vs. all-party consent** for audio monitoring | ☐ Verified | [e.g., wiretap statute §____] |
| **Workplace/educational surveillance notice** posted | ☐ Verified | [e.g., state labor code §____] |
| **Data protection impact assessment (DPIA)** completed | ☐ Verified | [GDPR Art. 35 / local equivalent] |
| **Lawful basis** documented (legitimate interest / consent) | ☐ Verified | [GDPR Art. 6 / local equivalent] |
| **Retention schedule** compliant with records law | ☐ Verified | [e.g., FERPA, state education code] |
| **Minor/student consent** process (parent/guardian) | ☐ Verified | [if participants < 18] |
| **Accessibility** — alternative for hearing-impaired | ☐ Verified | [ADA / local equivalent] |
| **Signage wording** approved (see §5) | ☐ Verified | |
| **Withdrawal mechanism** tested and documented | ☐ Verified | |
| **Data breach notification** plan in place | ☐ Verified | [GDPR Art. 33 / local equivalent] |

**Legal counsel sign-off:** _________________________  **Date:** _______________

---

## 5. Signage Wording (Post at Exam Room Entrance)

> **NOTICE: ACOUSTIC ACTIVITY MONITORING IN USE**
>
> This room uses a local acoustic activity monitor during examinations. The system detects **sound level and pattern anomalies only** — it does **not** record conversations, identify speakers, or transcribe speech.
>
> Metadata (time, energy level) is logged locally for 30 days for exam integrity review.
>
> **Questions/Withdrawal:** Contact [NAME] at [EMAIL/PHONE].
>
> [INSTITUTION LOGO]  [DATE]

---

## 6. Approval Checklist — MUST ALL BE TICKED BEFORE FIRST HUMAN AUDIO SESSION

| # | Requirement | Owner | Done |
|---|-------------|-------|------|
| 1 | Legal checklist (§4) fully verified by counsel | Legal | ☐ |
| 2 | Consent form (§3) approved by IRB/Ethics board | PI | ☐ |
| 3 | Signage (§5) posted at all entry points | Facilities | ☐ |
| 4 | `AudioMonitor` class code-reviewed: **no raw audio persistence** | Lead Eng | ☐ |
| 5 | Unit tests confirm: zero audio files written, metadata-only output | QA | ☐ |
| 6 | Withdrawal mechanism tested: stops monitor, deletes session data | QA | ☐ |
| 7 | Retention auto-delete job configured (30 days) | DevOps | ☐ |
| 8 | Participant consent forms collected and filed | PI | ☐ |
| 9 | Incident response plan for accidental audio capture | Security | ☐ |
| 10 | **Final GO/NO-GO** sign-off by PI + Legal + Security | All | ☐ |

**NO HUMAN AUDIO MAY BE RECORDED UNTIL ALL 10 ITEMS ARE TICKED.**

---

## 7. Change Control

Any modification to audio collection scope, retention, or processing requires:
1. Updated legal checklist (§4)
2. Re-approval of consent form (§3)
3. Re-sign-off of all 10 checklist items (§6)

---

**Document version:** 0.1-draft  
**Last updated:** 2026-10-05  
**Next review:** Before SD1 implementation