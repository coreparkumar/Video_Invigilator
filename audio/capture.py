"""Audio capture sources: microphone, WAV file, and fake source for tests."""

from __future__ import annotations
import queue
import wave
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class AudioSource(ABC):
    """Protocol for audio sources."""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Return next chunk of mono float32 audio in [-1, 1]. Empty array if none available."""
        ...

    @property
    @abstractmethod
    def finished(self) -> bool:
        """True if source is exhausted."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...


def list_input_devices() -> List[Tuple[int, str, int]]:
    """Return [(index, name, max_input_channels)] for devices that can record."""
    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError(
            "sounddevice not available. Install with: pip install sounddevice==0.4.7"
        ) from e

    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append((i, dev.get("name", f"Device {i}"), dev["max_input_channels"]))
    return devices


class MicSource(AudioSource):
    """Microphone input using sounddevice with a non-blocking queue."""

    def __init__(
        self,
        device: Optional[int] = None,
        sr: int = 16000,
        max_chunks: int = 200,
    ):
        try:
            import sounddevice as sd
        except Exception as e:
            raise RuntimeError(
                "sounddevice not available. Install with: pip install sounddevice==0.4.7"
            ) from e

        self._sr = sr
        self._frame = 512
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_chunks)
        self._dropped = 0
        self._stream = sd.InputStream(
            device=device,
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=self._frame,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass  # ignore underruns/overruns in callback
        chunk = indata[:, 0].copy()  # mono
        try:
            self._q.put_nowait(chunk)
        except queue.Full:
            # Drop the OLDEST block to keep latency bounded
            try:
                self._q.get_nowait()
                self._q.put_nowait(chunk)
            except queue.Empty:
                pass
            self._dropped += 1

    def read(self) -> np.ndarray:
        """Return all queued samples concatenated (empty array if none)."""
        chunks = []
        while True:
            try:
                chunks.append(self._q.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    @property
    def finished(self) -> bool:
        return False  # microphone runs until closed

    @property
    def dropped(self) -> int:
        return self._dropped

    def close(self) -> None:
        if hasattr(self, "_stream") and self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def release(self) -> None:
        """Alias for close() for compatibility with engine."""
        self.close()


class FakeAudioSource(AudioSource):
    """Fake source for tests: yields fixed chunks from a sample array."""

    def __init__(self, samples: np.ndarray, chunk: int = 512):
        if samples.ndim != 1:
            raise ValueError("samples must be 1D")
        self._samples = samples.astype(np.float32)
        self._chunk = chunk
        self._pos = 0

    def read(self) -> np.ndarray:
        if self._pos >= len(self._samples):
            return np.array([], dtype=np.float32)
        end = min(self._pos + self._chunk, len(self._samples))
        chunk = self._samples[self._pos:end]
        self._pos = end
        return chunk

    @property
    def finished(self) -> bool:
        return self._pos >= len(self._samples)

    def close(self) -> None:
        pass

    def release(self) -> None:
        """Alias for close() for compatibility with engine."""
        self.close()


def open_camera(device: Optional[int] = None, sr: int = 16000) -> MicSource:
    """Open a microphone source (convenience function matching video open_camera)."""
    return MicSource(device=device, sr=sr)


def _linear_resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear interpolation resample."""
    if orig_sr == target_sr:
        return x
    duration = len(x) / orig_sr
    target_len = int(duration * target_sr)
    src_idx = np.linspace(0, len(x) - 1, target_len)
    return np.interp(src_idx, np.arange(len(x)), x).astype(np.float32)


def load_wav(path: str, sr: int = 16000) -> np.ndarray:
    """Load 16-bit PCM WAV, convert to mono float32 at target sr."""
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"Only 16-bit PCM supported, got {wf.getsampwidth()*8}-bit")
        orig_sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    elif n_channels > 2:
        raise ValueError(f"Unsupported channel count: {n_channels}")

    return _linear_resample(audio, orig_sr, sr)


class WavSource(AudioSource):
    """WAV file source with real-time pacing via injectable clock."""

    def __init__(self, path: str, sr: int = 16000, clock=None):
        self._sr = sr
        self._frame = 512
        self._samples = load_wav(path, sr)
        self._pos = 0
        self._clock = clock or (lambda: __import__("time").monotonic())
        self._start_time = self._clock()  # Set start time on creation

    def read(self) -> np.ndarray:
        now = self._clock()
        # Compute how many samples should have elapsed
        elapsed = now - self._start_time
        target_pos = int(elapsed * self._sr)
        if target_pos <= self._pos:
            return np.array([], dtype=np.float32)
        if target_pos > len(self._samples):
            target_pos = len(self._samples)
        chunk = self._samples[self._pos:target_pos]
        self._pos = target_pos
        return chunk

    @property
    def finished(self) -> bool:
        return self._pos >= len(self._samples)

    def close(self) -> None:
        pass