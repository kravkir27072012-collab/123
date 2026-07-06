"""
audio_analyzer.py
=================

Audio-driven event detection for smart video clipping.

The analyzer works purely on the audio track extracted from a video and looks
for events that usually indicate an interesting / funny moment:

* Sudden loudness spikes (RMS energy jumps)      -> highlights
* Onset / transient density                      -> action, hits, cuts
* Laughter / applause                            -> funny moments, crowd reaction

Everything is built on top of ``librosa`` so it stays dependency-light and CPU
friendly. Laughter/applause detection uses a spectral heuristic (broadband,
noisy, rhythmically bursty energy in the 1-4 kHz band) which is robust enough
for clip selection without training a dedicated classifier.

The public entry point is :meth:`AudioAnalyzer.analyze` which returns a list of
:class:`AudioEvent` objects with timestamps and confidence scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import librosa
except ImportError as exc:  # pragma: no cover - guidance for the user
    raise ImportError(
        "librosa is required for audio analysis. Install with `pip install librosa`."
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class AudioEvent:
    """A single detected audio event.

    Attributes
    ----------
    start, end:
        Event boundaries in seconds.
    kind:
        One of ``"loudness"``, ``"onset"``, ``"laughter"``, ``"applause"``.
    score:
        Confidence / intensity in the ``[0, 1]`` range.
    """

    start: float
    end: float
    kind: str
    score: float = 0.0

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class AudioAnalysisResult:
    """Container bundling all events plus the raw curves for visualisation."""

    events: List[AudioEvent] = field(default_factory=list)
    sr: int = 22050
    hop_length: int = 512
    rms: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None

    def events_of(self, kind: str) -> List[AudioEvent]:
        return [e for e in self.events if e.kind == kind]


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class AudioAnalyzer:
    """Detect interesting acoustic events in an audio signal.

    Parameters
    ----------
    sr:
        Target sample rate for analysis. 22.05 kHz is plenty for event
        detection and keeps everything fast.
    hop_length:
        STFT hop in samples. Controls the temporal resolution of the curves.
    loudness_z:
        How many standard deviations above the mean RMS a frame must be to
        count as a loudness spike.
    min_event_gap:
        Minimum spacing (seconds) between two events of the same kind; closer
        detections are merged.
    """

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        loudness_z: float = 1.8,
        min_event_gap: float = 0.8,
    ) -> None:
        self.sr = sr
        self.hop_length = hop_length
        self.loudness_z = loudness_z
        self.min_event_gap = min_event_gap

    # -- loading ----------------------------------------------------------
    def load_audio(self, path: str) -> np.ndarray:
        """Load an audio *or* video file into a mono waveform.

        ``librosa`` can pull the audio track straight out of an ``.mp4`` when a
        system ``ffmpeg`` is on the PATH. When it isn't (e.g. only the bundled
        ``imageio-ffmpeg`` binary is present), we fall back to extracting the
        audio with MoviePy — which knows where its own ffmpeg lives — and read
        the resulting WAV.
        """
        logger.info("Loading audio from %s", path)
        try:
            y, _ = librosa.load(path, sr=self.sr, mono=True)
            return y
        except Exception as exc:  # noqa: BLE001 - broad on purpose (backend errors vary)
            logger.warning("Direct audio load failed (%s); extracting via MoviePy.", exc)
            return self._load_audio_via_moviepy(path)

    def _load_audio_via_moviepy(self, path: str) -> np.ndarray:
        import tempfile

        try:
            try:
                from moviepy import VideoFileClip  # MoviePy >= 2.0
            except ImportError:
                from moviepy.editor import VideoFileClip  # MoviePy 1.x
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Could not load audio: no ffmpeg backend for librosa and MoviePy "
                "is not installed."
            ) from exc

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            with VideoFileClip(path) as clip:
                if clip.audio is None:
                    logger.warning("Video has no audio track.")
                    return np.zeros(0, dtype=np.float32)
                clip.audio.write_audiofile(tmp.name, fps=self.sr, logger=None)
            y, _ = librosa.load(tmp.name, sr=self.sr, mono=True)
        return y

    # -- public API -------------------------------------------------------
    def analyze(self, source, y: Optional[np.ndarray] = None) -> AudioAnalysisResult:
        """Run the full detection pipeline.

        Parameters
        ----------
        source:
            Path to an audio/video file. Ignored when ``y`` is supplied.
        y:
            Optional pre-loaded mono waveform (skips loading).
        """
        if y is None:
            y = self.load_audio(source)

        if y.size == 0:
            logger.warning("Empty audio signal; returning no events.")
            return AudioAnalysisResult(sr=self.sr, hop_length=self.hop_length)

        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        times = librosa.frames_to_time(
            np.arange(len(rms)), sr=self.sr, hop_length=self.hop_length
        )

        events: List[AudioEvent] = []
        events += self._detect_loudness_spikes(rms, times)
        events += self._detect_onsets(y)
        events += self._detect_laughter_applause(y)

        events = self._merge_events(events)
        events.sort(key=lambda e: e.start)

        logger.info("Audio analysis found %d events", len(events))
        return AudioAnalysisResult(
            events=events,
            sr=self.sr,
            hop_length=self.hop_length,
            rms=rms,
            times=times,
        )

    # -- detectors --------------------------------------------------------
    def _detect_loudness_spikes(
        self, rms: np.ndarray, times: np.ndarray
    ) -> List[AudioEvent]:
        """Frames whose RMS is well above the running mean are loudness spikes."""
        if rms.std() < 1e-8:
            return []

        z = (rms - rms.mean()) / (rms.std() + 1e-8)
        peaks = np.where(z > self.loudness_z)[0]

        events: List[AudioEvent] = []
        for idx in peaks:
            t = float(times[idx])
            score = float(np.clip(z[idx] / (self.loudness_z * 2.0), 0.0, 1.0))
            events.append(
                AudioEvent(start=max(0.0, t - 0.5), end=t + 0.5, kind="loudness", score=score)
            )
        return events

    def _detect_onsets(self, y: np.ndarray) -> List[AudioEvent]:
        """Onset transients (impacts, hits, sudden action)."""
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self.sr, hop_length=self.hop_length, backtrack=False
        )
        if len(onset_frames) == 0:
            return []

        onset_times = librosa.frames_to_time(
            onset_frames, sr=self.sr, hop_length=self.hop_length
        )
        strengths = onset_env[onset_frames]
        smax = float(strengths.max()) if strengths.size else 1.0

        events: List[AudioEvent] = []
        for t, s in zip(onset_times, strengths):
            score = float(s / (smax + 1e-8))
            # Only keep reasonably strong onsets to avoid flooding the timeline.
            if score < 0.35:
                continue
            events.append(
                AudioEvent(start=max(0.0, float(t) - 0.3), end=float(t) + 0.3,
                           kind="onset", score=score)
            )
        return events

    def _detect_laughter_applause(self, y: np.ndarray) -> List[AudioEvent]:
        """Heuristic laughter / applause detector.

        Both share a signature: broadband, noisy energy concentrated in the
        1-4 kHz band with high zero-crossing rate and low spectral flatness
        variation. Applause tends to be more sustained and flatter; laughter is
        more rhythmically bursty. We separate the two by burst rate.
        """
        frame_length = 2048
        hop = self.hop_length

        S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=frame_length)

        band = (freqs >= 1000) & (freqs <= 4000)
        band_energy = S[band, :].sum(axis=0)
        total_energy = S.sum(axis=0) + 1e-8
        band_ratio = band_energy / total_energy

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop)[0]
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame_length, hop_length=hop)[0]

        # Align lengths (feature extractors can differ by a frame).
        n = min(len(band_ratio), len(zcr), len(flatness))
        band_ratio, zcr, flatness = band_ratio[:n], zcr[:n], flatness[:n]
        times = librosa.frames_to_time(np.arange(n), sr=self.sr, hop_length=hop)

        # A frame is "crowd-like" when mid-band energy, zero crossings and
        # spectral flatness are all elevated.
        mask = (
            (band_ratio > np.percentile(band_ratio, 75))
            & (zcr > np.percentile(zcr, 60))
            & (flatness > np.percentile(flatness, 60))
        )

        events: List[AudioEvent] = []
        for start_idx, end_idx in self._contiguous_regions(mask):
            duration = float(times[end_idx] - times[start_idx])
            if duration < 0.4:  # too short to be a genuine crowd reaction
                continue

            region_zcr = zcr[start_idx:end_idx]
            burstiness = float(np.std(region_zcr) / (np.mean(region_zcr) + 1e-8))
            score = float(np.clip(band_ratio[start_idx:end_idx].mean() * 2.0, 0.0, 1.0))

            # Bursty -> laughter, sustained/flat -> applause.
            kind = "laughter" if burstiness > 0.35 else "applause"
            events.append(
                AudioEvent(
                    start=float(times[start_idx]),
                    end=float(times[end_idx]),
                    kind=kind,
                    score=score,
                )
            )
        return events

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _contiguous_regions(mask: np.ndarray):
        """Yield ``(start, end)`` index pairs of contiguous ``True`` runs."""
        if mask.size == 0:
            return
        idx = np.flatnonzero(np.diff(mask.astype(int)))
        edges = np.concatenate(([0], idx + 1, [mask.size]))
        for a, b in zip(edges[:-1], edges[1:]):
            if mask[a]:
                yield int(a), int(b - 1)

    def _merge_events(self, events: List[AudioEvent]) -> List[AudioEvent]:
        """Merge same-kind events that are closer than ``min_event_gap``."""
        merged: List[AudioEvent] = []
        for kind in {e.kind for e in events}:
            group = sorted((e for e in events if e.kind == kind), key=lambda e: e.start)
            for ev in group:
                if merged and merged[-1].kind == kind and ev.start - merged[-1].end <= self.min_event_gap:
                    last = merged[-1]
                    last.end = max(last.end, ev.end)
                    last.score = max(last.score, ev.score)
                else:
                    merged.append(ev)
        return merged


__all__ = ["AudioAnalyzer", "AudioEvent", "AudioAnalysisResult"]
