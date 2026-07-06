"""
visual_processor.py
===================

Frame-level visual analysis for smart clipping and masking.

Responsibilities
----------------
1. **Motion analysis** via dense Optical Flow (Farneback) -> high-action moments.
2. **Face emotion detection** (FER or DeepFace, auto-detected) -> funny moments
   (a burst of "happy"/laughing faces).
3. **Segmentation & masking** via the Segment Anything Model (SAM) with a
   graceful ``rembg`` fallback -> masks for background removal / replacement.

Heavy models (SAM, FER, DeepFace) are optional. The module degrades gracefully:
if a dependency or checkpoint is missing it logs a warning and the corresponding
feature returns empty / passthrough results instead of crashing, so the motion
pipeline (pure OpenCV) always works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python is required. Install with `pip install opencv-python`."
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class MotionSample:
    """Mean optical-flow magnitude at a given timestamp."""

    time: float
    magnitude: float


@dataclass
class VisualEvent:
    """A detected visual event (motion burst or emotion burst)."""

    start: float
    end: float
    kind: str  # "motion" | "emotion"
    score: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


# ---------------------------------------------------------------------------
# Motion analysis (dense optical flow)
# ---------------------------------------------------------------------------
class MotionAnalyzer:
    """Estimate per-frame motion energy using Farneback dense optical flow."""

    def __init__(self, sample_fps: float = 5.0, resize_width: int = 320) -> None:
        """
        Parameters
        ----------
        sample_fps:
            How many frames per second to actually analyse. Optical flow is
            expensive; 5 fps captures action dynamics cheaply.
        resize_width:
            Frames are downscaled to this width before flow computation.
        """
        self.sample_fps = sample_fps
        self.resize_width = resize_width

    def analyze(self, video_path: str) -> List[MotionSample]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / self.sample_fps)))

        samples: List[MotionSample] = []
        prev_gray: Optional[np.ndarray] = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                gray = self._prep(frame)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    t = frame_idx / src_fps
                    samples.append(MotionSample(time=t, magnitude=float(mag.mean())))
                prev_gray = gray
            frame_idx += 1

        cap.release()
        logger.info("Motion analysis produced %d samples", len(samples))
        return samples

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = self.resize_width / float(w)
        resized = cv2.resize(frame, (self.resize_width, max(1, int(h * scale))))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def detect_events(
        self, samples: List[MotionSample], z_thresh: float = 1.5, min_gap: float = 1.0
    ) -> List[VisualEvent]:
        """Convert a motion curve into discrete high-action events."""
        if len(samples) < 3:
            return []

        mags = np.array([s.magnitude for s in samples])
        times = np.array([s.time for s in samples])
        mean, std = mags.mean(), mags.std() + 1e-8
        z = (mags - mean) / std

        events: List[VisualEvent] = []
        for i in np.where(z > z_thresh)[0]:
            t = float(times[i])
            score = float(np.clip(z[i] / (z_thresh * 2.0), 0.0, 1.0))
            if events and t - events[-1].end <= min_gap:
                events[-1].end = t + 0.5
                events[-1].score = max(events[-1].score, score)
            else:
                events.append(
                    VisualEvent(start=max(0.0, t - 0.5), end=t + 0.5,
                                kind="motion", score=score)
                )
        logger.info("Detected %d motion events", len(events))
        return events


# ---------------------------------------------------------------------------
# Face emotion analysis (funny-moment detection)
# ---------------------------------------------------------------------------
class EmotionAnalyzer:
    """Detect bursts of happy / laughing faces.

    Backend auto-detection order:
    1. ``fer.FER`` (lightweight, MTCNN optional).
    2. ``deepface.DeepFace`` (heavier but accurate).
    If neither is installed, emotion analysis is disabled gracefully.
    """

    HAPPY_LABELS = {"happy", "surprise"}

    def __init__(self, sample_fps: float = 2.0, backend: str = "auto") -> None:
        self.sample_fps = sample_fps
        self.backend = backend
        self._detector = None
        self._impl = self._init_backend(backend)

    def _init_backend(self, backend: str) -> Optional[str]:
        if backend in ("auto", "fer"):
            try:
                from fer import FER  # type: ignore
                self._detector = FER(mtcnn=False)
                logger.info("EmotionAnalyzer using FER backend")
                return "fer"
            except Exception as exc:  # noqa: BLE001
                if backend == "fer":
                    logger.warning("FER requested but unavailable: %s", exc)
        if backend in ("auto", "deepface"):
            try:
                import deepface  # noqa: F401
                logger.info("EmotionAnalyzer using DeepFace backend")
                return "deepface"
            except Exception as exc:  # noqa: BLE001
                if backend == "deepface":
                    logger.warning("DeepFace requested but unavailable: %s", exc)
        logger.warning("No emotion backend available; emotion detection disabled.")
        return None

    @property
    def available(self) -> bool:
        return self._impl is not None

    def _score_frame(self, frame_bgr: np.ndarray) -> float:
        """Return the max 'happy' probability across faces in a frame."""
        if self._impl == "fer":
            try:
                results = self._detector.detect_emotions(frame_bgr)
            except Exception:  # noqa: BLE001
                return 0.0
            best = 0.0
            for face in results or []:
                emo = face.get("emotions", {})
                best = max(best, sum(emo.get(l, 0.0) for l in self.HAPPY_LABELS))
            return float(min(best, 1.0))

        if self._impl == "deepface":
            try:
                from deepface import DeepFace
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                analysis = DeepFace.analyze(
                    rgb, actions=["emotion"], enforce_detection=False, silent=True
                )
                if isinstance(analysis, dict):
                    analysis = [analysis]
                best = 0.0
                for a in analysis:
                    emo = a.get("emotion", {})
                    best = max(best, sum(emo.get(l, 0.0) for l in self.HAPPY_LABELS) / 100.0)
                return float(min(best, 1.0))
            except Exception:  # noqa: BLE001
                return 0.0
        return 0.0

    def analyze(self, video_path: str) -> List[VisualEvent]:
        if not self.available:
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / self.sample_fps)))

        times, scores = [], []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                times.append(frame_idx / src_fps)
                scores.append(self._score_frame(frame))
            frame_idx += 1
        cap.release()

        return self._scores_to_events(np.array(times), np.array(scores))

    def _scores_to_events(
        self, times: np.ndarray, scores: np.ndarray, thresh: float = 0.5, min_gap: float = 1.5
    ) -> List[VisualEvent]:
        events: List[VisualEvent] = []
        for i in np.where(scores > thresh)[0]:
            t = float(times[i])
            if events and t - events[-1].end <= min_gap:
                events[-1].end = t + 0.5
                events[-1].score = max(events[-1].score, float(scores[i]))
            else:
                events.append(
                    VisualEvent(start=max(0.0, t - 0.7), end=t + 0.7,
                                kind="emotion", score=float(scores[i]),
                                meta={"emotion": "happy"})
                )
        logger.info("Detected %d emotion (funny) events", len(events))
        return events


# ---------------------------------------------------------------------------
# Segmentation & masking (SAM with rembg fallback)
# ---------------------------------------------------------------------------
class Segmenter:
    """Produce binary masks separating foreground subject from background.

    Uses the Segment Anything Model when a checkpoint is provided, otherwise
    falls back to ``rembg`` (u2net) and finally to a GrabCut heuristic so the
    pipeline never hard-fails.
    """

    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_h",
        device: str = "cpu",
    ) -> None:
        self.device = device
        self._sam_predictor = None
        self._rembg_session = None
        if sam_checkpoint:
            self._init_sam(sam_checkpoint, sam_model_type, device)
        if self._sam_predictor is None:
            self._init_rembg()

    def _init_sam(self, checkpoint: str, model_type: str, device: str) -> None:
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)
            self._sam_predictor = SamPredictor(sam)
            logger.info("SAM segmenter loaded (%s on %s)", model_type, device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load SAM (%s); falling back to rembg.", exc)
            self._sam_predictor = None

    def _init_rembg(self) -> None:
        try:
            from rembg import new_session
            self._rembg_session = new_session("u2net")
            logger.info("rembg fallback segmenter ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning("rembg unavailable (%s); using GrabCut heuristic.", exc)
            self._rembg_session = None

    def segment(
        self, frame_bgr: np.ndarray, point: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Return a ``uint8`` mask (255 = foreground) for a single frame.

        ``point`` optionally provides a SAM foreground prompt; if omitted the
        image centre is used.
        """
        h, w = frame_bgr.shape[:2]

        if self._sam_predictor is not None:
            return self._segment_sam(frame_bgr, point or (w // 2, h // 2))
        if self._rembg_session is not None:
            return self._segment_rembg(frame_bgr)
        return self._segment_grabcut(frame_bgr)

    def _segment_sam(self, frame_bgr: np.ndarray, point: Tuple[int, int]) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._sam_predictor.set_image(rgb)
        masks, scores, _ = self._sam_predictor.predict(
            point_coords=np.array([point]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best = masks[int(np.argmax(scores))]
        return (best.astype(np.uint8) * 255)

    def _segment_rembg(self, frame_bgr: np.ndarray) -> np.ndarray:
        from rembg import remove
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cut = remove(rgb, session=self._rembg_session)  # RGBA
        alpha = cut[..., 3]
        return (alpha > 127).astype(np.uint8) * 255

    def _segment_grabcut(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(frame_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
            return out.astype(np.uint8)
        except Exception:  # noqa: BLE001
            full = np.full((h, w), 255, np.uint8)
            return full


__all__ = [
    "MotionAnalyzer",
    "MotionSample",
    "EmotionAnalyzer",
    "Segmenter",
    "VisualEvent",
]
