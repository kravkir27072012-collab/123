"""
video_editor_core.py
====================

Orchestrates the full smart-clipping pipeline and does the final rendering
with MoviePy.

Pipeline (matches the technical spec)
-------------------------------------
1. **Audio Analysis** — loudness spikes, onsets, laughter/applause
   (``audio_analyzer``).
2. **Visual Analysis** — optical-flow motion + face-emotion "funny" moments
   (``visual_processor``).
3. **Fusion & Scoring** — audio and visual events are merged onto a common
   timeline and scored to rank highlights / funny moments.
4. **Segmentation & Effects** — optional background replacement, object
   removal, colour grading and overlays per clip (``visual_processor`` +
   ``effect_engine``).
5. **Rendering** — auto-trim around each highlight and export short clips with
   MoviePy.

Public entry point: :class:`VideoEditorCore`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from audio_analyzer import AudioAnalyzer, AudioEvent
from visual_processor import (
    MotionAnalyzer,
    EmotionAnalyzer,
    Segmenter,
    VisualEvent,
)
from effect_engine import EffectPipeline, replace_background, inpaint_object

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Highlight model
# ---------------------------------------------------------------------------
@dataclass
class Highlight:
    """A ranked segment of the source video worth turning into a clip."""

    start: float
    end: float
    score: float
    reasons: List[str] = field(default_factory=list)
    is_funny: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ClipConfig:
    """Auto-trimming configuration."""

    pad_before: float = 1.5      # seconds of context before the peak
    pad_after: float = 1.5       # seconds of context after the peak
    min_duration: float = 3.0    # clips shorter than this get extended
    max_duration: float = 30.0   # hard cap on a single clip
    merge_gap: float = 1.0       # merge highlights closer than this
    max_clips: int = 10          # keep only the top-N by score


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
class VideoEditorCore:
    """End-to-end smart video editor."""

    def __init__(
        self,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        motion_analyzer: Optional[MotionAnalyzer] = None,
        emotion_analyzer: Optional[EmotionAnalyzer] = None,
        segmenter: Optional[Segmenter] = None,
        clip_config: Optional[ClipConfig] = None,
    ) -> None:
        self.audio = audio_analyzer or AudioAnalyzer()
        self.motion = motion_analyzer or MotionAnalyzer()
        self.emotion = emotion_analyzer or EmotionAnalyzer()
        self.segmenter = segmenter  # lazily created only if effects need it
        self.cfg = clip_config or ClipConfig()

    # -- 1+2: analysis ----------------------------------------------------
    def analyze(self, video_path: str):
        """Run audio + visual analysis and return the raw event lists."""
        logger.info("=== Analyzing %s ===", video_path)

        audio_result = self.audio.analyze(video_path)

        motion_samples = self.motion.analyze(video_path)
        motion_events = self.motion.detect_events(motion_samples)

        emotion_events = self.emotion.analyze(video_path)

        return audio_result.events, motion_events, emotion_events

    # -- 3: fusion + scoring ---------------------------------------------
    def find_highlights(
        self,
        audio_events: List[AudioEvent],
        motion_events: List[VisualEvent],
        emotion_events: List[VisualEvent],
        funny_only: bool = False,
    ) -> List[Highlight]:
        """Fuse events onto a timeline and rank the best moments.

        Different event types carry different weights toward the "highlight"
        and "funny" scores. Overlapping events reinforce each other.
        """
        weights = {
            "loudness": 0.6,
            "onset": 0.4,
            "applause": 0.8,
            "laughter": 1.0,
            "motion": 0.7,
            "emotion": 1.0,
        }
        funny_kinds = {"laughter", "emotion"}

        # Build a flat list of (center, half_width, kind, score).
        spans = []
        for e in audio_events:
            spans.append((e.center, max(0.75, e.duration / 2), e.kind, e.score))
        for e in motion_events + emotion_events:
            spans.append((e.center, max(0.75, (e.end - e.start) / 2), e.kind, e.score))

        if not spans:
            logger.warning("No events detected; no highlights.")
            return []

        # Cluster spans that overlap in time into candidate highlights.
        spans.sort(key=lambda s: s[0])
        clusters: List[List[tuple]] = [[spans[0]]]
        for span in spans[1:]:
            last = clusters[-1][-1]
            if span[0] - last[0] <= self.cfg.merge_gap + last[1] + span[1]:
                clusters[-1].append(span)
            else:
                clusters.append([span])

        highlights: List[Highlight] = []
        for cluster in clusters:
            centers = [c[0] for c in cluster]
            start = min(c[0] - c[1] for c in cluster)
            end = max(c[0] + c[1] for c in cluster)

            score = 0.0
            funny_score = 0.0
            reasons = set()
            for _, _, kind, s in cluster:
                w = weights.get(kind, 0.3)
                score += w * (0.5 + 0.5 * s)
                if kind in funny_kinds:
                    funny_score += w * (0.5 + 0.5 * s)
                reasons.add(kind)

            is_funny = funny_score > 0.6
            if funny_only and not is_funny:
                continue

            highlights.append(
                Highlight(
                    start=max(0.0, start),
                    end=end,
                    score=funny_score if funny_only else score,
                    reasons=sorted(reasons),
                    is_funny=is_funny,
                )
            )

        highlights.sort(key=lambda h: h.score, reverse=True)
        logger.info("Ranked %d candidate highlights", len(highlights))
        return highlights

    # -- 4: auto-trimming -------------------------------------------------
    def auto_trim(self, highlights: List[Highlight], video_duration: float) -> List[Highlight]:
        """Apply padding, min/max duration and top-N selection."""
        trimmed: List[Highlight] = []
        for h in highlights:
            start = max(0.0, h.start - self.cfg.pad_before)
            end = min(video_duration, h.end + self.cfg.pad_after)

            if end - start < self.cfg.min_duration:
                deficit = self.cfg.min_duration - (end - start)
                start = max(0.0, start - deficit / 2)
                end = min(video_duration, end + deficit / 2)

            if end - start > self.cfg.max_duration:
                center = (start + end) / 2
                start = center - self.cfg.max_duration / 2
                end = center + self.cfg.max_duration / 2

            trimmed.append(
                Highlight(start=start, end=end, score=h.score,
                          reasons=h.reasons, is_funny=h.is_funny)
            )

        trimmed = self._merge_overlaps(trimmed)
        trimmed.sort(key=lambda h: h.score, reverse=True)
        return trimmed[: self.cfg.max_clips]

    def _merge_overlaps(self, highlights: List[Highlight]) -> List[Highlight]:
        ordered = sorted(highlights, key=lambda h: h.start)
        merged: List[Highlight] = []
        for h in ordered:
            if merged and h.start <= merged[-1].end:
                prev = merged[-1]
                prev.end = max(prev.end, h.end)
                prev.score = max(prev.score, h.score)
                prev.reasons = sorted(set(prev.reasons) | set(h.reasons))
                prev.is_funny = prev.is_funny or h.is_funny
            else:
                merged.append(h)
        return merged

    # -- 5: rendering -----------------------------------------------------
    def render_clips(
        self,
        video_path: str,
        highlights: List[Highlight],
        output_dir: str,
        effects: Optional[EffectPipeline] = None,
        prefix: str = "clip",
        fps: Optional[int] = None,
    ) -> List[str]:
        """Cut and export each highlight as a short clip with MoviePy.

        An optional :class:`EffectPipeline` is applied to every clip.
        """
        try:
            from moviepy.editor import VideoFileClip
        except ImportError as exc:  # pragma: no cover
            raise ImportError("moviepy is required for rendering.") from exc

        os.makedirs(output_dir, exist_ok=True)
        source = VideoFileClip(video_path)
        paths: List[str] = []

        try:
            for i, h in enumerate(highlights):
                start = max(0.0, h.start)
                end = min(source.duration, h.end)
                if end - start < 0.5:
                    continue

                subclip = source.subclip(start, end)
                if effects is not None:
                    subclip = effects.apply_to_clip(subclip)

                tag = "funny" if h.is_funny else "highlight"
                out_path = os.path.join(output_dir, f"{prefix}_{i:03d}_{tag}.mp4")
                logger.info("Rendering %s [%.1fs-%.1fs] score=%.2f",
                            out_path, start, end, h.score)

                subclip.write_videofile(
                    out_path,
                    codec="libx264",
                    audio_codec="aac",
                    fps=fps or source.fps,
                    logger=None,
                )
                paths.append(out_path)
        finally:
            source.close()

        return paths

    # -- one-shot convenience --------------------------------------------
    def process(
        self,
        video_path: str,
        output_dir: str = "output_clips",
        effects: Optional[EffectPipeline] = None,
        funny_only: bool = False,
    ) -> List[str]:
        """Run the entire pipeline end-to-end and return the clip paths."""
        from moviepy.editor import VideoFileClip

        audio_events, motion_events, emotion_events = self.analyze(video_path)
        highlights = self.find_highlights(
            audio_events, motion_events, emotion_events, funny_only=funny_only
        )

        with VideoFileClip(video_path) as vc:
            duration = vc.duration
        highlights = self.auto_trim(highlights, duration)

        if not highlights:
            logger.warning("No highlights survived trimming; nothing to render.")
            return []

        return self.render_clips(video_path, highlights, output_dir, effects=effects)

    # -- effect helpers ---------------------------------------------------
    def _ensure_segmenter(self, **kwargs) -> Segmenter:
        if self.segmenter is None:
            self.segmenter = Segmenter(**kwargs)
        return self.segmenter

    def build_background_replacer(self, background_path: str, **seg_kwargs):
        """Return a ``frame->frame`` callable that swaps the background.

        Wrap it in a :class:`~effect_engine.LambdaEffect` to add it to a
        pipeline. Segmentation runs per frame via the configured
        :class:`~visual_processor.Segmenter`.
        """
        import cv2

        seg = self._ensure_segmenter(**seg_kwargs)
        background = cv2.imread(background_path)
        if background is None:
            raise IOError(f"Cannot read background image: {background_path}")

        def _replace(frame_bgr: np.ndarray) -> np.ndarray:
            mask = seg.segment(frame_bgr)
            return replace_background(frame_bgr, mask, background)

        return _replace

    def build_object_remover(self, mask_provider, method: str = "auto"):
        """Return a ``frame->frame`` callable that inpaints an object.

        ``mask_provider`` is a callable ``frame -> mask`` (e.g. a fixed region
        or a per-frame tracker) describing the object to erase.
        """
        def _remove(frame_bgr: np.ndarray) -> np.ndarray:
            mask = mask_provider(frame_bgr)
            return inpaint_object(frame_bgr, mask, method=method)

        return _remove


__all__ = ["VideoEditorCore", "Highlight", "ClipConfig"]
