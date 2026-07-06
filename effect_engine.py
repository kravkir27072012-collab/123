"""
effect_engine.py
================

The AI Effects Engine: colour grading, blur, overlays, background
removal/replacement and object removal (inpainting).

Two layers of API:

* **Per-frame primitives** (pure NumPy/OpenCV) — e.g. :func:`color_grade`,
  :func:`gaussian_blur`, :func:`inpaint_object`, :func:`replace_background`.
  These operate on single ``BGR`` frames and are trivially unit-testable.
* **Composable effects** — :class:`Effect` subclasses that wrap the primitives
  and can be chained in an :class:`EffectPipeline`, then applied to a MoviePy
  clip via :meth:`EffectPipeline.apply_to_clip`.

MoviePy is used at the clip level; OpenCV does the pixel work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python is required for effect_engine.") from exc

logger = logging.getLogger(__name__)


# ===========================================================================
# Per-frame primitives (operate on BGR uint8 frames)
# ===========================================================================
def color_grade(
    frame: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    temperature: float = 0.0,
) -> np.ndarray:
    """Colour-correct a frame.

    Parameters
    ----------
    brightness:
        Additive brightness in ``[-1, 1]`` (scaled to +-127).
    contrast:
        Multiplicative contrast around mid-grey (``1.0`` = unchanged).
    saturation:
        Saturation multiplier in HSV (``1.0`` = unchanged).
    temperature:
        Warm/cool shift in ``[-1, 1]``; positive = warmer (more red/less blue).
    """
    out = frame.astype(np.float32)

    # Contrast around 128, then brightness.
    out = (out - 128.0) * contrast + 128.0 + brightness * 127.0

    if temperature != 0.0:
        out[..., 2] += temperature * 40.0  # R (BGR order)
        out[..., 0] -= temperature * 40.0  # B

    out = np.clip(out, 0, 255).astype(np.uint8)

    if saturation != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out


def gaussian_blur(frame: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Whole-frame Gaussian blur. ``ksize`` is forced odd."""
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def background_blur(frame: np.ndarray, mask: np.ndarray, ksize: int = 31) -> np.ndarray:
    """Keep the masked foreground sharp and blur everything else (bokeh)."""
    k = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    m = _normalise_mask(mask, frame.shape[:2])[..., None] / 255.0
    return (frame * m + blurred * (1 - m)).astype(np.uint8)


def overlay_image(
    frame: np.ndarray,
    overlay: np.ndarray,
    position: Tuple[int, int] = (0, 0),
    opacity: float = 1.0,
) -> np.ndarray:
    """Alpha-composite an RGBA/BGR ``overlay`` onto ``frame`` at ``position``."""
    out = frame.copy()
    x, y = position
    oh, ow = overlay.shape[:2]
    fh, fw = frame.shape[:2]

    if x >= fw or y >= fh:
        return out
    ow, oh = min(ow, fw - x), min(oh, fh - y)
    ov = overlay[:oh, :ow]

    if ov.shape[2] == 4:
        alpha = (ov[..., 3:4] / 255.0) * opacity
        rgb = ov[..., :3]
    else:
        alpha = np.full((oh, ow, 1), opacity)
        rgb = ov

    roi = out[y:y + oh, x:x + ow].astype(np.float32)
    out[y:y + oh, x:x + ow] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
    return out


def replace_background(
    frame: np.ndarray, mask: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """Composite the masked foreground over a new ``background`` image.

    ``background`` is resized to the frame, and the mask edges are feathered
    for a natural cut-out.
    """
    h, w = frame.shape[:2]
    bg = cv2.resize(background, (w, h))
    m = _normalise_mask(mask, (h, w))
    m = cv2.GaussianBlur(m, (7, 7), 0)[..., None] / 255.0
    return (frame * m + bg * (1 - m)).astype(np.uint8)


def remove_background(
    frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Replace the background with a flat colour (default chroma green)."""
    bg = np.full_like(frame, color, dtype=np.uint8)
    return replace_background(frame, mask, bg)


def inpaint_object(
    frame: np.ndarray, mask: np.ndarray, method: str = "auto", radius: int = 3
) -> np.ndarray:
    """Remove the masked region by inpainting the surrounding content.

    ``method``:
    * ``"lama"``  — use ``simple-lama-inpainting`` if installed (best quality).
    * ``"telea"`` / ``"ns"`` — OpenCV inpainting variants.
    * ``"auto"``  — LaMa if available, else Telea.
    """
    m = _normalise_mask(mask, frame.shape[:2])

    if method in ("auto", "lama"):
        result = _inpaint_lama(frame, m)
        if result is not None:
            return result
        if method == "lama":
            logger.warning("LaMa unavailable; using OpenCV Telea inpainting.")

    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    return cv2.inpaint(frame, m, radius, flag)


def _inpaint_lama(frame: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    try:
        from simple_lama_inpainting import SimpleLama
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None
    try:
        lama = _get_lama_singleton()
        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        m = Image.fromarray(mask).convert("L")
        out = lama(rgb, m)
        return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LaMa inpainting failed (%s).", exc)
        return None


_LAMA_INSTANCE = None


def _get_lama_singleton():
    global _LAMA_INSTANCE
    if _LAMA_INSTANCE is None:
        from simple_lama_inpainting import SimpleLama
        _LAMA_INSTANCE = SimpleLama()
    return _LAMA_INSTANCE


def _normalise_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Coerce a mask to a single-channel ``uint8`` array matching ``shape``."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    return mask


# ===========================================================================
# Composable effects
# ===========================================================================
class Effect:
    """Base class: an effect maps a BGR frame + timestamp to a BGR frame."""

    name = "effect"

    def apply(self, frame: np.ndarray, t: float = 0.0) -> np.ndarray:  # noqa: D401
        raise NotImplementedError


class ColorGradeEffect(Effect):
    name = "color_grade"

    def __init__(self, **params) -> None:
        self.params = params

    def apply(self, frame: np.ndarray, t: float = 0.0) -> np.ndarray:
        return color_grade(frame, **self.params)


class BlurEffect(Effect):
    name = "blur"

    def __init__(self, ksize: int = 15) -> None:
        self.ksize = ksize

    def apply(self, frame: np.ndarray, t: float = 0.0) -> np.ndarray:
        return gaussian_blur(frame, self.ksize)


class OverlayEffect(Effect):
    name = "overlay"

    def __init__(self, overlay: np.ndarray, position=(0, 0), opacity: float = 1.0) -> None:
        self.overlay = overlay
        self.position = position
        self.opacity = opacity

    def apply(self, frame: np.ndarray, t: float = 0.0) -> np.ndarray:
        return overlay_image(frame, self.overlay, self.position, self.opacity)


class LambdaEffect(Effect):
    """Wrap an arbitrary ``frame -> frame`` callable as an effect."""

    name = "lambda"

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], name: str = "lambda") -> None:
        self.fn = fn
        self.name = name

    def apply(self, frame: np.ndarray, t: float = 0.0) -> np.ndarray:
        return self.fn(frame)


@dataclass
class EffectPipeline:
    """An ordered chain of effects applied to every frame."""

    effects: List[Effect]

    def __init__(self, effects: Optional[Sequence[Effect]] = None) -> None:
        self.effects = list(effects or [])

    def add(self, effect: Effect) -> "EffectPipeline":
        self.effects.append(effect)
        return self

    def process_frame(self, frame: np.ndarray, t: float = 0.0) -> np.ndarray:
        for eff in self.effects:
            frame = eff.apply(frame, t)
        return frame

    def apply_to_clip(self, clip):
        """Return a new MoviePy clip with the pipeline applied frame-by-frame.

        MoviePy frames are RGB; OpenCV primitives here are BGR-agnostic (they
        operate channel-wise) except colour-temperature which assumes BGR, so
        we convert to keep semantics consistent.
        """
        def _transform(get_frame, t):
            rgb = get_frame(t)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            out = self.process_frame(bgr, t)
            return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        # fl() signature differs across MoviePy versions; both accept a
        # (get_frame, t) callable.
        return clip.fl(_transform)


__all__ = [
    "color_grade",
    "gaussian_blur",
    "background_blur",
    "overlay_image",
    "replace_background",
    "remove_background",
    "inpaint_object",
    "Effect",
    "ColorGradeEffect",
    "BlurEffect",
    "OverlayEffect",
    "LambdaEffect",
    "EffectPipeline",
]
