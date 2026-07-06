"""
cli.py
======

Command-line front-end for the smart video clipping & effects pipeline.

Examples
--------
Auto-cut highlights from a long video::

    python cli.py input.mp4 --output clips/

Only extract funny moments (laughter + happy faces)::

    python cli.py input.mp4 --output clips/ --funny-only

Add colour grading and background blur to every clip::

    python cli.py input.mp4 --output clips/ --grade --bg-blur

Replace the background using SAM (needs a checkpoint)::

    python cli.py input.mp4 --output clips/ \\
        --replace-bg new_bg.jpg --sam-checkpoint sam_vit_h.pth
"""

from __future__ import annotations

import argparse
import logging
import sys

from audio_analyzer import AudioAnalyzer
from visual_processor import MotionAnalyzer, EmotionAnalyzer, Segmenter
from effect_engine import (
    EffectPipeline,
    ColorGradeEffect,
    LambdaEffect,
    background_blur,
)
from video_editor_core import VideoEditorCore, ClipConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smart video clipping + AI effects engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Path to the source .mp4 video")
    p.add_argument("-o", "--output", default="output_clips", help="Output directory")
    p.add_argument("--funny-only", action="store_true",
                   help="Only keep funny moments (laughter / happy faces)")
    p.add_argument("--max-clips", type=int, default=10, help="Max number of clips")
    p.add_argument("--min-duration", type=float, default=3.0, help="Min clip length (s)")
    p.add_argument("--max-duration", type=float, default=30.0, help="Max clip length (s)")

    # Effects
    p.add_argument("--grade", action="store_true", help="Apply cinematic colour grade")
    p.add_argument("--bg-blur", action="store_true", help="Blur the background (bokeh)")
    p.add_argument("--replace-bg", metavar="IMG",
                   help="Replace background with this image")
    p.add_argument("--sam-checkpoint", help="SAM checkpoint for segmentation")
    p.add_argument("--sam-model-type", default="vit_h", help="SAM model type")
    p.add_argument("--device", default="cpu", help="Torch device for SAM (cpu/cuda)")

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def build_effects(args, editor: VideoEditorCore) -> EffectPipeline:
    pipeline = EffectPipeline()

    if args.grade:
        pipeline.add(ColorGradeEffect(contrast=1.1, saturation=1.15,
                                      brightness=0.03, temperature=0.1))

    seg_kwargs = dict(
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device,
    )

    if args.replace_bg:
        replacer = editor.build_background_replacer(args.replace_bg, **seg_kwargs)
        pipeline.add(LambdaEffect(replacer, name="replace_bg"))
    elif args.bg_blur:
        seg = editor._ensure_segmenter(**seg_kwargs)

        def _blur_bg(frame):
            return background_blur(frame, seg.segment(frame))

        pipeline.add(LambdaEffect(_blur_bg, name="bg_blur"))

    return pipeline


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    editor = VideoEditorCore(
        clip_config=ClipConfig(
            max_clips=args.max_clips,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    )

    effects = build_effects(args, editor)
    if not effects.effects:
        effects = None

    clips = editor.process(
        args.input,
        output_dir=args.output,
        effects=effects,
        funny_only=args.funny_only,
    )

    if clips:
        print(f"\nGenerated {len(clips)} clip(s):")
        for c in clips:
            print(f"  - {c}")
    else:
        print("No clips were generated (no highlights detected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
