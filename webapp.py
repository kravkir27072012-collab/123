"""
webapp.py
=========

Flask web front-end for the smart video clipping & AI effects pipeline.

Upload a long video in the browser, pick options (funny-only, colour grade,
background blur, number of clips), and the server runs the pipeline in a
background thread while the page polls for progress. When it finishes, the
generated clips are shown inline with `<video>` players and download links.

Run::

    pip install -r requirements.txt flask
    python webapp.py                      # -> http://127.0.0.1:5000

Endpoints
---------
GET  /                       upload page (single-page app)
POST /api/upload             accept a video + options, start a job
GET  /api/status/<job_id>    JSON job status / progress / result clips
GET  /clips/<job_id>/<name>  serve a rendered clip
GET  /uploads/<job_id>/<name> serve the original upload (for comparison)
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
    abort,
)
from werkzeug.utils import secure_filename

from effect_engine import (
    EffectPipeline,
    ColorGradeEffect,
    LambdaEffect,
    background_blur,
)
from video_editor_core import VideoEditorCore, ClipConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("webapp")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(BASE_DIR, "web_workspace")
UPLOAD_DIR = os.path.join(WORK_DIR, "uploads")
CLIPS_DIR = os.path.join(WORK_DIR, "clips")
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_CONTENT_MB = 512

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    filename: str
    state: str = "queued"        # queued | analyzing | rendering | done | error
    progress: float = 0.0        # 0..1 (coarse)
    message: str = ""
    clips: List[dict] = field(default_factory=list)
    error: Optional[str] = None


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _set(job: Job, **kw) -> None:
    with JOBS_LOCK:
        for k, v in kw.items():
            setattr(job, k, v)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
def _build_effects(editor: VideoEditorCore, opts: dict) -> Optional[EffectPipeline]:
    pipe = EffectPipeline()
    if opts.get("grade"):
        pipe.add(ColorGradeEffect(contrast=1.1, saturation=1.15,
                                  brightness=0.03, temperature=0.1))
    if opts.get("bg_blur"):
        seg = editor._ensure_segmenter()

        def _blur_bg(frame):
            return background_blur(frame, seg.segment(frame))

        pipe.add(LambdaEffect(_blur_bg, name="bg_blur"))
    return pipe if pipe.effects else None


def _run_job(job: Job, video_path: str, opts: dict) -> None:
    try:
        editor = VideoEditorCore(
            clip_config=ClipConfig(
                max_clips=int(opts.get("max_clips", 6)),
                min_duration=float(opts.get("min_duration", 3.0)),
                max_duration=float(opts.get("max_duration", 30.0)),
            )
        )

        _set(job, state="analyzing", progress=0.1,
             message="Анализ аудио и видео (громкость, смех, движение, эмоции)…")
        audio_events, motion_events, emotion_events = editor.analyze(video_path)

        _set(job, progress=0.55, message="Поиск и ранжирование ключевых моментов…")
        highlights = editor.find_highlights(
            audio_events, motion_events, emotion_events,
            funny_only=bool(opts.get("funny_only")),
        )

        # duration for trimming
        VideoFileClip = _import_vfc()
        with VideoFileClip(video_path) as vc:
            duration = vc.duration
        highlights = editor.auto_trim(highlights, duration)

        if not highlights:
            _set(job, state="done", progress=1.0,
                 message="Ключевые моменты не найдены — попробуйте другое видео "
                         "или отключите «только смешные».", clips=[])
            return

        _set(job, state="rendering", progress=0.7,
             message=f"Рендер {len(highlights)} клип(ов)…")
        effects = _build_effects(editor, opts)

        out_dir = os.path.join(CLIPS_DIR, job.id)
        paths = editor.render_clips(video_path, highlights, out_dir, effects=effects)

        clips = []
        for h, p in zip(highlights, paths):
            clips.append({
                "name": os.path.basename(p),
                "url": f"/clips/{job.id}/{os.path.basename(p)}",
                "start": round(h.start, 1),
                "end": round(h.end, 1),
                "duration": round(h.duration, 1),
                "score": round(float(h.score), 2),
                "funny": bool(h.is_funny),
                "reasons": h.reasons,
            })

        _set(job, state="done", progress=1.0,
             message=f"Готово: создано {len(clips)} клип(ов).", clips=clips)
        logger.info("Job %s finished with %d clips", job.id, len(clips))

    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed", job.id)
        _set(job, state="error", error=str(exc),
             message=f"Ошибка обработки: {exc}")


def _import_vfc():
    try:
        from moviepy import VideoFileClip
    except ImportError:  # pragma: no cover
        from moviepy.editor import VideoFileClip
    return VideoFileClip


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", max_mb=MAX_CONTENT_MB)


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "Файл не выбран"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Пустое имя файла"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Неподдерживаемый формат {ext}. "
                                 f"Разрешено: {', '.join(sorted(ALLOWED_EXT))}"}), 400

    job_id = uuid.uuid4().hex[:12]
    safe_name = secure_filename(file.filename) or f"upload{ext}"
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)
    video_path = os.path.join(job_upload_dir, safe_name)
    file.save(video_path)

    opts = {
        "funny_only": request.form.get("funny_only") == "true",
        "grade": request.form.get("grade") == "true",
        "bg_blur": request.form.get("bg_blur") == "true",
        "max_clips": request.form.get("max_clips", "6"),
    }

    job = Job(id=job_id, filename=safe_name)
    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(target=_run_job, args=(job, video_path, opts), daemon=True).start()
    logger.info("Queued job %s (%s) opts=%s", job_id, safe_name, opts)
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Неизвестная задача"}), 404
        return jsonify({
            "id": job.id,
            "state": job.state,
            "progress": job.progress,
            "message": job.message,
            "clips": job.clips,
            "error": job.error,
            "source_url": f"/uploads/{job.id}/{job.filename}",
        })


@app.route("/clips/<job_id>/<path:name>")
def serve_clip(job_id: str, name: str):
    directory = os.path.join(CLIPS_DIR, secure_filename(job_id))
    if not os.path.isdir(directory):
        abort(404)
    return send_from_directory(directory, name)


@app.route("/uploads/<job_id>/<path:name>")
def serve_upload(job_id: str, name: str):
    directory = os.path.join(UPLOAD_DIR, secure_filename(job_id))
    if not os.path.isdir(directory):
        abort(404)
    return send_from_directory(directory, name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
