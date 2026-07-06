# Smart Video Clipping & AI Effects Engine

Автоматическая нарезка длинного видео на короткие клипы вокруг «крутых» и
«смешных» моментов, плюс модуль продвинутой визуальной обработки (замена фона,
удаление объектов, фильтры и эффекты).

Аудио анализируется через **Librosa**, кадры — через **OpenCV** (Optical Flow),
эмоции лиц — через **FER/DeepFace**, сегментация — через **Segment Anything
(SAM)**, финальный рендер и склейка — через **MoviePy**.

## Возможности

### 1. Smart Clipping — автоматическая нарезка
- **Highlight Detection** — резкие изменения громкости (RMS), детекция
  онсетов/ударов, аплодисментов и смеха через `librosa`; высокая динамика
  движения в кадре через плотный Optical Flow (Farneback).
- **Funny Moments Detection** — смех (аудио-эвристика по 1–4 кГц полосе +
  bursty ZCR) и «счастливые/смеющиеся» лица через `FER` или `DeepFace`.
- **Auto-trimming** — обрезка видео вокруг ключевых точек с паддингом,
  ограничением мин/макс длины, слиянием пересечений и отбором топ-N.

### 2. Advanced Visual Processing — визуальная обработка
- **Background Removal / Replacement** — точные маски через **SAM**
  (fallback: `rembg` → GrabCut), замена фона на новое изображение.
- **Object Removal** — inpainting через **LaMa** (`simple-lama-inpainting`,
  если установлен) или `cv2.inpaint` (Telea / Navier-Stokes).
- **AI Effects Engine** — цветокоррекция, размытие, боке-размытие фона,
  наложение оверлеев через OpenCV, применяемые к клипу через MoviePy.

## Архитектура / модули

| Модуль | Назначение |
|--------|-----------|
| `audio_analyzer.py` | Аудио-анализ: громкость, онсеты, смех/аплодисменты (Librosa). |
| `visual_processor.py` | Optical Flow (движение), эмоции лиц (FER/DeepFace), сегментация (SAM/rembg). |
| `effect_engine.py` | Фильтры, эффекты, замена фона, inpainting; композиция эффектов. |
| `video_editor_core.py` | Оркестрация пайплайна, фьюжн событий, скоринг, auto-trim, рендер (MoviePy). |
| `cli.py` | Командная строка. |
| `webapp.py` + `templates/` | Веб-интерфейс (Flask): загрузка видео, прогресс, просмотр/скачивание клипов. |

### Технический пайплайн

```
Input (.mp4 + аудио-дорожка)
   │
   ├─ 1. Audio Analysis   (loudness / onsets / laughter / applause)   ── audio_analyzer
   ├─ 2. Visual Analysis  (optical-flow motion / face emotions)       ── visual_processor
   ├─ 3. Fusion & Scoring (события на общий таймлайн, ранжирование)    ── video_editor_core
   ├─ 4. Segmentation & Effects (SAM-маски, фон, inpainting, фильтры)  ── visual_processor + effect_engine
   └─ 5. Rendering        (auto-trim + склейка коротких клипов)        ── video_editor_core (MoviePy)
   │
   └─▶ Output: набор коротких, отредактированных, улучшенных клипов
```

## Установка

```bash
pip install -r requirements.txt
```

Тяжёлые модели опциональны — код деградирует мягко (если SAM/FER/DeepFace/LaMa
недоступны, соответствующая функция отключается с предупреждением, а не падает):

- **SAM**: скачайте чекпоинт, напр. `sam_vit_h_4b8939.pth`
  (`https://github.com/facebookresearch/segment-anything#model-checkpoints`)
  и передайте через `--sam-checkpoint`. Без него используется `rembg`.
- **LaMa**: `pip install simple-lama-inpainting` для качественного inpainting;
  иначе используется `cv2.inpaint`.
- **Эмоции**: `FER` идёт в `requirements.txt`; `DeepFace` — альтернатива.

## Веб-интерфейс

Самый простой способ «посмотреть на приложение» — запустить веб-сайт: загрузить
видео в браузере, выбрать опции и получить готовые клипы с плеерами и кнопками
скачивания.

**Вариант 1 — Docker (проще всего, ffmpeg уже внутри):**

```bash
docker build -t video-clipper .
docker run --rm -p 5000:5000 video-clipper
# откройте http://localhost:5000
```

**Вариант 2 — скрипт (создаёт venv и ставит зависимости):**

```bash
./run_web.sh
# откройте http://localhost:5000
```

**Вариант 3 — вручную:**

```bash
pip install -r requirements.txt   # нужен также системный ffmpeg
python webapp.py                  # -> http://127.0.0.1:5000
```

Возможности страницы:
- Drag-and-drop загрузка видео (MP4/MOV/AVI/MKV/WEBM).
- Опции: «только смешное», цветокоррекция, размытие фона, число клипов.
- Фоновая обработка с индикатором прогресса (задачи выполняются в отдельном
  потоке, страница опрашивает статус).
- Результат: клипы с `<video>`-плеерами, бейджами 🔥HIGHLIGHT / 😂FUNNY,
  таймкодами, score и ссылками на скачивание.

Модуль `webapp.py` — тонкая обёртка над `VideoEditorCore`; вся логика нарезки
живёт в основных модулях.

## Использование (CLI)

```bash
# Авто-нарезка хайлайтов
python cli.py input.mp4 --output clips/

# Только смешные моменты (смех + счастливые лица)
python cli.py input.mp4 --output clips/ --funny-only

# Цветокоррекция + боке-размытие фона на каждом клипе
python cli.py input.mp4 --output clips/ --grade --bg-blur

# Замена фона через SAM
python cli.py input.mp4 --output clips/ \
    --replace-bg new_bg.jpg --sam-checkpoint sam_vit_h_4b8939.pth --device cuda
```

## Использование (Python API)

```python
from video_editor_core import VideoEditorCore, ClipConfig
from effect_engine import EffectPipeline, ColorGradeEffect

editor = VideoEditorCore(clip_config=ClipConfig(max_clips=8, min_duration=4))

# Полный пайплайн: анализ → хайлайты → auto-trim → рендер
clips = editor.process("long_video.mp4", output_dir="clips/")

# С эффектами (цветокоррекция на каждом клипе)
fx = EffectPipeline([ColorGradeEffect(contrast=1.1, saturation=1.15)])
clips = editor.process("long_video.mp4", output_dir="clips/", effects=fx, funny_only=True)
```

Точечная работа с примитивами обработки кадра:

```python
import cv2
from visual_processor import Segmenter
from effect_engine import replace_background, inpaint_object

seg = Segmenter(sam_checkpoint="sam_vit_h_4b8939.pth")
frame = cv2.imread("frame.png")
mask = seg.segment(frame)                                    # маска переднего плана
new_frame = replace_background(frame, mask, cv2.imread("bg.jpg"))   # замена фона
clean = inpaint_object(frame, object_mask, method="auto")    # удаление объекта
```

## Замечания по реализации
- Optical Flow считается на прореженных кадрах (по умолчанию 5 fps, ширина 320px)
  для скорости.
- Детекция смеха/аплодисментов — спектральная эвристика (полоса 1–4 кГц, ZCR,
  spectral flatness), разделение по «бёрстности» сигнала.
- Скоринг фьюзит аудио- и видео-события на общий таймлайн: пересекающиеся
  события усиливают друг друга; смех и счастливые лица дают «funny»-скор.
