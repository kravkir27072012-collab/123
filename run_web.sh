#!/usr/bin/env bash
# Запуск веб-сайта локально одной командой.
#   ./run_web.sh
# Затем откройте http://localhost:5000 в браузере.
set -e

cd "$(dirname "$0")"

# Виртуальное окружение (создаётся при первом запуске)
if [ ! -d ".venv" ]; then
  echo ">> Создаю виртуальное окружение .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> Устанавливаю зависимости (может занять пару минут при первом запуске)"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Проверка ffmpeg (нужен moviepy/librosa)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "!! ffmpeg не найден в системе."
  echo "   macOS:   brew install ffmpeg"
  echo "   Ubuntu:  sudo apt install ffmpeg"
  echo "   (или используйте Docker: docker build -t video-clipper . && docker run --rm -p 5000:5000 video-clipper)"
fi

echo ">> Запускаю веб-сайт на http://localhost:5000  (Ctrl+C для остановки)"
python webapp.py
