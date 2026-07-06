# Smart Video Clipper — web app image.
# Build:  docker build -t video-clipper .
# Run:    docker run --rm -p 5000:5000 video-clipper
# Open:   http://localhost:5000
FROM python:3.11-slim

# ffmpeg is required by moviepy/librosa for reading and writing video/audio.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
ENV PORT=5000
CMD ["python", "webapp.py"]
