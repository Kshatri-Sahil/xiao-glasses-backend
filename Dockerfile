FROM python:3.11-slim

# Install system dependencies for OpenCV, FFmpeg, and audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-compiled binary dlib and face-recognition
RUN pip install --no-cache-dir dlib-bin face-recognition-models && \
    pip install --no-cache-dir --no-deps face-recognition

# Install rest of requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy assets and app code
COPY . .

# Bind Gunicorn to dynamic Render $PORT (single bind to avoid duplicate port conflict)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 300 app:app"]
