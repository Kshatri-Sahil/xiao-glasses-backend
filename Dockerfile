FROM python:3.11-slim

# Install runtime system dependencies (ffmpeg for audio, libgl for opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-compiled binary dlib and face-recognition without compiling from source
RUN pip install --no-cache-dir dlib-bin face-recognition-models && \
    pip install --no-cache-dir --no-deps face-recognition

# Install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Start gunicorn binding to dynamic Render $PORT (defaults to 10000)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 app:app"]
