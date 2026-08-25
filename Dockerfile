FROM python:3.11-slim

# Install system dependencies: ffmpeg (for audio), build-essential & cmake (for dlib/face_recognition)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files and assets (PDF, player images, etc.)
COPY . .

# Set default port
ENV PORT=5000
EXPOSE 5000

# Start gunicorn with 1 worker and 4 threads
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120 app:app"]
