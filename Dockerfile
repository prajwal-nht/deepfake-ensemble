# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy core requirements first
COPY requirements-core.txt .

# Install core Python packages first
RUN pip install --no-cache-dir -r requirements-core.txt

# Now copy the full requirements
COPY requirements.txt .

# Install remaining packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/app/temp_uploads /app/app/processed_videos /app/temp_frames

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
