# Stage 1: Base Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app_root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dri \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Install Python and PyTorch first (base dependencies)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face Transformers and related libraries
RUN pip install --no-cache-dir \
    transformers==4.35.0 \
    datasets==2.14.5 \
    accelerate==0.23.0 \
    timm==0.9.2

# Install TensorFlow and Keras with compatible versions
RUN pip install --no-cache-dir \
    tensorflow==2.9.3 \
    tensorflow-addons==0.17.1 \
    keras==2.9.0 \
    tensorflow-hub==0.13.0

# Install face processing libraries
RUN pip install --no-cache-dir \
    retina-face==0.0.13 \
    deepface==0.0.81 \
    insightface==0.7.1

# Copy requirements and install other dependencies
COPY requirements_fixed.txt .
COPY retinaface_compat.py .

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements_fixed.txt && \
    pip install --no-cache-dir \
        timm==0.9.12 \
        transformers[torch]>=4.35.0 \
        datasets>=2.14.5 \
        accelerate>=0.23.0 \
        ultralytics>=8.0.0 \
        einops==0.7.0 \
        pytubefix==8.13.1 \
        Pillow>=10.0.0 \
        scikit-image==0.22.0 \
        opencv-python-headless==4.9.0.80 \
        numpy==1.26.4 \
        onnxruntime==1.17.0 \
        transformers>=4.36.0 && \
    # Clean up
    rm -rf /root/.cache/pip

# Create necessary directories
RUN mkdir -p /app_root/app/temp_uploads \
    /app_root/app/processed_videos \
    /app_root/app/logs \
    /root/.insightface/models/

# Create models directory structure
RUN mkdir -p /root/.insightface/models/buffalo_l

# Copy pre-downloaded models from the models directory
COPY models/insightface/buffalo_l/ /root/.insightface/models/buffalo_l/

# Set environment variables
ENV INSIGHTFACE_MODELS_ROOT=/root/.insightface/models
ENV INSIGHTFACE_HOME=/root/.insightface

# Copy application code
COPY . .

# Verify model files
RUN ls -la /root/.insightface/models/buffalo_l/

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
