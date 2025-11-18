# Deepfake Detection API

A production-ready FastAPI service for detecting deepfakes in videos and images using state-of-the-art models. The service features parallel and sequential processing pipelines, demographic analysis, and comprehensive API documentation.

## 🚀 Key Features

- **Multi-Model Detection**: Utilizes XceptionNet, MesoNet, Face X-ray, EfficientNetV2, and GenConvViT models
- **Parallel & Sequential Pipelines**: Choose between parallel processing or sequential pipeline with dynamic weighting
- **Demographic Analysis**: Age, gender, race, and accessory detection using DeepFace and YOLO
- **Face Detection & Tracking**: Advanced face detection with RetinaFace and tracking across video frames
- **Containerized Deployment**: Ready-to-deploy Docker container with all dependencies pre-installed
- **RESTful API**: Fully documented OpenAPI/Swagger interface
- **Performance Optimized**: Supports both CPU and GPU acceleration

## Features

* REST API powered by FastAPI and Uvicorn
* Parallel deepfake detection on video and image inputs
* Support for multiple pre-trained models from Hugging Face
* Temporary storage for uploads and processing outputs
* Model checkpoints and preprocessing scripts included

## 🛠️ System Requirements

- **Minimum**:
  - 8GB RAM
  - 20GB free disk space
  - Docker 20.10+
  - x86_64/amd64 CPU

- **Recommended**:
  - 16GB+ RAM (32GB for production)
  - NVIDIA GPU with 8GB+ VRAM
  - Ubuntu 20.04+ or Windows 10/11 with WSL2
  - Docker with NVIDIA Container Toolkit

- **Disk Space**:
  - Base image: ~2GB
  - Models: ~3GB
  - Temporary files: 5-10GB during processing

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/deepfake-detection-api.git
cd deepfake-detection-api/Deepfake

# 2. Build the Docker image
docker build -t deepfake-api .

# 3. Run the container
docker run -d \
  -p 8000:8000 \
  --gpus all \
  --shm-size=8g \
  --name deepfake-container \
  deepfake-api

# 4. Access the API documentation
#    http://localhost:8000/docs
```

### Local Development

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-pipeline.txt

# 3. Set environment variables
export INSIGHTFACE_HOME=$(pwd)/models

# 4. Run the API
uvicorn main_updated:app --host 0.0.0.0 --port 8000 --reload
```

## 🔧 Configuration

Create a `.env` file in the project root:

```ini
# Application
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=true

# Model Settings
MODEL_DEVICE=cuda  # or 'cpu'
FRAME_SKIP=5       # Process every 5th frame in videos
MAX_FRAMES=100     # Max frames to process per video

# Paths
MODELS_DIR=./models
TEMP_UPLOADS=./temp_uploads
PROCESSED_VIDEOS=./processed_videos

# Performance
WORKERS=4          # Number of worker processes
THREADS=2          # Threads per worker
```

## 🛠️ API Endpoints

### Parallel Detection
Process multiple models in parallel for maximum throughput:
- `POST /api/parallel/detect/image` - Analyze a single image
- `POST /api/parallel/detect/video` - Process a video file
- `GET /api/parallel/models` - List available parallel models

### Sequential Pipeline
Process frames through a sequential pipeline with dynamic weighting:
- `POST /api/sequential/detect/image` - Process image through pipeline
- `POST /api/sequential/detect/video` - Process video through pipeline
- `GET /api/sequential/models` - List pipeline components

### Unified Endpoint
Single endpoint for all detection needs:
- `POST /api/unified/detect` - Unified detection interface
- `GET /api/unified/models` - List all available models

### Available Endpoints

- `POST /api/parallel/detect/image` - Process a single image
- `POST /api/parallel/detect/video` - Process a video file
- `GET /api/parallel/models` - List available models

### Example Usage

```python
import requests

# 1. List available models
response = requests.get("http://localhost:8000/api/parallel/models")
print(response.json())

# 2. Detect deepfake in an image
with open("test.jpg", "rb") as f:
    files = {"file": ("test.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/api/parallel/detect/image", files=files)
    print(response.json())

# 3. Detect deepfake in a video
with open("test.mp4", "rb") as f:
    files = {"file": ("test.mp4", f, "video/mp4")}
    response = requests.post(
        "http://localhost:8000/api/parallel/detect/video",
        files=files,
        data={"frame_skip": 10, "max_frames": 100}
    )
    print(response.json())
```

### Response Format

#### Image Detection
```json
{
  "status": "success",
  "processing_time_seconds": 1.23,
  "models_run": 5,
  "successful_models": 5,
  "results": {
    "model1": {
      "success": true,
      "logits": [...],
      "probabilities": [0.1, 0.9],
      "error": null
    },
    "model2": { ... }
  }
}
```

#### Video Detection
```json
{
  "status": "success",
  "video_metadata": {
    "filename": "test.mp4",
    "total_frames": 300,
    "processed_frames": 30,
    "fps": 30.0,
    "duration_seconds": 10.0
  },
  "processing_stats": {
    "total_time_seconds": 15.67,
    "frames_per_second": 1.92,
    "frame_skip": 10
  },
  "average_scores": {
    "model1": {
      "average_score": 0.85,
      "std_dev": 0.12,
      "frames_processed": 30
    }
  },
  "frame_results": [...]
}
```

## 📁 Project Structure

```
.
├── app/                           # FastAPI application
│   ├── api/                      # API endpoints
│   │   ├── parallel_detection.py # Parallel model endpoints
│   │   ├── sequential.py         # Sequential pipeline endpoints
│   │   ├── unified_detection.py  # Unified detection endpoints
│   │   └── face_indexing.py      # Face recognition and indexing
│   ├── models/                   # Model implementations
│   └── main_updated.py           # Main application entry point
│
├── pipeline/                     # Video processing pipeline
│   ├── demographic_analyzer.py   # Age/gender/race analysis
│   ├── face_detector.py         # Face detection and tracking
│   ├── video_processor.py       # Video frame processing
│   └── pipeline.py              # Main pipeline orchestrator
│
├── models/                       # Model weights and configs
│   └── insightface/             # Face recognition models
│       └── buffalo_l/           # Pre-downloaded InsightFace models
│
├── requirements-core.txt         # Core dependencies
├── requirements-pipeline.txt     # Pipeline dependencies
├── Dockerfile                   # Container configuration
└── .env                         # Environment variables
```

## 🚀 Deployment

### Docker Compose (Production)

```yaml
version: '3.8'

services:
  deepfake-api:
    image: deepfake-api:latest
    build: .
    ports:
      - "8000:8000"
    environment:
      - APP_HOST=0.0.0.0
      - APP_PORT=8000
      - MODEL_DEVICE=${MODEL_DEVICE:-cuda}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/app_root/models
      - ./temp_uploads:/app_root/temp_uploads
      - ./processed_videos:/app_root/processed_videos
    restart: unless-stopped
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepfake-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: deepfake-api
  template:
    metadata:
      labels:
        app: deepfake-api
    spec:
      containers:
      - name: deepfake-api
        image: deepfake-api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: deepfake-api
spec:
  selector:
    app: deepfake-api
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
  type: LoadBalancer
```

## 📊 Performance

### Resource Usage
- **CPU**: 4-8 cores recommended
- **Memory**: 8GB minimum, 16GB+ recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM for optimal performance

### Expected Throughput
- **Images**: 2-5 seconds per image (depending on model)
- **Video (1 min, 30fps)**: 30-60 seconds with frame skipping

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for face analysis models
- [Hugging Face](https://huggingface.co/) for model hosting
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Docker](https://www.docker.com/) for containerization

## Running Locally (Without Docker)

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Start the server:

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
