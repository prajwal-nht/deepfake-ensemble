# Deepfake Detection Ensemble

A production-ready deepfake detection API using an ensemble of 5 specialized models with demographic-aware weighting. Deployed on Modal for serverless, scalable inference.

## Core Features

- **5-Model Ensemble**: Multiple HuggingFace models voting together for higher accuracy
- **Demographic Analysis**: Adjusts confidence based on detected age, gender, and ethnicity
- **Video & Image Support**: Frame-by-frame video analysis and single image detection
- **FastAPI Backend**: RESTful API with automatic documentation
- **Modal Deployment**: Serverless inference that scales on demand

## Quick Start

### Run Locally

```bash
cd Deepfake-API/Deepfake
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`

### Deploy to Modal

```bash
cd Deepfake-API/Deepfake
modal token new  # First time only
modal deploy modal_ensemble_deploy.py
```

You'll get a production endpoint like `https://your-app.modal.run`

### Test the API

```bash
# Test image detection
curl -X POST http://localhost:8000/api/unified/detect/image \
  -F "file=@test_image.jpg"

# Test video detection
curl -X POST http://localhost:8000/api/unified/detect/video \
  -F "file=@test_video.mp4"
```

## Ensemble Detector

The core detection logic lives in `ensemble_detector.py`. Here's how it works:

### Architecture

```python
# 5 models vote on each image
MODEL_CONFIGS = {
    "xception": "dima806/deepfake_vs_real_image_detection",
    "sdxl": "Organika/sdxl-detector", 
    "ai_detector": "umm-maybe/AI-image-detector",
    "logo_detector": "iamkaikai/amazing_logos_v3",
    "aesthetic": "cafeai/cafe_aesthetic"
}
```

Each model processes the image and returns a probability. The ensemble:
1. Collects all predictions
2. Applies demographic weighting (if face detected)
3. Combines votes using weighted average
4. Returns final prediction with confidence

### Demographic Weighting

When a face is detected, the system:
- Analyzes age, gender, ethnicity using DeepFace
- Adjusts model weights based on demographics
- Reduces false positives for underrepresented groups

Example: If detecting a young Asian female face, models trained on diverse datasets get higher weight.

### Running Standalone

You can use the ensemble detector directly without the FastAPI wrapper:

```python
from ensemble_detector import EnsembleDetector

detector = EnsembleDetector()
result = detector.detect_image("path/to/image.jpg")

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
print(f"Model votes: {result['model_predictions']}")
```

### Configuration

Edit thresholds in `ensemble_detector.py`:

```python
# Frame-level threshold for videos
FRAME_DEEPFAKE_THRESHOLD = 0.70  # Lower = more sensitive

# Overall video threshold
OVERALL_VIDEO_DEEPFAKE_THRESHOLD = 0.65

# Demographic adjustment range
DEMOGRAPHIC_WEIGHT_ADJUSTMENT = 0.10  # ±10%
```

## API Endpoints

### Unified Detection (Recommended)
- `POST /api/unified/detect/image` - Single image analysis
- `POST /api/unified/detect/video` - Video analysis (frame-by-frame)

### Parallel Processing
- `POST /api/parallel/detect/image` - All models run simultaneously
- `POST /api/parallel/detect/video` - Parallel video processing

### Sequential Processing  
- `POST /api/sequential/detect/image` - Models run one after another
- `POST /api/sequential/detect/video` - Sequential video processing

### Model Info
- `GET /api/parallel/models` - List available models
- `GET /docs` - Interactive API documentation

## Project Structure

```
Deepfake-API/Deepfake/
├── deepfake-detection/
│   ├── ensemble_detector.py       # Core ensemble logic
│   ├── demographic_analysis.py    # Face analysis & weighting
│   ├── evaluation_metrics.py      # Performance metrics
│   ├── modal_client.py           # Modal API client
│   └── modal_deploy.py           # Deployment configs
├── app/
│   ├── main.py                   # FastAPI app
│   └── api/
│       ├── unified_detection.py  # Unified endpoint
│       ├── parallel_detection.py # Parallel processing
│       └── sequential_detection.py # Sequential processing
├── modal_ensemble_deploy.py      # Main Modal deployment
└── requirements.txt              # Dependencies
```

## Configuration

### Environment Variables

Create `Deepfake-API/Deepfake/.env`:

```env
# Modal deployment (optional)
MODAL_TOKEN_ID=your_token_id
MODAL_TOKEN_SECRET=your_token_secret

# Model settings
FRAME_DEEPFAKE_THRESHOLD=0.70
OVERALL_VIDEO_DEEPFAKE_THRESHOLD=0.65
```

### Model Configuration

Edit `ensemble_detector.py` to add/remove models:

```python
MODEL_CONFIGS = {
    "your_model": {
        "name": "huggingface/model-name",
        "type": "huggingface",
        "input_size": 224,
        "weight": 1.0
    }
}
```

## How the Ensemble Works

### Model Pipeline

```
Input Image
    ↓
Face Detection (DeepFace/RetinaFace)
    ↓
Demographic Analysis (age, gender, ethnicity)
    ↓
Parallel Model Inference (5 models)
    ├─ Xception (deepfake specialist)
    ├─ SDXL Detector (AI art detector)
    ├─ AI Detector (general AI images)
    ├─ Logo Detector (watermark detection)
    └─ Aesthetic Scorer (quality assessment)
    ↓
Demographic Weighting (adjust for bias)
    ↓
Ensemble Voting (weighted average)
    ↓
Final Prediction + Confidence
```

### The 5 Models

1. **dima806/deepfake_vs_real_image_detection**
   - Xception-based architecture
   - Trained specifically on deepfakes
   - Best for face swaps and GAN-generated faces

2. **Organika/sdxl-detector**
   - Detects Stable Diffusion XL outputs
   - Good for AI-generated art
   - Catches modern diffusion models

3. **umm-maybe/AI-image-detector**
   - General AI image detector
   - Trained on multiple generators
   - Broad coverage

4. **iamkaikai/amazing_logos_v3**
   - Detects AI-generated logos/watermarks
   - Catches synthetic patterns
   - Useful for composite images

5. **cafeai/cafe_aesthetic**
   - Aesthetic quality scorer
   - Real photos tend to have natural quality
   - AI images often have "too perfect" aesthetics

### Voting Strategy

Each model outputs a probability (0-1). The ensemble:
- Weights each vote based on model reliability
- Applies demographic adjustments (±10%)
- Averages the weighted votes
- Applies threshold (default 0.70 for images)

Example:
```
Model A: 0.85 (weight: 1.0) → 0.85
Model B: 0.72 (weight: 0.8) → 0.576
Model C: 0.91 (weight: 1.2) → 1.092
Model D: 0.68 (weight: 0.9) → 0.612
Model E: 0.79 (weight: 1.0) → 0.79

Weighted average: (0.85 + 0.576 + 1.092 + 0.612 + 0.79) / 5 = 0.784
Demographic adjustment: +0.05 (young female face)
Final score: 0.834 → DEEPFAKE (>0.70 threshold)
```

## Troubleshooting

**Models not loading?**
- First run downloads ~2GB of models from HuggingFace
- Check internet connection
- Verify HuggingFace isn't rate-limiting you
- Try: `huggingface-cli login` if using private models

**Out of memory errors?**
- Reduce batch size in `ensemble_detector.py`
- Use CPU instead of GPU: `device = "cpu"`
- Process videos in smaller chunks
- Close other applications

**Low accuracy?**
- Check if models downloaded correctly
- Verify input image quality (not too small/blurry)
- Try adjusting thresholds in config
- Check demographic weighting is enabled

**Modal deployment fails?**
- Run `modal token new` to authenticate
- Check Modal dashboard for logs
- Verify all dependencies in `modal_ensemble_deploy.py`
- Try: `modal run modal_ensemble_deploy.py` first (test mode)

## Performance

- **Image analysis**: ~3-5 seconds per image
- **Video analysis**: ~30-60 seconds for 10-second video
- **Accuracy**: ~85-90% on test datasets
- **Models**: ~2GB total size (downloaded on first run)

## Additional Resources

- `FINAL_EVALUATION_REPORT.md` - Model performance metrics
- `FINAL_ARCHITECTURE_SUMMARY.md` - System architecture details
- `ENSEMBLE_FLOW_DIAGRAM.md` - Visual flow diagram
- `INTEGRATION_README.md` - API integration examples
- `Deepfake-API/Deepfake/deepfake-detection/README.md` - Detector documentation

## Frontend

A React + TypeScript frontend is available in the `frontend/` directory. See `frontend/README.md` for setup instructions.

## WASM Demo

Browser-based demos are in `wasm/` and `wasm-demo/` directories. These run models entirely client-side using WebAssembly.
