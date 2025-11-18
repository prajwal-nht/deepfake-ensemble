# Ensemble Deepfake Detection API

This is a production-ready deployment of an ensemble deepfake detection system using 5 different models, designed to run on Modal with GPU acceleration.

## Models Included

1. **Xception-based** - Specialized in detecting GAN-generated faces
2. **EfficientNet** - Efficient architecture for deepfake detection
3. **Vision Transformer (ViT)** - Transformer-based approach for image classification
4. **ResNet** - Deep residual network for image analysis
5. **Face X-ray** - Detects blending boundaries in deepfake images

## Prerequisites

- Python 3.10+
- Modal account (sign up at [modal.com](https://modal.com))
- GPU with CUDA support (handled by Modal)

## Project Structure

```
deepfake-detection/
├── ensemble_detector.py    # Main API implementation
├── deploy.py              # Deployment script
├── requirements.txt        # Python dependencies
├── test_ensemble_api.py    # Test script (auto-generated)
├── training/              # Model weights and configs
│   ├── weights/
│   └── config/
└── preprocessing/         # Any preprocessing utilities
```

## Deployment Instructions

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install modal
   ```

2. **Set up your Modal account**:
   ```bash
   modal token new
   ```

3. **Prepare model files**:
   - Place your model weights in `training/weights/`
   - Add configuration files to `training/config/detector/`

4. **Deploy to Modal**:
   ```bash
   python deploy.py
   ```

5. **Test the API**:
   ```bash
   python test_ensemble_api.py <your_modal_url> <path_to_image>
   ```

## API Endpoints

- `GET /` - Health check
- `POST /detect` - Process an image for deepfake detection
- `GET /health` - Service health status

## Request Format

```http
POST /detect
Content-Type: multipart/form-data

file: <image_file>
```

## Response Format

```json
{
  "status": "success",
  "faces": [
    {
      "bbox": [x1, y1, x2, y2],
      "models": {
        "xception": {
          "prediction": "real|fake",
          "confidence": 0.95,
          "all_scores": [0.1, 0.9]
        },
        "efficientnet": { ... },
        ...
      }
    }
  ]
}
```

## Local Development

To run the API locally for testing:

```bash
uvicorn ensemble_detector:web_app --reload --host 0.0.0.0 --port 8000
```

## Monitoring and Logs

View logs and monitor your deployment in the [Modal dashboard](https://modal.com/dashboard).

## Troubleshooting

1. **Missing model files**: Ensure all required model files are in the correct directories
2. **CUDA errors**: Check that your Modal environment has GPU access
3. **Memory issues**: The T4 GPU has 16GB VRAM - reduce batch size if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.
