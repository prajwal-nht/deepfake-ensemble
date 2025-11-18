"""
Ensemble Deepfake Detection API Deployment for Modal

This script deploys an ensemble of 5 deepfake detection models:
1. Xception-based model
2. EfficientNet-based model
3. Vision Transformer (ViT)
4. ResNet-based model
5. Face X-ray model
"""
import os
import sys
import logging
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict, Any, List
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, ViTForImageClassification
from PIL import Image
import cv2
import modal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal app setup
app = modal.App("deepfake-ensemble-api")

# Define image with required dependencies
image = (
    modal.Image
    .debian_slim(python_version="3.10")
    .pip_install([
        "torch>=2.0.0",
        "torchvision",
        "opencv-python-headless",
        "numpy",
        "fastapi",
        "python-multipart",
        "pillow",
        "transformers",
        "timm",
        "deepface",
        "tf-keras",
        "retina-face",
        "facenet-pytorch",
        "scikit-image"
    ])
)

# Model configurations - Updated with working models
MODEL_CONFIGS = {
    "xception": {
        "name": "dima806/deepfake_vs_real_image_detection",
        "type": "huggingface",
        "input_size": 299,
        "description": "Xception-based deepfake detector (trained for deepfake)"
    },
    "vit_deepfake": {
        "name": "prithivMLmods/Deep-Fake-Detector-Model",
        "type": "huggingface",
        "input_size": 224,
        "description": "ViT-based deepfake detector (trained for deepfake)"
    },
    "vit_wvolf": {
        "name": "Wvolf/ViT_Deepfake_Detection",
        "type": "huggingface",
        "input_size": 224,
        "description": "ViT deepfake detector by Wvolf (trained for deepfake)"
    },
    "vit_deepfake_v2": {
        "name": "prithivMLmods/deepfake-detector-model-v1",
        "type": "huggingface",
        "input_size": 224,
        "description": "ViT-based deepfake detector v2 (trained for deepfake)"
    },
    "efficientnet_b4": {
        "name": "google/efficientnet-b4",
        "type": "huggingface",
        "input_size": 380,
        "description": "EfficientNet-B4 (general model, adds diversity)"
    }
}

# Thresholds - Lowered to reduce false negatives
FRAME_DEEPFAKE_THRESHOLD = 0.70
OVERALL_VIDEO_DEEPFAKE_THRESHOLD = 0.65

class EnsembleDeepfakeDetector:
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all models in the ensemble"""
        logger.info(f"Initializing models on {self.device}...")
        
        for model_name, config in MODEL_CONFIGS.items():
            try:
                logger.info(f"Loading {model_name}...")
                
                if config["type"] == "huggingface":
                    model = AutoModelForImageClassification.from_pretrained(
                        config["name"],
                        num_labels=2  # Binary classification (real/fake)
                    )
                    feature_extractor = AutoFeatureExtractor.from_pretrained(config["name"])
                    
                    self.models[model_name] = {
                        'model': model.to(self.device).eval(),
                        'feature_extractor': feature_extractor,
                        'config': config
                    }
                
                logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {str(e)}")
                raise
    
    def preprocess_image(self, image: np.ndarray, model_name: str) -> torch.Tensor:
        """Preprocess image for a specific model"""
        model_info = self.models[model_name]
        config = model_info['config']
        
        # Resize image
        target_size = (config["input_size"], config["input_size"])
        img_resized = cv2.resize(image, target_size)
        
        # Convert to RGB if needed
        if len(img_resized.shape) == 2:  # Grayscale
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif img_resized.shape[2] == 4:  # RGBA
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def predict_single(self, image: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Run prediction using a single model"""
        try:
            model_info = self.models[model_name]
            model = model_info['model']
            
            # Preprocess image
            inputs = self.preprocess_image(image, model_name)
            
            # Run inference
            with torch.no_grad():
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                probs = torch.softmax(logits, dim=-1)
                confidence, pred = torch.max(probs, 1)
            
            return {
                'model': model_name,
                'prediction': 'fake' if pred.item() == 1 else 'real',
                'confidence': confidence.item(),
                'all_scores': probs.cpu().numpy().tolist()[0]
            }
            
        except Exception as e:
            logger.error(f"Prediction error with {model_name}: {str(e)}")
            return {
                'model': model_name,
                'error': str(e),
                'prediction': 'error',
                'confidence': 0.0
            }
    
    def predict_ensemble(self, image: np.ndarray) -> Dict[str, Any]:
        """Run prediction using all models and combine results"""
        results = {}
        model_predictions = []
        
        # Get predictions from all models
        for model_name in self.models:
            result = self.predict_single(image, model_name)
            model_predictions.append(result)
        
        # Calculate ensemble prediction (simple average)
        confidences = [p.get('confidence', 0) for p in model_predictions 
                      if 'confidence' in p and p.get('prediction') != 'error']
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            ensemble_prediction = 'fake' if avg_confidence > 0.5 else 'real'
        else:
            avg_confidence = 0.0
            ensemble_prediction = 'error'
        
        return {
            'status': 'success',
            'ensemble_prediction': ensemble_prediction,
            'ensemble_confidence': avg_confidence,
            'model_predictions': model_predictions,
            'timestamp': str(datetime.now())
        }

# Initialize FastAPI app
web_app = FastAPI(title="Ensemble Deepfake Detection API")

@web_app.get("/")
async def root():
    return {"message": "Ensemble Deepfake Detection API is running"}

@web_app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # Read and validate image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Initialize detector if not already done
        if not hasattr(web_app, 'detector'):
            web_app.detector = EnsembleDeepfakeDetector()
        
        # Get predictions
        result = web_app.detector.predict_ensemble(img)
        return result
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount FastAPI app to Modal
@app.function(
    gpu="T4",
    image=image,
    timeout=600,
    memory=8192,
    cpu=4.0
)
@modal.asgi_app()
def fastapi_app():
    return web_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8000)
