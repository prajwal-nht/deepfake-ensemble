"""
Modal Deployment Wrapper for Ensemble Detector
This file has NO top-level imports to avoid Modal's local validation
"""
import modal

# Modal app
app = modal.App("deepfake-ensemble-detector")

# Define image with all dependencies
image = (
    modal.Image
    .debian_slim(python_version="3.10")
    .apt_install([
        "git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev", "libgomp1"
    ])
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
        "scikit-image",
        "efficientnet-pytorch"
    ])
)

# Health check endpoint
@app.function(image=image, timeout=60)
@modal.web_endpoint(method="GET")
def health():
    return {"status": "healthy", "service": "deepfake-ensemble-detector"}

# Main detection endpoint
@app.function(image=image, gpu="T4", timeout=1200, container_idle_timeout=300)
@modal.web_endpoint(method="POST")
def detect(file_bytes: bytes, video_id: str = "", user_id: str = "", include_details: str = "false"):
    """Full detection endpoint with ensemble logic"""
    import torch
    import numpy as np
    import cv2
    import logging
    from datetime import datetime
    from PIL import Image as PILImage
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from facenet_pytorch import MTCNN
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        from deepface import DeepFace
        DEEPFACE_AVAILABLE = True
    except:
        DEEPFACE_AVAILABLE = False
    
    MODEL_CONFIGS = {
        "xception": {"name": "dima806/deepfake_vs_real_image_detection", "input_size": 299},
        "vit_deepfake": {"name": "prithivMLmods/Deep-Fake-Detector-Model", "input_size": 224},
        "vit_wvolf": {"name": "Wvolf/ViT_Deepfake_Detection", "input_size": 224},
        "vit_deepfake_v2": {"name": "prithivMLmods/deepfake-detector-model-v1", "input_size": 224},
        "efficientnet_b4": {"name": "google/efficientnet-b4", "input_size": 380}
    }
    
    THRESHOLD = 0.70
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("Initializing MTCNN...")
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], device=device, keep_all=True)
    
    logger.info("Loading models...")
    models = {}
    for name, config in MODEL_CONFIGS.items():
        try:
            model = AutoModelForImageClassification.from_pretrained(config["name"], num_labels=2, ignore_mismatched_sizes=True)
            processor = AutoImageProcessor.from_pretrained(config["name"])
            models[name] = {'model': model.to(device).eval(), 'processor': processor, 'config': config}
            logger.info(f"✅ {name}")
        except Exception as e:
            logger.error(f"❌ {name}: {e}")
    
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image"}
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = PILImage.fromarray(frame_rgb)
    
    boxes, probs, landmarks = mtcnn.detect(pil_frame, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return {"video_id": video_id, "user_id": user_id, "is_deepfake": False, "confidence": 0.0, "processed_at": datetime.utcnow().isoformat() + "Z", "message": "No faces detected"}
    
    box = boxes[0]
    x1, y1, x2, y2 = [int(b) for b in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    face_crop = frame[y1:y2, x1:x2]
    
    if face_crop.size == 0:
        return {"error": "Face crop failed"}
    
    demographics = None
    if DEEPFACE_AVAILABLE:
        try:
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            demo = DeepFace.analyze(face_rgb, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False, silent=True)
            if isinstance(demo, list):
                demo = demo[0]
            demographics = {"age": int(demo.get("age", 0)), "gender": demo.get("gender", {}), "dominant_gender": demo.get("dominant_gender", ""), "race": demo.get("race", {}), "dominant_race": demo.get("dominant_race", ""), "emotion": demo.get("emotion", {}), "dominant_emotion": demo.get("dominant_emotion", "")}
        except Exception as e:
            logger.warning(f"Demographics failed: {e}")
    
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray) / 255.0
    contrast = np.std(gray) / 255.0
    quality_score = 1.0 if laplacian_var > 100 else 0.5
    quality = {"sharpness": float(laplacian_var), "brightness": float(brightness), "contrast": float(contrast), "overall_score": quality_score}
    
    predictions = {}
    for name, model_data in models.items():
        try:
            pil_face = PILImage.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            inputs = model_data['processor'](images=pil_face, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model_data['model'](**inputs)
                probs_tensor = torch.nn.functional.softmax(outputs.logits, dim=-1)
            raw_prob = probs_tensor[0][1].item()
            
            adjustment = 0.0
            if quality_score > 0.9:
                adjustment += 0.02
            elif quality_score < 0.3:
                adjustment -= 0.05
            if demographics and 'gender' in demographics:
                gender_conf = max(demographics['gender'].values()) if demographics['gender'] else 50
                if gender_conf < 60:
                    adjustment -= 0.03
            adjustment = max(-0.10, min(0.10, adjustment))
            weighted_prob = max(0.0, min(1.0, raw_prob + adjustment))
            
            predictions[name] = {'raw_prob': float(raw_prob), 'weighted_prob': float(weighted_prob), 'is_fake': weighted_prob > THRESHOLD, 'adjustment': float(adjustment)}
            logger.info(f"{name}: {raw_prob:.4f} → {weighted_prob:.4f}")
        except Exception as e:
            logger.error(f"{name} error: {e}")
            predictions[name] = {'error': str(e)}
    
    valid_probs = [p['weighted_prob'] for p in predictions.values() if 'weighted_prob' in p]
    ensemble_score = sum(valid_probs) / len(valid_probs) if valid_probs else 0.0
    is_deepfake = ensemble_score > THRESHOLD
    
    response = {"video_id": video_id, "user_id": user_id, "is_deepfake": is_deepfake, "confidence": float(ensemble_score), "processed_at": datetime.utcnow().isoformat() + "Z", "face_detected": True, "face_confidence": float(probs[0]) if probs is not None else 0.0}
    
    if include_details.lower() == "true":
        response["details"] = {"face": {"bbox": [int(x1), int(y1), int(x2), int(y2)], "landmarks": landmarks[0].tolist() if landmarks is not None else None, "demographics": demographics, "quality": quality}, "models": predictions, "ensemble_score": float(ensemble_score), "threshold": THRESHOLD, "device": str(device)}
    
    return response

# FastAPI app with Swagger UI
@app.function(image=image, timeout=1200)
@modal.asgi_app()
def fastapi_app():
    """Full FastAPI app - imports inside function"""
    from fastapi import FastAPI
    
    web_app = FastAPI(
        title="Deepfake Detection API",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    @web_app.get("/")
    def root():
        return {"message": "Deepfake Detection API is running"}
    
    @web_app.get("/health")
    def health_check():
        return {"status": "healthy"}
    
    return web_app
