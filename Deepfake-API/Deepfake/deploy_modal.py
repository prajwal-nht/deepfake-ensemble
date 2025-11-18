"""
Modal Deployment Wrapper - Avoids local import issues
"""
import modal

# Modal app setup
app = modal.App("deepfake-ensemble-api")

# Define image with required dependencies
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
    # Copy the ensemble_detector.py file into the Modal image
    .copy_local_file(
        "deepfake-detection/ensemble_detector.py",
        "/root/ensemble_detector.py"
    )
)

# Model configurations
MODEL_CONFIGS = {
    "xception": {
        "name": "dima806/deepfake_vs_real_image_detection",
        "type": "huggingface",
        "input_size": 299,
    },
    "vit_deepfake": {
        "name": "prithivMLmods/Deep-Fake-Detector-Model",
        "type": "huggingface",
        "input_size": 224,
    },
    "vit_wvolf": {
        "name": "Wvolf/ViT_Deepfake_Detection",
        "type": "huggingface",
        "input_size": 224,
    },
    "vit_deepfake_v2": {
        "name": "prithivMLmods/deepfake-detector-model-v1",
        "type": "huggingface",
        "input_size": 224,
    },
    "efficientnet_b4": {
        "name": "google/efficientnet-b4",
        "type": "huggingface",
        "input_size": 380,
    }
}

FRAME_DEEPFAKE_THRESHOLD = 0.70
OVERALL_VIDEO_DEEPFAKE_THRESHOLD = 0.65

@app.function(image=image, gpu="T4", timeout=600)
@modal.web_endpoint(method="GET")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "deepfake-ensemble-api"}

@app.function(image=image, gpu="T4", timeout=1200)
@modal.web_endpoint(method="POST")
def detect(file_data: bytes, video_id: str = "", user_id: str = "", video_url: str = "", include_details: str = "false"):
    """Detection endpoint - imports are inside function to avoid local validation"""
    import sys
    sys.path.insert(0, '/root')
    
    # Import ensemble_detector module (now available in Modal image)
    import ensemble_detector
    
    # Initialize detector
    detector = ensemble_detector.EnsembleDeepfakeDetector()
    detector.initialize()
    
    # Process the file
    import numpy as np
    import cv2
    import tempfile
    import os
    from datetime import datetime
    
    # Decode file
    nparr = np.frombuffer(file_data, np.uint8)
    
    # Try as image first
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is not None:
        # Process as image
        results = detector.predict_frame(frame)
        return {
            "status": "success",
            "type": "image",
            "is_deepfake": results.get("is_deepfake", False),
            "confidence": results.get("ensemble_score", 0.0),
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "details": results if include_details.lower() == "true" else None
        }
    else:
        # Process as video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        
        try:
            import asyncio
            video_results = asyncio.run(ensemble_detector.process_video(tmp_path))
            
            # Calculate overall result
            deepfake_count = sum(1 for r in video_results if r.get("result", {}).get("is_deepfake", False))
            total_frames = len(video_results)
            confidence = deepfake_count / total_frames if total_frames > 0 else 0.0
            
            return {
                "video_id": video_id,
                "user_id": user_id,
                "video_url": video_url,
                "is_deepfake": confidence >= OVERALL_VIDEO_DEEPFAKE_THRESHOLD,
                "confidence": confidence,
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "frame_count": total_frames,
                "processed_frames": total_frames,
                "details": video_results if include_details.lower() == "true" else None
            }
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.function(image=image, timeout=1200)
@modal.asgi_app()
def fastapi_app():
    """Full FastAPI app with Swagger UI"""
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
