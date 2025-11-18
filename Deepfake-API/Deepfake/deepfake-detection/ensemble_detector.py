"""
Ensemble Deepfake Detection API with demographic analysis and quality assessment
"""
import os
import sys
import logging
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter, Form
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image as PILImage

# Initialize FastAPI app and router (explicit docs to ensure exposure on Modal)
web_app = FastAPI(
    title="Deepfake Detection API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
router = APIRouter()
from modal import Image as ModalImage, App, web_endpoint
import cv2
import tempfile
import modal
from pathlib import Path
import time
from datetime import datetime
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Optional deps: DeepFace / RetinaFace
try:
    from deepface import DeepFace  # type: ignore
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False
    DeepFace = None  # type: ignore
    logging.warning("DeepFace not available. Demographic analysis will be skipped.")

try:
    from retinaface import RetinaFace  # type: ignore
    RETINAFACE_AVAILABLE = True
except Exception:
    RETINAFACE_AVAILABLE = False
    RetinaFace = None  # type: ignore
    logging.warning("RetinaFace not available. Falling back to MTCNN for face detection.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize dlib availability flag
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logger.warning("dlib not available. Facial landmarks will be disabled.")

# Modal app setup
app = modal.App("deepfake-ensemble-detector")

# Define image with required dependencies
image = (
    ModalImage
    .debian_slim(python_version="3.10")
    .apt_install([
        "git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev", "libgomp1",
        "libfontconfig1-dev", "libfreetype6-dev", "pkg-config",
        "libavcodec-dev", "libavformat-dev", "libswscale-dev",
        "cmake", "build-essential"
    ])
    .pip_install([
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "opencv-python-headless",
        "numpy",
        "fastapi",
        "python-multipart",
        "pillow",
        "transformers",
        "timm",
        "scikit-image",
        "facenet-pytorch",
        "efficientnet-pytorch",
        "deepface",
        "tf-keras",
        "retina-face",
        "git+https://github.com/openai/CLIP.git"
    ])
    .add_local_dir("./training", remote_path="/root/training")
    .add_local_dir("./preprocessing", remote_path="/root/preprocessing")
)

# Model configurations aligned with manager's requirements
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

# Thresholds - Balanced for production use
FRAME_DEEPFAKE_THRESHOLD = 0.78  # Balanced threshold for frame-level detection
OVERALL_VIDEO_DEEPFAKE_THRESHOLD = 0.75  # Balanced threshold for video-level detection

class MTCNNFaceDetector:
    """GPU-optimized face detection using MTCNN with face alignment"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.detector = None
        
    def initialize(self):
        """Initialize MTCNN face detector"""
        try:
            from facenet_pytorch import MTCNN
            logger.info("Initializing MTCNN face detector...")
            
            self.detector = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=self.device,
                keep_all=True,
                select_largest=False
            )
            logger.info("MTCNN initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {e}")
            raise
    
    def detect_faces(self, image):
        """Detect faces in image and return bounding boxes and landmarks"""
        try:
            if isinstance(image, np.ndarray):
                if image.shape[2] == 3:  # BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = PILImage.fromarray(image)
            
            # Detect faces and landmarks
            boxes, probs, landmarks = self.detector.detect(image, landmarks=True)
            
            faces = []
            if boxes is not None:
                for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                    if prob > 0.9:  # High confidence threshold
                        faces.append({
                            'bbox': box.tolist(),
                            'confidence': float(prob),
                            'landmarks': landmark.tolist() if landmark is not None else None
                        })
            
            return faces
            
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
            return []

class EnsembleDeepfakeDetector:
    """Ensemble of deepfake detection models"""
    
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_detector = None
        self.face_predictor = None
        self.initialized = False
        self.demographic_models = {
            'age': 'Age',
            'gender': 'Gender',
            'race': 'Race',
            'emotion': 'Emotion'
        }
        
    def initialize(self):
        """Initialize all models in the ensemble"""
        if self.initialized:
            return
            
        logger.info(f"Initializing models on {self.device}...")
        
        # Initialize face detector
        self.face_detector = MTCNNFaceDetector(device=self.device)
        self.face_detector.initialize()
        
        # Initialize deepfake detection models
        for model_name, config in MODEL_CONFIGS.items():
            try:
                logger.info(f"Loading {model_name}...")
                
                if config["type"] == "huggingface":
                    # Special handling for different model types
                    if "resnet" in model_name:
                        from transformers import ResNetForImageClassification
                        model = ResNetForImageClassification.from_pretrained(
                            config["name"],
                            num_labels=2,
                            ignore_mismatched_sizes=True
                        )
                    elif "efficientnet" in model_name:
                        from transformers import EfficientNetForImageClassification
                        model = EfficientNetForImageClassification.from_pretrained(
                            config["name"],
                            num_labels=2,
                            ignore_mismatched_sizes=True
                        )
                    else:
                        model = AutoModelForImageClassification.from_pretrained(
                            config["name"],
                            num_labels=2,  # Binary classification (real/fake)
                            ignore_mismatched_sizes=True
                        )
                    # Use AutoImageProcessor (replaces deprecated AutoFeatureExtractor)
                    feature_extractor = AutoImageProcessor.from_pretrained(config["name"])
                    
                    self.models[model_name] = {
                        'model': model.to(self.device).eval(),
                        'image_processor': feature_extractor,
                        'config': config
                    }
                
                logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {str(e)}")
                raise
                
        self.initialized = True
    
    # Removed duplicate initialize_models method - using initialize() instead
    
    def detect_faces(self, frame):
        """Detect faces in an image frame using the initialized face detector"""
        if not self.face_detector:
            logger.error("Face detector not initialized")
            return []
        
        # Convert frame to RGB if needed
        if frame.shape[2] == 3:  # BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
            
        results: List[Dict[str, Any]] = []
        # Prefer RetinaFace if available
        if RETINAFACE_AVAILABLE:
            try:
                faces = RetinaFace.detect_faces(rgb_frame)
                if isinstance(faces, dict):
                    for _, face_data in faces.items():
                        facial_area = face_data['facial_area']
                        bbox = [
                            int(facial_area[0]),
                            int(facial_area[1]),
                            int(facial_area[2] - facial_area[0]),
                            int(facial_area[3] - facial_area[1])
                        ]
                        landmarks = face_data.get('landmarks', {})
                        results.append({
                            'bbox': bbox,
                            'landmarks': landmarks,
                            'confidence': face_data.get('score', 0.9)
                        })
                    return results
            except Exception as e:
                logger.warning(f"RetinaFace detection failed, falling back to MTCNN: {e}")

        # Fallback: MTCNN detector
        boxes = None
        try:
            boxes, probs, landmarks = self.face_detector.detector.detect(PILImage.fromarray(rgb_frame), landmarks=True)  # type: ignore
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
            return []

        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob and prob > 0.5:
                    x1, y1, x2, y2 = map(int, box)
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    results.append({
                        'bbox': bbox,
                        'landmarks': landmarks[i].tolist() if landmarks is not None else [],
                        'confidence': float(prob)
                    })
        return results
        
    def _convert_to_python_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def analyze_demographics(self, frame, face_bbox):
        """Analyze demographics and quality for a face region"""
        try:
            x, y, w, h = face_bbox
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                return None
                
            # Convert to RGB for DeepFace
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Analyze demographics
            demography = None
            if DEEPFACE_AVAILABLE:
                try:
                    demography = DeepFace.analyze(
                        face_rgb,
                        actions=list(self.demographic_models.keys()),
                        enforce_detection=False,
                        silent=True
                    )
                    # Convert numpy types to Python types
                    demography = self._convert_to_python_types(demography)
                except Exception as e:
                    logger.warning(f"DeepFace analysis failed: {e}")
            
            # Get quality metrics
            quality = self.assess_quality(face_img)
            
            # Get face landmarks
            landmarks = self.detect_landmarks(face_rgb)
            
            return {
                'demographics': demography[0] if isinstance(demography, list) else demography,
                'quality': quality,
                'landmarks': landmarks
            }
            
        except Exception as e:
            logger.error(f"Error in demographic analysis: {str(e)}")
            return None
            
    def assess_quality(self, face_img):
        """Assess quality of face image"""
        try:
            # Convert to grayscale for quality assessment
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Calculate quality metrics
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Normalize metrics to 0-1 range
            quality_score = min(1.0, laplacian_var / 100.0)  # Normalize by typical good value
            
            return {
                'sharpness': float(laplacian_var),
                'brightness': float(brightness / 255.0),  # Normalize to 0-1
                'contrast': float(contrast / 128.0),      # Normalize to ~0-1
                'overall_score': quality_score
            }
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {str(e)}")
            return {
                'sharpness': 0.0,
                'brightness': 0.0,
                'contrast': 0.0,
                'overall_score': 0.0
            }
            
    def detect_landmarks(self, face_img):
        """Detect facial landmarks using dlib if available"""
        if not DLIB_AVAILABLE:
            return []
            
        try:
            if self.face_predictor is None:
                # Initialize dlib's face detector and landmark predictor
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                
                # Convert to grayscale for dlib
                gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                
                # Detect faces
                faces = detector(gray, 1)
                if len(faces) > 0:
                    landmarks = predictor(gray, faces[0])
                    return [(p.x, p.y) for p in landmarks.parts()]
                    
        except Exception as e:
            logger.error(f"Error in landmark detection: {str(e)}")
            
        return []

    def apply_demographic_weighting(self, raw_prob, demographics, quality):
        """Apply demographic and quality-based weighting to prediction probabilities
        
        CONSERVATIVE APPROACH: Only make small adjustments (max ±10%) to avoid false negatives
        """
        if not demographics or not quality:
            return raw_prob
            
        # Calculate adjustment factor (much more conservative)
        adjustment = 0.0
        
        # Quality-based adjustment (±5% max)
        quality_score = quality.get('overall_score', 0.5)
        if quality_score < 0.3:  # Very low quality
            adjustment -= 0.05  # Reduce confidence slightly
        elif quality_score > 0.9:  # Very high quality
            adjustment += 0.02  # Increase confidence slightly
        
        # Gender confidence adjustment (±3% max)
        if 'gender' in demographics:
            gender_data = demographics.get('gender', {})
            # Get the confidence of the dominant gender
            if isinstance(gender_data, dict):
                max_gender_conf = max(gender_data.values()) if gender_data else 50.0
                if max_gender_conf < 60:  # Low confidence in gender detection
                    adjustment -= 0.03
        
        # Apply conservative adjustment (max ±10%)
        adjustment = max(-0.10, min(0.10, adjustment))
        
        # Apply adjustment to raw probability
        weighted_prob = raw_prob + adjustment
        
        # Ensure result is within valid range [0, 1]
        weighted_prob = max(0.0, min(1.0, weighted_prob))
            
        return weighted_prob
        
    def _generate_summary(self, face_results):
        """Generate a summary of detection results"""
        if not face_results:
            return {
                'total_faces': 0,
                'deepfake_faces': 0,
                'deepfake_ratio': 0.0,
                'avg_confidence': 0.0,
                'avg_quality': 0.0
            }
            
        total_faces = len(face_results)
        deepfake_faces = sum(1 for f in face_results if f.get('is_deepfake', False))
        avg_confidence = sum(f.get('confidence', 0) for f in face_results) / total_faces
        
        # Calculate average quality
        total_quality = 0
        valid_qualities = 0
        for f in face_results:
            if f.get('quality') and 'overall_score' in f['quality']:
                total_quality += f['quality']['overall_score']
                valid_qualities += 1
                
        avg_quality = total_quality / valid_qualities if valid_qualities > 0 else 0.0
        
        return {
            'total_faces': total_faces,
            'deepfake_faces': deepfake_faces,
            'deepfake_ratio': deepfake_faces / total_faces if total_faces > 0 else 0.0,
            'avg_confidence': avg_confidence,
            'avg_quality': avg_quality
        }
        
    def predict_frame(self, frame):
        """Run prediction on a single frame using all models with demographic weighting"""
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            
            if not faces:
                return {
                    "status": "no_face",
                    "message": "No faces detected in frame",
                    "demographics": None,
                    "quality": None,
                    "faces": []
                }
            
            # Process each face
            results = []
            for face in faces:
                bbox = face['bbox']
                x1, y1, w, h = map(int, bbox)
                x2, y2 = x1 + w, y1 + h
                
                # Get face crop
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                
                # Analyze demographics and quality
                analysis = self.analyze_demographics(frame, (x1, y1, w, h))
                
                # Prepare face result
                face_result = {
                    "bbox": [x1, y1, w, h],
                    "landmarks": face.get('landmarks', []),
                    "confidence": face.get('confidence', 0.0),
                    "demographics": analysis['demographics'] if analysis else None,
                    "quality": analysis['quality'] if analysis else None,
                    "models": {}
                }
                
                # Get predictions from each model
                for model_name, model_data in self.models.items():
                    try:
                        # Convert to PIL Image for transformers
                        pil_img = PILImage.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        
                        # Preprocess
                        inputs = model_data['image_processor'](
                            images=pil_img, 
                            return_tensors="pt"
                        ).to(self.device)
                        
                        # Predict
                        with torch.no_grad():
                            outputs = model_data['model'](**inputs)
                            logits = outputs.logits
                            probs = torch.nn.functional.softmax(logits, dim=-1)
                            
                        # Get prediction (assuming binary classification: 0=real, 1=fake)
                        fake_prob = probs[0][1].item()
                        
                        # Log raw prediction before weighting
                        logger.info(f"🔍 Model: {model_name} | Raw Prob: {fake_prob:.4f}")
                        
                        # Apply demographic and quality weighting
                        weighted_prob = self.apply_demographic_weighting(
                            fake_prob, 
                            face_result.get('demographics', {}),
                            face_result.get('quality', {})
                        )
                        
                        # Log after weighting
                        adjustment = weighted_prob - fake_prob
                        logger.info(f"⚖️  Model: {model_name} | Weighted Prob: {weighted_prob:.4f} | Adjustment: {adjustment:+.4f}")
                        
                        face_result['models'][model_name] = {
                            'raw_prob': fake_prob,
                            'weighted_prob': weighted_prob,
                            'is_fake': weighted_prob > FRAME_DEEPFAKE_THRESHOLD,
                            'all_scores': probs[0].cpu().numpy().tolist()
                        }
                        
                    except Exception as e:
                        logger.error(f"Error in {model_name} prediction: {str(e)}")
                        face_result['models'][model_name] = {
                            'error': str(e),
                            'raw_prob': None,
                            'weighted_prob': None,
                            'is_fake': None,
                            'all_scores': None
                        }
                
                # Calculate ensemble score with demographic and quality weighting
                model_probs = [
                    m['weighted_prob'] for m in face_result['models'].values() 
                    if isinstance(m, dict) and 'weighted_prob' in m and m['weighted_prob'] is not None
                ]
                
                if model_probs:
                    face_result['ensemble_score'] = sum(model_probs) / len(model_probs)
                    face_result['is_deepfake'] = face_result['ensemble_score'] > FRAME_DEEPFAKE_THRESHOLD
                    
                    # Log ensemble result
                    logger.info(f"📊 ENSEMBLE RESULT:")
                    logger.info(f"   Ensemble Score: {face_result['ensemble_score']:.4f}")
                    logger.info(f"   Threshold: {FRAME_DEEPFAKE_THRESHOLD}")
                    logger.info(f"   Decision: {'🚨 DEEPFAKE' if face_result['is_deepfake'] else '✅ REAL'}")
                    logger.info(f"   Demographics: Age={face_result.get('demographics', {}).get('age', 'N/A')}, Gender={face_result.get('demographics', {}).get('dominant_gender', 'N/A')}")
                    logger.info(f"   Quality Score: {face_result.get('quality', {}).get('overall_score', 0):.2f}")
                    
                    # Add confidence based on quality and number of models
                    quality_score = face_result.get('quality', {}).get('overall_score', 0.5)
                    confidence = (face_result['ensemble_score'] * 0.7 + 
                                quality_score * 0.3)
                    face_result['confidence'] = min(1.0, max(0.0, confidence))
                else:
                    face_result['ensemble_score'] = None
                    face_result['is_deepfake'] = None
                    face_result['confidence'] = 0.0
                
                results.append(face_result)
                
            # Prepare final result
            if not results:
                return {
                    "status": "error",
                    "message": "No valid faces processed",
                    "faces": []
                }
                
            return {
                "status": "success",
                "message": f"Processed {len(results)} face(s)",
                "faces": results,
                "summary": self._generate_summary(results)
            }
            
        except Exception as e:
            logger.error(f"Error in predict_frame: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing frame: {str(e)}",
                "faces": []
            }

async def process_video(video_path: str, frame_interval: int = 10) -> List[Dict]:
    """Process video file and return analysis results"""
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        detector = EnsembleDeepfakeDetector()
        detector.initialize()
        
        frame_count = 0
        results = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process every nth frame
                if frame_count % frame_interval == 0:
                    try:
                        result = detector.predict_frame(frame)
                        results.append({
                            'frame_number': frame_count,
                            'result': result
                        })
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_count}: {str(e)}")
                
                frame_count += 1
        finally:
            cap.release()
            
        return results
        
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing video: {str(e)}"
        )

@router.get("/")
async def root():
    return {"message": "Ensemble Deepfake Detection API is running"}

@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    video_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    video_url: Optional[str] = Form(None),
    include_details: Optional[bool] = Form(False)
):
    """Detect deepfakes in an uploaded image or video"""
    try:
        # Check file type
        content_type = file.content_type
        is_video = content_type and 'video' in content_type
        
        if not is_video:
            # Handle image
            start_time = time.time()
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise HTTPException(status_code=400, detail="Invalid image")
                
            # Run detection
            detector = EnsembleDeepfakeDetector()
            detector.initialize()
            result = detector.predict_frame(frame)

            # Aggregate to manager schema
            faces = result.get("faces", [])
            max_conf = max((f.get("confidence", 0.0) for f in faces), default=0.0)
            any_deepfake = any(bool(f.get("is_deepfake", False)) for f in faces)
            response: Dict[str, Any] = {
                "video_id": video_id or "",
                "user_id": user_id or "",
                "video_url": video_url or "",
                "is_deepfake": bool(any_deepfake),
                "confidence": float(max_conf),
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "face_matches": [],  # Optional, not available in current pipeline
                "frame_count": 1,
                "processed_frames": 1
            }
            if include_details:
                response["details"] = {
                    "faces": faces,
                    "summary": result.get("summary", {}),
                    "thresholds": {
                        "frame_deepfake": FRAME_DEEPFAKE_THRESHOLD,
                        "video_overall": OVERALL_VIDEO_DEEPFAKE_THRESHOLD
                    },
                    "processing": {
                        "elapsed_ms": int((time.time() - start_time) * 1000),
                        "frame_indices": [0]
                    },
                    "environment": {
                        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
                        "cuda_available": bool(torch.cuda.is_available()),
                        "models_loaded": list(detector.models.keys())
                    },
                    "errors": []
                }
            return response
            
        else:
            # Handle video
            start_time = time.time()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                contents = await file.read()
                temp_video.write(contents)
                temp_video_path = temp_video.name
            
            try:
                # Process video
                results = await process_video(temp_video_path)

                # results is a list of {frame_number, result}
                processed_frames = len(results)
                frame_count = results[-1]["frame_number"] + 1 if processed_frames > 0 else 0

                # Collect confidences per frame (max per frame)
                per_frame_max = []
                any_deepfake = False
                for item in results:
                    faces = (item.get("result") or {}).get("faces", [])
                    if not faces:
                        continue
                    max_conf = max((f.get("confidence", 0.0) for f in faces), default=0.0)
                    per_frame_max.append(max_conf)
                    if any(bool(f.get("is_deepfake", False)) for f in faces):
                        any_deepfake = True

                overall_conf = float(np.mean(per_frame_max)) if per_frame_max else 0.0
                is_deepfake = bool(any_deepfake or (overall_conf >= OVERALL_VIDEO_DEEPFAKE_THRESHOLD))

                response: Dict[str, Any] = {
                    "video_id": video_id or "",
                    "user_id": user_id or "",
                    "video_url": video_url or "",
                    "is_deepfake": is_deepfake,
                    "confidence": overall_conf,
                    "processed_at": datetime.utcnow().isoformat() + "Z",
                    "face_matches": [],  # Optional, not available in current pipeline
                    "frame_count": int(frame_count),
                    "processed_frames": int(processed_frames)
                }
                if include_details:
                    # Summarize last frame's faces if available
                    last_faces = (results[-1].get("result") or {}).get("faces", []) if results else []
                    last_summary = (results[-1].get("result") or {}).get("summary", {}) if results else {}
                    response["details"] = {
                        "faces": last_faces,
                        "summary": last_summary,
                        "thresholds": {
                            "frame_deepfake": FRAME_DEEPFAKE_THRESHOLD,
                            "video_overall": OVERALL_VIDEO_DEEPFAKE_THRESHOLD
                        },
                        "processing": {
                            "elapsed_ms": int((time.time() - start_time) * 1000),
                            "frame_indices": [r["frame_number"] for r in results] if results and isinstance(results[0], dict) else []
                        },
                        "environment": {
                            "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
                            "cuda_available": bool(torch.cuda.is_available())
                        },
                        "errors": []
                    }
                return response
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_video_path)
                except Exception as e:
                    logger.error(f"Error deleting temp file: {e}")
    
    except Exception as e:
        logger.error(f"Error in detect endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ensemble-deepfake-detection",
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    }

# Include the router before exposing (used for local uvicorn)
web_app.include_router(router)

# Modal-native simple endpoints (compatible with older Modal SDKs)
@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.fastapi_endpoint(method="GET")
def root():
    return {"message": "Ensemble Deepfake Detection API is running"}

# Expose full FastAPI app with Swagger UI on Modal (ASGI app)
@app.function(
    image=image,
    timeout=1200,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.asgi_app()
def fastapi_app():
    return web_app

@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.fastapi_endpoint(method="GET")
def health():
    """Modal-exposed health endpoint"""
    return {
        "status": "healthy",
        "service": "ensemble-deepfake-detection",
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    }

@app.function(
    image=image,
    gpu="T4",
    timeout=1200,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.fastapi_endpoint(method="POST")
def detect(body: bytes = b"", content_type: str = "application/octet-stream"):
    """Modal-exposed POST detect endpoint.

    Supports:
    - application/octet-stream: raw file bytes (image or video)
    - application/json: {"image_base64": str} or {"video_base64": str}

    For images, runs single-frame detection. For videos, writes to a temp file
    and processes at intervals.
    """
    import base64
    import json

    try:
        if not body:
            return {"status": "error", "message": "Empty request body"}

        # Determine content type if not provided
        if not content_type:
            content_type = "application/octet-stream"
        
        # Handle JSON with base64
        if content_type.startswith("application/json"):
            data = json.loads(body.decode("utf-8")) if body else {}
            image_b64 = data.get("image_base64")
            video_b64 = data.get("video_base64")
            video_id = data.get("video_id") or ""
            user_id = data.get("user_id") or ""
            video_url = data.get("video_url") or ""
            include_details = bool(data.get("include_details", False))

            if image_b64:
                # Run detection on image
                img_bytes = base64.b64decode(image_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                start_time = time.time()
                detector = EnsembleDeepfakeDetector()
                detector.initialize()
                result = detector.predict_frame(frame)
                faces = result.get("faces", [])
                max_conf = max((f.get("confidence", 0.0) for f in faces), default=0.0)
                any_deepfake = any(bool(f.get("is_deepfake", False)) for f in faces)
                resp: Dict[str, Any] = {
                    "video_id": video_id,
                    "user_id": user_id,
                    "video_url": video_url,
                    "is_deepfake": bool(any_deepfake),
                    "confidence": float(max_conf),
                    "processed_at": datetime.utcnow().isoformat() + "Z",
                    "face_matches": [],
                    "frame_count": 1,
                    "processed_frames": 1
                }
                if include_details:
                    resp["details"] = {
                        "faces": faces,
                        "summary": result.get("summary", {}),
                        "thresholds": {
                            "frame_deepfake": FRAME_DEEPFAKE_THRESHOLD,
                            "video_overall": OVERALL_VIDEO_DEEPFAKE_THRESHOLD
                        },
                        "processing": {
                            "elapsed_ms": int((time.time() - start_time) * 1000),
                            "frame_indices": [0]
                        }
                    }
                return resp

            if video_b64:
                # Write to temp and process video
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
                    tmp_vid.write(base64.b64decode(video_b64))
                    tmp_video_path = tmp_vid.name
                try:
                    import asyncio
                    video_results = asyncio.run(process_video(tmp_video_path))
                finally:
                    try:
                        os.unlink(tmp_video_path)
                    except Exception:
                        pass
                processed_frames = len(video_results)
                frame_count = video_results[-1]["frame_number"] + 1 if processed_frames > 0 else 0
                per_frame_max = []
                any_deepfake = False
                for item in video_results:
                    faces = (item.get("result") or {}).get("faces", [])
                    if not faces:
                        continue
                    max_conf = max((f.get("confidence", 0.0) for f in faces), default=0.0)
                    per_frame_max.append(max_conf)
                    if any(bool(f.get("is_deepfake", False)) for f in faces):
                        any_deepfake = True
                overall_conf = float(np.mean(per_frame_max)) if per_frame_max else 0.0
                is_deepfake = bool(any_deepfake or (overall_conf >= OVERALL_VIDEO_DEEPFAKE_THRESHOLD))
                resp: Dict[str, Any] = {
                    "video_id": video_id,
                    "user_id": user_id,
                    "video_url": video_url,
                    "is_deepfake": is_deepfake,
                    "confidence": overall_conf,
                    "processed_at": datetime.utcnow().isoformat() + "Z",
                    "face_matches": [],
                    "frame_count": int(frame_count),
                    "processed_frames": int(processed_frames)
                }
                if include_details:
                    # attach details from last frame if possible
                    last_faces = (video_results[-1].get("result") or {}).get("faces", []) if video_results else []
                    last_summary = (video_results[-1].get("result") or {}).get("summary", {}) if video_results else []
                    resp["details"] = {
                        "faces": last_faces,
                        "summary": last_summary,
                        "thresholds": {
                            "frame_deepfake": FRAME_DEEPFAKE_THRESHOLD,
                            "video_overall": OVERALL_VIDEO_DEEPFAKE_THRESHOLD
                        }
                    }
                return resp

            return {"status": "error", "message": "JSON must include image_base64 or video_base64"}

        # Otherwise, assume raw bytes (image preferred; fallback to video)
        nparr = np.frombuffer(body, np.uint8)
        frame = None
        try:
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            frame = None

        if frame is not None:
            detector = EnsembleDeepfakeDetector()
            detector.initialize()
            results = detector.predict_frame(frame)
            return {"status": "success", "type": "image", "results": results}

        # If not decodable as image, treat as video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(body)
            tmp_video_path = tmp_vid.name
        try:
            import asyncio
            video_results = asyncio.run(process_video(tmp_video_path))
        finally:
            try:
                os.unlink(tmp_video_path)
            except Exception:
                pass
        return {"status": "success", "type": "video", "results": video_results}

    except Exception as e:
        logger.error(f"Error in Modal detect endpoint: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8001)
