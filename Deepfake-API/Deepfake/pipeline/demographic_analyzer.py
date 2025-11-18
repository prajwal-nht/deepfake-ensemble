"""
Enhanced Demographic Analysis Module
Analyzes face images for demographic information with improved models and quality assessment.
"""
from dataclasses import dataclass, field
import os
import cv2
import numpy as np
import logging
import logging.handlers
import sys
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image
import torch
from retinaface import RetinaFace
from deepface import DeepFace
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default face detection model
DEFAULT_FACE_DETECTION_MODEL = 'retinaface'

# Configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = 'logs'
log_file = os.path.join(log_dir, 'demographic_analyzer.log')
try:
    os.makedirs(log_dir, exist_ok=True)
    
    # Create file handler which logs debug messages
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # Add the handlers to the logger
    if not logger.handlers:  # Avoid adding handlers multiple times
        logger.addHandler(ch)
        logger.addHandler(fh)
        
except Exception as e:
    logger.warning(f"Could not set up file logging: {str(e)}. Using console logging only.")
    # Fall back to just console logging if file logging fails
    if not logger.handlers:
        logger.addHandler(ch)

logger.info("Demographic analyzer logger configured")

@dataclass
class DemographicResult:
    """Container for demographic analysis results for a single face."""
    face_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    age: int
    gender: str
    gender_confidence: float
    race: Dict[str, float]
    emotion: Dict[str, float]
    accessories: Dict[str, float]
    quality_metrics: Dict[str, float]
    landmarks: Optional[List[List[float]]] = None
    quality_score: float = 0.0

@dataclass
class DemographicData:
    """Container for demographic analysis results."""
    age: int
    gender: str
    gender_confidence: float
    race: Dict[str, float]
    emotion: Dict[str, float]
    accessories: Dict[str, float]
    quality_metrics: Optional[Dict[str, float]] = None
    landmarks: Optional[np.ndarray] = None  # Facial landmarks if detected
    quality_score: float = 0.0

class DemographicAnalyzer:
    """Enhanced analyzer for face demographics with quality assessment and adaptive models."""
    
    def __init__(self, face_detection_model: str = None, device: str = None):
        """
        Initialize the demographic analyzer with enhanced models.
        
        Args:
            face_detection_model: The face detection model to use.
            device: The device to run the models on ('cuda' or 'cpu').
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize face detection
        self.face_detection_model = face_detection_model or DEFAULT_FACE_DETECTION_MODEL
        try:
            # Initialize RetinaFace with proper configuration
            self.face_detector = RetinaFace.build_model()
            self.face_detector_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Initialized face detector with {self.face_detection_model} on {self.face_detector_device}")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {str(e)}", exc_info=True)
            # Fall back to OpenCV's Haar Cascade if RetinaFace fails
            self.face_detection_model = 'haarcascade'
            self.face_detector = None
            logger.warning("Falling back to Haar Cascade face detector")
        
        # Initialize demographic models
        self.demographic_models = {
            'age': 'DenseNet121',  # Better age estimation
            'gender': 'Gender',
            'race': 'Race',
            'emotion': 'Emotion'
        }
        
        # Initialize emotion model
        self.emotion_model = self._load_emotion_model()
        
        # Preload DeepFace models
        self._preload_deepface_models()
        
        # Load models
        self.demography_model = self._load_demography_model()
        self.face_quality_model = self._load_face_quality_model()
        self.face_landmark_model = self._load_landmark_model()
        self.accessory_model = self._load_accessory_model()
        
        logger.info("All models loaded successfully")
    
    def _load_demography_model(self):
        """Load an enhanced demographic analysis model."""
        try:
            from deepface import DeepFace
            # Use a model that supports age, gender, and race analysis
            # Available models: 'VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace'
            return {
                'model': DeepFace.build_model('Facenet'),
                'analyze': self._analyze_face_with_deepface
            }
        except ImportError:
            logger.error("DeepFace not installed. Install with: pip install deepface")
            return None
            
    def _analyze_face_with_deepface(self, face_image, **kwargs):
        """Analyze face using DeepFace with proper error handling."""
        try:
            from deepface import DeepFace
            
            # Convert to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:  # If RGB
                face_rgb = face_image
            else:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
            # Analyze returns a list of results (one per face)
            result = DeepFace.analyze(
                face_rgb,
                actions=['age', 'gender', 'race'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list) and len(result) > 0:
                return result[0]  # Take first face
                
        except Exception as e:
            logger.error(f"Error in DeepFace analysis: {str(e)}")
            
        # Return default values on error
        return {
            'age': 30,
            'gender': 'unknown',
            'gender_confidence': 0.5,
            'race': {'unknown': 1.0}
        }
            
    def _load_face_quality_model(self):
        """Load a model to assess face image quality."""
        # Using a simple model for quality assessment
        return None  # Will use traditional CV methods for now
        
    def _load_emotion_model(self):
        """Load the emotion recognition model."""
        try:
            from transformers import pipeline
            model = pipeline(
                "image-classification",
                model="trpakov/vit-face-expression",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Loaded VIT-based emotion model")
            return model
        except Exception as e:
            logger.error(f"Error loading emotion model: {str(e)}")
            return None
            
    def _preload_deepface_models(self):
        """Preload DeepFace models to reduce latency during analysis."""
        try:
            from deepface import DeepFace
            
            # Define models to preload
            models = {
                'age': 'DenseNet121',
                'gender': 'Gender',
                'race': 'Race',
                'emotion': 'Emotion'
            }
            
            # Preload each model
            for model_name, model_type in models.items():
                try:
                    DeepFace.build_model(model_name=model_type)
                    logger.info(f"Preloaded {model_type} model for {model_name} detection")
                except Exception as e:
                    logger.error(f"Failed to preload {model_type} model: {str(e)}")
                    
        except ImportError as e:
            logger.error("DeepFace not available. Some demographic features may be limited.")
            
    def _load_landmark_model(self):
        """Load a facial landmark detector."""
        try:
            from facenet_pytorch import MTCNN
            return MTCNN(
                keep_all=True,
                device=self.device,
                post_process=False
            )
        except ImportError:
            logger.warning("facenet-pytorch not installed. Landmark detection disabled.")
            return None
            
        # Load better emotion model
        try:
            from transformers import pipeline
            self.emotion_model = pipeline(
                "image-classification", 
                model="trpakov/vit-face-expression"  # Better emotion recognition
            )
            logger.info("Loaded VIT-based emotion model")
        except Exception as e:
            logger.error(f"Error loading emotion model: {str(e)}")
            self.emotion_model = None
            
        # Load better gender and race model
        try:
            from deepface import DeepFace
            # Preload models to avoid cold start
            DeepFace.verify(
                img1_path=np.zeros((224, 224, 3), dtype=np.uint8),
                model_name='ArcFace',
                detector_backend='retinaface',
                enforce_detection=False
            )
            logger.info("Preloaded DeepFace models")
        except Exception as e:
            logger.error(f"Error preloading DeepFace models: {str(e)}")
    
    def _load_accessory_model(self):
        """Load accessory detection model."""
        try:
            from ultralytics import YOLO
            return YOLO('yolov8n.pt')  # Nano version for speed
        except ImportError:
            logger.error("YOLOv8 not installed. Install with: pip install ultralytics")
            return None
    
    def analyze(self, image: Union[str, np.ndarray, Image.Image], 
               face_locations: Optional[List[Tuple[int, int, int, int]]] = None) -> List[DemographicResult]:
        """
        Analyze an image for demographic information with enhanced features.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            face_locations: Optional list of (top, right, bottom, left) tuples for face locations
            
        Returns:
            List of DemographicResult objects, one per face, with enhanced attributes
        """
        # Convert image to numpy array if it's a file path or PIL Image
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                logger.error(f"Could not read image from path: {image}")
                return []
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 2:  # Grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if not isinstance(image, np.ndarray):
            logger.error("Invalid image format. Must be file path, numpy array, or PIL Image.")
            return []
            
        if not face_locations:
            try:
                logger.info(f"Starting face detection. Image type: {type(image)}")
                
                # Load the image if it's a file path
                if isinstance(image, str):  # It's a file path
                    logger.info(f"Loading image from path: {image}")
                    try:
                        img = cv2.imread(image)
                        if img is None:
                            raise ValueError(f"Failed to load image from path: {image}")
                        # Convert from BGR to RGB (OpenCV loads as BGR by default)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        logger.error(f"Error loading image from path {image}: {str(e)}")
                        return {"error": f"Failed to load image: {str(e)}"}
                else:  # It's an in-memory image (numpy array)
                    logger.info("Using in-memory image for face detection")
                    img_rgb = image
                    if len(img_rgb.shape) == 2:  # Grayscale
                        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
                    elif img_rgb.shape[2] == 4:  # RGBA
                        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
                    elif img_rgb.shape[2] == 3:  # RGB or BGR
                        # Check if it's BGR (OpenCV default)
                        if np.argmax(img_rgb[0, 0]) == 0:  # If blue channel has the highest value
                            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                
                # Save the original image for debugging
                debug_img = img_rgb.copy()
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                
                # Equalize histogram to improve contrast
                gray = cv2.equalizeHist(gray)
                
                # Load the pre-trained face detector (Haar Cascade)
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                logger.info(f"Loading Haar Cascade from: {cascade_path}")
                
                if not os.path.exists(cascade_path):
                    error_msg = f"Haar Cascade file not found at: {cascade_path}"
                    logger.error(error_msg)
                    logger.error(f"Current working directory: {os.getcwd()}")
                    logger.error(f"cv2.data.haarcascades: {cv2.data.haarcascades}")
                    logger.error(f"Files in haarcascades directory: {os.listdir(os.path.dirname(cascade_path)) if os.path.exists(os.path.dirname(cascade_path)) else 'Directory not found'}")
                    return []  # Return empty list to maintain consistent return type
                
                face_cascade = cv2.CascadeClassifier(cascade_path)
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                
                # Apply Gaussian blur to reduce noise
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Use optimal parameters from testing
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                logger.info(f"Detected {len(faces)} faces with optimized parameters")
                
                # Save debug image with detection results
                debug_img = img_rgb.copy()
                debug_img_path = os.path.join('debug', 'face_detection_debug.jpg')
                os.makedirs(os.path.dirname(debug_img_path), exist_ok=True)
                
                # Draw all detected faces on the debug image
                for (x, y, w, h) in faces:
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Save the debug image
                cv2.imwrite(debug_img_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                
                # Log detailed information about the image and detection
                logger.info(f"Image shape: {img_rgb.shape}, Type: {img_rgb.dtype}")
                logger.info(f"Number of faces detected: {len(faces)}")
                logger.info(f"Faces coordinates: {faces}")
                logger.info(f"Saved debug image to: {os.path.abspath(debug_img_path)}")
                
                # Save the grayscale image used for detection for comparison
                gray_img_path = os.path.join('debug', 'grayscale_debug.jpg')
                cv2.imwrite(gray_img_path, gray)
                logger.info(f"Saved grayscale debug image to: {os.path.abspath(gray_img_path)}")
                
                if len(faces) == 0:
                    logger.warning("No faces detected in the image after trying different parameters")
                    logger.info(f"Debug image saved to: {os.path.abspath(debug_img_path)}")
                    logger.info(f"Image dimensions: {img_rgb.shape}")
                    logger.info(f"Image type: {img_rgb.dtype}")
                    return []  # Return empty list when no faces are detected
                
                # Convert OpenCV rectangles to DeepFace format
                face_objs = []
                for (x, y, w, h) in faces:
                    # Create a face object similar to what DeepFace would return
                    face_obj = {
                        'face': img_rgb[y:y+h, x:x+w],
                        'facial_area': {
                            'x': int(x),
                            'y': int(y),
                            'w': int(w),
                            'h': int(h),
                            'left_eye': (int(x + w * 0.3), int(y + h * 0.35)),
                            'right_eye': (int(x + w * 0.7), int(y + h * 0.35))
                        },
                        'confidence': 1.0  # Placeholder confidence
                    }
                    face_objs.append(face_obj)
                
                # Convert to (top, right, bottom, left) format
                face_locations = []
                for i, face in enumerate(face_objs):
                    try:
                        logger.debug(f"Processing face {i+1}: {face}")
                        if 'facial_area' in face:
                            x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], \
                                        face['facial_area']['w'], face['facial_area']['h']
                            logger.info(f"Face {i+1} detected at (x:{x}, y:{y}, w:{w}, h:{h})")
                            # Add some padding to ensure we capture the full face
                            padding = int(min(w, h) * 0.1)
                            y1 = max(0, y - padding)
                            y2 = min(image.shape[0], y + h + padding)
                            x1 = max(0, x - padding)
                            x2 = min(image.shape[1], x + w + padding)
                            face_locations.append((y1, x2, y2, x1))
                            logger.info(f"Added face {i+1} to face_locations: (top:{y1}, right:{x2}, bottom:{y2}, left:{x1})")
                        else:
                            logger.warning(f"Face {i+1} has no 'facial_area' key. Face object: {face}")
                    except Exception as e:
                        logger.error(f"Error processing face {i+1}: {str(e)}", exc_info=True)
            except Exception as e:
                logger.error(f"Error detecting faces: {str(e)}", exc_info=True)
                return []
        
        results = []
        
        for i, (top, right, bottom, left) in enumerate(face_locations):
            try:
                # Extract face region with safety checks
                if (bottom <= top) or (right <= left):
                    logger.warning(f"Invalid face coordinates: top={top}, right={right}, bottom={bottom}, left={left}")
                    continue
                    
                face_image = image[top:bottom, left:right]
                if face_image.size == 0:
                    logger.warning(f"Empty face image at location {i}")
                    continue
                
                # Get all demographic and quality information
                demography = self._get_demography(face_image)
                emotion = self._get_emotion(face_image)  # Now accepts numpy array
                accessories = self._detect_accessories(face_image)
                quality_metrics = self._assess_face_quality(face_image)
                landmarks = self._detect_landmarks(face_image)
                
                # Calculate face quality score (weighted average of metrics)
                quality_weights = {
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'sharpness': 0.3,
                    'blur': -0.3,  # Negative because lower blur is better
                    'illumination': 0.2,
                    'noise': -0.2  # Negative because lower noise is better
                }
                
                quality_score = 0.0
                for metric, weight in quality_weights.items():
                    if metric in quality_metrics:
                        quality_score += quality_metrics[metric] * weight
                
                # Normalize to [0, 1] range
                quality_score = max(0.0, min(1.0, quality_score))
                quality_metrics['overall'] = quality_score
                
                # Create result object with enhanced attributes
                result = DemographicResult(
                    face_id=i,
                    bbox=(left, top, right - left, bottom - top),
                    age=demography.get('age', 30),
                    gender=demography.get('gender', 'unknown'),
                    gender_confidence=float(demography.get('gender_confidence', 0.5)),
                    race=demography.get('race', {'unknown': 1.0}),
                    emotion=emotion,
                    accessories=accessories,
                    quality_metrics=quality_metrics,
                    landmarks=landmarks.tolist() if landmarks is not None else None,
                    quality_score=quality_score
                )
                
                results.append(result)
                
                logger.debug(f"Processed face {i} - Age: {result.age}, Gender: {result.gender} "
                           f"(Confidence: {result.gender_confidence:.2f}), "
                           f"Quality: {quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing face {i}: {str(e)}", exc_info=True)
                continue
                
        return results
    
    def _get_demography(self, face_image: np.ndarray) -> Dict[str, Any]:
        """Get enhanced demographic information using DeepFace with better error handling."""
        def _analyze_face(self, face_img, face_location=None):
            """Analyze a single face for demographic attributes with enhanced accuracy."""
            try:
                # Convert to RGB if needed and resize for better model performance
                if len(face_img.shape) == 3 and face_img.shape[2] == 3:  # BGR to RGB
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Resize for better model performance (maintain aspect ratio)
                target_size = (224, 224)
                height, width = face_img.shape[:2]
                scale = min(target_size[0] / width, target_size[1] / height)
                new_size = (int(width * scale), int(height * scale))
                resized_img = cv2.resize(face_img, new_size)
                
                # Pad to target size
                delta_w = target_size[0] - new_size[0]
                delta_h = target_size[1] - new_size[1]
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                padded_img = cv2.copyMakeBorder(
                    resized_img, top, bottom, left, right, 
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
                
                # Convert to PIL Image for better compatibility
                pil_img = Image.fromarray(padded_img)
                
                # Enhanced emotion detection using VIT model
                emotion_result = {"emotion": {"neutral": 1.0}}
                if self.emotion_model:
                    try:
                        emotion_pred = self.emotion_model(pil_img)
                        emotion_result = {"emotion": {}}
                        for pred in emotion_pred:
                            emotion_result["emotion"][pred["label"]] = float(pred["score"])
                    except Exception as e:
                        logger.warning(f"Error in emotion detection: {str(e)}")
                
                # Get other attributes using DeepFace
                try:
                    analysis = DeepFace.analyze(
                        img_path=padded_img,
                        actions=['age', 'gender', 'race'],
                        enforce_detection=False,
                        detector_backend='retinaface',
                        prog_bar=False,
                        silent=True
                    )
                    
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    
                    # Merge emotion results
                    analysis.update(emotion_result)
                    
                    # Enhance gender detection confidence
                    if 'gender' in analysis:
                        # Convert gender to lowercase for consistency
                        gender = analysis['gender'].lower()
                        analysis['gender'] = gender
                        
                        # If confidence is too low, mark as unknown
                        if 'gender_confidence' in analysis and analysis['gender_confidence'] < 0.7:
                            analysis['gender'] = 'unknown'
                    
                    # Enhance race detection
                    if 'race' in analysis and isinstance(analysis['race'], dict):
                        # Filter out low confidence races
                        analysis['race'] = {k: v for k, v in analysis['race'].items() if v > 0.1}
                        
                        # If no high confidence races, mark as unknown
                        if not analysis['race']:
                            analysis['race'] = {'unknown': 1.0}
                    
                    # Add face location if provided
                    if face_location:
                        analysis['face_location'] = face_location
                    
                    return analysis
                    
                except Exception as e:
                    logger.error(f"Error in DeepFace analysis: {str(e)}")
                    return None
                
            except Exception as e:
                logger.error(f"Error in face analysis: {str(e)}", exc_info=True)
                return None
            
        if self.demography_model is None or not callable(self.demography_model.get('analyze')):
            return {
                'age': 30,
                'gender': 'unknown',
                'gender_confidence': 0.5,
                'race': {'unknown': 1.0}
            }
            
        try:
            # Convert to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:  # If RGB
                face_rgb = face_image
            else:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
            # Call the analyze method from the demography_model dictionary
            result = self.demography_model['analyze'](face_rgb)
            
            if isinstance(result, dict):
                # Process and validate results
                gender = str(result.get('gender', 'unknown')).lower()
                gender_confidence = float(result.get('gender_confidence', 0.5))
                
                # Ensure race is in the correct format
                race_results = result.get('race', {'unknown': 1.0})
                if not isinstance(race_results, dict):
                    race_results = {'unknown': 1.0}
                    
                return {
                    'age': int(result.get('age', 30)),
                    'gender': gender,
                    'gender_confidence': gender_confidence,
                    'race': {str(k).lower(): float(v) for k, v in race_results.items()}
                }
            
        except Exception as e:
            logger.error(f"Error in demographic analysis: {str(e)}", exc_info=True)
            return {
                'age': 30,
                'gender': 'unknown',
                'gender_confidence': 0.5,
                'race': {'unknown': 1.0}
            }
    
    def _get_emotion(self, face_image: np.ndarray) -> Dict[str, float]:
        """Get enhanced emotion predictions with confidence."""
        if self.emotion_model is None:
            return {'neutral': 1.0}
            
        try:
            # Convert to PIL Image if needed
            if not isinstance(face_image, Image.Image):
                pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = face_image
                
            # Get predictions
            results = self.emotion_model(pil_image)
            
            # Convert to standard format
            emotions = {}
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and 'label' in item and 'score' in item:
                        emotions[item['label'].lower()] = float(item['score'])
            
            # Ensure we have at least neutral
            if not emotions:
                emotions = {'neutral': 1.0}
                
            # Normalize to sum to 1
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total for k, v in emotions.items()}
            
            # Create a default DemographicResult with all required fields
            return [DemographicResult(
                face_id=0,
                bbox=(0, 0, face_image.shape[1], face_image.shape[0]),
                age=30,
                gender='unknown',
                gender_confidence=0.5,
                race={'unknown': 1.0},
                emotion=emotions,
                accessories={},
                quality_metrics={},
                quality_score=0.5
            )]
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}", exc_info=True)
            # Ensure we return an empty list on error
            return []
    
    def _detect_landmarks(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Detect facial landmarks if model is available."""
        if self.face_landmark_model is None:
            return None
            
        try:
            # Convert to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:  # If RGB
                face_rgb = face_image
            else:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
            # Detect landmarks
            boxes, probs, landmarks = self.face_landmark_model.detect(face_rgb, landmarks=True)
            
            if landmarks is not None and len(landmarks) > 0:
                # Return landmarks for the first face
                return landmarks[0].astype(np.float32)
                
        except Exception as e:
            logger.error(f"Error in landmark detection: {str(e)}", exc_info=True)
            
        return None
    
    def _assess_face_quality(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Assess the quality of a face image using various metrics.
        
        Args:
            face_image: Input face image in RGB format
            
        Returns:
            Dictionary containing quality metrics (brightness, contrast, sharpness, blur, noise)
        """
        metrics = {
            'brightness': 0.0,
            'contrast': 0.0,
            'sharpness': 0.0,
            'blur': 0.0,
            'noise': 0.0
        }
        
        if face_image.size == 0:
            return metrics
            
        try:
            # Convert to grayscale for some metrics
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image
                
            # 1. Brightness (0-255, ideal around 127)
            brightness = np.mean(gray)
            metrics['brightness'] = float(1.0 - abs(brightness - 127) / 127.0)
            
            # 2. Contrast (standard deviation of pixel intensities)
            contrast = np.std(gray)
            metrics['contrast'] = float(min(1.0, contrast / 50.0))
            
            # 3. Sharpness (variance of Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            metrics['sharpness'] = float(min(1.0, sharpness / 200.0))
            
            # 4. Blur (inverse of sharpness)
            metrics['blur'] = 1.0 - metrics['sharpness']
            
            # 5. Noise (variance of the Laplacian)
            noise = np.var(laplacian)
            metrics['noise'] = float(min(1.0, noise / 1000.0))
            
            # Ensure all values are in [0, 1]
            for k in metrics:
                metrics[k] = max(0.0, min(1.0, metrics[k]))
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {str(e)}", exc_info=True)
            return metrics
    
    def _detect_accessories(self, image: np.ndarray) -> Dict[str, float]:
        """Detect accessories like glasses, hats, etc. with improved accuracy."""
        if self.accessory_model is None:
            return {}
            
        try:
            # Ensure image is in BGR format for YOLO
            if len(image.shape) == 3 and image.shape[2] == 3:  # If RGB
                image_bgr = image[..., ::-1]  # Convert RGB to BGR
            else:
                image_bgr = image
                
            # Run inference with confidence threshold
            results = self.accessory_model(
                image_bgr, 
                verbose=False,
                conf=0.3  # Confidence threshold
            )
            
            # Map class IDs to accessory names (focused on common face accessories)
            accessory_map = {
                15: 'hat',  # Hat (using 'hat' for any headwear)
                16: 'glasses',  # Glasses (using 'glasses' for any eyewear)
                28: 'handbag',
                29: 'tie',
                30: 'suitcase',
                31: 'frisbee',
                32: 'skis',
                33: 'snowboard',
                34: 'sports_ball',
                35: 'kite',
                36: 'baseball_bat',
                37: 'baseball_glove',
                38: 'skateboard',
                39: 'surfboard',
                40: 'tennis_racket',
                41: 'bottle',
                42: 'wine_glass',
                43: 'cup',
                44: 'fork',
                45: 'knife',
                46: 'spoon',
                47: 'bowl',
                48: 'banana',
                49: 'apple',
                50: 'sandwich',
                51: 'orange',
                52: 'broccoli',
                53: 'carrot',
                54: 'hot_dog',
                55: 'pizza',
                56: 'donut',
                57: 'cake',
                58: 'chair',
                59: 'couch',
                60: 'potted_plant',
                61: 'bed',
                62: 'dining_table',
                63: 'toilet',
                64: 'tv',
                65: 'laptop',
                66: 'mouse',
                67: 'remote',
                68: 'keyboard',
                69: 'cell_phone',
                70: 'microwave',
                71: 'oven',
                72: 'toaster',
                73: 'sink',
                74: 'refrigerator',
                75: 'book',
                76: 'clock',
                77: 'vase',
                78: 'scissors',
                79: 'teddy_bear',
                80: 'hair_dryer',
                81: 'toothbrush'
            }
            
            # Process results
            accessories = {}
            if hasattr(results, 'xyxy') and len(results.xyxy) > 0:  # YOLOv5 format
                for result in results.xyxy[0]:  # Get detections for the first image
                    if len(result) >= 6:  # [x1, y1, x2, y2, conf, class_id, ...]
                        class_id = int(result[5].item())
                        confidence = float(result[4].item())
                        
                        if class_id in accessory_map:
                            accessory_name = accessory_map[class_id]
                            # Only keep the highest confidence detection for each accessory
                            if accessory_name not in accessories or confidence > accessories[accessory_name]:
                                accessories[accessory_name] = confidence
            
            return accessories
            
        except Exception as e:
            logger.error(f"Error in accessory detection: {str(e)}", exc_info=True)
            return {}
