"""
Face detection module for the deepfake detection pipeline.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Face alignment settings
ENABLE_ALIGNMENT = True

@dataclass
class FaceDetection:
    """Data class for storing face detection results."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    landmarks: np.ndarray  # 5 points [x,y] for eyes, nose, mouth corners
    confidence: float = 1.0
    embedding: Optional[np.ndarray] = None
    quality: float = 1.0

class FaceDetector:
    """Face detection using OpenCV's Haar Cascade."""
    
    def __init__(self, model_name: str = 'haarcascade_frontalface_default.xml', 
                 scale_factor: float = 1.05, min_neighbors: int = 4):
        """
        Initialize the face detector.
        
        Args:
            model_name: Name of the Haar Cascade model file
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
        """
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.model = self._load_model()
        logger.info(f"Initialized FaceDetector with model: {model_name}")
    
    def _load_model(self):
        """Load the Haar Cascade face detection model."""
        try:
            # Try to find the cascade file in the current directory first
            cascade_path = Path(__file__).parent / 'models' / self.model_name
            
            if not cascade_path.exists():
                # If not found, try to use OpenCV's built-in cascades
                cascade_path = Path(cv2.data.haarcascades) / self.model_name
                
                if not cascade_path.exists():
                    # If still not found, try to download it
                    logger.warning(f"Cascade file not found at {cascade_path}")
                    logger.info("Attempting to use OpenCV's built-in cascade...")
                    cascade_path = cv2.data.haarcascades + self.model_name
            
            logger.info(f"Loading Haar Cascade model from: {cascade_path}")
            model = cv2.CascadeClassifier(str(cascade_path))
            
            if model.empty():
                raise ValueError(f"Failed to load cascade classifier from {cascade_path}")
                
            logger.info("Successfully loaded Haar Cascade model")
            return model
            
        except Exception as e:
            logger.error(f"Error loading Haar Cascade model: {str(e)}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in the input image.
        
        Args:
            image: Input image in BGR format (OpenCV default)
            
        Returns:
            List of detected FaceDetection objects
        """
        if image is None or image.size == 0:
            logger.warning("Received empty or invalid image")
            return []
            
        try:
            # Convert to grayscale (required for Haar Cascade)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with more permissive parameters
            faces = self.model.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(20, 20),  # Reduced minimum size to detect smaller faces
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Format detections
            detections = []
            for (x, y, w, h) in faces:
                # Convert from [x, y, w, h] to [x1, y1, x2, y2] format
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                
                # Create default landmarks (not provided by Haar Cascade)
                landmarks = np.array([
                    [0, 0],  # right eye
                    [0, 0],  # left eye
                    [0, 0],  # nose
                    [0, 0],  # right mouth
                    [0, 0]   # left mouth
                ])
                
                detections.append(FaceDetection(
                    bbox=np.array([x1, y1, x2, y2]),
                    landmarks=landmarks,
                    confidence=1.0
                ))
                
            return detections
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def extract_face_chips(self, image: np.ndarray, face: FaceDetection, size: int = 160) -> np.ndarray:
        """Extract and align face chip using OpenCV for face alignment."""
        try:
            if not ENABLE_ALIGNMENT or face.landmarks is None or len(face.landmarks) < 5:
                # Fallback to simple crop if alignment is disabled or not enough landmarks
                return image[
                    int(face.bbox[1]):int(face.bbox[3]),
                    int(face.bbox[0]):int(face.bbox[2])
                ]
            
            # Define template face landmarks (normalized to [0, 1])
            template = np.array([
                [0.31556875, 0.46157411],  # Left eye
                [0.68262236, 0.46157411],  # Right eye
                [0.50026208, 0.64050537],  # Nose
                [0.3494713, 0.82469112],   # Left mouth corner
                [0.65304688, 0.82469112]   # Right mouth corner
            ], dtype=np.float32) * size
            
            # Convert landmarks to numpy array if they aren't already
            src_points = np.array(face.landmarks, dtype=np.float32)
            
            # Calculate the transformation matrix
            transform = cv2.estimateAffine2D(
                src_points, 
                template, 
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )[0]
            
            if transform is None:
                logger.warning("Failed to estimate affine transform, using simple crop")
                return image[
                    int(face.bbox[1]):int(face.bbox[3]),
                    int(face.bbox[0]):int(face.bbox[2])
                ]
            
            # Apply the transformation
            warped = cv2.warpAffine(
                image,
                transform,
                (size, size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            return warped
            
        except Exception as e:
            logger.error(f"Error in face alignment: {str(e)}")
            # Fallback to simple crop
            try:
                return image[
                    max(0, int(face.bbox[1])):min(image.shape[0], int(face.bbox[3])),
                    max(0, int(face.bbox[0])):min(image.shape[1], int(face.bbox[2]))
                ]
            except Exception as crop_err:
                logger.error(f"Error in fallback crop: {str(crop_err)}")
                # Return black image as last resort
                return np.zeros((size, size, 3), dtype=np.uint8)
