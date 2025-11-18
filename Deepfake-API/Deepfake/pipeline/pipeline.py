"""
Main Pipeline Orchestrator
Coordinates the entire deepfake detection process.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
import json
import time

from .video_processor import VideoProcessor
from .face_detector import FaceDetector, FaceDetection
from .demographic_analyzer import DemographicAnalyzer, DemographicData

logger = logging.getLogger(__name__)

@dataclass
class FrameAnalysis:
    """Container for frame analysis results."""
    frame_number: int
    faces: List[Dict[str, Any]]
    quality_score: float
    processing_time: float

@dataclass
class DetectionResult:
    """Container for final detection results."""
    is_deepfake: bool
    confidence: float
    frame_results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def to_dict(self):
        def convert_numpy(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(x) for x in obj]
            return obj
            
        result = {
            'is_deepfake': bool(self.is_deepfake),
            'confidence': float(self.confidence),
            'frame_results': convert_numpy(self.frame_results),
            'metadata': convert_numpy(self.metadata)
        }
        return result
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

class DeepfakePipeline:
    """
    Main pipeline for deepfake detection.
    Orchestrates video processing, face detection, and analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the pipeline with optional configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        
        # Initialize components
        self.video_processor = VideoProcessor(
            frame_interval=self.config.get('frame_interval', 10),
            target_size=tuple(self.config.get('target_size', (640, 480)))
        )
        
        # Configure face detector with Haar Cascade settings
        self.face_detector = FaceDetector(
            model_name=self.config.get('face_detector_model', 'haarcascade_frontalface_default.xml'),
            scale_factor=self.config.get('face_detector_scale_factor', 1.1),
            min_neighbors=self.config.get('face_detector_min_neighbors', 5)
        )
        
        self.demographic_analyzer = DemographicAnalyzer()
        
        # Initialize models (to be implemented)
        self.models = self._initialize_models()
        
        logger.info("Deepfake pipeline initialized")
    
    def _initialize_models(self) -> Dict:
        """Initialize deepfake detection models."""
        # TODO: Implement model loading
        return {}
    
    def process_video(self, video_path: str) -> DetectionResult:
        """
        Process a video file for deepfake detection.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            DetectionResult containing analysis results
        """
        start_time = time.time()
        frame_results = []
        
        # Get video properties
        video_info = self.video_processor.get_video_properties(video_path)
        if not video_info:
            raise ValueError(f"Could not read video: {video_path}")
        
        logger.info(f"Processing video: {os.path.basename(video_path)}")
        logger.info(f"Duration: {video_info['duration_seconds']:.1f}s, "
                   f"Frames: {video_info['frame_count']}, FPS: {video_info['fps']:.1f}")
        
        # Process each frame
        for frame_idx, frame in self.video_processor.extract_frames(video_path):
            frame_start = time.time()
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            # Analyze each face
            face_results = []
            for face in faces:
                # Extract face chip
                face_chip = self.face_detector.extract_face_chips(frame, face)
                
                # Get demographic info
                demo_data = self.demographic_analyzer.analyze(face_chip)
                
                # TODO: Run deepfake detection models with dynamic weights
                model_results = self._run_models(face_chip, demo_data)
                
                face_results.append({
                    'bounding_box': face.bbox.tolist(),
                    'landmarks': face.landmarks.tolist(),
                    'quality': face.quality,
                    'demographics': asdict(demo_data),
                    'model_results': model_results,
                    'is_deepfake': None,  # Will be set by ensemble
                    'confidence': None    # Will be set by ensemble
                })
            
            # Apply ensemble method
            frame_result = self._apply_ensemble(face_results)
            frame_results.append({
                'frame_number': frame_idx,
                'timestamp': frame_idx / video_info['fps'],
                'faces': frame_result,
                'processing_time': time.time() - frame_start
            })
        
        # Aggregate results across frames
        return self._aggregate_results(frame_results, {
            'processing_time': time.time() - start_time,
            'video_info': video_info
        })
    
    def _run_models(self, face_image: np.ndarray, demo_data: DemographicData) -> Dict[str, Any]:
        """
        Run deepfake detection models with dynamic weights.
        
        Args:
            face_image: Face image to analyze
            demo_data: Demographic data for adaptive weighting
            
        Returns:
            Dictionary of model results
        """
        results = {}
        
        # Calculate dynamic weights based on demographics and quality
        weights = self._calculate_weights(demo_data)
        
        # TODO: Implement model inference with dynamic weights
        # This is a placeholder - actual implementation will depend on model APIs
        for model_name, model in self.models.items():
            try:
                # Run model inference
                # result = model.predict(face_image)
                # results[model_name] = {
                #     'prediction': result['prediction'],
                #     'confidence': result['confidence'],
                #     'weight': weights[model_name]
                # }
                pass
            except Exception as e:
                logger.error(f"Error running {model_name}: {str(e)}")
                results[model_name] = {
                    'error': str(e),
                    'weight': 0.0
                }
        
        return results
    
    def _calculate_weights(self, demo_data: DemographicData) -> Dict[str, float]:
        """
        Calculate dynamic weights for models based on demographic data.
        
        Args:
            demo_data: Demographic data for the face
            
        Returns:
            Dictionary of model names to weights
        """
        # Base weights (can be configured)
        weights = {
            'xception': 0.25,
            'mesonet': 0.2,
            'face_xray': 0.25,
            'efficientnet': 0.2,
            'genconvvit': 0.1
        }
        
        # Adjust weights based on demographics
        if demo_data.age < 18 or demo_data.age > 60:
            # Younger and older faces might benefit from different models
            weights['xception'] += 0.1
            weights['face_xray'] += 0.1
        
        # Adjust for gender if confident
        if demo_data.gender_confidence > 0.8:
            if demo_data.gender.lower() == 'female':
                weights['efficientnet'] += 0.1
            else:
                weights['mesonet'] += 0.1
        
        # Adjust for accessories
        if 'glasses' in demo_data.accessories:
            weights['face_xray'] += 0.15  # Better with occlusions
        
        # Normalize to sum to 1
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def _apply_ensemble(self, face_results: List[Dict]) -> List[Dict]:
        """
        Apply weighted voting ensemble to model results.
        
        Args:
            face_results: List of face analysis results
            
        Returns:
            Updated face results with ensemble predictions
        """
        for face in face_results:
            if 'model_results' not in face or not face['model_results']:
                continue
                
            total_weight = 0.0
            weighted_sum = 0.0
            
            # Calculate weighted average
            for model_name, result in face['model_results'].items():
                if 'confidence' not in result or 'weight' not in result:
                    continue
                    
                # For binary classification, we can use the confidence directly
                # Assuming 0.5 is the decision threshold
                prediction = 1.0 if result['confidence'] > 0.5 else 0.0
                weight = result['weight']
                
                weighted_sum += prediction * weight
                total_weight += weight
            
            if total_weight > 0:
                final_confidence = weighted_sum / total_weight
                face['is_deepfake'] = final_confidence > 0.5
                face['confidence'] = max(final_confidence, 1 - final_confidence)  # Distance from 0.5
            
        return face_results
    
    def _aggregate_results(
        self, 
        frame_results: List[Dict],
        metadata: Dict
    ) -> DetectionResult:
        """
        Aggregate results across frames to make final decision.
        
        Args:
            frame_results: List of frame analysis results
            metadata: Additional metadata about the processing
            
        Returns:
            Final detection result
        """
        if not frame_results:
            return DetectionResult(
                is_deepfake=False,
                confidence=0.0,
                frame_results=[],
                metadata=metadata
            )
        
        # Simple majority voting across frames
        deepfake_votes = 0
        total_frames = 0
        total_confidence = 0.0
        
        for frame in frame_results:
            for face in frame.get('faces', []):
                if 'is_deepfake' in face and 'confidence' in face and face['confidence'] is not None:
                    total_frames += 1
                    if face['is_deepfake']:
                        deepfake_votes += 1
                    # Ensure confidence is a float and within valid range [0, 1]
                    confidence = float(face['confidence'])
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                    total_confidence += confidence
        
        if total_frames == 0:
            logger.warning("No valid face detections with confidence values found in any frame")
            return DetectionResult(
                is_deepfake=False,
                confidence=0.0,
                frame_results=frame_results,
                metadata=metadata
            )
            
        # Calculate final confidence and make decision
        final_confidence = total_confidence / total_frames if total_frames > 0 else 0.0
        is_deepfake = deepfake_votes > (total_frames / 2)
        
        logger.info(f"Aggregated results: {total_frames} faces, "
                  f"{deepfake_votes} deepfake votes, "
                  f"final confidence: {final_confidence:.2f}")
        
        return DetectionResult(
            is_deepfake=is_deepfake,
            confidence=final_confidence,
            frame_results=frame_results,
            metadata=metadata
        )

def create_pipeline(config: Optional[Dict] = None) -> DeepfakePipeline:
    """
    Factory function to create a configured pipeline instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured DeepfakePipeline instance
    """
    return DeepfakePipeline(config=config or {})
