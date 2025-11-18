"""
Parallel Processor for running multiple deepfake detection models simultaneously.
"""
import concurrent.futures
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import logging
import time
import cv2
import sys
import os

# Add the parent directory to the path to import from pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.demographic_analyzer import DemographicAnalyzer
from pipeline.face_detector import FaceDetector

from .hf_models import HFModelLoader
from .model_interfaces import BaseModel
from .ensemble import create_ensemble, WeightedEnsemble

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelProcessor:
    """
    Handles parallel execution of multiple deepfake detection models.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the parallel processor with ensemble support.
        
        Args:
            device: Device to run models on ('cuda' or 'cpu').
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loader = HFModelLoader(device=self.device)
        self.available_models = self.model_loader.get_available_models()
        
        # Initialize model adapters
        self.model_adapters: Dict[str, BaseModel] = {}
        for model_id in self.available_models:
            try:
                self.model_adapters[model_id] = self.model_loader.get_model_adapter(model_id)
                logger.info(f"Initialized model adapter for {model_id}")
            except Exception as e:
                logger.error(f"Failed to initialize model adapter for {model_id}: {str(e)}")
        
        # Initialize ensemble with all available models
        try:
            self.ensemble = create_ensemble(self.model_loader, list(self.available_models.keys()))
            logger.info(f"Initialized ensemble with {len(self.available_models)} models")
        except Exception as e:
            logger.error(f"Failed to initialize ensemble: {str(e)}")
            self.ensemble = None
        
        # Initialize face detection and demographic analysis
        self.face_detector = FaceDetector()
        try:
            self.demographic_analyzer = DemographicAnalyzer()
            logger.info("Successfully initialized DemographicAnalyzer")
        except Exception as e:
            logger.warning(f"Could not initialize DemographicAnalyzer: {str(e)}")
            self.demographic_analyzer = None
        
        logger.info(f"Initialized ParallelProcessor on {self.device}")
        logger.info(f"Available models: {', '.join(self.available_models.keys())}")
        if self.demographic_analyzer is not None:
            logger.info("Demographic analysis is available")
        else:
            logger.warning("Demographic analysis is not available")
    
    def _analyze_demographics(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze demographic information from a face image.
        
        Args:
            face_image: Cropped face image as numpy array (H, W, C)
            
        Returns:
            Dictionary containing demographic analysis results with default values if analysis fails
        """
        if self.demographic_analyzer is None:
            return self._get_default_demographics()
            
        try:
            # Convert to RGB if needed
            if face_image.shape[-1] == 3 and len(face_image.shape) == 3:
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_image_rgb = face_image
                
            # Analyze demographics
            demo_results = self.demographic_analyzer.analyze(face_image_rgb)
            
            # Handle case where analyze returns a list of results
            if isinstance(demo_results, list) and demo_results:
                # Take the first face result
                demo_data = demo_results[0]
                
                # Extract attributes with safe defaults
                return {
                    'demographics': {
                        'age': getattr(demo_data, 'age', 30),
                        'gender': getattr(demo_data, 'gender', 'unknown'),
                        'gender_confidence': float(getattr(demo_data, 'gender_confidence', 0.5)),
                        'race': getattr(demo_data, 'race', {'unknown': 1.0}),
                        'emotion': getattr(demo_data, 'emotion', {'neutral': 1.0}),
                        'accessories': getattr(demo_data, 'accessories', {})
                    },
                    'quality_metrics': getattr(demo_data, 'quality_metrics', {}),
                    'quality_score': float(getattr(demo_data, 'quality_score', 0.5))
                }
            else:
                logger.warning("No demographic data returned from analyzer")
                return self._get_default_demographics()
                
        except Exception as e:
            logger.error(f"Error in demographic analysis: {str(e)}")
            return self._get_default_demographics()
            
    def _get_default_demographics(self) -> Dict[str, Any]:
        """Return default demographic values when analysis fails."""
        return {
            'demographics': {
                'age': 30,
                'gender': 'unknown',
                'gender_confidence': 0.5,
                'race': {'unknown': 1.0},
                'emotion': {'neutral': 1.0},
                'accessories': {}
            },
            'quality_metrics': {},
            'quality_score': 0.5
        }
            
    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image and return their bounding boxes and landmarks.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of dictionaries containing face detection results
        """
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            results = []
            for face in faces:
                # Extract face region
                x1, y1, x2, y2 = face.bbox.astype(int)
                face_region = image[y1:y2, x1:x2]
                
                # Analyze demographics if analyzer is available
                demographics = {}
                if self.demographic_analyzer is not None and face_region.size > 0:
                    demographics = self._analyze_demographics(face_region)
                
                # Add to results
                results.append({
                    'bbox': face.bbox.tolist(),
                    'landmarks': face.landmarks.tolist() if face.landmarks is not None else [],
                    'confidence': float(face.confidence) if hasattr(face, 'confidence') else 1.0,
                    **demographics
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []
    
    def _run_single_model(self, model_id: str, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference with a single model on an image using the standardized interface.
        
        Args:
            model_id: ID of the model to run
            image: Input image as numpy array (H, W, C) in RGB format
            
        Returns:
            Dictionary containing model outputs and metadata
        """
        try:
            if model_id not in self.model_adapters:
                raise ValueError(f"No adapter found for model: {model_id}")
                
            logger.info(f"Running inference with model: {model_id}")
            
            # Get the model adapter and run prediction
            model_adapter = self.model_adapters[model_id]
            result = model_adapter.predict(image)
            
            # Standardize the output format
            return {
                'model_id': model_id,
                'logits': result.get('logits'),
                'probabilities': result.get('probabilities'),
                'class_label': result.get('class_label', 'unknown'),
                'confidence': float(result.get('confidence', 0.0)),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error running model {model_id}: {str(e)}", exc_info=True)
            return {
                'model_id': model_id,
                'success': False,
                'error': str(e),
                'class_label': 'error',
                'confidence': 0.0
            }
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame with all available models in parallel.
        
        Args:
            frame: Input image as numpy array (H, W, C) in RGB format
            
        Returns:
            Dictionary containing results from all models and face analysis
        """
        start_time = time.time()
        
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("Frame is None or not a numpy array")
            raise ValueError("Invalid frame input: expected numpy array")
            
        logger.info(f"Processing frame with shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
        logger.info(f"Available models: {list(self.available_models.keys())}")
        
        # Detect faces
        face_results = []
        try:
            face_boxes = self.face_detector.detect_faces(frame)
            logger.info(f"Detected {len(face_boxes)} faces in the frame")
            
            # Convert face detections to the expected format
            for i, face in enumerate(face_boxes):
                x1, y1, w, h = face.bbox
                face_results.append({
                    'face_id': i,
                    'bbox': [int(x1), int(y1), int(x1 + w), int(y1 + h)],
                    'confidence': face.confidence,
                    'landmarks': face.landmarks.tolist() if hasattr(face, 'landmarks') else []
                })
        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")
        
        # Initialize results storage
        results = {
            'frame_analysis': {},
            'face_analyses': []
        }
        
        # Process the whole frame with the ensemble if no faces detected
        if not face_results:
            logger.info("No faces detected, analyzing full frame with ensemble")
            if self.ensemble is not None:
                try:
                    # For full frame analysis, we don't have demographic info
                    ensemble_result = self.ensemble.predict(frame)
                    results['frame_analysis'] = {
                        'success': True,
                        'prediction': 'fake' if ensemble_result['prediction'] == 1 else 'real',
                        'confidence': float(ensemble_result['confidence']),
                        'model_predictions': ensemble_result['model_predictions'],
                        'weights': ensemble_result['weights'],
                        'demographic_info': 'full_frame_analysis',
                        'note': 'No demographic-specific weighting applied for full frame analysis'
                    }
                    logger.info(f"Full frame analysis completed with confidence: {ensemble_result['confidence']:.4f}")
                except Exception as e:
                    logger.error(f"Error in ensemble prediction: {str(e)}", exc_info=True)
                    results['frame_analysis'] = {
                        'success': False,
                        'error': str(e),
                        'demographic_info': 'error'
                    }
        
        # Process each detected face with the ensemble
        for face in face_results:
            x1, y1, x2, y2 = face['bbox']
            
            # Extract face region with boundary checks
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid face coordinates: {face['bbox']}")
                continue
                
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                logger.warning(f"Empty face region for bbox: {face['bbox']}")
                continue
            
            # Initialize face analysis with basic info
            face_analysis = {
                'face_id': face['face_id'],
                'bbox': face['bbox'],
                'landmarks': face.get('landmarks', []),
                'demographics': {}
            }
            
            # Get demographic analysis first (needed for ensemble prediction)
            if self.demographic_analyzer is not None:
                try:
                    demo_result = self._analyze_demographics(face_region)
                    face_analysis['demographics'] = demo_result
                    
                    # Extract demographic info for ensemble
                    demo_info = demo_result.get('demographics', {})
                    race_info = demo_info.get('race', {})
                    # Get the dominant race if available, otherwise use 'unknown'
                    dominant_race = max(race_info.items(), key=lambda x: x[1])[0] if isinstance(race_info, dict) and race_info else 'unknown'
                    
                    demographics = {
                        'age': demo_info.get('age', 30),  # Default to 30 if age not available
                        'gender': str(demo_info.get('gender', 'unknown')).lower(),
                        'race': str(dominant_race).lower()
                    }
                except Exception as e:
                    logger.error(f"Error in demographic analysis: {str(e)}", exc_info=True)
                    demographics = None
            else:
                demographics = None
            
            # Get ensemble prediction for face with demographic info
            if self.ensemble is not None:
                try:
                    # Pass demographics to ensemble for dynamic weighting
                    ensemble_result = self.ensemble.predict(
                        face_region,
                        demographics=demographics
                    )
                    
                    # Format the prediction result
                    face_analysis.update({
                        'success': True,
                        'prediction': 'fake' if ensemble_result['prediction'] == 1 else 'real',
                        'confidence': float(ensemble_result['confidence']),
                        'model_predictions': ensemble_result['model_predictions'],
                        'weights': ensemble_result['weights'],
                        'demographic_weights': ensemble_result.get('demographics')
                    })
                    
                    # Add debug info about weights
                    logger.info(f"Model weights for face {face['face_id']} (age={demographics['age'] if demographics else 'N/A'}, "
                               f"gender={demographics['gender'] if demographics else 'N/A'}): {ensemble_result['weights']}")
                    
                except Exception as e:
                    logger.error(f"Error in ensemble prediction for face {face['face_id']}: {str(e)}", exc_info=True)
                    face_analysis.update({
                        'success': False,
                        'error': str(e)
                    })
            
            results['face_analyses'].append(face_analysis)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Format the final result
        result = {
            'frame_shape': frame.shape,
            'processing_time_seconds': processing_time,
            'num_faces': len(face_results),
            'frame_analysis': results['frame_analysis'],
            'face_analyses': results['face_analyses']
        }
        
        # Add summary statistics
        if results['face_analyses']:
            confidences = [f.get('confidence', 0) for f in results['face_analyses'] 
                         if f.get('success', False) and 'confidence' in f]
            if confidences:
                result['average_confidence'] = sum(confidences) / len(confidences)
            
            predictions = [f.get('prediction', '').lower() 
                         for f in results['face_analyses'] 
                         if f.get('success', False) and 'prediction' in f]
            
            if predictions:
                fake_count = sum(1 for p in predictions if p == 'fake')
                result['summary'] = {
                    'total_faces': len(predictions),
                    'fake_faces': fake_count,
                    'real_faces': len(predictions) - fake_count,
                    'fake_ratio': fake_count / len(predictions) if predictions else 0.0
                }
        
        return result
        
    async def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process a single image with all available models.
        This is an async wrapper around process_frame for use with FastAPI endpoints.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format (OpenCV default)
            
        Returns:
            Dictionary containing results from all models with consistent format,
            including face detection and demographic analysis
        """
        try:
            logger.info(f"Processing image with shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
            
            # Validate input image
            if not isinstance(image, np.ndarray) or image.size == 0:
                logger.error("Invalid image input: expected non-empty numpy array")
                raise ValueError("Invalid image input: expected non-empty numpy array")
                
            # Convert from BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:  # If it's a color image
                try:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    logger.debug("Converted image from BGR to RGB")
                except Exception as e:
                    logger.warning(f"Error converting BGR to RGB, using as-is: {str(e)}")
                    image_rgb = image
            else:
                image_rgb = image
            
            # Process the frame with all models and face analysis
            logger.info("Starting frame processing with all models and face analysis")
            result = self.process_frame(image_rgb)
            logger.info(f"Frame processing completed")
            
            # Format the results for the API response
            formatted_results = {}
            
            # If we have predictions from the ensemble, format them
            if 'predictions' in result and result['predictions']:
                for i, pred in enumerate(result['predictions']):
                    model_key = f"face_{i}"
                    formatted_results[model_key] = {
                        'success': 'error' not in pred,
                        'bbox': pred.get('bbox', []),
                        'prediction': pred.get('prediction', -1),
                        'confidence': float(pred.get('confidence', 0.0)),
                        'model_details': pred.get('model_details', {})
                    }
                    
                    # Add demographic info if available
                    if 'demographics' in result and i < len(result['demographics']):
                        formatted_results[model_key]['demographics'] = result['demographics'][i]
            
            return {
                'success': True,
                'results': formatted_results,
                'num_faces': len(result.get('predictions', [])),
                'demographics_available': 'demographics' in result and bool(result['demographics'])
            }
            
        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'results': {}
            }

# Example usage
if __name__ == "__main__":
    import cv2
    
    # Initialize processor
    processor = ParallelProcessor()
    
    # Load a test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)  # Replace with actual image
    
    # Process the frame
    print("Processing test frame...")
    results = processor.process_frame(test_image)
    
    # Print summary
    print(f"\nProcessed frame in {results['processing_time_seconds']:.2f} seconds")
    print(f"Ran {results['models_run']} models ({results['successful_models']} successful)")
    
    # Print results for each model
    for model_id, result in results['results'].items():
        status = "SUCCESS" if result['success'] else f"FAILED: {result.get('error', 'Unknown error')}"
        print(f"\n{model_id}: {status}")
        
        if result['success']:
            print(f"Logits shape: {result['logits'].shape}")
            if result['probabilities'] is not None:
                print(f"Class probabilities: {result['probabilities']}")
