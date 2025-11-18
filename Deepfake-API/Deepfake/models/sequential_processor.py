"""
Sequential Processor for running deepfake detection models in a pipeline.
"""
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from PIL import Image
import logging
import cv2
import sys
import os
import time

# Add the parent directory to the path to import from pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.demographic_analyzer import DemographicAnalyzer, DemographicData
from pipeline.face_detector import FaceDetector, FaceDetection
from .hf_models import HFModelLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequentialProcessor:
    """
    Handles sequential execution of deepfake detection models with weighted decisions.
    """
    
    # Model weights for deepfake detection models
    MODEL_WEIGHTS = {
        'xception': 0.35,      # Slightly reduced weight as it might be overconfident
        'mesonet': 0.3,        # Kept same weight
        'face_xray': 0.25,     # Increased weight as it's good at detecting artifacts
        'vit': 0.1             # Kept as is for diversity
    }
    
    # Weights for demographic and quality features
    DEMOGRAPHIC_WEIGHTS = {
        'age': 0.1,           # Younger faces might be more likely to be fake
        'gender': 0.1,        # Certain demographics might be targeted more
        'race': 0.1,          # Racial bias in training data
        'emotion': 0.05,      # Unnatural emotions
        'quality': 0.1,       # Image quality metrics
        'color': 0.05         # Color distribution anomalies
    }
    
    # Lowered confidence threshold for early stopping to ensure all models get a chance
    CONFIDENCE_THRESHOLD = 0.95
    
    # New: Minimum number of models that must agree for early stopping
    MIN_MODELS_FOR_CONSENSUS = 2
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the sequential processor with all required models and analyzers.
        
        Args:
            device: Device to run models on ('cuda' or 'cpu').
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize deepfake detection models
        self.model_loader = HFModelLoader(device=self.device)
        self.available_models = self._get_available_models()
        
        # Initialize face detection and demographic analysis
        self.face_detector = FaceDetector()
        try:
            self.demographic_analyzer = DemographicAnalyzer()
            logger.info("Successfully initialized DemographicAnalyzer")
        except Exception as e:
            logger.warning(f"Could not initialize DemographicAnalyzer: {str(e)}")
            self.demographic_analyzer = None
        
        logger.info(f"Initialized SequentialProcessor on {self.device}")
        logger.info(f"Available models in sequence: {', '.join(self.available_models.keys())}")
        if self.demographic_analyzer is not None:
            logger.info("Demographic analysis is available")
        else:
            logger.warning("Demographic analysis is not available")
    
    def _get_available_models(self) -> Dict[str, float]:
        """Get available models with their weights."""
        all_models = self.model_loader.get_available_models()
        return {model_id: self.MODEL_WEIGHTS.get(model_id, 0.1) 
                for model_id in all_models 
                if model_id in self.MODEL_WEIGHTS}
                
    def _calculate_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate image quality metrics.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        try:
            # Convert to grayscale for some metrics
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Calculate blur (variance of Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            metrics['sharpness'] = float(laplacian.var())
            
            # Calculate brightness (mean pixel intensity)
            metrics['brightness'] = float(np.mean(gray))
            
            # Calculate contrast (standard deviation of pixel intensities)
            metrics['contrast'] = float(np.std(gray))
            
            # Calculate noise (variance of the Laplacian of the image)
            metrics['noise'] = float(laplacian.var())
            
            # Calculate dynamic range (max - min pixel intensity)
            metrics['dynamic_range'] = float(gray.max() - gray.min())
            
            # Calculate overall quality score (0-100)
            quality_score = 0.0
            
            # Higher sharpness is better (up to a point)
            sharpness_score = min(metrics['sharpness'] / 100.0, 1.0) * 40
            
            # Good brightness is around 127 (mid-range)
            brightness_score = 1.0 - abs(metrics['brightness'] - 127) / 127.0 * 30
            
            # Higher contrast is better (up to a point)
            contrast_score = min(metrics['contrast'] / 50.0, 1.0) * 20
            
            # Lower noise is better
            noise_score = max(0, 1.0 - (metrics['noise'] / 1000.0)) * 10
            
            quality_score = sharpness_score + brightness_score + contrast_score + noise_score
            metrics['quality_score'] = min(max(quality_score, 0), 100)
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {str(e)}")
            metrics['quality_score'] = 50.0  # Default quality score
            
        return metrics
        
    def _analyze_demographics(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze demographic information from a face image.
        
        Args:
            face_image: Cropped face image as numpy array (H, W, C)
            
        Returns:
            Dictionary containing demographic analysis results
        """
        if self.demographic_analyzer is None:
            return {}
            
        try:
            # Convert to RGB if needed
            if face_image.shape[-1] == 3 and len(face_image.shape) == 3:
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_image_rgb = face_image
                
            # Analyze demographics
            demo_data = self.demographic_analyzer.analyze(face_image_rgb)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(face_image_rgb)
            
            # Combine results
            return {
                'demographics': {
                    'age': demo_data.age,
                    'gender': demo_data.gender,
                    'gender_confidence': demo_data.gender_confidence,
                    'race': demo_data.race,
                    'emotion': demo_data.emotion,
                    'accessories': demo_data.accessories
                },
                'quality': quality_metrics
            }
        except Exception as e:
            logger.error(f"Error in demographic analysis: {str(e)}")
            return {}
            
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
                
                # Skip if face region is too small
                if face_region.size == 0 or min(face_region.shape[:2]) < 20:
                    continue
                    
                # Analyze demographics and quality
                analysis = self._analyze_demographics(face_region)
                
                # Add to results
                results.append({
                    'bbox': face.bbox.tolist(),
                    'landmarks': face.landmarks.tolist() if face.landmarks is not None else [],
                    'confidence': float(face.confidence) if hasattr(face, 'confidence') else 1.0,
                    'demographics': analysis.get('demographics', {}),
                    'quality': analysis.get('quality', {})
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []
    
    def _calculate_demographic_scores(self, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate demographic-based scores that might indicate deepfake likelihood.
        
        Args:
            faces: List of detected faces with demographic analysis
            
        Returns:
            Dictionary of demographic-based scores and face information
        """
        if not faces:
            return {}
            
        scores = {}
        total_weight = 0.0
        
        # Calculate demographic-based scores for each face
        for face_idx, face in enumerate(faces):
            demo = face.get('demographics', {})
            quality = face.get('quality', {})
            bbox = face.get('bbox', [0, 0, 100, 100])
            
            # Calculate face area for weighting
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if len(bbox) == 4 else 10000
            
            # Age-based score (younger faces might be more likely to be faked)
            age = demo.get('age', 30)  # Default to 30 if not available
            age_score = min(age / 30.0, 1.0)  # Younger than 30 gets higher score
            
            # Gender-based score (certain demographics might be targeted more)
            gender = str(demo.get('gender', 'unknown')).lower()
            gender_score = 0.5  # Neutral by default
            if gender == 'female':
                gender_score = 0.7  # Females might be targeted more
            elif gender == 'male':
                gender_score = 0.3
                
            # Emotion-based score (unnatural emotions might indicate deepfakes)
            emotion = demo.get('emotion', {})
            emotion_score = 0.5
            if emotion and isinstance(emotion, dict):
                # Higher score for less common emotions in real photos
                common_emotions = ['neutral', 'happy', 'sad']
                uncommon_emotions = ['angry', 'surprise', 'disgust', 'fear']
                
                for emo in uncommon_emotions:
                    if emo in emotion and isinstance(emotion[emo], (int, float)) and emotion[emo] > 0.5:
                        emotion_score = 0.7
                        break
                        
            # Quality-based score (low quality might indicate manipulation)
            quality_score = float(quality.get('quality_score', 50)) / 100.0  # Normalize to 0-1
            
            # Calculate weighted score for this face
            face_score = (
                age_score * self.DEMOGRAPHIC_WEIGHTS['age'] +
                gender_score * self.DEMOGRAPHIC_WEIGHTS['gender'] +
                emotion_score * self.DEMOGRAPHIC_WEIGHTS['emotion'] +
                (1.0 - quality_score) * self.DEMOGRAPHIC_WEIGHTS['quality']
            )
            
            # Store face information
            face_key = f'face_{face_idx}'
            scores[face_key] = {
                'score': float(face_score),
                'weight': float(face_area),
                'age': int(age),
                'gender': gender,
                'quality': quality_score,
                'bbox': [float(x) for x in bbox] if isinstance(bbox, (list, tuple, np.ndarray)) else [0.0, 0.0, 100.0, 100.0]
            }
            
            total_weight += face_area
        
        # Calculate weighted average of all face scores
        if scores and total_weight > 0:
            try:
                weighted_sum = sum(s['score'] * s['weight'] for s in scores.values() 
                                if isinstance(s, dict) and 'score' in s and 'weight' in s)
                scores['overall_score'] = float(weighted_sum / total_weight)
            except (TypeError, ZeroDivisionError):
                scores['overall_score'] = 0.5  # Fallback neutral score
        else:
            scores['overall_score'] = 0.5  # Neutral score if no faces
            
        # Ensure all values are JSON serializable
        return {k: (float(v) if isinstance(v, (int, float, np.number)) else v) 
               for k, v in scores.items()}
        
    def _calculate_overall_quality(self, faces: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate overall quality metrics for the frame.
        
        Args:
            faces: List of detected faces with quality metrics
            
        Returns:
            Dictionary of overall quality metrics
        """
        if not faces:
            return {}
            
        # Calculate average quality metrics across all faces
        avg_quality = {
            'sharpness': 0.0,
            'brightness': 0.0,
            'contrast': 0.0,
            'noise': 0.0,
            'dynamic_range': 0.0,
            'quality_score': 0.0
        }
        
        valid_faces = 0
        for face in faces:
            quality = face.get('quality', {})
            if quality:
                for k in avg_quality:
                    if k in quality:
                        avg_quality[k] += quality[k]
                valid_faces += 1
                
        if valid_faces > 0:
            for k in avg_quality:
                avg_quality[k] /= valid_faces
                
        return avg_quality
        
    def preprocess_image(self, image: np.ndarray, model_id: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess an image for a specific model.
        
        Args:
            image: Input image as numpy array (H, W, C)
            model_id: ID of the target model
            
        Returns:
            Dictionary of model inputs
        """
        try:
            processor = self.model_loader.get_processor(model_id)
            pil_image = Image.fromarray(image)
            inputs = processor(images=pil_image, return_tensors="pt")
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs
            
        except Exception as e:
            logger.error(f"Error preprocessing image for {model_id}: {str(e)}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame through the sequential pipeline with face detection,
        demographic analysis, and quality assessment.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            
        Returns:
            Dictionary containing:
            - predictions: List of individual model predictions with confidence and weights
            - faces: List of detected faces with detailed analysis including:
                - bbox: Bounding box coordinates [x1, y1, x2, y2]
                - confidence: Face detection confidence
                - demographics: Age, gender, gender_confidence, race
                - quality: Quality metrics including quality_score, sharpness, etc.
                - landmarks: Facial landmarks if available
                - accessories: Detected accessories if any
            - final_decision: Weighted final decision (0-1)
            - is_deepfake: Boolean based on threshold (0.5)
            - confidence: Average confidence score (0-1)
            - metadata: Additional metadata including processing time and face count
        """
        start_time = time.time()
        if not self.available_models:
            raise ValueError("No models available for processing")
        
        # Convert BGR to RGB if needed (OpenCV uses BGR by default)
        if frame.shape[-1] == 3 and len(frame.shape) == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Detect faces and analyze demographics
        faces = self._detect_faces(frame_rgb)
        
        # Process with deepfake detection models
        model_results = []
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_id, weight in self.available_models.items():
            try:
                # Preprocess
                inputs = self.preprocess_image(frame_rgb, model_id)
                
                # Get model and predict
                model = self.model_loader.get_model(model_id)
                model_config = self.model_loader.get_model_config(model_id)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Handle different model output formats
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    logits = outputs[0]  # Assume first element is logits
                else:
                    logits = outputs
                
                # Get probabilities
                if logits.dim() > 2:  # Handle sequence outputs
                    logits = logits.mean(dim=1)  # Average over sequence length
                    
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence, pred = torch.max(probs, dim=-1)
                confidence = confidence.item()
                pred = pred.item()
                
                # Get fake probability based on model type
                if model_id == 'vit':
                    # For ViT, use the predicted class probability
                    fake_prob = confidence if pred == 1 else 1 - confidence
                elif model_id == 'efficientnet':
                    # For EfficientNet, use the predicted class probability
                    fake_prob = confidence if pred == 1 else 1 - confidence
                elif probs.shape[-1] == 2:  # Binary classification
                    fake_prob = probs[0][1].item()
                else:
                    # For multi-class, use the predicted probability
                    fake_prob = confidence
                
                # Store results
                result = {
                    'model': model_id,
                    'prediction': pred,
                    'confidence': confidence,
                    'fake_probability': fake_prob,
                    'weight': weight
                }
                model_results.append(result)
                
                # Update weighted sum
                weighted_sum += fake_prob * weight
                total_weight += weight
                
                # Early stopping only if we have consensus from multiple models
                if (confidence >= self.CONFIDENCE_THRESHOLD and 
                    len([p for p in model_results if p['confidence'] >= 0.7]) >= self.MIN_MODELS_FOR_CONSENSUS):
                    logger.info(f"Early stopping with consensus at {model_id} (confidence: {confidence:.2f})")
                    break
                    
            except Exception as e:
                logger.error(f"Error processing with {model_id}: {str(e)}")
                
        # Calculate final decision
        final_decision = (weighted_sum / total_weight) if total_weight > 0 else 0.5
        is_deepfake = final_decision >= 0.5
        
        # Calculate confidence as average of model confidences
        if model_results:
            avg_confidence = sum(r['confidence'] for r in model_results) / len(model_results)
        else:
            avg_confidence = 0.5
            
        # Calculate demographic and quality scores
        demo_scores = self._calculate_demographic_scores(faces)
        
        # Format faces with detailed information
        formatted_faces = []
        for face in faces:
            formatted_face = {
                'bbox': [int(x) for x in face.get('bbox', [0, 0, 0, 0])],
                'confidence': float(face.get('confidence', 0.0)),
                'demographics': {
                    'age': face.get('demographics', {}).get('age', 0),
                    'gender': face.get('demographics', {}).get('gender', 'unknown'),
                    'gender_confidence': float(face.get('demographics', {}).get('gender_confidence', 0.0)),
                    'race': face.get('demographics', {}).get('race', {})
                },
                'quality': {
                    'quality_score': float(face.get('quality', {}).get('quality_score', 0.0)),
                    'sharpness': float(face.get('quality', {}).get('sharpness', 0.0)),
                    'brightness': float(face.get('quality', {}).get('brightness', 0.0)),
                    'contrast': float(face.get('quality', {}).get('contrast', 0.0)),
                    'noise': float(face.get('quality', {}).get('noise', 0.0)),
                    'dynamic_range': float(face.get('quality', {}).get('dynamic_range', 0.0))
                },
                'landmarks': face.get('landmarks', []),
                'accessories': face.get('demographics', {}).get('accessories', {})
            }
            formatted_faces.append(formatted_face)
        
        # Add processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Combine all results
        result = {
            'predictions': model_results,
            'faces': formatted_faces,
            'final_decision': float(final_decision),
            'is_deepfake': bool(is_deepfake),
            'confidence': float(avg_confidence),
            'metadata': {
                'num_faces': len(formatted_faces),
                'demographics': demo_scores,
                'quality': self._calculate_overall_quality(faces),
                'processing_time_ms': processing_time_ms
            }
        }
        
        # Ensure all numeric values are converted to Python native types for JSON serialization
        def convert_floats(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (list, tuple)):
                return [convert_floats(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            return obj
            
        return convert_floats(result)
        if len(results) > 0:
            # Calculate average confidence across all models
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            
            # If models are very confident but disagree, be more cautious
            if avg_confidence > 0.8 and abs(weighted_sum/total_weight - 0.5) < 0.2:
                logger.warning("High confidence but models disagree - applying caution")
                # Bias towards 'fake' if there's any significant indication
                if weighted_sum/total_weight > 0.4:  # If any significant indication of fake
                    weighted_sum = min(weighted_sum * 1.3, total_weight)  # Bias towards fake
            
            final_decision = (weighted_sum / total_weight)
        else:
            final_decision = 0.5  # Default to uncertain
        
        return {
            'predictions': results,
            'final_decision': final_decision,
            'is_deepfake': final_decision >= 0.5,
            'confidence': abs(final_decision - 0.5) * 2  # Convert to 0-1 confidence
        }

# Example usage
if __name__ == "__main__":
    import cv2
    
    # Initialize processor
    processor = SequentialProcessor()
    
    # Load a test image
    image_path = "test_image.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image {image_path}")
    else:
        # Process the image
        result = processor.process_frame(image)
        
        # Print results
        print("\n=== Individual Model Predictions ===")
        for pred in result['predictions']:
            print(f"{pred['model']}: "
                  f"Fake Prob: {pred['fake_probability']:.4f}, "
                  f"Confidence: {pred['confidence']:.4f}, "
                  f"Weight: {pred['weight']}")
        
        print("\n=== Final Decision ===")
        print(f"Final Score: {result['final_decision']:.4f}")
        print(f"Is Deepfake: {result['is_deepfake']}")
        print(f"Confidence: {result['confidence']:.4f}")
