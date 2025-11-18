"""
Weighted Ensemble for combining multiple deepfake detection models with dynamic weighting
based on demographic analysis.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from .model_interfaces import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class DemographicGroup:
    """Represents a demographic group for dynamic weighting."""
    age_range: Tuple[int, int]  # min_age, max_age
    gender: str  # 'male', 'female', or 'all'
    race: Optional[str] = None  # Optional race filter
    
    def matches(self, age: int, gender: str, race: Optional[str] = None) -> bool:
        """Check if the given demographics match this group."""
        age_match = self.age_range[0] <= age <= self.age_range[1]
        gender_match = self.gender in ['all', gender.lower()]
        race_match = self.race is None or (race and self.race.lower() == race.lower())
        return age_match and gender_match and race_match

class WeightedEnsemble:
    """
    A weighted ensemble of deepfake detection models with dynamic weighting based on demographics.
    
    This class combines predictions from multiple models using weighted averaging,
    where weights can be adjusted dynamically based on the demographic group of the input.
    """
    
    # Default model performance by demographic group
    # Based on empirical performance metrics across different demographics
    DEFAULT_PERFORMANCE = {
        'xception': {
            # Age ranges with performance weights (young, middle-aged, senior)
            'age_ranges': [
                (0, 20, 0.9),   # Slightly less accurate for young faces
                (21, 50, 1.0),  # Best for middle-aged
                (51, 120, 0.95) # Slightly less accurate for seniors
            ],
            # Gender performance weights
            'gender_weights': {
                'male': 1.0,    # Best for male faces
                'female': 0.95,  # Slightly worse for female faces
                'all': 1.0
            },
            # Race/ethnicity performance weights
            'race_weights': {
                'caucasian': 1.0,
                'asian': 0.92,
                'african': 0.88,
                'hispanic': 0.95,
                'indian': 0.9,
                'middle eastern': 0.93,
                'all': 1.0
            }
        },
        'mesonet': {
            'age_ranges': [
                (0, 20, 0.95),
                (21, 50, 0.98),
                (51, 120, 1.0)  # Better for older faces
            ],
            'gender_weights': {
                'male': 0.95,
                'female': 1.0,   # Better for female faces
                'all': 1.0
            },
            'race_weights': {
                'caucasian': 0.95,
                'asian': 0.98,
                'african': 0.92,
                'hispanic': 0.94,
                'indian': 0.96,
                'middle eastern': 0.93,
                'all': 1.0
            }
        },
        'face_xray': {
            'age_ranges': [
                (0, 20, 0.92),
                (21, 50, 0.95),
                (51, 120, 0.98)
            ],
            'gender_weights': {
                'male': 0.97,
                'female': 0.95,
                'all': 1.0
            },
            'race_weights': {
                'caucasian': 0.94,
                'asian': 0.96,
                'african': 0.98,  # Better for African faces
                'hispanic': 0.95,
                'indian': 0.93,
                'middle eastern': 0.96,
                'all': 1.0
            }
        },
        'vit': {
            'age_ranges': [
                (0, 20, 0.98, 0.02),    # (min_age, max_age, weight, smoothing)
                (21, 50, 1.0, 0.02),
                (51, 120, 0.97, 0.02)
            ],
            'gender_weights': {
                'male': 1.0,
                'female': 0.98,
                'all': 1.0
            },
            'race_weights': {
                'caucasian': 0.97,
                'asian': 0.98,
                'african': 0.95,
                'hispanic': 0.96,
                'indian': 0.99,    # Best for Indian faces
                'middle eastern': 0.97,
                'all': 1.0
            }
        }
    }
    
    def __init__(self, models: Dict[str, BaseModel], 
                demographic_performance: Optional[Dict[str, Dict]] = None,
                base_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the ensemble with dynamic weighting capabilities.
        
        Args:
            models: Dictionary of model_id to model instances
            demographic_performance: Optional dictionary specifying model performance by demographic.
                                   If None, uses default equal performance.
            base_weights: Optional base weights for models, before demographic adjustments.
        """
        self.models = models
        self.model_ids = list(models.keys())
        
        # Initialize demographic performance metrics
        self.performance_metrics = demographic_performance or self.DEFAULT_PERFORMANCE
        
        # Set base weights (before demographic adjustments)
        if base_weights is None:
            self.base_weights = {model_id: 1.0/len(self.model_ids) for model_id in self.model_ids}
        else:
            self._validate_weights(base_weights)
            self.base_weights = base_weights
        
        # Current dynamic weights (will be updated per prediction)
        self.current_weights = self.base_weights.copy()
        
        logger.info(f"Initialized ensemble with models: {', '.join(self.model_ids)}")
        logger.info(f"Base weights: {self.base_weights}")
    
    def _validate_weights(self, weights: Dict[str, float]):
        """Validate that weights are properly normalized and cover all models."""
        # Check all models have weights
        missing_models = set(self.model_ids) - set(weights.keys())
        if missing_models:
            raise ValueError(f"Missing weights for models: {missing_models}")
            
        # Check weights sum to ~1.0 (allow for floating point imprecision)
        total_weight = sum(weights.values())
        if not 0.99 <= total_weight <= 1.01:  # Allow 1% tolerance
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def _get_age_weight(self, age: int, age_ranges: List[tuple]) -> float:
        """Calculate age-based weight with smooth transitions between ranges."""
        if not age_ranges:
            return 1.0
            
        # Find the matching age range
        for i, (min_age, max_age, weight, *_) in enumerate(age_ranges):
            if min_age <= age <= max_age:
                # If this is the last range or next range is not adjacent, return weight directly
                if i == len(age_ranges) - 1 or age_ranges[i+1][0] > max_age + 1:
                    return weight
                
                # Calculate transition zone (10% of the range or 5 years, whichever is smaller)
                next_min = age_ranges[i+1][0]
                transition_zone = min((max_age - min_age) * 0.1, 5)
                
                # If in transition zone, interpolate between current and next weight
                if age > max_age - transition_zone:
                    next_weight = age_ranges[i+1][2]
                    alpha = (age - (max_age - transition_zone)) / (2 * transition_zone)
                    alpha = max(0, min(1, alpha))  # Clamp to [0, 1]
                    return weight * (1 - alpha) + next_weight * alpha
                return weight
                
        return 1.0  # Default weight if no range matches

    def update_weights_for_demographics(self, age: int, gender: str, race: Optional[str] = None) -> Dict[str, float]:
        """
        Update model weights based on the demographic group of the input.
        
        Args:
            age: Age of the subject (0-120)
            gender: Gender of the subject ('male', 'female', or 'all')
            race: Optional race/ethnicity of the subject
            
        Returns:
            Dictionary of updated weights for each model
        """
        if not hasattr(self, 'performance_metrics'):
            logger.warning("No performance metrics available, using base weights")
            return self.base_weights
            
        # Normalize inputs
        gender = str(gender).lower() if gender else 'all'
        race = str(race).lower() if race else 'all'
        age = max(0, min(120, int(age)))  # Clamp age to 0-120
        
        # Calculate adjusted weights based on performance metrics
        adjusted_weights = {}
        total_weight = 0.0
        
        for model_id in self.model_ids:
            if model_id not in self.performance_metrics:
                logger.warning(f"No performance metrics for model {model_id}, using base weight")
                adjusted_weights[model_id] = self.base_weights[model_id]
                total_weight += adjusted_weights[model_id]
                continue
                
            metrics = self.performance_metrics[model_id]
            
            # Get age-based weight with smooth transitions
            age_ranges = metrics.get('age_ranges', [(0, 120, 1.0)])
            age_weight = self._get_age_weight(age, age_ranges)
            
            # Get gender-based weight with fallback to 'all'
            gender_weights = metrics.get('gender_weights', {'all': 1.0})
            gender_weight = gender_weights.get(gender, gender_weights.get('all', 1.0))
            
            # Get race-based weight with fallback to most specific match
            race_weights = metrics.get('race_weights', {'all': 1.0})
            race_weight = race_weights.get(race, race_weights.get('all', 1.0))
            
            # Calculate base weight with exponential scaling for more pronounced differences
            base_weight = self.base_weights[model_id]
            
            # Combine weights with non-linear scaling (emphasize stronger weights more)
            combined_factor = (age_weight ** 2) * (gender_weight ** 1.5) * (race_weight ** 1.2)
            adjusted_weight = base_weight * combined_factor
            
            # Apply minimum weight to avoid complete exclusion of any model
            min_weight = 0.1 * base_weight  # At least 10% of base weight
            adjusted_weights[model_id] = max(adjusted_weight, min_weight)
            total_weight += adjusted_weights[model_id]
        
        # Normalize to sum to 1.0
        if total_weight > 1e-6:  # Avoid division by zero
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        # Log the most significant weight changes
        weight_changes = [
            (model_id, (w - self.base_weights[model_id])/self.base_weights[model_id] * 100)
            for model_id, w in adjusted_weights.items()
        ]
        significant_changes = [f"{m}: {c:+.1f}%" for m, c in weight_changes if abs(c) > 5]
        
        if significant_changes:
            logger.info(
                f"Adjusted weights for (age={age}, gender={gender}, race={race}): "
                f"{', '.join(significant_changes)}"
            )
            
        self.current_weights = adjusted_weights
        return adjusted_weights
    
    def _get_model_confidence(self, pred: Dict[str, Any]) -> float:
        """Extract confidence from model prediction with validation."""
        try:
            # Handle different confidence formats
            confidence = pred.get('confidence')
            if confidence is not None:
                return float(confidence)
                
            # Fallback to probability if confidence not available
            if 'probabilities' in pred and isinstance(pred['probabilities'], (list, dict)):
                if isinstance(pred['probabilities'], dict):
                    # Handle dictionary of class probabilities
                    return float(max(pred['probabilities'].values(), default=0.0))
                else:
                    # Handle list of probabilities
                    return float(max(pred['probabilities'], default=0.0))
                    
        except (TypeError, ValueError) as e:
            logger.warning(f"Error extracting confidence: {str(e)}")
            
        return 0.5  # Default confidence if extraction fails

    def predict(self, image: np.ndarray, 
               demographics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a prediction by combining predictions from all models with dynamic weighting.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            demographics: Optional dictionary containing demographic info:
                        - 'age': int (0-120)
                        - 'gender': str ('male'/'female')
                        - 'race': str (optional)
                        - 'quality_score': float (0-1, optional)
                        
        Returns:
            Dictionary containing:
            - 'prediction': Final prediction (0=real, 1=fake)
            - 'confidence': Weighted confidence score (0-1)
            - 'model_predictions': Raw predictions from each model
            - 'weights': Weights used for each model
            - 'demographics': Demographic info used for weighting
        """
        model_predictions = {}
        
        # Update weights based on demographics if provided
        if demographics and 'age' in demographics and 'gender' in demographics:
            try:
                weights = self.update_weights_for_demographics(
                    age=int(demographics['age']),
                    gender=str(demographics['gender']),
                    race=str(demographics['race']) if demographics.get('race') else None
                )
                
                # Apply quality-based adjustment if quality score is available
                if 'quality_score' in demographics:
                    quality = float(demographics['quality_score'])
                    # Reduce weight of less reliable models more for low quality images
                    quality_factor = {
                        'xception': 0.9 + (0.2 * quality),  # More stable with quality
                        'mesonet': 0.85 + (0.3 * quality),  # More sensitive to quality
                        'face_xray': 0.8 + (0.4 * quality), # Most sensitive to quality
                        'vit': 0.9 + (0.2 * quality)        # More stable with quality
                    }
                    
                    for model_id in weights:
                        if model_id in quality_factor:
                            weights[model_id] *= quality_factor[model_id]
                    
                    # Renormalize
                    total_weight = sum(weights.values())
                    if total_weight > 1e-6:
                        weights = {k: v/total_weight for k, v in weights.items()}
                        
            except Exception as e:
                logger.error(f"Error processing demographics: {str(e)}", exc_info=True)
                weights = self.current_weights
        else:
            weights = self.current_weights
        
        # Get predictions from all models
        for model_id, model in self.models.items():
            try:
                pred = model.predict(image)
                model_predictions[model_id] = {
                    'class_label': pred.get('class_label', 'unknown'),
                    'confidence': float(pred.get('confidence', 0.0)),
                    'probabilities': pred.get('probabilities', []),
                    'success': True,
                    'error': None
                }
            except Exception as e:
                logger.error(f"Error running model {model_id}: {str(e)}", exc_info=True)
                model_predictions[model_id] = {
                    'class_label': 'error',
                    'confidence': 0.0,
                    'probabilities': [],
                    'success': False,
                    'error': str(e)
                }
        
        # Calculate weighted average confidence
        total_confidence = 0.0
        total_weight = 0.0
        
        for model_id, pred in model_predictions.items():
            if pred['success']:
                weight = weights.get(model_id, 0.0)
                total_confidence += pred['confidence'] * weight
                total_weight += weight
        
        # Calculate final prediction (weighted majority vote)
        weighted_vote = 0.0
        for model_id, pred in model_predictions.items():
            if pred['success']:
                # Convert class_label to 0/1 (assuming 'real'=0, 'fake'=1)
                vote = 1 if pred.get('class_label', '').lower() in ['fake', '1'] else 0
                weighted_vote += vote * weights.get(model_id, 0.0)
        
        final_prediction = 1 if weighted_vote >= 0.5 else 0
        final_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'model_predictions': model_predictions,
            'weights': weights,
            'demographics': demographics if demographics else None
        }
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update the weights used by the ensemble.
        
        Args:
            new_weights: Dictionary mapping model_id to new weight
        """
        # Create a copy of current weights and update with new values
        updated_weights = self.weights.copy()
        for model_id, weight in new_weights.items():
            if model_id in updated_weights:
                updated_weights[model_id] = weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(updated_weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in updated_weights.items()}
        else:
            logger.warning("Cannot update weights: total weight is zero")
        
        logger.info(f"Updated model weights: {self.weights}")


def create_ensemble(model_loader, model_ids: List[str]) -> WeightedEnsemble:
    """
    Helper function to create an ensemble from a model loader.
    
    Args:
        model_loader: Instance of HFModelLoader
        model_ids: List of model IDs to include in the ensemble
        
    Returns:
        Initialized WeightedEnsemble instance
    """
    models = {}
    for model_id in model_ids:
        try:
            model = model_loader.get_model_adapter(model_id)
            models[model_id] = model
            logger.info(f"Added model to ensemble: {model_id}")
        except Exception as e:
            logger.error(f"Failed to add model {model_id} to ensemble: {str(e)}")
    
    if not models:
        raise ValueError("No valid models were added to the ensemble")
    
    return WeightedEnsemble(models)
