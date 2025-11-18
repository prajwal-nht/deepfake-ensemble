"""
Standardized interfaces for deepfake detection models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base interface that all deepfake detection models must implement."""
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on an input image.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            
        Returns:
            Dictionary containing:
            - 'logits': Raw model outputs (numpy array)
            - 'probabilities': Class probabilities (numpy array)
            - 'class_label': Predicted class (str)
            - 'confidence': Prediction confidence (float)
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Preprocess an image for the model.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Dictionary of model inputs (tensors on correct device)
        """
        pass

class HFModelWrapper(BaseModel):
    """Adapter for Hugging Face models to use the standard interface."""
    
    def __init__(self, model, processor, model_id: str, device: Optional[str] = None):
        """
        Initialize the wrapper.
        
        Args:
            model: Loaded Hugging Face model
            processor: Model's feature extractor/processor
            model_id: Identifier for the model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Configure based on model type
        self._setup_model_specifics()
    
    def _setup_model_specifics(self):
        """Configure model-specific settings."""
        # Default configuration
        self.id2label = getattr(self.model.config, 'id2label', {0: 'real', 1: 'fake'})
        self.num_labels = getattr(self.model.config, 'num_labels', 2)
        
        # Special handling for specific model types
        if 'efficientnet' in self.model_id:
            self._setup_efficientnet()
        elif 'xception' in self.model_id:
            self._setup_xception()
        elif 'mesonet' in self.model_id.lower():
            self._setup_mesonet()
    
    def _setup_efficientnet(self):
        """Configure EfficientNet specific settings."""
        self.id2label = {0: 'real', 1: 'fake'}
    
    def _setup_xception(self):
        """Configure Xception specific settings."""
        self.id2label = {0: 'real', 1: 'fake'}
    
    def _setup_mesonet(self):
        """Configure MesoNet specific settings."""
        self.id2label = {0: 'real', 1: 'fake'}
    
    def preprocess(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """Preprocess image for the model."""
        try:
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
                
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Move to device
            return {k: v.to(self.device) for k, v in inputs.items()}
            
        except Exception as e:
            logger.error(f"Error preprocessing image for {self.model_id}: {str(e)}")
            raise
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run inference on an input image."""
        try:
            # Preprocess
            inputs = self.preprocess(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get logits and probabilities
            if hasattr(outputs, 'logits'):
                logits = outputs.logits.cpu().numpy()
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            else:
                logits = outputs.logits
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            # Get prediction
            pred_idx = np.argmax(probs, axis=-1)[0]
            confidence = float(probs[0, pred_idx])
            class_label = self.id2label.get(int(pred_idx), f"class_{pred_idx}")
            
            return {
                'logits': logits,
                'probabilities': probs,
                'class_label': class_label,
                'confidence': confidence,
                'model_id': self.model_id
            }
            
        except Exception as e:
            logger.error(f"Error running {self.model_id} prediction: {str(e)}")
            raise

def create_model_adapter(model, processor, model_id: str, device: Optional[str] = None) -> BaseModel:
    """
    Factory function to create the appropriate model adapter.
    
    Args:
        model: Loaded model
        processor: Model's feature extractor/processor
        model_id: Identifier for the model
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        Model adapter implementing BaseModel interface
    """
    return HFModelWrapper(model, processor, model_id, device)
