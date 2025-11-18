"""
Hugging Face Model Loader for Parallel Deepfake Detection
"""
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from typing import Dict, Any, Optional, List, Union
import logging

# Import the new model interface
from .model_interfaces import BaseModel, create_model_adapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HFModelLoader:
    """
    Loads and manages multiple Hugging Face models for parallel inference.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            device: Device to load models on ('cuda' or 'cpu').
                   If None, will use CUDA if available.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        
        # Updated model configurations with better defaults and fixes
        self.model_configs = {
            'xception': {
                'model_name': 'dima806/deepfake_vs_real_image_detection',
                'description': 'Xception-based deepfake detector',
                'requires_special_handling': False,
                'num_labels': 2,
                'id2label': {0: 'real', 1: 'fake'}
            },
            'mesonet': {
                'model_name': 'prithivMLmods/Deep-Fake-Detector-v2-Model',
                'description': 'MesoNet-based deepfake detector',
                'requires_special_handling': False,
                'num_labels': 2,
                'id2label': {0: 'real', 1: 'fake'}
            },
            'face_xray': {
                'model_name': 'microsoft/resnet-50',
                'description': 'Face X-ray detection model',
                'requires_special_handling': True,
                'num_labels': 2,
                'id2label': {0: 'real', 1: 'fake'}
            },
            'efficientnet': {
                'model_name': 'nateraw/tf-efficientnet-b4-1k',  # Using timm model with safetensors
                'description': 'EfficientNet for deepfake detection',
                'requires_special_handling': True,
                'num_labels': 1000,
                'from_timm': True  # Flag to indicate this is a timm model
            },
            'vit': {
                'model_name': 'google/vit-base-patch16-224',
                'description': 'Vision Transformer for deepfake detection',
                'requires_special_handling': False,
                'num_labels': 1000,
                'id2label': {0: 'real', 1: 'fake'},
                'class_index_map': {0: 0, 1: 1}  # Map ImageNet classes to binary
            }
        }
        
        # Load all models
        self._load_all_models()
        
    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific model.
        
        Args:
            model_id: ID of the model to get config for
            
        Returns:
            Dict containing model configuration or None if not found
        """
        return self.model_configs.get(model_id)
    
    def _load_model(self, model_name: str) -> bool:
        """
        Load a single model and its processor with detailed error handling and logging.
        
        Args:
            model_name: Name of the model to load from Hugging Face hub
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        model_id = model_name.split('/')[-1].lower()
        logger.info(f"\n{'='*80}")
        logger.info(f"ATTEMPTING TO LOAD MODEL: {model_id} ({model_name})")
        logger.info(f"{'='*80}")
        
        # Get model config
        model_config = None
        for config_id, config in self.model_configs.items():
            if config['model_name'] == model_name:
                model_config = config
                break
        
        if not model_config:
            logger.error(f"No configuration found for model: {model_name}")
            return False
            
        # Log model configuration being loaded
        logger.info(f"Model configuration for {model_id}:")
        logger.info(f"- Model name: {model_name}")
        logger.info(f"- Description: {model_config.get('description', 'No description')}")
        logger.info(f"- Special handling: {model_config.get('requires_special_handling', False)}")
        
        try:
            # Special handling for specific models
            if model_config.get('requires_special_handling', False):
                if 'efficientnet' in model_id:
                    logger.info(f"Using specialized loader for EfficientNet model: {model_id}")
                    return self._load_efficientnet_model(model_name, model_id)
                elif 'resnet' in model_id or 'face_xray' in model_id:
                    logger.info(f"Using specialized loader for ResNet model: {model_id}")
                    return self._load_face_xray_model(model_name, model_id)
            
            logger.info(f"Using default model loading for {model_id}")
            return self._load_default_model(model_name, model_id, model_config)
            
        except Exception as e:
            logger.error(f"Error in model loading pipeline for {model_id}: {str(e)}")
            return False
        
        try:
            # Log device information
            logger.info(f"Using device: {self.device}")
            if 'cuda' in self.device:
                logger.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                    logger.info(f"CUDA device memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
            
            # Try loading with default parameters first
            logger.info(f"Loading model: {model_name}")
            try:
                model = AutoModelForImageClassification.from_pretrained(
                    model_name,
                    trust_remote_code=True,  # Needed for some custom models
                    resume_download=True,    # Resume interrupted downloads
                    local_files_only=False,  # Allow downloading if not in cache
                    device_map="auto"        # Let transformers handle device placement
                )
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                logger.info("Trying with device_map=None...")
                model = AutoModelForImageClassification.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    resume_download=True,
                    local_files_only=False,
                    device_map=None  # Try without device_map
                )
            
            logger.info(f"Model loaded successfully, loading processor...")
            try:
                processor = AutoFeatureExtractor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    resume_download=True,
                    local_files_only=False
                )
            except Exception as e:
                logger.error(f"Error loading processor for {model_name}: {str(e)}")
                logger.info("Trying with default processor...")
                # Try with a default processor if the specific one fails
                from transformers import ViTFeatureExtractor
                processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            
            # Move model to device if not already done by device_map
            if not hasattr(model, 'device'):
                logger.info(f"Moving model to {self.device}...")
                model = model.to(self.device)
            model = model.eval()
            
            # Get the original model_id from config (not the derived model_id)
            original_model_id = None
            for config_id, config in self.model_configs.items():
                if config['model_name'] == model_name:
                    original_model_id = config_id
                    break
            
            if original_model_id is None:
                logger.warning(f"Could not find original model_id for {model_name}, using derived ID: {model_id}")
                original_model_id = model_id
            
            # Store model and processor with original model_id
            self.models[original_model_id] = model
            self.processors[original_model_id] = processor
            
            # Log model architecture and parameters
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"✅ Successfully loaded model: {original_model_id} ({model_name}) to {self.device}")
            logger.info(f"Model architecture: {type(model).__name__}")
            logger.info(f"Total parameters: {total_params:,}")
            device = next(model.parameters()).device if hasattr(model, 'parameters') and len(list(model.parameters())) > 0 else 'unknown'
            logger.info(f"Model device: {device}")
            
            return True
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Failed to load model {model_name}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("Traceback:")
            logger.error(traceback.format_exc())
            
            # Log specific error details
            error_msg = str(e).lower()
            if "cuda out of memory" in error_msg:
                logger.error("CUDA out of memory error detected")
                if torch.cuda.is_available():
                    logger.error(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                    logger.error(f"CUDA memory cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
            elif "no module named" in error_msg:
                logger.error("Missing dependency detected. Please install the required package.")
            elif "couldn't connect to" in error_msg or "timed out" in error_msg:
                logger.error("Network connection issue. Please check your internet connection.")
            
            logger.error(f"Skipping model {model_id} due to error")
            return False
    
    def _load_default_model(self, model_name: str, model_id: str, model_config: dict) -> bool:
        """
        Load a model using the default Hugging Face pipeline.
        
        Args:
            model_name: Name of the model to load
            model_id: Short ID for the model
            model_config: Model configuration dictionary
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        from transformers import AutoModelForImageClassification, AutoFeatureExtractor
        import torch
        
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Try loading with device_map='auto' first
            try:
                model = AutoModelForImageClassification.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    resume_download=True,
                    num_labels=model_config.get('num_labels', 2)
                ).to(self.device)
            except Exception as e:
                logger.warning(f"Device map 'auto' failed, falling back to manual device placement: {str(e)}")
                model = AutoModelForImageClassification.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    resume_download=True,
                    num_labels=model_config.get('num_labels', 2)
                ).to(self.device)
            
            # Load processor
            try:
                processor = AutoFeatureExtractor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    resume_download=True
                )
            except Exception as e:
                logger.warning(f"Could not load feature extractor, using default: {str(e)}")
                from transformers import ViTFeatureExtractor
                processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            
            # Store model and processor
            config_id = next((k for k, v in self.model_configs.items() 
                            if v['model_name'] == model_name), model_id)
            self.models[config_id] = model.eval()
            self.processors[config_id] = processor
            
            logger.info(f"✅ Successfully loaded model: {config_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def _load_efficientnet_model(self, model_name: str, model_id: str) -> bool:
        """
        Specialized loader for EfficientNet models.
        
        Args:
            model_name: Name of the model to load
            model_id: Short ID for the model
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification
            import torch
            
            logger.info(f"Loading EfficientNet model: {model_name}")
            
            # Try loading with timm first (more reliable for EfficientNet)
            try:
                import timm
                model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2)
                model = model.to(self.device).eval()
                logger.info("Loaded EfficientNet using timm")
            except ImportError:
                logger.warning("timm not available, trying with transformers...")
                model = AutoModelForImageClassification.from_pretrained(
                    'google/efficientnet-b4',
                    num_labels=2,
                    trust_remote_code=True
                ).to(self.device).eval()
            
            # Load processor
            try:
                processor = AutoFeatureExtractor.from_pretrained(
                    'google/efficientnet-b4',
                    trust_remote_code=True
                )
            except Exception as e:
                logger.warning(f"Could not load feature extractor, using default: {str(e)}")
                from transformers import ViTFeatureExtractor
                processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            
            # Store model and processor
            self.models[model_id] = model
            self.processors[model_id] = processor
            
            logger.info(f"✅ Successfully loaded EfficientNet model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load EfficientNet model {model_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _load_face_xray_model(self, model_name: str, model_id: str) -> bool:
        """
        Specialized loader for Face X-ray models.
        
        Args:
            model_name: Name of the model to load
            model_id: Short ID for the model
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            from transformers import AutoFeatureExtractor, ResNetForImageClassification
            import torch
            
            logger.info(f"Loading Face X-ray model: {model_name}")
            
            # Load model
            model = ResNetForImageClassification.from_pretrained(
                model_name,
                num_labels=2,  # Binary classification: real or fake
                trust_remote_code=True
            )
            model = model.to(self.device).eval()
            
            # Load processor
            try:
                processor = AutoFeatureExtractor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            except Exception as e:
                logger.warning(f"Could not load feature extractor, using default: {str(e)}")
                from transformers import ViTFeatureExtractor
                processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            
            # Store model and processor
            self.models[model_id] = model
            self.processors[model_id] = processor
            
            logger.info(f"✅ Successfully loaded Face X-ray model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Face X-ray model {model_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _load_all_models(self) -> None:
        """
        Load all configured models with detailed error reporting and memory management.
        
        This method attempts to load each model one by one, with error handling and
        memory management between each load attempt.
        """
        from transformers.utils import logging as tf_logging
        
        # Enable more verbose logging from transformers
        tf_logging.set_verbosity_info()
        logger.info("\n" + "="*80)
        logger.info("STARTING MODEL LOADING PROCESS")
        logger.info("="*80)
        
        # Log system information
        logger.info(f"System device: {self.device}")
        if 'cuda' in self.device and torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
        
        # Track loaded models and failures
        loaded_models = []
        failed_models = []
        
        for model_id, config in self.model_configs.items():
            model_name = config['model_name']
            
            # Try loading the model
            success = self._load_model(model_name)
            
            if success:
                loaded_models.append(model_id)
                logger.info(f"✅ Successfully loaded model: {model_id} ({model_name})")
                
                # Clean up memory between model loads
                if 'cuda' in self.device and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"Freed GPU memory. Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            else:
                failed_models.append((model_id, model_name))
                logger.error(f"❌ Failed to load model: {model_id} ({model_name})")
        
        # Log summary
        logger.info("\n" + "="*80)
        logger.info("MODEL LOADING SUMMARY")
        logger.info("="*80)
        logger.info(f"Total models configured: {len(self.model_configs)}")
        logger.info(f"Successfully loaded: {len(loaded_models)} models")
        logger.info(f"Failed to load: {len(failed_models)} models")
        
        if loaded_models:
            logger.info("\nSuccessfully loaded models:\n- " + "\n- ".join(loaded_models))
            
        if failed_models:
            failed_models_str = "\n- " + "\n- ".join([f"{m[0]} ({m[1]})" for m in failed_models])
            logger.error("\nFailed to load models:" + failed_models_str)
            
        if not loaded_models:
            raise RuntimeError("Failed to load any models. Check logs for details.")
        
        logger.info("\n" + "="*80)
        logger.info("MODEL LOADING COMPLETED")
        logger.info("="*80)
    
    def get_model(self, model_id: str) -> Any:
        """Get a loaded model by ID."""
        model_id = model_id.lower()
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found. Available models: {list(self.models.keys())}")
        return self.models[model_id]
    
    def get_processor(self, model_id: str) -> Any:
        """Get a processor by model ID."""
        model_id = model_id.lower()
        if model_id not in self.processors:
            raise ValueError(f"Processor for {model_id} not found.")
        return self.processors[model_id]
    
    def _load_face_xray_model(self, model_name: str, model_id: str) -> bool:
        """Special handling for the Face X-ray model (Wvolf/ViT_Deepfake_Detection)."""
        try:
            from transformers import ViTForImageClassification, ViTFeatureExtractor
            
            logger.info(f"Loading Face X-ray model: {model_name}")
            
            # Load model with specific parameters for this architecture
            model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=2,  # Binary classification (real vs fake)
                ignore_mismatched_sizes=True,
                local_files_only=False,
                trust_remote_code=True
            )
            
            # Load processor
            processor = ViTFeatureExtractor.from_pretrained(
                'google/vit-base-patch16-224',  # Use standard ViT processor
                size=224,
                do_resize=True,
                do_normalize=True,
                return_tensors='pt'
            )
            
            # Move model to device
            model = model.to(self.device).eval()
            
            # Get the original model_id from config (not the derived model_id)
            original_model_id = None
            for config_id, config in self.model_configs.items():
                if config['model_name'] == model_name:
                    original_model_id = config_id
                    break
            
            if original_model_id is None:
                logger.warning(f"Could not find original model_id for {model_name}, using derived ID: {model_id}")
                original_model_id = model_id
            
            # Store model and processor with original model_id
            self.models[original_model_id] = model
            self.processors[original_model_id] = processor
            
            logger.info(f"✅ Successfully loaded Face X-ray model: {original_model_id} ({model_name})")
            logger.info(f"Model architecture: {model.__class__.__name__}")
            logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            logger.info(f"Model device: {next(model.parameters()).device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Face X-ray model {model_name}")
            logger.error(f"Error: {str(e)}")
            logger.exception("Full traceback:")
            return False
    
    def _load_efficientnet_model(self, model_name: str):
        """Special loader for EfficientNet models using timm."""
        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            from torchvision import transforms
            
            logger.info(f"Loading EfficientNet model: {model_name}")
            
            # Load model from timm
            model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=1000,
                checkpoint_path='',
                exportable=True,
                scriptable=True
            )
            model.eval()
            model = model.to(self.device)
            
            # Get model config and create transform
            config = resolve_data_config({}, model=model)
            transform = create_transform(
                input_size=config['input_size'],
                is_training=False,
                mean=config['mean'],
                std=config['std']
            )
            
            # Ensure transform includes ToTensor and Normalize
            if not any(isinstance(t, transforms.Normalize) for t in transform.transforms):
                transform = transforms.Compose([
                    *transform.transforms,
                    transforms.Normalize(mean=config['mean'], std=config['std'])
                ])
            model = model.to(self.device).eval()
            
            # Get the original model_id from config (not the derived model_id)
            original_model_id = None
            for config_id, config in self.model_configs.items():
                if config['model_name'] == model_name:
                    original_model_id = config_id
                    break
            
            if original_model_id is None:
                logger.warning(f"Could not find original model_id for {model_name}, using derived ID: {model_id}")
                original_model_id = model_id
            
            # Store model and processor with original model_id
            self.models[original_model_id] = model
            self.processors[original_model_id] = processor
            
            logger.info(f"✅ Successfully loaded EfficientNet model: {original_model_id} ({model_name})")
            logger.info(f"Model architecture: {model.__class__.__name__}")
            logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            logger.info(f"Model device: {next(model.parameters()).device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load EfficientNet model {model_name}")
            logger.error(f"Error: {str(e)}")
            logger.exception("Full traceback:")
            return False
    
    def get_available_models(self) -> Dict[str, str]:
        """
        Get a dictionary of available model IDs and their descriptions.
        
        Returns:
            Dict mapping model IDs to their descriptions
        """
        result = {}
        
        # First check which models are actually loaded
        loaded_model_ids = list(self.models.keys())
        logger.info(f"Loaded model IDs: {loaded_model_ids}")
        logger.info(f"Model configs: {list(self.model_configs.keys())}")
        
        for model_id, config in self.model_configs.items():
            # Try matching by original model_id first
            if model_id in self.models:
                result[model_id] = config.get('description', 'No description')
            else:
                # Fallback to matching by model name
                model_name = config['model_name'].split('/')[-1].lower().replace('_', '-')
                if model_name in self.models:
                    result[model_id] = config.get('description', 'No description')
        
        logger.info(f"Available models after matching: {list(result.keys())}")
        return result
        
    def get_model_adapter(self, model_id: str) -> BaseModel:
        """
        Get a model adapter that implements the standard interface.
        
        Args:
            model_id: ID of the model to get
            
        Returns:
            Model adapter implementing BaseModel interface
        """
        model_id = model_id.lower()
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found. Available models: {list(self.models.keys())}")
            
        model = self.models[model_id]
        processor = self.processors.get(model_id, self.processors.get('default'))
        
        if processor is None:
            raise ValueError(f"No processor found for model {model_id}")
            
        return create_model_adapter(model, processor, model_id, self.device)

# Example usage
if __name__ == "__main__":
    # Initialize the model loader
    loader = HFModelLoader()
    
    # Print available models
    print("\nAvailable models:")
    for model_id, desc in loader.get_available_models().items():
        print(f"- {model_id}: {desc}")
    
    # Test model loading
    if 'xception' in loader.models:
        print("\nXception model loaded successfully!")
        print(f"Device: {next(loader.models['xception'].parameters()).device}")
