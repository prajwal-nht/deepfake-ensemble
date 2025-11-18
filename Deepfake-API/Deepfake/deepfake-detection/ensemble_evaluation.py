"""
5-Model Weighted Ensemble Evaluation System
Based on previous performance data for optimal model selection
"""
import numpy as np
import torch
import cv2
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
import logging
from ensemble_detector import EnsembleDeepfakeDetector
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 5-Model Configuration - Updated with working models
FIVE_MODEL_CONFIG = {
    "xception": {
        "name": "dima806/deepfake_vs_real_image_detection",
        "type": "huggingface",
        "input_size": 299,
        "description": "Xception-based deepfake detector (trained for deepfake)",
        "expected_auroc": 0.85,
        "expected_auprc": 0.866,
        "weight": 0.25
    },
    "vit_deepfake": {
        "name": "prithivMLmods/Deep-Fake-Detector-Model", 
        "type": "huggingface",
        "input_size": 224,
        "description": "ViT-based deepfake detector (trained for deepfake)",
        "expected_auroc": 0.80,
        "expected_auprc": 0.82,
        "weight": 0.22
    },
    "vit_wvolf": {
        "name": "Wvolf/ViT_Deepfake_Detection",
        "type": "huggingface", 
        "input_size": 224,
        "description": "ViT deepfake detector by Wvolf (trained for deepfake)",
        "expected_auroc": 0.80,
        "expected_auprc": 0.82,
        "weight": 0.22
    },
    "vit_deepfake_v2": {
        "name": "prithivMLmods/deepfake-detector-model-v1",
        "type": "huggingface",
        "input_size": 224,
        "description": "ViT-based deepfake detector v2 (trained for deepfake)", 
        "expected_auroc": 0.80,
        "expected_auprc": 0.81,
        "weight": 0.21
    },
    "efficientnet_b4": {
        "name": "google/efficientnet-b4",
        "type": "huggingface",
        "input_size": 380,
        "description": "EfficientNet-B4 (general model, adds diversity)",
        "expected_auroc": 0.70,  # Lower since not trained for deepfake
        "expected_auprc": 0.72,
        "weight": 0.10  # Lower weight for general model
    }
}

class WeightedEnsembleEvaluator:
    """Weighted ensemble evaluation with AUROC and AUPRC metrics"""
    
    def __init__(self, model_config: Dict = FIVE_MODEL_CONFIG):
        self.model_config = model_config
        self.detector = None
        self.results = {}
        
    def initialize_detector(self):
        """Initialize the ensemble detector with 5 models"""
        logger.info("Initializing 5-model ensemble detector...")
        self.detector = EnsembleDeepfakeDetector()
        
        # Override the MODEL_CONFIGS in detector
        self.detector.MODEL_CONFIGS = {
            name: {k: v for k, v in config.items() if k != 'weight' and k != 'expected_auroc' and k != 'expected_auprc'}
            for name, config in self.model_config.items()
        }
        
        self.detector.initialize()
        logger.info(f"Successfully initialized {len(self.model_config)} models")
        
    def evaluate_single_model(self, model_name: str, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate a single model on test data"""
        logger.info(f"Evaluating {model_name}...")
        
        y_true = []
        y_scores = []
        
        for sample in test_data:
            image_path = sample['image_path']
            true_label = sample['label']  # 0 = real, 1 = fake
            
            # Load and process image
            frame = cv2.imread(image_path)
            if frame is None:
                continue
                
            # Get prediction from ensemble detector
            result = self.detector.predict_frame(frame)
            
            if result['status'] == 'success' and result['faces']:
                # Get the specific model's prediction
                face = result['faces'][0]  # Use first face
                if model_name in face.get('models', {}):
                    model_result = face['models'][model_name]
                    if 'weighted_prob' in model_result and model_result['weighted_prob'] is not None:
                        y_true.append(true_label)
                        y_scores.append(model_result['weighted_prob'])
        
        if len(y_true) < 10:  # Need minimum samples
            return {'auroc': 0.0, 'auprc': 0.0, 'samples': len(y_true)}
            
        # Calculate metrics
        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
        
        return {
            'auroc': auroc,
            'auprc': auprc, 
            'samples': len(y_true),
            'y_true': y_true,
            'y_scores': y_scores
        }
    
    def calculate_optimal_weights(self, individual_results: Dict) -> Dict[str, float]:
        """Calculate optimal weights based on individual model performance"""
        logger.info("Calculating optimal ensemble weights...")
        
        # Weight based on AUROC performance
        auroc_scores = {name: results['auroc'] for name, results in individual_results.items()}
        total_auroc = sum(auroc_scores.values())
        
        if total_auroc == 0:
            # Equal weights if no valid scores
            return {name: 1.0/len(self.model_config) for name in self.model_config.keys()}
        
        # Normalize weights based on performance
        optimal_weights = {name: score/total_auroc for name, score in auroc_scores.items()}
        
        # Ensure weights sum to 1
        weight_sum = sum(optimal_weights.values())
        optimal_weights = {name: weight/weight_sum for name, weight in optimal_weights.items()}
        
        return optimal_weights
    
    def evaluate_weighted_ensemble(self, test_data: List[Dict], weights: Dict[str, float]) -> Dict[str, float]:
        """Evaluate the weighted ensemble on test data"""
        logger.info("Evaluating weighted ensemble...")
        
        y_true = []
        y_scores = []
        
        for sample in test_data:
            image_path = sample['image_path']
            true_label = sample['label']
            
            frame = cv2.imread(image_path)
            if frame is None:
                continue
                
            result = self.detector.predict_frame(frame)
            
            if result['status'] == 'success' and result['faces']:
                face = result['faces'][0]
                model_scores = []
                model_weights = []
                
                # Collect weighted scores from all models
                for model_name, weight in weights.items():
                    if model_name in face.get('models', {}):
                        model_result = face['models'][model_name]
                        if 'weighted_prob' in model_result and model_result['weighted_prob'] is not None:
                            model_scores.append(model_result['weighted_prob'])
                            model_weights.append(weight)
                
                if model_scores:
                    # Calculate weighted ensemble score
                    weighted_score = sum(score * weight for score, weight in zip(model_scores, model_weights))
                    weighted_score /= sum(model_weights)  # Normalize
                    
                    y_true.append(true_label)
                    y_scores.append(weighted_score)
        
        if len(y_true) < 10:
            return {'auroc': 0.0, 'auprc': 0.0, 'samples': len(y_true)}
        
        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
        
        return {
            'auroc': auroc,
            'auprc': auprc,
            'samples': len(y_true),
            'y_true': y_true,
            'y_scores': y_scores
        }
    
    def run_full_evaluation(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        logger.info("Starting full 5-model ensemble evaluation...")
        
        # Initialize detector
        self.initialize_detector()
        
        # Evaluate individual models
        individual_results = {}
        for model_name in self.model_config.keys():
            individual_results[model_name] = self.evaluate_single_model(model_name, test_data)
        
        # Calculate optimal weights
        optimal_weights = self.calculate_optimal_weights(individual_results)
        
        # Evaluate weighted ensemble
        ensemble_results = self.evaluate_weighted_ensemble(test_data, optimal_weights)
        
        # Compile final results
        final_results = {
            'individual_models': individual_results,
            'optimal_weights': optimal_weights,
            'ensemble_performance': ensemble_results,
            'model_config': self.model_config
        }
        
        return final_results
    
    def generate_report(self, results: Dict[str, Any], output_path: str = "ensemble_evaluation_report.json"):
        """Generate detailed evaluation report"""
        logger.info(f"Generating evaluation report: {output_path}")
        
        # Create summary
        summary = {
            'ensemble_auroc': results['ensemble_performance']['auroc'],
            'ensemble_auprc': results['ensemble_performance']['auprc'],
            'total_samples': results['ensemble_performance']['samples'],
            'individual_performance': {
                name: {
                    'auroc': res['auroc'],
                    'auprc': res['auprc'],
                    'weight': results['optimal_weights'][name]
                }
                for name, res in results['individual_models'].items()
            }
        }
        
        # Save detailed results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_path = output_path.replace('.json', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("5-MODEL WEIGHTED ENSEMBLE EVALUATION RESULTS")
        print("="*60)
        print(f"Ensemble AUROC: {summary['ensemble_auroc']:.4f}")
        print(f"Ensemble AUPRC: {summary['ensemble_auprc']:.4f}")
        print(f"Total Samples: {summary['total_samples']}")
        print("\nIndividual Model Performance:")
        print("-"*60)
        for name, perf in summary['individual_performance'].items():
            print(f"{name:20} | AUROC: {perf['auroc']:.4f} | AUPRC: {perf['auprc']:.4f} | Weight: {perf['weight']:.3f}")
        print("="*60)
        
        return summary

def prepare_test_data(dataset_path: str) -> List[Dict]:
    """Prepare test data from UADFV dataset"""
    test_data = []
    dataset_path = Path(dataset_path)
    
    print(f"📂 Loading dataset from: {dataset_path}")
    
    # Check for organized structure first
    if (dataset_path / "organized").exists():
        dataset_path = dataset_path / "organized"
    
    # Load real images
    real_dir = dataset_path / "real"
    if real_dir.exists():
        real_images = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.jpeg")) + list(real_dir.glob("*.png"))
        for img_path in real_images:
            test_data.append({
                'image_path': str(img_path),
                'label': 0,  # Real
                'category': 'real'
            })
        print(f"✅ Loaded {len(real_images)} real images")
    
    # Load fake images  
    fake_dir = dataset_path / "fake"
    if fake_dir.exists():
        fake_images = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.jpeg")) + list(fake_dir.glob("*.png"))
        for img_path in fake_images:
            test_data.append({
                'image_path': str(img_path),
                'label': 1,  # Fake
                'category': 'fake'
            })
        print(f"✅ Loaded {len(fake_images)} fake images")
    
    # If no organized structure, try to find images in subdirectories
    if not test_data:
        print("🔍 Searching for images in subdirectories...")
        for img_path in dataset_path.rglob("*.jpg"):
            path_str = str(img_path).lower()
            if any(keyword in path_str for keyword in ['real', 'original', 'authentic']):
                test_data.append({
                    'image_path': str(img_path),
                    'label': 0,
                    'category': 'real'
                })
            elif any(keyword in path_str for keyword in ['fake', 'deepfake', 'synthetic', 'manipulated']):
                test_data.append({
                    'image_path': str(img_path),
                    'label': 1,
                    'category': 'fake'
                })
    
    print(f"📊 Total samples loaded: {len(test_data)}")
    real_count = sum(1 for item in test_data if item['label'] == 0)
    fake_count = sum(1 for item in test_data if item['label'] == 1)
    print(f"📊 Real: {real_count}, Fake: {fake_count}")
    
    return test_data

if __name__ == "__main__":
    # Example usage
    evaluator = WeightedEnsembleEvaluator()
    
    # You'll need to provide your test dataset path
    test_data = prepare_test_data("path/to/your/test/dataset")
    
    # Run evaluation
    results = evaluator.run_full_evaluation(test_data)
    
    # Generate report
    summary = evaluator.generate_report(results)