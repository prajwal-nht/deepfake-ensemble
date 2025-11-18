"""
Demographic Analysis for Deepfake Detection

This script analyzes model performance across different demographic groups
using DeepFace for demographic prediction.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from deepface import DeepFace
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib for better visualization
plt.style.use('seaborn')
sns.set_palette("pastel")

class DemographicAnalyzer:
    """Analyze deepfake detection performance across demographic groups"""
    
    def __init__(self, results_path: str):
        """
        Initialize the analyzer with path to evaluation results
        
        Args:
            results_path: Path to the JSON file containing evaluation results
        """
        self.results_path = Path(results_path)
        self.results = self._load_results()
        self.demographic_metrics = {
            'gender': defaultdict(list),
            'race': defaultdict(list),
            'age': defaultdict(list)
        }
    
    def _load_results(self) -> list:
        """Load evaluation results from JSON file"""
        with open(self.results_path, 'r') as f:
            return json.load(f)
    
    def analyze_demographics(self, image_path: str) -> dict:
        """
        Analyze demographics of a face image using DeepFace
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing demographic attributes
        """
        try:
            # Analyze image with DeepFace
            demography = DeepFace.analyze(
                img_path=image_path,
                actions=['age', 'gender', 'race'],
                enforce_detection=False,
                silent=True
            )
            
            # Return the first face analysis (assuming single face per image)
            if isinstance(demography, list):
                return demography[0]
            return demography
            
        except Exception as e:
            print(f"Error analyzing {image_path}: {str(e)}")
            return None
    
    def process_results(self):
        """Process evaluation results and analyze demographics"""
        print("🚀 Starting demographic analysis...")
        
        for i, result in enumerate(tqdm(self.results, desc="Processing images")):
            image_path = result.get('image_path')
            if not image_path or not os.path.exists(image_path):
                continue
                
            # Get demographic analysis
            demography = self.analyze_demographics(image_path)
            if not demography:
                continue
                
            # Get prediction and ground truth
            prediction = result.get('prediction', 0.5)  # Default to 0.5 if not available
            is_real = result.get('is_real', True)  # Default to True if not available
            
            # Store metrics by demographic group
            self._store_metrics(demography, prediction, is_real)
    
    def _store_metrics(self, demography: dict, prediction: float, is_real: bool):
        """Store metrics for each demographic group"""
        # Gender metrics
        gender = demography.get('dominant_gender', 'unknown').lower()
        self.demographic_metrics['gender'][gender].append({
            'prediction': prediction,
            'is_real': is_real,
            'correct': (prediction > 0.5) == is_real
        })
        
        # Race metrics
        race = demography.get('dominant_race', 'unknown').lower()
        self.demographic_metrics['race'][race].append({
            'prediction': prediction,
            'is_real': is_real,
            'correct': (prediction > 0.5) == is_real
        })
        
        # Age metrics (grouped by decades)
        age = int(demography.get('age', 0))
        age_group = f"{age // 10 * 10}-{age // 10 * 10 + 9}"
        self.demographic_metrics['age'][age_group].append({
            'prediction': prediction,
            'is_real': is_real,
            'correct': (prediction > 0.5) == is_real
        })
    
    def calculate_metrics(self) -> dict:
        """Calculate performance metrics per demographic group"""
        metrics = {}
        
        for category, groups in self.demographic_metrics.items():
            metrics[category] = {}
            for group, samples in groups.items():
                if not samples:
                    continue
                    
                # Calculate metrics
                correct = sum(1 for s in samples if s['correct'])
                total = len(samples)
                accuracy = correct / total if total > 0 else 0
                
                # Store metrics
                metrics[category][group] = {
                    'accuracy': accuracy,
                    'samples': total,
                    'avg_confidence': np.mean([s['prediction'] for s in samples])
                }
        
        return metrics
    
    def plot_metrics(self, metrics: dict, output_dir: str = 'demographic_plots'):
        """Generate and save plots of demographic metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        for category, groups in metrics.items():
            if not groups:
                continue
                
            # Create DataFrame for plotting
            df = pd.DataFrame([
                {'group': group, 'accuracy': data['accuracy'], 'samples': data['samples']}
                for group, data in groups.items()
            ])
            
            if df.empty:
                continue
                
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Create bar plot
            ax = sns.barplot(x='group', y='accuracy', data=df)
            
            # Add sample counts to the bars
            for i, p in enumerate(ax.patches):
                ax.annotate(
                    f"{p.get_height():.2f}\n(n={df['samples'].iloc[i]})",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10),
                    textcoords='offset points'
                )
            
            # Customize plot
            plt.title(f'Accuracy by {category.capitalize()}')
            plt.xlabel(category.capitalize())
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(output_dir, f'{category}_accuracy.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved {category} plot to {output_path}")

def main():
    # Configuration
    RESULTS_PATH = "evaluation_results/detailed_results.json"  # Update this path
    
    # Initialize analyzer
    analyzer = DemographicAnalyzer(RESULTS_PATH)
    
    # Process results and calculate metrics
    analyzer.process_results()
    metrics = analyzer.calculate_metrics()
    
    # Save metrics to file
    with open('demographic_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✅ Saved metrics to demographic_metrics.json")
    
    # Generate and save plots
    analyzer.plot_metrics(metrics)
    
    print("\n📊 Demographic Analysis Complete!")
    print("Check the 'demographic_plots' directory for visualizations.")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import deepface
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "deepface", "matplotlib", "seaborn", "pandas"])
    
    main()
