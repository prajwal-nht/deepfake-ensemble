# 5-Model Weighted Ensemble Evaluation System

## 🎯 Overview

This evaluation system implements a **5-model weighted ensemble** for deepfake detection with comprehensive AUROC and AUPRC analysis, based on your previous model performance data.

## 📊 Model Configuration

### Your 5-Model Ensemble:

1. **Xception** (`dima806/deepfake_vs_real_image_detection`)
   - Expected AUROC: 0.85, AUPRC: 0.866
   - Input: 299x299

2. **ResNet-50** (MesoNet alternative)
   - Expected AUROC: 0.78, AUPRC: 0.8  
   - Input: 224x224

3. **ViT** (Face X-Ray alternative)
   - Expected AUROC: 0.8, AUPRC: 0.82
   - Input: 224x224

4. **EfficientNet-V2**
   - Expected AUROC: 0.83, AUPRC: 0.845
   - Input: 384x384

5. **SigLIP** (GenConvIT alternative)
   - Expected AUROC: 0.8, AUPRC: 0.81
   - Input: 512x512

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install -r evaluation_requirements.txt
```

### 2. Prepare Your Dataset

Organize your test dataset in this structure:
```
your_dataset/
├── real/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── fake/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 3. Run Evaluation
```bash
python run_evaluation.py --dataset_path /path/to/your/dataset --output_dir ./results
```

### 4. Generate Visualizations
```bash
python visualize_results.py --results_path ./results/ensemble_evaluation_results.json --output_dir ./plots
```

## 📋 What You Need to Provide

To run the evaluation, I need:

1. **Test Dataset Path**: Directory with real/fake images
2. **Dataset Structure**: Confirm if your dataset follows the expected structure
3. **Sample Size**: How many images you want to test (for initial testing, 100-500 images per class)

## 📈 Output Files

The evaluation will generate:

### Results Files:
- `ensemble_evaluation_results.json` - Complete detailed results
- `ensemble_evaluation_results_summary.json` - Summary metrics
- `evaluation_log.txt` - Processing log

### Visualizations:
- `model_comparison.png` - AUROC/AUPRC bar charts
- `model_weights.png` - Pie chart of optimal weights
- `performance_vs_weight.png` - Scatter plot analysis
- `performance_table.png` - Summary table
- `roc_pr_curves.png` - ROC and PR curves

## 🔧 Customization

### Modify Dataset Loading
Edit the `prepare_test_data()` function in `ensemble_evaluation.py` to match your dataset structure:

```python
def prepare_test_data(dataset_path: str) -> List[Dict]:
    test_data = []
    dataset_path = Path(dataset_path)
    
    # Customize based on your structure
    # Example for CSV-based dataset:
    # df = pd.read_csv(dataset_path / "labels.csv")
    # for _, row in df.iterrows():
    #     test_data.append({
    #         'image_path': row['image_path'],
    #         'label': row['label']  # 0=real, 1=fake
    #     })
    
    return test_data
```

### Adjust Model Weights
Modify `FIVE_MODEL_CONFIG` in `ensemble_evaluation.py`:

```python
FIVE_MODEL_CONFIG = {
    "xception": {
        # ... existing config ...
        "weight": 0.3  # Increase weight for better models
    },
    # ... other models ...
}
```

## 📊 Expected Output Format

```json
{
  "ensemble_auroc": 0.8542,
  "ensemble_auprc": 0.8734,
  "total_samples": 1000,
  "individual_performance": {
    "xception": {
      "auroc": 0.8500,
      "auprc": 0.8660,
      "weight": 0.267
    },
    "resnet_mesonet": {
      "auroc": 0.7800,
      "auprc": 0.8000,
      "weight": 0.195
    }
    // ... other models
  }
}
```

## 🎯 Next Steps

1. **Provide your dataset path**
2. **Confirm dataset structure**
3. **Run initial evaluation with small sample**
4. **Review results and optimize weights**
5. **Run full evaluation**
6. **Generate final report**

## 🔍 Troubleshooting

### Common Issues:

1. **CUDA Memory Error**: Reduce batch size or use CPU
2. **Model Loading Error**: Check internet connection for Hugging Face models
3. **Dataset Path Error**: Verify dataset structure matches expected format

### Debug Mode:
```bash
python run_evaluation.py --dataset_path /path/to/dataset --sample_limit 50
```

## 📞 Ready to Start?

Please provide:
1. **Your test dataset path**
2. **Confirmation of dataset structure** (real/fake folders or CSV with labels)
3. **Preferred sample size** for initial testing

Then we can run the complete evaluation and get your AUROC/AUPRC results! 🚀