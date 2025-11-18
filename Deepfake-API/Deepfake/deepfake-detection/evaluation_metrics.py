"""
Comprehensive evaluation metrics for deepfake detection models (Refactored)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os


def calculate_metrics(y_true, y_scores, threshold=0.5):
    """
    Calculate classification metrics and curves.
    
    Args:
        y_true: List/array of true labels (0 or 1)
        y_scores: List/array of prediction scores (0-1)
        threshold: Decision threshold for binary classification
        
    Returns:
        Dictionary containing metrics, confusion matrix, and curve data
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    # Confusion matrix
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    # Base metrics
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # ROC and PR curves
    thresholds = np.linspace(0, 1, 101)
    tpr, fpr, precision_list, recall_list = [], [], [], []

    for thresh in thresholds:
        y_pred_thresh = (y_scores >= thresh).astype(int)
        tp_ = np.sum((y_pred_thresh == 1) & (y_true == 1))
        tn_ = np.sum((y_pred_thresh == 0) & (y_true == 0))
        fp_ = np.sum((y_pred_thresh == 1) & (y_true == 0))
        fn_ = np.sum((y_pred_thresh == 0) & (y_true == 1))

        tpr.append(tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0)
        fpr.append(fp_ / (fp_ + tn_) if (fp_ + tn_) > 0 else 0)

        p = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
        r = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
        precision_list.append(p)
        recall_list.append(r)

    # AUROC & AUPRC
    # Ensure x is in ascending order for proper integration
    auroc = np.abs(np.trapezoid(tpr, fpr))
    auprc = np.abs(np.trapezoid(precision_list, recall_list))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'auroc': auroc,
        'auprc': auprc,
        'thresholds': thresholds.tolist(),
        'tpr': tpr,
        'fpr': fpr,
        'precision_curve': precision_list,
        'recall_curve': recall_list,
        'true_labels': y_true.tolist(),
        'predictions': y_scores.tolist(),
    }


def plot_roc_curve(metrics_dict, output_dir='evaluation_plots'):
    """Plot ROC curve using precomputed metrics"""
    plt.figure(figsize=(10, 8))

    for model_name, metrics in metrics_dict.items():
        if 'tpr' in metrics and 'fpr' in metrics:
            plt.plot(metrics['fpr'], metrics['tpr'],
                     label=f"{model_name} (AUROC = {metrics['auroc']:.4f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/roc_curve.png')
    plt.close()


def plot_pr_curve(metrics_dict, output_dir='evaluation_plots'):
    """Plot Precision-Recall curve using precomputed metrics"""
    plt.figure(figsize=(10, 8))

    for model_name, metrics in metrics_dict.items():
        if 'precision_curve' in metrics and 'recall_curve' in metrics:
            plt.plot(metrics['recall_curve'], metrics['precision_curve'],
                     label=f"{model_name} (AUPRC = {metrics['auprc']:.4f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/pr_curve.png')
    plt.close()


def generate_evaluation_report(metrics_dict, output_file='evaluation_report.json'):
    """Generate a comprehensive evaluation report"""
    report = {}

    for model_name, metrics in metrics_dict.items():
        report[model_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'true_positives': metrics['true_positives'],
            'true_negatives': metrics['true_negatives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives'],
            'auroc': metrics['auroc'],
            'auprc': metrics['auprc'],
        }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=4)

    return report


def print_evaluation_summary(metrics_dict):
    """Print a summary of evaluation metrics"""
    print("\n📊 MODEL EVALUATION SUMMARY")
    print("=" * 120)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} "
          f"{'F1-Score':<10} {'TP':<5} {'FP':<5} {'FN':<5} {'TN':<5} {'AUROC':<10} {'AUPRC':<10}")
    print("-" * 120)

    for model_name, metrics in metrics_dict.items():
        print(f"{model_name:<20} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f} "
              f"{metrics['true_positives']:<5} "
              f"{metrics['false_positives']:<5} "
              f"{metrics['false_negatives']:<5} "
              f"{metrics['true_negatives']:<5} "
              f"{metrics['auroc']:<10.4f} "
              f"{metrics['auprc']:<10.4f}")

    print("=" * 120)


if __name__ == "__main__":
    from simple_real_evaluation import evaluate_on_real_dataset, FIVE_MODEL_CONFIG

    results = evaluate_on_real_dataset("datasets/uadfv/organized")

    if results:
        metrics_dict = {}

        # Individual models
        for model_name in FIVE_MODEL_CONFIG.keys():
            model_data = results['individual_models'][model_name]
            if 'true_labels' in model_data and 'predictions' in model_data:
                metrics_dict[model_name] = calculate_metrics(
                    model_data['true_labels'],
                    model_data['predictions']
                )

        # Ensemble (simple average)
        model_names = list(FIVE_MODEL_CONFIG.keys())
        if model_names:
            true_labels = np.array(results['individual_models'][model_names[0]]['true_labels'])
            preds = [np.array(results['individual_models'][m]['predictions'])
                     for m in model_names if 'predictions' in results['individual_models'][m]]
            if preds:
                ensemble_preds = np.mean(preds, axis=0)
                metrics_dict['ensemble'] = calculate_metrics(true_labels, ensemble_preds)

        # Generate plots and reports
        plot_roc_curve(metrics_dict)
        plot_pr_curve(metrics_dict)
        report = generate_evaluation_report(metrics_dict)
        print_evaluation_summary(metrics_dict)

        print(f"\n✅ Evaluation complete!")
        print(f"📊 ROC and PR curves saved to: evaluation_plots/")
        print(f"📄 Full report saved to: evaluation_report.json")
