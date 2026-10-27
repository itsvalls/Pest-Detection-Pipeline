"""
Model evaluation module for generic image classification.
Handles testing, metrics calculation, and result analysis.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Evaluates trained models on test set.
    """
    
    def __init__(self, model: nn.Module, config, device: str = 'cuda'):
        """
        Args:
            model: Trained PyTorch model
            config: Configuration dictionary
            device: Device to use (cuda or cpu)
        """
        self.model = model
        self.config = config
        self.device = device
    
    def evaluate(self, test_loader: DataLoader, 
                class_to_idx: Dict[str, int]) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: DataLoader for test set
            class_to_idx: Mapping of class names to indices
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\n✓ Evaluating model on test set...")
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Testing")
            
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_preds, all_labels, all_probs, class_to_idx)
        
        return metrics, all_preds, all_labels, all_probs
    
    def _calculate_metrics(self, preds: np.ndarray, labels: np.ndarray, 
                          probs: np.ndarray, class_to_idx: Dict[str, int]) -> Dict:
        """Calculate evaluation metrics."""
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        # Overall metrics
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_metrics = {}
        for class_idx, class_name in idx_to_class.items():
            class_mask = labels == class_idx
            if class_mask.sum() > 0:
                class_acc = accuracy_score(
                    labels[class_mask],
                    preds[class_mask]
                )
                per_class_metrics[class_name] = {
                    'accuracy': float(class_acc),
                    'samples': int(class_mask.sum())
                }
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Classification report
        class_report = classification_report(
            labels, preds,
            target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
            zero_division=0,
            output_dict=True
        )
        
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
            },
            'per_class': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
        }
        
        return metrics
    
    def save_results(self, metrics: Dict, preds: np.ndarray, labels: np.ndarray,
                    class_to_idx: Dict[str, int]):
        """Save evaluation results."""
        results_dir = self.config.RESULTS_DIR
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        # Save metrics as JSON
        metrics_file = results_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"✓ Metrics saved to {metrics_file}")
        
        # Save predictions to CSV
        predictions_file = results_dir / 'predictions.csv'
        with open(predictions_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Ground Truth', 'Prediction', 'Correct'])
            for true_label, pred_label in zip(labels, preds):
                correct = true_label == pred_label
                writer.writerow([
                    idx_to_class[true_label],
                    idx_to_class[pred_label],
                    correct
                ])
        print(f"✓ Predictions saved to {predictions_file}")
        
        # Save confusion matrix plot
        cm = np.array(metrics['confusion_matrix'])
        self._plot_confusion_matrix(cm, idx_to_class, 
                                   results_dir / 'confusion_matrix.png')
        
        # Save classification report
        report_file = results_dir / 'classification_report.txt'
        with open(report_file, 'w') as f:
            f.write(classification_report(
                labels, preds,
                target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
                zero_division=0
            ))
        print(f"✓ Classification report saved to {report_file}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray, idx_to_class: Dict, 
                              save_path: Path):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(idx_to_class.values()),
                   yticklabels=list(idx_to_class.values()))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Confusion matrix plot saved to {save_path}")
    
    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics."""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Overall metrics
        overall = metrics['overall']
        print("\nOverall Metrics:")
        print(f"  Accuracy:  {overall['accuracy']:.4f}")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1 Score:  {overall['f1']:.4f}")
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(
                f"  {class_name}: Acc={class_metrics['accuracy']:.4f}, "
                f"Samples={class_metrics['samples']}"
            )
        
        print("="*50 + "\n")
