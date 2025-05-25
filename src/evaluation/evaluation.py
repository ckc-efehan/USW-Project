"""
Evaluation module for 4-day work week argument analysis.

This module handles:
1. Calculating evaluation metrics (F1-score, accuracy)
2. Performing error analysis
3. Visualizing results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluator for argument classification models."""
    
    def __init__(self, output_dir='data/processed'):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir (str): Directory to save evaluation results
        """
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(self, predictions_df, true_label_column='label', pred_label_column='final'):
        """
        Evaluate model predictions against true labels.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions and true labels
            true_label_column (str): Name of the column with true labels
            pred_label_column (str): Name of the column with predicted labels
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating model predictions for {len(predictions_df)} samples")
        
        # Check if required columns exist
        if true_label_column not in predictions_df.columns:
            logger.error(f"True label column '{true_label_column}' not found in DataFrame")
            return None
        
        if pred_label_column not in predictions_df.columns:
            logger.error(f"Prediction column '{pred_label_column}' not found in DataFrame")
            return None
        
        # Get true labels and predictions
        y_true = predictions_df[true_label_column].values
        y_pred = predictions_df[pred_label_column].values
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate F1 scores for each class
        f1_pro = f1_score(y_true, y_pred, labels=['pro'], average='micro')
        f1_contra = f1_score(y_true, y_pred, labels=['contra'], average='micro')
        f1_neutral = f1_score(y_true, y_pred, labels=['neutral'], average='micro')
        
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['pro', 'contra', 'neutral'])
        
        # Store results
        metrics = {
            'accuracy': accuracy,
            'f1_pro': f1_pro,
            'f1_contra': f1_contra,
            'f1_neutral': f1_neutral,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        logger.info(f"Evaluation results: Accuracy = {accuracy:.4f}, F1 (pro) = {f1_pro:.4f}, F1 (contra) = {f1_contra:.4f}, F1 (neutral) = {f1_neutral:.4f}")
        
        return metrics
    
    def error_analysis(self, predictions_df, true_label_column='label', pred_label_column='final', text_column='sentence'):
        """
        Perform error analysis on model predictions.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame with predictions and true labels
            true_label_column (str): Name of the column with true labels
            pred_label_column (str): Name of the column with predicted labels
            text_column (str): Name of the column with text data
            
        Returns:
            pd.DataFrame: DataFrame with error analysis
        """
        logger.info("Performing error analysis")
        
        # Check if required columns exist
        required_columns = [true_label_column, pred_label_column, text_column]
        for col in required_columns:
            if col not in predictions_df.columns:
                logger.error(f"Required column '{col}' not found in DataFrame")
                return None
        
        # Find misclassified samples
        misclassified = predictions_df[predictions_df[true_label_column] != predictions_df[pred_label_column]].copy()
        
        # Add error type column
        misclassified['error_type'] = misclassified.apply(
            lambda row: f"{row[true_label_column]} -> {row[pred_label_column]}", axis=1
        )
        
        # Group by error type
        error_counts = misclassified['error_type'].value_counts().reset_index()
        error_counts.columns = ['error_type', 'count']
        
        # Calculate error rate
        error_rate = len(misclassified) / len(predictions_df) * 100
        
        logger.info(f"Found {len(misclassified)} misclassified samples ({error_rate:.2f}% error rate)")
        
        # Save error analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'error_analysis_{timestamp}.csv')
        misclassified.to_csv(output_file, index=False)
        
        logger.info(f"Saved error analysis to {output_file}")
        
        return misclassified, error_counts
    
    def visualize_results(self, metrics, output_prefix='evaluation'):
        """
        Visualize evaluation results.
        
        Args:
            metrics (dict): Evaluation metrics from evaluate()
            output_prefix (str): Prefix for output files
            
        Returns:
            None
        """
        logger.info("Visualizing evaluation results")
        
        # Create timestamp for file names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['pro', 'contra', 'neutral'],
            yticklabels=['pro', 'contra', 'neutral']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save confusion matrix plot
        cm_file = os.path.join(self.output_dir, f'{output_prefix}_confusion_matrix_{timestamp}.png')
        plt.savefig(cm_file)
        plt.close()
        
        # Plot F1 scores
        plt.figure(figsize=(8, 6))
        f1_scores = [metrics['f1_pro'], metrics['f1_contra'], metrics['f1_neutral']]
        sns.barplot(x=['Pro', 'Contra', 'Neutral'], y=f1_scores)
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.title('F1 Scores by Class')
        
        # Save F1 scores plot
        f1_file = os.path.join(self.output_dir, f'{output_prefix}_f1_scores_{timestamp}.png')
        plt.savefig(f1_file)
        plt.close()
        
        logger.info(f"Saved visualization plots to {self.output_dir}")
    
    def save_metrics(self, metrics, output_prefix='evaluation'):
        """
        Save evaluation metrics to file.
        
        Args:
            metrics (dict): Evaluation metrics from evaluate()
            output_prefix (str): Prefix for output files
            
        Returns:
            None
        """
        # Create timestamp for file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create metrics summary
        summary = {
            'accuracy': metrics['accuracy'],
            'f1_pro': metrics['f1_pro'],
            'f1_contra': metrics['f1_contra'],
            'f1_neutral': metrics['f1_neutral'],
            'timestamp': timestamp
        }
        
        # Save metrics summary
        summary_file = os.path.join(self.output_dir, f'{output_prefix}_metrics_{timestamp}.csv')
        pd.DataFrame([summary]).to_csv(summary_file, index=False)
        
        # Save detailed classification report
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        report_file = os.path.join(self.output_dir, f'{output_prefix}_report_{timestamp}.csv')
        report_df.to_csv(report_file)
        
        logger.info(f"Saved metrics to {summary_file} and {report_file}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Example data for testing
    example_data = {
        'sentence': [
            "Studien zeigen, dass Mitarbeiter an vier Tagen mehr leisten.",
            "Viele Branchen können den Arbeitsausfall nicht kompensieren.",
            "Island hat mehrere Pilotprojekte zur 4-Tage-Woche getestet.",
            "Die Produktivität steigt durch ausgeruhte Mitarbeiter.",
            "Die Work-Life-Balance verbessert sich deutlich.",
            "Die Belastung an den Arbeitstagen wird höher sein.",
            "In der Pflege ist eine 4-Tage-Woche kaum umsetzbar."
        ],
        'label': ['pro', 'contra', 'neutral', 'pro', 'pro', 'contra', 'contra'],
        'final': ['pro', 'contra', 'pro', 'pro', 'pro', 'neutral', 'contra']
    }
    
    # Create test DataFrame
    test_df = pd.DataFrame(example_data)
    
    # Run evaluator
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(test_df)
    
    if metrics:
        # Perform error analysis
        misclassified, error_counts = evaluator.error_analysis(test_df)
        
        # Visualize results
        evaluator.visualize_results(metrics)
        
        # Save metrics
        evaluator.save_metrics(metrics)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score (Pro): {metrics['f1_pro']:.4f}")
        print(f"F1 Score (Contra): {metrics['f1_contra']:.4f}")
        print(f"F1 Score (Neutral): {metrics['f1_neutral']:.4f}")
        
        print("\nError Analysis:")
        for _, row in error_counts.iterrows():
            print(f"{row['error_type']}: {row['count']} instances")