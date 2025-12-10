"""
Week 3: Evaluation Module
Computes precision, recall, F1, and AUROC for hallucination detectors
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List
import yaml

class HallucinationEvaluator:
    """
    Evaluates hallucination detection performance
    """

    def __init__(self, config):
        self.config = config

    def load_results(self, quality_tier: str) -> pd.DataFrame:
        """Load results for a quality tier"""
        results_path = Path(self.config['paths']['results_dir']) / f"results_{quality_tier}.jsonl"

        results = []
        with open(results_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))

        return pd.DataFrame(results)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray = None
    ) -> Dict:
        """
        Compute evaluation metrics

        Args:
            y_true: Ground truth labels (1 = hallucinated, 0 = faithful)
            y_pred: Predicted labels
            y_scores: Prediction scores for AUROC (optional)

        Returns:
            Dictionary of metrics
        """
        # Handle NaN values
        valid_mask = ~(pd.isna(y_pred) | pd.isna(y_true))
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        if len(y_true_valid) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auroc': 0.0,
                'accuracy': 0.0
            }

        # Compute metrics
        precision = precision_score(y_true_valid, y_pred_valid, zero_division=0)
        recall = recall_score(y_true_valid, y_pred_valid, zero_division=0)
        f1 = f1_score(y_true_valid, y_pred_valid, zero_division=0)
        accuracy = (y_true_valid == y_pred_valid).mean()

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'num_samples': len(y_true_valid)
        }

        # Compute AUROC if scores provided
        if y_scores is not None:
            y_scores_valid = y_scores[valid_mask]
            score_valid_mask = ~pd.isna(y_scores_valid)

            if score_valid_mask.sum() > 0 and len(np.unique(y_true_valid[score_valid_mask])) > 1:
                try:
                    auroc = roc_auc_score(
                        y_true_valid[score_valid_mask],
                        y_scores_valid[score_valid_mask]
                    )
                    metrics['auroc'] = auroc
                except Exception as e:
                    print(f"Warning: Could not compute AUROC: {e}")
                    metrics['auroc'] = 0.0
            else:
                metrics['auroc'] = 0.0
        else:
            metrics['auroc'] = 0.0

        return metrics

    def evaluate_detector(
        self,
        df: pd.DataFrame,
        detector_name: str,
        pred_col: str,
        score_col: str = None
    ) -> Dict:
        """
        Evaluate a specific detector

        Args:
            df: DataFrame with results
            detector_name: Name of detector
            pred_col: Column name for predictions
            score_col: Column name for scores (optional)

        Returns:
            Dictionary of metrics
        """
        # Handle None/NaN values - fill with False (not hallucinated) as default
        y_true = df['ground_truth_hallucinated'].fillna(False).astype(int).values
        y_pred = df[pred_col].fillna(False).astype(int).values

        y_scores = None
        if score_col and score_col in df.columns:
            y_scores = df[score_col].values

            # Convert scores to probabilities if needed
            # For NLI: low entailment prob = high hallucination prob
            if 'entailment' in score_col:
                # Handle NaN scores
                y_scores = pd.Series(y_scores).fillna(0.5).values
                y_scores = 1 - y_scores  # Invert for hallucination probability

        metrics = self.compute_metrics(y_true, y_pred, y_scores)
        metrics['detector'] = detector_name

        return metrics

    def evaluate_all_detectors(self, quality_tier: str) -> pd.DataFrame:
        """
        Evaluate all detectors for a quality tier

        Args:
            quality_tier: 'high', 'medium', or 'low'

        Returns:
            DataFrame with metrics for all detectors
        """
        print(f"\nEvaluating detectors for {quality_tier} quality tier...")

        # Load results
        df = self.load_results(quality_tier)

        # Evaluate each detector
        detectors = [
            ('RAGAS', 'ragas_hallucinated', 'ragas_faithfulness'),
            ('NLI', 'nli_hallucinated', 'nli_entailment_prob'),
            ('Lexical', 'lexical_hallucinated', 'lexical_overlap')
        ]

        all_metrics = []

        for detector_name, pred_col, score_col in detectors:
            if pred_col in df.columns:
                metrics = self.evaluate_detector(df, detector_name, pred_col, score_col)
                metrics['quality_tier'] = quality_tier
                all_metrics.append(metrics)
            else:
                print(f"Warning: {pred_col} not found in results")

        return pd.DataFrame(all_metrics)

    def evaluate_all_tiers(self) -> pd.DataFrame:
        """
        Evaluate all detectors across all quality tiers

        Returns:
            DataFrame with complete evaluation results
        """
        all_results = []

        for tier in ['high', 'medium', 'low']:
            try:
                tier_results = self.evaluate_all_detectors(tier)
                all_results.append(tier_results)
            except FileNotFoundError:
                print(f"Warning: Results not found for {tier} tier")

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()

    def print_evaluation_summary(self, results_df: pd.DataFrame):
        """Print formatted evaluation summary"""
        print("\n" + "="*80)
        print("HALLUCINATION DETECTION EVALUATION SUMMARY")
        print("="*80)

        for tier in ['high', 'medium', 'low']:
            tier_results = results_df[results_df['quality_tier'] == tier]

            if len(tier_results) == 0:
                continue

            print(f"\n{'='*80}")
            print(f"{tier.upper()} Quality Tier")
            print('='*80)

            print(f"\n{'Detector':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUROC':<12} {'Samples':<10}")
            print('-'*80)

            for _, row in tier_results.iterrows():
                print(f"{row['detector']:<15} "
                      f"{row['precision']:<12.3f} "
                      f"{row['recall']:<12.3f} "
                      f"{row['f1']:<12.3f} "
                      f"{row['auroc']:<12.3f} "
                      f"{row['num_samples']:<10.0f}")

        print("\n" + "="*80)

    def save_evaluation_results(self, results_df: pd.DataFrame):
        """Save evaluation results"""
        output_dir = Path(self.config['paths']['results_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        csv_path = output_dir / "evaluation_metrics.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved evaluation metrics to {csv_path}")

        # Save as JSON
        json_path = output_dir / "evaluation_metrics.json"
        results_df.to_json(json_path, orient='records', indent=2)
        print(f"Saved evaluation metrics to {json_path}")

    def compute_confusion_matrices(self, quality_tier: str) -> Dict:
        """
        Compute confusion matrices for all detectors

        Args:
            quality_tier: Quality tier to analyze

        Returns:
            Dictionary of confusion matrices
        """
        df = self.load_results(quality_tier)
        y_true = df['ground_truth_hallucinated'].astype(int).values

        detectors = {
            'RAGAS': 'ragas_hallucinated',
            'NLI': 'nli_hallucinated',
            'Lexical': 'lexical_hallucinated'
        }

        confusion_matrices = {}

        for detector_name, pred_col in detectors.items():
            if pred_col in df.columns:
                y_pred = df[pred_col].astype(int).values

                # Handle NaN values
                valid_mask = ~(pd.isna(y_pred) | pd.isna(y_true))

                if valid_mask.sum() > 0:
                    cm = confusion_matrix(y_true[valid_mask], y_pred[valid_mask])
                    confusion_matrices[detector_name] = cm

        return confusion_matrices


def main():
    """Main evaluation function"""
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create evaluator
    evaluator = HallucinationEvaluator(config)

    # Evaluate all detectors across all tiers
    results_df = evaluator.evaluate_all_tiers()

    # Print summary
    evaluator.print_evaluation_summary(results_df)

    # Save results
    evaluator.save_evaluation_results(results_df)

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
