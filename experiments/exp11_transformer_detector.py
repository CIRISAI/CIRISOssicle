#!/usr/bin/env python3
"""
Experiment 11: Transformer-based Attack Detection

Tests whether a small transformer can learn to detect GPU tamper attacks
from chaotic oscillator time series, compared to statistical baseline.

Usage:
    # Quick test (small dataset)
    python exp11_transformer_detector.py --quick

    # Full training
    python exp11_transformer_detector.py --collect --train

    # Train on existing data
    python exp11_transformer_detector.py --train --data path/to/data.npz

Author: CIRIS L3C
License: BSL 1.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

from data_collector import DataCollector, quick_collect
from attack_transformer import (
    AttackDetector,
    TransformerConfig,
    create_small_model,
    create_medium_model
)


class StatisticalBaseline:
    """
    Statistical baseline detector using z-scores and correlation analysis.
    Mirrors the approach from exp9_tamper_detector.py.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.baseline_mean = None
        self.baseline_std = None

    def fit(self, X_clean: np.ndarray):
        """Fit baseline on clean data."""
        # Compute correlations for each sequence
        correlations = []

        for seq in X_clean:
            # Compute rolling correlation between crystal A and B
            mean_a = seq[:, 0]
            mean_b = seq[:, 1]

            for i in range(self.window_size, len(mean_a), self.window_size // 4):
                window_a = mean_a[i-self.window_size:i]
                window_b = mean_b[i-self.window_size:i]
                if len(window_a) >= 2:
                    rho = np.corrcoef(window_a, window_b)[0, 1]
                    if not np.isnan(rho):
                        correlations.append(rho)

        self.baseline_mean = np.mean(correlations)
        self.baseline_std = np.std(correlations)

        print(f"StatisticalBaseline fitted: ρ = {self.baseline_mean:.4f} ± {self.baseline_std:.4f}")

    def predict(self, X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Predict using z-score threshold."""
        predictions = []

        for seq in X:
            mean_a = seq[:, 0]
            mean_b = seq[:, 1]

            # Check if any window exceeds threshold
            alert = 0
            for i in range(self.window_size, len(mean_a), self.window_size // 4):
                window_a = mean_a[i-self.window_size:i]
                window_b = mean_b[i-self.window_size:i]
                if len(window_a) >= 2:
                    rho = np.corrcoef(window_a, window_b)[0, 1]
                    if not np.isnan(rho):
                        z = abs(rho - self.baseline_mean) / (self.baseline_std + 1e-10)
                        if z > threshold:
                            alert = 1
                            break

            predictions.append(alert)

        return np.array(predictions)

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 3.0) -> dict:
        """Evaluate baseline detector."""
        preds = self.predict(X, threshold)

        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        tn = ((preds == 0) & (y == 0)).sum()

        accuracy = (tp + tn) / len(y)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}
        }


def load_dataset(path: str) -> tuple:
    """Load dataset from npz file."""
    data = np.load(path, allow_pickle=True)
    X = data['sequences']
    y = data['labels']
    metadata = json.loads(str(data['metadata']))
    return X, y, metadata


def run_experiment(
    collect_data: bool = False,
    data_path: str = None,
    quick: bool = False,
    model_size: str = "small"
):
    """Run the transformer detection experiment."""

    print("="*70)
    print("EXPERIMENT 11: TRANSFORMER-BASED ATTACK DETECTION")
    print("="*70)
    print()

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect or load data
    if collect_data or quick:
        print("[PHASE 1] COLLECTING DATA")
        print("-"*50)

        if quick:
            # Quick test: 10 sequences each, 5 seconds
            dataset = quick_collect(n_each=10, duration=5.0)
            X, y = dataset['sequences'], dataset['labels']
        else:
            collector = DataCollector(output_dir=str(output_dir / "data"))
            dataset = collector.collect_dataset(
                n_clean_sequences=50,
                n_attack_sequences=50,
                sequence_duration=10.0
            )
            X, y = dataset['sequences'], dataset['labels']

    elif data_path:
        print("[PHASE 1] LOADING DATA")
        print("-"*50)
        X, y, _ = load_dataset(data_path)
        print(f"Loaded {len(X)} sequences from {data_path}")

    else:
        # Generate synthetic data for testing
        print("[PHASE 1] GENERATING SYNTHETIC DATA")
        print("-"*50)
        print("No data path provided, generating synthetic data...")

        n_samples = 100
        seq_len = 500
        X = np.random.randn(n_samples, seq_len, 3).astype(np.float32)

        # Make attack sequences have higher variance
        y = np.array([0] * 50 + [1] * 50)
        X[50:] *= 1.5  # Attack sequences have more variance
        X[50:] += np.random.randn(50, seq_len, 3) * 0.2

        print(f"Generated {n_samples} synthetic sequences")

    print(f"\nDataset shape: {X.shape}")
    print(f"Labels: {np.sum(y==0)} clean, {np.sum(y==1)} attack")

    # Subsample sequences to fixed length for transformer
    max_seq_len = 512
    if X.shape[1] > max_seq_len:
        print(f"Subsampling sequences from {X.shape[1]} to {max_seq_len}")
        # Uniform subsampling
        indices = np.linspace(0, X.shape[1] - 1, max_seq_len, dtype=int)
        X = X[:, indices, :]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Train statistical baseline
    print("\n" + "="*70)
    print("[PHASE 2] STATISTICAL BASELINE")
    print("-"*50)

    baseline = StatisticalBaseline(window_size=50)
    baseline.fit(X_train[y_train == 0])  # Fit on clean only

    baseline_metrics = baseline.evaluate(X_test, y_test)
    print(f"\nBaseline Test Results:")
    print(f"  Accuracy:  {baseline_metrics['accuracy']:.3f}")
    print(f"  Precision: {baseline_metrics['precision']:.3f}")
    print(f"  Recall:    {baseline_metrics['recall']:.3f}")
    print(f"  F1:        {baseline_metrics['f1']:.3f}")

    # Train transformer
    print("\n" + "="*70)
    print("[PHASE 3] TRANSFORMER TRAINING")
    print("-"*50)

    if model_size == "small":
        detector = create_small_model()
    else:
        detector = create_medium_model()

    print(f"\nTraining {model_size} model ({detector.model.num_parameters:,} params)...")
    start_time = time.time()

    history = detector.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100 if not quick else 20,
        batch_size=16,
        early_stopping_patience=15
    )

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s")

    # Evaluate transformer
    print("\n" + "="*70)
    print("[PHASE 4] EVALUATION")
    print("-"*50)

    transformer_metrics = detector.evaluate(X_test, y_test)
    print(f"\nTransformer Test Results:")
    print(f"  Accuracy:  {transformer_metrics['accuracy']:.3f}")
    print(f"  Precision: {transformer_metrics['precision']:.3f}")
    print(f"  Recall:    {transformer_metrics['recall']:.3f}")
    print(f"  F1:        {transformer_metrics['f1']:.3f}")

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print()
    print(f"{'Metric':<12} {'Baseline':<12} {'Transformer':<12} {'Improvement'}")
    print("-"*50)

    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        bl = baseline_metrics[metric]
        tf = transformer_metrics[metric]
        imp = tf - bl
        print(f"{metric:<12} {bl:<12.3f} {tf:<12.3f} {imp:+.3f}")

    # Inference speed
    print("\n" + "-"*50)
    print("Inference Speed:")

    # Baseline timing
    start = time.time()
    for _ in range(10):
        baseline.predict(X_test[:10])
    baseline_time = (time.time() - start) / 100

    # Transformer timing
    start = time.time()
    for _ in range(10):
        detector.predict(X_test[:10])
    transformer_time = (time.time() - start) / 100

    print(f"  Baseline:    {baseline_time*1000:.2f} ms/sample")
    print(f"  Transformer: {transformer_time*1000:.2f} ms/sample")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        'experiment': 'exp11_transformer_detector',
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'n_samples': len(X),
            'seq_len': X.shape[1],
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test)
        },
        'model': {
            'size': model_size,
            'n_params': detector.model.num_parameters,
            'config': {
                'd_model': detector.config.d_model,
                'n_heads': detector.config.n_heads,
                'n_layers': detector.config.n_layers
            }
        },
        'baseline_metrics': baseline_metrics,
        'transformer_metrics': transformer_metrics,
        'training_time_sec': train_time,
        'inference_time_ms': {
            'baseline': baseline_time * 1000,
            'transformer': transformer_time * 1000
        }
    }

    results_file = output_dir / f"exp11_transformer_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save model
    model_file = output_dir / f"attack_detector_{timestamp}.pt"
    detector.save(str(model_file))

    print(f"\nResults saved to: {results_file}")
    print(f"Model saved to: {model_file}")

    # Final verdict
    print("\n" + "="*70)
    if transformer_metrics['f1'] > baseline_metrics['f1'] + 0.05:
        print("*** TRANSFORMER OUTPERFORMS BASELINE ***")
        print(f"F1 improvement: +{transformer_metrics['f1'] - baseline_metrics['f1']:.3f}")
    elif transformer_metrics['f1'] > baseline_metrics['f1']:
        print("Transformer slightly better than baseline")
    else:
        print("Baseline performs comparably to transformer")
        print("(More data or larger model may help)")
    print("="*70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Transformer attack detection experiment')
    parser.add_argument('--collect', action='store_true', help='Collect new training data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data', type=str, help='Path to existing dataset')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--model', type=str, default='small', choices=['small', 'medium'],
                        help='Model size')

    args = parser.parse_args()

    run_experiment(
        collect_data=args.collect,
        data_path=args.data,
        quick=args.quick,
        model_size=args.model
    )
