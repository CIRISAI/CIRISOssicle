#!/usr/bin/env python3
"""
Transformer-based Attack Detection Model

A lightweight transformer that learns temporal patterns in chaotic oscillator
correlations to detect unauthorized GPU workloads.

Architecture:
    Input: [batch, seq_len, 3] (mean_a, mean_b, mean_c)
    -> Positional Encoding
    -> Transformer Encoder (2-4 layers)
    -> Global Average Pooling
    -> Classification Head
    -> Output: [batch, 2] (clean vs attack)

Design goals:
    - Small enough to run alongside monitored workload (~100K-500K params)
    - Fast inference (<10ms per classification)
    - Learns patterns statistical methods miss

Author: CIRIS L3C
License: BSL 1.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for the attack detection transformer."""
    # Input
    n_features: int = 3          # mean_a, mean_b, mean_c
    max_seq_len: int = 512       # Max sequence length

    # Architecture
    d_model: int = 64            # Embedding dimension
    n_heads: int = 4             # Attention heads
    n_layers: int = 2            # Transformer layers
    d_ff: int = 128              # Feed-forward dimension
    dropout: float = 0.1

    # Output
    n_classes: int = 2           # clean vs attack

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttackDetectionTransformer(nn.Module):
    """
    Transformer for detecting GPU tamper attacks.

    Takes time series of chaotic oscillator means and classifies
    as clean (0) or attack (1).
    """

    def __init__(self, config: TransformerConfig = None):
        super().__init__()
        self.config = config or TransformerConfig()
        cfg = self.config

        # Input projection
        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, n_features]
            mask: Optional attention mask [batch, seq_len]

        Returns:
            Logits [batch, n_classes]
        """
        # Project input to model dimension
        x = self.input_proj(x)  # [batch, seq, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        if mask is not None:
            # Convert padding mask to attention mask
            src_key_padding_mask = ~mask  # True where padded
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Global average pooling over sequence
        if mask is not None:
            # Masked mean
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits

    def predict(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x, mask)
        return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x, mask)
        return F.softmax(logits, dim=-1)

    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttackDetector:
    """
    High-level interface for attack detection.

    Wraps the transformer model with training and inference utilities.
    """

    def __init__(self, config: TransformerConfig = None, device: str = "cuda"):
        self.config = config or TransformerConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = AttackDetectionTransformer(self.config).to(self.device)

        print(f"AttackDetector initialized on {self.device}")
        print(f"  Parameters: {self.model.num_parameters:,}")
        print(f"  Config: d_model={self.config.d_model}, layers={self.config.n_layers}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> dict:
        """
        Train the model.

        Args:
            X_train: Training sequences [n_samples, seq_len, n_features]
            y_train: Training labels [n_samples]
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Stop if val loss doesn't improve

        Returns:
            Training history dict
        """
        self.model.train()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)

        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.LongTensor(y_val).to(self.device)

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0

        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(n_samples)
            X_train_t = X_train_t[perm]
            y_train_t = y_train_t[perm]

            epoch_loss = 0
            epoch_correct = 0

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_train_t[start:end]
                y_batch = y_train_t[start:end]

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * (end - start)
                epoch_correct += (logits.argmax(dim=-1) == y_batch).sum().item()

            scheduler.step()

            train_loss = epoch_loss / n_samples
            train_acc = epoch_correct / n_samples
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                    val_acc = (val_logits.argmax(dim=-1) == y_val_t).float().mean().item()
                self.model.train()

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}: loss={train_loss:.4f} acc={train_acc:.3f} "
                          f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}: loss={train_loss:.4f} acc={train_acc:.3f}")

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model on test data."""
        self.model.eval()

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            accuracy = (preds == y_t).float().mean().item()

            # Per-class metrics
            tp = ((preds == 1) & (y_t == 1)).sum().item()
            fp = ((preds == 1) & (y_t == 0)).sum().item()
            fn = ((preds == 0) & (y_t == 1)).sum().item()
            tn = ((preds == 0) & (y_t == 0)).sum().item()

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            preds = self.model.predict(X_t)

        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(X_t)

        return probs.cpu().numpy()

    def save(self, path: str):
        """Save model weights and config."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.model = AttackDetectionTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


def create_small_model() -> AttackDetector:
    """Create a small model (~50K params) for minimal overhead."""
    config = TransformerConfig(
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        dropout=0.1
    )
    return AttackDetector(config)


def create_medium_model() -> AttackDetector:
    """Create a medium model (~200K params) for better accuracy."""
    config = TransformerConfig(
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        dropout=0.1
    )
    return AttackDetector(config)


if __name__ == "__main__":
    # Quick test
    print("Testing AttackDetectionTransformer...")

    detector = create_small_model()

    # Dummy data
    X = np.random.randn(100, 500, 3).astype(np.float32)
    y = np.random.randint(0, 2, 100)

    # Train
    history = detector.train(X[:80], y[:80], X[80:], y[80:], epochs=20)

    # Evaluate
    metrics = detector.evaluate(X[80:], y[80:])
    print(f"\nTest metrics: {metrics}")
