#!/usr/bin/env python3
"""
Data Collector for Transformer-based Tamper Detection

Collects labeled time series data from the strain sensor during:
- Clean workloads (label=0)
- Attack workloads (label=1)

Outputs sequences suitable for transformer training.

Author: CIRIS L3C
License: BSL 1.1
"""

import numpy as np
import cupy as cp
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from strain_sensor import StrainSensor, SensorConfig


@dataclass
class Sample:
    """A single training sample."""
    timestamp: float
    mean_a: float
    mean_b: float
    mean_c: float
    label: int  # 0=clean, 1=attack


@dataclass
class Sequence:
    """A sequence of samples for transformer input."""
    samples: List[Sample]
    label: int
    workload_type: str
    attack_type: Optional[str]


class WorkloadGenerator:
    """Generate GPU workloads for data collection."""

    def __init__(self, workload_type: str, intensity: float = 0.7):
        self.workload_type = workload_type
        self.intensity = intensity
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self):
        size = int(1024 + 3072 * self.intensity)

        if self.workload_type == "transformer":
            # Simulate LLM inference
            q = cp.random.randn(size, 64, dtype=cp.float32)
            k = cp.random.randn(size, 64, dtype=cp.float32)
            v = cp.random.randn(size, 64, dtype=cp.float32)
            while self.running:
                attn = cp.matmul(q, k.T)
                attn = cp.exp(attn - cp.max(attn, axis=-1, keepdims=True))
                attn = attn / cp.sum(attn, axis=-1, keepdims=True)
                out = cp.matmul(attn, v)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.005)

        elif self.workload_type == "matmul":
            # Dense matrix operations
            a = cp.random.randn(size, size, dtype=cp.float32)
            b = cp.random.randn(size, size, dtype=cp.float32)
            while self.running:
                c = cp.matmul(a, b)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.01)

        elif self.workload_type == "conv":
            # Convolution-like workload
            x = cp.random.randn(1, 64, size, size, dtype=cp.float32)
            while self.running:
                # Simulate conv via im2col + matmul pattern
                y = cp.sum(x, axis=(2, 3))
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.01)


class AttackGenerator:
    """Generate attack workloads."""

    def __init__(self, attack_type: str, intensity: float = 0.8):
        self.attack_type = attack_type
        self.intensity = intensity
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self):
        size = int(1024 * 8 * self.intensity)

        if self.attack_type == "mining":
            # Crypto mining simulation (hash operations)
            data = cp.random.randint(0, 2**32, size=(size,), dtype=cp.uint32)
            while self.running:
                for _ in range(200):
                    data = data ^ (data << 13)
                    data = data ^ (data >> 17)
                    data = data ^ (data << 5)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.002)

        elif self.attack_type == "memory":
            # Memory bandwidth attack
            a = cp.random.randn(size * 128, dtype=cp.float32)
            b = cp.zeros_like(a)
            while self.running:
                cp.copyto(b, a)
                cp.copyto(a, b)
                cp.cuda.Stream.null.synchronize()

        elif self.attack_type == "compute":
            # SFU operations (sin/cos/exp)
            a = cp.random.randn(size * 64, dtype=cp.float32)
            while self.running:
                b = cp.sin(a) * cp.cos(a) * cp.exp(-a * 0.001)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.001)


class DataCollector:
    """Collect training data for transformer attack detection."""

    def __init__(self, output_dir: str = "data/training"):
        self.sensor = StrainSensor(SensorConfig(n_iterations=5000))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_sequence(
        self,
        duration: float,
        label: int,
        workload_type: str,
        attack_type: Optional[str] = None
    ) -> Sequence:
        """Collect a single sequence of sensor readings."""
        samples = []
        start = time.time()

        while time.time() - start < duration:
            mean_a, mean_b, mean_c = self.sensor.read_raw()
            samples.append(Sample(
                timestamp=time.time(),
                mean_a=mean_a,
                mean_b=mean_b,
                mean_c=mean_c,
                label=label
            ))

        return Sequence(
            samples=samples,
            label=label,
            workload_type=workload_type,
            attack_type=attack_type
        )

    def collect_dataset(
        self,
        n_clean_sequences: int = 50,
        n_attack_sequences: int = 50,
        sequence_duration: float = 10.0,
        workload_types: List[str] = ["transformer", "matmul"],
        attack_types: List[str] = ["mining", "memory", "compute"],
        warmup_time: float = 2.0
    ) -> dict:
        """Collect a full dataset with clean and attack sequences."""

        print("="*60)
        print("DATA COLLECTION FOR TRANSFORMER ATTACK DETECTION")
        print("="*60)
        print(f"Clean sequences: {n_clean_sequences}")
        print(f"Attack sequences: {n_attack_sequences}")
        print(f"Sequence duration: {sequence_duration}s")
        print()

        all_sequences = []

        # Collect clean sequences
        print("[PHASE 1] Collecting CLEAN sequences")
        print("-"*40)

        for i in range(n_clean_sequences):
            workload_type = workload_types[i % len(workload_types)]
            print(f"  [{i+1}/{n_clean_sequences}] {workload_type}...", end=" ", flush=True)

            # Start workload
            workload = WorkloadGenerator(workload_type, intensity=0.7)
            workload.start()
            time.sleep(warmup_time)

            # Collect
            seq = self.collect_sequence(
                duration=sequence_duration,
                label=0,
                workload_type=workload_type
            )
            all_sequences.append(seq)

            workload.stop()
            print(f"{len(seq.samples)} samples")
            time.sleep(0.5)

        # Collect attack sequences
        print()
        print("[PHASE 2] Collecting ATTACK sequences")
        print("-"*40)

        for i in range(n_attack_sequences):
            workload_type = workload_types[i % len(workload_types)]
            attack_type = attack_types[i % len(attack_types)]
            print(f"  [{i+1}/{n_attack_sequences}] {workload_type} + {attack_type}...", end=" ", flush=True)

            # Start both workloads
            workload = WorkloadGenerator(workload_type, intensity=0.7)
            attack = AttackGenerator(attack_type, intensity=0.8)
            workload.start()
            attack.start()
            time.sleep(warmup_time)

            # Collect
            seq = self.collect_sequence(
                duration=sequence_duration,
                label=1,
                workload_type=workload_type,
                attack_type=attack_type
            )
            all_sequences.append(seq)

            attack.stop()
            workload.stop()
            print(f"{len(seq.samples)} samples")
            time.sleep(0.5)

        # Convert to numpy arrays for saving
        print()
        print("[PHASE 3] Saving dataset")
        print("-"*40)

        dataset = self._sequences_to_arrays(all_sequences)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"tamper_dataset_{timestamp}.npz"

        np.savez_compressed(
            output_file,
            sequences=dataset['sequences'],
            labels=dataset['labels'],
            metadata=json.dumps(dataset['metadata'])
        )

        print(f"Saved to: {output_file}")
        print(f"  Sequences: {len(all_sequences)}")
        print(f"  Shape: {dataset['sequences'].shape}")

        return dataset

    def _sequences_to_arrays(self, sequences: List[Sequence]) -> dict:
        """Convert sequences to numpy arrays."""

        # Find max sequence length
        max_len = max(len(s.samples) for s in sequences)
        n_features = 3  # mean_a, mean_b, mean_c

        # Create arrays
        X = np.zeros((len(sequences), max_len, n_features), dtype=np.float32)
        y = np.zeros(len(sequences), dtype=np.int32)

        metadata = []

        for i, seq in enumerate(sequences):
            for j, sample in enumerate(seq.samples):
                X[i, j, 0] = sample.mean_a
                X[i, j, 1] = sample.mean_b
                X[i, j, 2] = sample.mean_c
            y[i] = seq.label
            metadata.append({
                'workload_type': seq.workload_type,
                'attack_type': seq.attack_type,
                'n_samples': len(seq.samples)
            })

        return {
            'sequences': X,
            'labels': y,
            'metadata': metadata
        }


def quick_collect(n_each: int = 10, duration: float = 5.0):
    """Quick data collection for testing."""
    collector = DataCollector()
    return collector.collect_dataset(
        n_clean_sequences=n_each,
        n_attack_sequences=n_each,
        sequence_duration=duration
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Collect training data')
    parser.add_argument('--clean', type=int, default=50, help='Number of clean sequences')
    parser.add_argument('--attack', type=int, default=50, help='Number of attack sequences')
    parser.add_argument('--duration', type=float, default=10.0, help='Sequence duration (seconds)')
    parser.add_argument('--quick', action='store_true', help='Quick test (10 each, 5s)')
    args = parser.parse_args()

    if args.quick:
        quick_collect()
    else:
        collector = DataCollector()
        collector.collect_dataset(
            n_clean_sequences=args.clean,
            n_attack_sequences=args.attack,
            sequence_duration=args.duration
        )
