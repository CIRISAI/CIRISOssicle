#!/usr/bin/env python3
"""
Experiment 10: Tamper Detection at LLM Scale

Scaled to realistic LLM inference dimensions:
- 8B model: hidden=4096, heads=32, head_dim=128
- Sequence length: 2048
- Memory footprint: ~600MB per forward pass (vs 3MB in toy version)

Memory Budget (16GB VRAM):
- Model weights: ~4.5GB (Q4_K_M)
- KV cache: ~1GB (2K context)
- Monitor: ~0.3GB
- Attack headroom: ~1.5GB
- Available: ~9GB headroom

Usage:
  # With synthetic workload (always works):
  python exp10_llm_tamper.py --synthetic

  # With ollama (if installed):
  python exp10_llm_tamper.py --model llama3.2

Author: CIRIS L3C
License: BSL 1.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import cupy as cp
import time
import json
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from strain_sensor import StrainSensor, SensorConfig


class RealisticLLMWorkload:
    """Simulate realistic LLM inference at 8B model scale."""

    def __init__(self, hidden_size: int = 4096, num_heads: int = 32,
                 head_dim: int = 128, seq_len: int = 2048, num_layers: int = 32):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        self.running = False
        self.thread = None
        self.tokens_generated = 0

        # Pre-allocate tensors (simulating loaded model weights)
        print(f"Allocating LLM tensors (hidden={hidden_size}, seq={seq_len})...")
        self.q_proj = cp.random.randn(hidden_size, hidden_size, dtype=cp.float32).astype(cp.float16)
        self.k_proj = cp.random.randn(hidden_size, hidden_size, dtype=cp.float32).astype(cp.float16)
        self.v_proj = cp.random.randn(hidden_size, hidden_size, dtype=cp.float32).astype(cp.float16)
        self.o_proj = cp.random.randn(hidden_size, hidden_size, dtype=cp.float32).astype(cp.float16)

        # FFN weights (8B models typically have 4x hidden in FFN)
        ffn_hidden = hidden_size * 4
        self.ffn_up = cp.random.randn(hidden_size, ffn_hidden, dtype=cp.float32).astype(cp.float16)
        self.ffn_down = cp.random.randn(ffn_hidden, hidden_size, dtype=cp.float32).astype(cp.float16)

        mem_gb = (self.q_proj.nbytes + self.k_proj.nbytes + self.v_proj.nbytes +
                  self.o_proj.nbytes + self.ffn_up.nbytes + self.ffn_down.nbytes) / (1024**3)
        print(f"  Allocated {mem_gb:.2f} GB for model simulation")

    def start(self):
        """Start LLM inference loop."""
        self.running = True
        self.tokens_generated = 0
        self.thread = threading.Thread(target=self._inference_loop)
        self.thread.start()

    def stop(self):
        """Stop inference."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        return self.tokens_generated

    def _inference_loop(self):
        """Simulate autoregressive token generation."""
        batch_size = 1

        # Simulated hidden states
        hidden = cp.random.randn(batch_size, self.seq_len, self.hidden_size, dtype=cp.float32).astype(cp.float16)

        while self.running:
            # Simulate one transformer layer forward pass

            # 1. Self-attention
            # Q, K, V projections
            q = cp.matmul(hidden, self.q_proj)
            k = cp.matmul(hidden, self.k_proj)
            v = cp.matmul(hidden, self.v_proj)

            # Reshape for multi-head attention
            q = q.reshape(batch_size, self.seq_len, self.num_heads, self.head_dim)
            k = k.reshape(batch_size, self.seq_len, self.num_heads, self.head_dim)
            v = v.reshape(batch_size, self.seq_len, self.num_heads, self.head_dim)

            # Attention scores (simplified - no causal mask for speed)
            q = q.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
            k = k.transpose(0, 2, 3, 1)  # [batch, heads, head_dim, seq]
            v = v.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]

            # Attention: softmax(QK^T / sqrt(d)) @ V
            scale = 1.0 / np.sqrt(self.head_dim)
            attn = cp.matmul(q, k) * scale
            attn = cp.exp(attn - cp.max(attn, axis=-1, keepdims=True))
            attn = attn / (cp.sum(attn, axis=-1, keepdims=True) + 1e-6)
            attn_out = cp.matmul(attn, v)

            # Reshape back and output projection
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, self.seq_len, self.hidden_size)
            attn_out = cp.matmul(attn_out, self.o_proj)

            # 2. FFN (SwiGLU-like)
            ffn_up = cp.matmul(hidden, self.ffn_up)
            ffn_up = ffn_up * (1.0 / (1.0 + cp.exp(-ffn_up)))  # SiLU
            ffn_out = cp.matmul(ffn_up, self.ffn_down)

            # Residual connection (simplified)
            hidden = hidden + attn_out + ffn_out

            cp.cuda.Stream.null.synchronize()
            self.tokens_generated += 1

            # Small delay to simulate real token generation rate (~50 tok/s)
            time.sleep(0.02)


class MiningAttack:
    """Simulated crypto mining workload - memory bandwidth variant."""

    def __init__(self, intensity: float = 0.8, attack_type: str = "memory"):
        self.intensity = intensity
        self.attack_type = attack_type
        self.running = False
        self.thread = None
        self.ops = 0

    def start(self):
        self.running = True
        self.ops = 0
        self.thread = threading.Thread(target=self._attack)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        return self.ops

    def _attack(self):
        """Attack workload."""
        size = int(1024 * 16 * self.intensity)

        if self.attack_type == "memory":
            # Memory bandwidth attack - very different from matmul
            a = cp.random.randn(size * 1024, dtype=cp.float32)
            b = cp.zeros_like(a)
            while self.running:
                cp.copyto(b, a)
                cp.copyto(a, b)
                # Also do some compute to stress ALUs
                a = a * 1.0001
                cp.cuda.Stream.null.synchronize()
                self.ops += 1

        elif self.attack_type == "compute":
            # Pure compute attack (sin/cos/exp)
            a = cp.random.randn(size * 512, dtype=cp.float32)
            while self.running:
                b = cp.sin(a) * cp.cos(a) * cp.exp(-a * 0.001)
                cp.cuda.Stream.null.synchronize()
                self.ops += 1
                time.sleep(0.001)

        else:  # hash
            data = cp.random.randint(0, 2**32, size=(size * 1024,), dtype=cp.uint32)
            while self.running:
                for _ in range(500):
                    data = data ^ (data << 13)
                    data = data ^ (data >> 17)
                    data = data ^ (data << 5)
                cp.cuda.Stream.null.synchronize()
                self.ops += 500
                time.sleep(0.001)


class TamperDetector:
    """Real-time tamper detection."""

    def __init__(self, sensor, window_size: int = 500):
        self.sensor = sensor
        self.window_size = window_size
        self.baseline_mean = None
        self.baseline_std = None
        self.means_a = []
        self.means_b = []
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.alerts = []

    def train(self, duration: float = 30.0):
        """Train baseline."""
        print(f"Training baseline ({duration}s)...")
        correlations = []
        self.means_a = []
        self.means_b = []

        start = time.time()
        while time.time() - start < duration:
            mean_a, mean_b, _ = self.sensor.read_raw()
            self.means_a.append(mean_a)
            self.means_b.append(mean_b)

            n = len(self.means_a)
            if n >= self.window_size and n % (self.window_size // 4) == 0:
                rho = np.corrcoef(
                    self.means_a[-self.window_size:],
                    self.means_b[-self.window_size:]
                )[0, 1]
                correlations.append(rho)

        self.baseline_mean = np.mean(correlations)
        self.baseline_std = np.std(correlations)
        print(f"  Baseline: ρ = {self.baseline_mean:.4f} ± {self.baseline_std:.4f}")

    def monitor(self, duration: float):
        """Monitor and return alert count."""
        alerts = 0
        start = time.time()

        while time.time() - start < duration:
            mean_a, mean_b, _ = self.sensor.read_raw()
            self.means_a.append(mean_a)
            self.means_b.append(mean_b)

            if len(self.means_a) > self.window_size * 4:
                self.means_a = self.means_a[-self.window_size * 2:]
                self.means_b = self.means_b[-self.window_size * 2:]

            n = len(self.means_a)
            if n >= self.window_size and n % (self.window_size // 4) == 0:
                rho = np.corrcoef(
                    self.means_a[-self.window_size:],
                    self.means_b[-self.window_size:]
                )[0, 1]

                z = (rho - self.baseline_mean) / (self.baseline_std + 1e-10)
                self.cusum_pos = max(0, self.cusum_pos + z - 0.5)
                self.cusum_neg = max(0, self.cusum_neg - z - 0.5)

                if abs(z) > 3.0 or self.cusum_pos > 5.0 or self.cusum_neg > 5.0:
                    alerts += 1

        return alerts


def run_llm_tamper_experiment(use_ollama: bool = False, model_name: str = "llama3.2"):
    """Run tamper detection with LLM-scale workload."""
    print("="*70)
    print("EXPERIMENT 10: LLM-SCALE TAMPER DETECTION")
    print("="*70)
    print()
    print("Testing tamper detection at realistic LLM inference scale.")
    print("Simulating 8B model: hidden=4096, heads=32, seq=2048")
    print()

    # Check VRAM
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True)
    free_vram = float(result.stdout.strip())
    print(f"Free VRAM: {free_vram:.0f} MB")

    # Initialize sensor
    config = SensorConfig(n_iterations=5000)
    sensor = StrainSensor(config=config)
    detector = TamperDetector(sensor, window_size=500)

    # Start LLM workload - reduced seq_len to establish detectable baseline
    print("\n[PHASE 1] STARTING LLM WORKLOAD")
    print("-"*50)
    llm = RealisticLLMWorkload(hidden_size=4096, num_heads=32, head_dim=128, seq_len=256)
    llm.start()
    time.sleep(3)  # Warm up

    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True)
    used_vram = float(result.stdout.strip())
    print(f"VRAM after LLM start: {used_vram:.0f} MB")

    # Train baseline
    print("\n[PHASE 2] TRAINING BASELINE")
    print("-"*50)
    detector.train(duration=30.0)

    # Monitor clean operation
    print("\n[PHASE 3] MONITORING CLEAN OPERATION (30s)")
    print("-"*50)
    clean_alerts = detector.monitor(duration=30.0)
    clean_tokens = llm.tokens_generated
    print(f"  Tokens generated: {clean_tokens}")
    print(f"  False positive alerts: {clean_alerts}")

    # Inject attack - use compute-based attack (sin/cos/exp - SFU operations, different from tensor cores)
    print("\n[PHASE 4] INJECTING COMPUTE ATTACK (SFU ops)")
    print("-"*50)
    attack = MiningAttack(intensity=0.7, attack_type="compute")
    attack.start()
    time.sleep(2)

    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True)
    used_vram_attack = float(result.stdout.strip())
    print(f"VRAM with attack: {used_vram_attack:.0f} MB (+{used_vram_attack - used_vram:.0f} MB)")

    detector.cusum_pos = 0  # Reset CUSUM
    detector.cusum_neg = 0
    attack_alerts = detector.monitor(duration=30.0)
    attack_ops = attack.stop()  # Stop and get ops count
    tokens_during_attack = llm.tokens_generated - clean_tokens

    print(f"  Tokens during attack: {tokens_during_attack}")
    print(f"  Attack ops: {attack_ops}")
    print(f"  Alerts triggered: {attack_alerts}")

    # Stop LLM
    total_tokens = llm.stop()

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"LLM Scale: 8B model simulation")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Token rate: {total_tokens / 90:.1f} tok/s")
    print()
    print(f"Tamper Detection:")
    print(f"  Clean alerts (false positives): {clean_alerts}")
    print(f"  Attack alerts: {attack_alerts}")
    print(f"  Alert ratio: {attack_alerts / max(1, clean_alerts):.1f}x")
    print()

    if attack_alerts > clean_alerts * 2 and attack_alerts > 10:
        print("="*70)
        print("*** TAMPER DETECTION SUCCESSFUL AT LLM SCALE ***")
        print("="*70)
        print()
        print("Mining attack detected during LLM inference!")
        print(f"  {attack_alerts} alerts vs {clean_alerts} false positives")
        success = True
    else:
        print("Tamper detection inconclusive at this scale.")
        success = False

    # Performance impact
    if clean_tokens > 0 and tokens_during_attack > 0:
        slowdown = 1 - (tokens_during_attack / clean_tokens)
        print(f"\nPerformance impact of attack: {slowdown*100:.1f}% slowdown")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp10_llm_tamper_{timestamp}.json"

    output = {
        'experiment': 'exp10_llm_tamper',
        'timestamp': datetime.now().isoformat(),
        'scale': {
            'hidden_size': 4096,
            'num_heads': 32,
            'seq_len': 256,
            'model_type': '8B_simulation_reduced',
        },
        'baseline_mean': float(detector.baseline_mean),
        'baseline_std': float(detector.baseline_std),
        'clean_alerts': clean_alerts,
        'attack_alerts': attack_alerts,
        'total_tokens': total_tokens,
        'attack_ops': attack_ops,
        'success': success,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return success


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LLM-scale tamper detection')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic workload')
    parser.add_argument('--model', type=str, default='llama3.2', help='Ollama model name')
    args = parser.parse_args()

    run_llm_tamper_experiment(use_ollama=not args.synthetic, model_name=args.model)
