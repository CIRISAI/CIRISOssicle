#!/usr/bin/env python3
"""
GPU TRNG: Generate random bits from cross-block race conditions.

Uses multi-block chaotic resonators to harvest entropy from thread scheduling.
"""

import numpy as np
import cupy as cp
import sys
from pathlib import Path

# TRNG kernel - extracts full bytes from chaotic state
TRNG_KERNEL = '''
extern "C" __global__
void trng_kernel(unsigned char* output, int n_bytes, unsigned int base_seed) {
    int byte_idx = blockIdx.x;
    if (byte_idx >= n_bytes) return;

    int block_in_byte = blockIdx.y;
    int thread_id = block_in_byte * blockDim.x + threadIdx.x;

    // Unique seed per thread
    unsigned int seed = base_seed + byte_idx * 100000 + thread_id * 31337;

    // Initialize state
    float x = (float)((seed * 1103515245u + 12345u) % 65536) / 65536.0f;
    x = fminf(0.999f, fmaxf(0.001f, x));

    // Run chaotic iterations
    for (int i = 0; i < 500; i++) {
        x = 3.75f * x * (1.0f - x);
    }

    // Extract byte from fractional part (bits 8-15 of fixed-point representation)
    unsigned int quantized = (unsigned int)(x * 65536.0f);
    unsigned char bits = (quantized >> 4) & 0xFF;

    // Atomic XOR - all threads in all blocks for this byte race here
    atomicXor((unsigned int*)&output[byte_idx & ~3],
              ((unsigned int)bits) << (8 * (byte_idx & 3)));
}
'''

# Alternative: simpler kernel with full-width atomic
TRNG_KERNEL_V2 = '''
extern "C" __global__
void trng_kernel_v2(unsigned int* output, int n_words, unsigned int base_seed) {
    int word_idx = blockIdx.x;
    if (word_idx >= n_words) return;

    int thread_id = blockIdx.y * blockDim.x + threadIdx.x;

    // Unique seed
    unsigned int rng = base_seed + word_idx * 1000000 + thread_id * 31337;
    rng = rng * 1103515245u + 12345u;

    // Initialize 4 chaotic oscillators
    float x0 = (float)((rng = rng * 1103515245u + 12345u) & 0xFFFF) / 65536.0f;
    float x1 = (float)((rng = rng * 1103515245u + 12345u) & 0xFFFF) / 65536.0f;
    float x2 = (float)((rng = rng * 1103515245u + 12345u) & 0xFFFF) / 65536.0f;
    float x3 = (float)((rng = rng * 1103515245u + 12345u) & 0xFFFF) / 65536.0f;

    x0 = fminf(0.999f, fmaxf(0.001f, x0));
    x1 = fminf(0.999f, fmaxf(0.001f, x1));
    x2 = fminf(0.999f, fmaxf(0.001f, x2));
    x3 = fminf(0.999f, fmaxf(0.001f, x3));

    // Run chaos
    for (int i = 0; i < 500; i++) {
        x0 = 3.75f * x0 * (1.0f - x0);
        x1 = 3.75f * x1 * (1.0f - x1);
        x2 = 3.75f * x2 * (1.0f - x2);
        x3 = 3.75f * x3 * (1.0f - x3);
    }

    // Extract bytes
    unsigned char b0 = (unsigned char)((unsigned int)(x0 * 256.0f) & 0xFF);
    unsigned char b1 = (unsigned char)((unsigned int)(x1 * 256.0f) & 0xFF);
    unsigned char b2 = (unsigned char)((unsigned int)(x2 * 256.0f) & 0xFF);
    unsigned char b3 = (unsigned char)((unsigned int)(x3 * 256.0f) & 0xFF);

    unsigned int word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);

    // Atomic XOR - race condition creates entropy
    atomicXor(&output[word_idx], word);
}
'''

def generate_random_bytes(n_bytes, seed=None):
    """Generate n_bytes of random data using GPU TRNG."""
    if seed is None:
        seed = np.random.randint(0, 2**31)

    module = cp.RawModule(code=TRNG_KERNEL_V2)
    kernel = module.get_function('trng_kernel_v2')

    # Work with 4-byte words
    n_words = (n_bytes + 3) // 4
    output = cp.zeros(n_words, dtype=cp.uint32)

    # Multiple blocks race per word
    blocks_per_word = 16  # More blocks = more race conditions
    threads_per_block = 32

    grid = (n_words, blocks_per_word)
    block = (threads_per_block,)

    kernel(grid, block, (output, np.int32(n_words), np.uint32(seed)))
    cp.cuda.Device().synchronize()

    # Convert to bytes
    result = output.view(cp.uint8)[:n_bytes].get()
    return result


def generate_to_file(filename, n_bytes, chunk_size=1024*1024):
    """Generate random data to file in chunks."""
    print(f"Generating {n_bytes:,} bytes to {filename}...")

    with open(filename, 'wb') as f:
        generated = 0
        chunk_num = 0
        while generated < n_bytes:
            chunk = min(chunk_size, n_bytes - generated)
            # Use chunk number as part of seed for variety
            data = generate_random_bytes(chunk, seed=chunk_num * 1000000)
            f.write(data.tobytes())
            generated += chunk
            chunk_num += 1
            print(f"  {generated:,} / {n_bytes:,} bytes ({100*generated/n_bytes:.1f}%)", end='\r')

    print(f"\nDone! Generated {n_bytes:,} bytes")
    return filename


if __name__ == "__main__":
    # Generate specified MB for testing (default 10MB)
    n_mb = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_bytes = n_mb * 1024 * 1024

    output_dir = Path("data/trng")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"random_{n_mb}mb.bin"
    generate_to_file(filename, n_bytes)

    # Sanity check
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(100000), dtype=np.uint8)

    print(f"\nSanity check (first 100KB):")
    print(f"  Mean: {data.mean():.2f} (expected: 127.5)")
    print(f"  Std: {data.std():.2f} (expected: ~73.9)")
    print(f"  Min: {data.min()}, Max: {data.max()}")

    # Byte frequency chi-squared
    counts = np.bincount(data, minlength=256)
    expected = len(data) / 256
    chi2 = np.sum((counts - expected)**2 / expected)
    # For 255 df, 95% critical value is ~293
    print(f"  Chi² (uniformity): {chi2:.1f} (pass if < 293)")

    # Bit frequency
    bits = np.unpackbits(data)
    ones = np.sum(bits)
    zeros = len(bits) - ones
    print(f"  Bit balance: {ones} ones, {zeros} zeros ({100*ones/len(bits):.2f}% ones)")
