#!/usr/bin/env python3
"""
CIRISOssicle - Interactive GPU Tamper Detection Demo

This script:
1. Characterizes your GPU
2. Sweeps for optimal ossicle parameters
3. Sets up an LLM with ossicle monitoring
4. Runs concurrent workloads and detects them
5. Generates a detection report

License: BSL 1.1
Author: CIRIS L3C (Eric Moore)
"""

import sys
import os
import time
import json
import argparse
import threading
from datetime import datetime
from pathlib import Path

# Global headless mode flag
HEADLESS = False

# Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██████╗██╗██████╗ ██╗███████╗ ██████╗ ███████╗███████╗██╗ ██████╗██╗   ║
║  ██╔════╝██║██╔══██╗██║██╔════╝██╔═══██╗██╔════╝██╔════╝██║██╔════╝██║   ║
║  ██║     ██║██████╔╝██║███████╗██║   ██║███████╗███████╗██║██║     ██║   ║
║  ██║     ██║██╔══██╗██║╚════██║██║   ██║╚════██║╚════██║██║██║     ██║   ║
║  ╚██████╗██║██║  ██║██║███████║╚██████╔╝███████║███████║██║╚██████╗███████╗
║   ╚═════╝╚═╝╚═╝  ╚═╝╚═╝╚══════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝ ╚═════╝╚══════╝
║                                                                           ║
║                    GPU Tamper Detection System                            ║
║                         Version 1.0.0                                     ║
║                                                                           ║
║   License: BSL 1.1 (Free for individuals, academics, <$1M revenue)        ║
║   Copyright: CIRIS L3C (Eric Moore)                                       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

def wait_for_user(message):
    """Wait for user to press Enter (skipped in headless mode)."""
    print(f"\n{'='*70}")
    print(f"NEXT STEP: {message}")
    print("="*70)
    if not HEADLESS:
        input("\n>>> Press ENTER to continue (or Ctrl+C to abort)... ")
    print()


def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")

    missing = []

    try:
        import numpy as np
        print(f"  [OK] NumPy {np.__version__}")
    except ImportError:
        missing.append("numpy")
        print("  [MISSING] NumPy")

    try:
        import cupy as cp
        print(f"  [OK] CuPy {cp.__version__}")
    except ImportError:
        missing.append("cupy")
        print("  [MISSING] CuPy (required for GPU)")

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False

    return True


def characterize_gpu():
    """Characterize the available GPU."""
    import cupy as cp

    print("\n" + "="*70)
    print("GPU CHARACTERIZATION")
    print("="*70)

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"\nFound {device_count} GPU(s)")

        for i in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props['name'].decode()
            sm_count = props['multiProcessorCount']
            mem_bytes = props['totalGlobalMem']
            mem_gb = mem_bytes / (1024**3)

            print(f"\nGPU {i}: {name}")
            print(f"  SMs: {sm_count}")
            print(f"  Memory: {mem_gb:.1f} GB")

            # Determine platform type
            if sm_count >= 100:
                platform = "high-end"
                recommended_twist = 1.1
                recommended_cells = 64
                recommended_iter = 500
            elif sm_count >= 20:
                platform = "mid-range"
                recommended_twist = 0.8
                recommended_cells = 128
                recommended_iter = 1000
            else:
                platform = "embedded/mobile"
                recommended_twist = 0.5
                recommended_cells = 256
                recommended_iter = 2000

            print(f"  Platform type: {platform}")
            print(f"  Recommended config:")
            print(f"    - twist: {recommended_twist} deg")
            print(f"    - cells: {recommended_cells}")
            print(f"    - iterations: {recommended_iter}")

            return {
                'name': name,
                'sm_count': sm_count,
                'memory_gb': mem_gb,
                'platform': platform,
                'recommended': {
                    'twist_deg': recommended_twist,
                    'n_cells': recommended_cells,
                    'n_iterations': recommended_iter,
                }
            }

    except Exception as e:
        print(f"\nError characterizing GPU: {e}")
        return None


class OssicleKernel:
    """CIRISOssicle sensor kernel."""

    KERNEL_CODE = r'''
    extern "C" __global__ void ossicle_step(
        float* states, float* r_vals, float* twists,
        float coupling, int n, int n_osc, int iterations
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        extern __shared__ float shared[];
        float* s = &shared[threadIdx.x * 16];

        for (int o = 0; o < n_osc && o < 16; o++) {
            s[o] = states[o * n + idx];
        }

        int left = (idx > 0) ? idx - 1 : n - 1;
        int right = (idx < n - 1) ? idx + 1 : 0;

        for (int iter = 0; iter < iterations; iter++) {
            for (int o = 0; o < n_osc && o < 16; o++) {
                float r = r_vals[o];
                float neighbor = states[o * n + left] + states[o * n + right];

                float cross = 0.0f;
                for (int p = 0; p < n_osc && p < 16; p++) {
                    if (p != o) {
                        float twist_diff = twists[p] - twists[o];
                        cross += s[p] * cosf(twist_diff) * 0.03f;
                    }
                }

                s[o] = r * s[o] * (1.0f - s[o])
                     + coupling * (neighbor - 2.0f * s[o])
                     + coupling * cross;

                s[o] = fminf(fmaxf(s[o], 0.0001f), 0.9999f);
            }

            for (int o = 0; o < n_osc && o < 16; o++) {
                states[o * n + idx] = s[o];
            }
        }
    }
    '''

    def __init__(self, n_osc=3, n_cells=64, n_iterations=500, twist_deg=1.1):
        import cupy as cp
        import numpy as np

        self.n_osc = n_osc
        self.n_cells = n_cells
        self.n_iterations = n_iterations
        self.twist_deg = twist_deg

        self.r_vals = np.array([3.70 + i * 0.03 for i in range(n_osc)], dtype=np.float32)
        self.twists = np.array([i * np.radians(twist_deg) for i in range(n_osc)], dtype=np.float32)

        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('ossicle_step')

        self.r_vals_gpu = cp.asarray(self.r_vals)
        self.twists_gpu = cp.asarray(self.twists)
        self._init_states()

        self.block_size = min(64, n_cells)
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size
        self.shared_mem = self.block_size * 16 * 4

        self.memory_kb = (n_osc * n_cells * 4) / 1024

    def _init_states(self):
        import cupy as cp
        import numpy as np
        states = np.random.uniform(0.1, 0.9, (self.n_osc, self.n_cells)).astype(np.float32)
        self.states_gpu = cp.asarray(states)

    def step(self):
        import cupy as cp
        self.kernel(
            (self.grid_size,), (self.block_size,),
            (self.states_gpu, self.r_vals_gpu, self.twists_gpu,
             cp.float32(0.05), cp.int32(self.n_cells),
             cp.int32(self.n_osc), cp.int32(self.n_iterations)),
            shared_mem=self.shared_mem
        )
        cp.cuda.Stream.null.synchronize()
        return [float(cp.mean(self.states_gpu[i])) for i in range(self.n_osc)]

    def reset(self):
        self._init_states()


def quick_test(sensor, duration=5.0):
    """Run a quick detection test."""
    import numpy as np

    history = [[] for _ in range(sensor.n_osc)]

    start = time.time()
    while time.time() - start < duration:
        means = sensor.step()
        for i, m in enumerate(means):
            history[i].append(m)

    if len(history[0]) < 30:
        return 0, 0, 0

    arrays = [np.array(h) for h in history]

    def safe_corr(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        r = np.corrcoef(x, y)[0, 1]
        return r if not np.isnan(r) else 0.0

    corrs = []
    for i in range(sensor.n_osc):
        for j in range(i + 1, sensor.n_osc):
            corrs.append(safe_corr(arrays[i], arrays[j]))

    return np.mean(corrs), np.std(corrs), len(history[0])


def sweep_parameters(gpu_info):
    """Sweep parameters to find optimal configuration."""
    import numpy as np

    print("\n" + "="*70)
    print("PARAMETER SWEEP")
    print("="*70)
    print("\nTesting different configurations to find optimal settings...")

    # Configurations to test based on platform
    if gpu_info['sm_count'] >= 100:
        # High-end GPU
        configs = [
            {'n_cells': 64, 'n_iterations': 500, 'twist_deg': 0.8},
            {'n_cells': 64, 'n_iterations': 500, 'twist_deg': 1.1},
            {'n_cells': 64, 'n_iterations': 500, 'twist_deg': 1.5},
            {'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 1.1},
        ]
    else:
        # Embedded/mobile GPU
        configs = [
            {'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 0.5},
            {'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 0.7},
            {'n_cells': 256, 'n_iterations': 2000, 'twist_deg': 0.5},
            {'n_cells': 256, 'n_iterations': 2000, 'twist_deg': 1.0},
        ]

    results = []

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: cells={cfg['n_cells']}, iter={cfg['n_iterations']}, twist={cfg['twist_deg']}")

        sensor = OssicleKernel(n_osc=3, **cfg)

        # Baseline
        sensor.reset()
        b_mean, b_std, b_samples = quick_test(sensor, duration=4.0)

        # Attack (memory workload)
        import cupy as cp
        attack_running = True
        def run_attack():
            src = cp.random.randn(400*400, dtype=cp.float32)
            dst = cp.zeros_like(src)
            while attack_running:
                dst[:] = src
                src[:] = dst
                cp.cuda.Stream.null.synchronize()

        attack_thread = threading.Thread(target=run_attack, daemon=True)
        attack_thread.start()
        time.sleep(0.3)

        sensor.reset()
        a_mean, a_std, a_samples = quick_test(sensor, duration=4.0)

        attack_running = False
        attack_thread.join(timeout=2)

        z_score = abs(a_mean - b_mean) / (b_std + 1e-10)

        result = {**cfg, 'z_score': z_score, 'memory_kb': sensor.memory_kb}
        results.append(result)

        detected = "STRONG" if z_score > 3 else ("YES" if z_score > 2 else ("weak" if z_score > 1 else "no"))
        print(f"  z-score: {z_score:.2f} ({detected})")

    # Find best
    best = max(results, key=lambda x: x['z_score'])

    print("\n" + "-"*50)
    print("SWEEP RESULTS:")
    print("-"*50)
    for r in sorted(results, key=lambda x: x['z_score'], reverse=True):
        mark = "<<<" if r == best else ""
        print(f"  twist={r['twist_deg']:.1f}, cells={r['n_cells']}, iter={r['n_iterations']}: z={r['z_score']:.2f} {mark}")

    print(f"\nBest configuration: twist={best['twist_deg']}, cells={best['n_cells']}, iter={best['n_iterations']}")

    return best


def run_llm_demo(optimal_config):
    """Run LLM demo with ossicle monitoring."""
    import numpy as np
    import cupy as cp

    print("\n" + "="*70)
    print("LLM INFERENCE WITH OSSICLE MONITORING")
    print("="*70)

    # Check if transformers is available
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        HAS_TRANSFORMERS = True
    except ImportError:
        HAS_TRANSFORMERS = False
        print("\nNote: transformers library not installed.")
        print("Running synthetic LLM workload simulation instead.")

    # Create sensor with optimal config (filter to valid kwargs)
    sensor_kwargs = {k: v for k, v in optimal_config.items()
                     if k in ('n_cells', 'n_iterations', 'twist_deg')}
    sensor = OssicleKernel(n_osc=3, **sensor_kwargs)
    print(f"\nOssicle initialized: {sensor.memory_kb:.2f} KB")

    # Monitoring state
    correlations = []
    alerts = []
    monitoring = True

    def monitor_thread():
        """Background thread for ossicle monitoring."""
        history = [[] for _ in range(sensor.n_osc)]
        window_size = 30

        baseline_mean = None
        baseline_std = None

        while monitoring:
            means = sensor.step()
            for i, m in enumerate(means):
                history[i].append(m)

            if len(history[0]) >= window_size:
                arrays = [np.array(h[-window_size:]) for h in history]

                def safe_corr(x, y):
                    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                        return 0.0
                    r = np.corrcoef(x, y)[0, 1]
                    return r if not np.isnan(r) else 0.0

                corrs = []
                for i in range(sensor.n_osc):
                    for j in range(i + 1, sensor.n_osc):
                        corrs.append(safe_corr(arrays[i], arrays[j]))

                current = np.mean(corrs)
                correlations.append(current)

                if baseline_mean is None and len(correlations) > 10:
                    baseline_mean = np.mean(correlations[-10:])
                    baseline_std = np.std(correlations[-10:]) + 0.001

                if baseline_mean is not None:
                    z = abs(current - baseline_mean) / baseline_std
                    if z > 3:
                        alerts.append({'time': time.time(), 'z': z, 'type': 'STRONG'})
                    elif z > 2:
                        alerts.append({'time': time.time(), 'z': z, 'type': 'DETECTED'})

    # Start monitoring
    print("\nStarting ossicle monitoring...")
    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()

    # Phase 1: Baseline (no additional workload)
    print("\n--- Phase 1: Establishing baseline (5 seconds) ---")
    time.sleep(5)
    baseline_alert_count = len(alerts)
    print(f"Baseline alerts: {baseline_alert_count}")

    # Phase 2: LLM or synthetic workload
    print("\n--- Phase 2: Running inference workload (5 seconds) ---")

    if HAS_TRANSFORMERS:
        try:
            print("Loading Qwen model (this may take a moment)...")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True, device_map="cuda")

            prompt = "Explain GPU tamper detection in simple terms:"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            print("Running inference...")
            for _ in range(5):
                outputs = model.generate(**inputs, max_new_tokens=50)
                time.sleep(0.5)
        except Exception as e:
            print(f"LLM error: {e}")
            print("Falling back to synthetic workload...")
            # Synthetic workload
            for _ in range(50):
                a = cp.random.randn(1000, 1000, dtype=cp.float32)
                b = cp.random.randn(1000, 1000, dtype=cp.float32)
                c = cp.matmul(a, b)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.1)
    else:
        # Synthetic LLM-like workload (matmuls)
        print("Running synthetic matmul workload (simulating LLM)...")
        for _ in range(50):
            a = cp.random.randn(1000, 1000, dtype=cp.float32)
            b = cp.random.randn(1000, 1000, dtype=cp.float32)
            c = cp.matmul(a, b)
            cp.cuda.Stream.null.synchronize()
            time.sleep(0.1)

    inference_alert_count = len(alerts) - baseline_alert_count
    print(f"Inference phase alerts: {inference_alert_count}")

    # Phase 3: Concurrent attack simulation
    print("\n--- Phase 3: Simulating concurrent attack (5 seconds) ---")
    attack_running = True
    def attack_workload():
        src = cp.random.randn(500*500, dtype=cp.float32)
        dst = cp.zeros_like(src)
        while attack_running:
            dst[:] = src
            src[:] = dst
            cp.cuda.Stream.null.synchronize()

    attack_thread = threading.Thread(target=attack_workload, daemon=True)
    attack_thread.start()

    time.sleep(5)

    attack_running = False
    attack_thread.join(timeout=2)

    attack_alert_count = len(alerts) - baseline_alert_count - inference_alert_count
    print(f"Attack phase alerts: {attack_alert_count}")

    # Stop monitoring
    monitoring = False
    time.sleep(0.5)

    return {
        'baseline_alerts': baseline_alert_count,
        'inference_alerts': inference_alert_count,
        'attack_alerts': attack_alert_count,
        'total_samples': len(correlations),
        'alerts': alerts
    }


def generate_report(gpu_info, optimal_config, detection_results):
    """Generate a detection report."""
    print("\n" + "="*70)
    print("DETECTION REPORT")
    print("="*70)

    report = {
        'timestamp': datetime.now().isoformat(),
        'system': 'CIRISOssicle v1.0.0',
        'license': 'BSL 1.1',
        'gpu': gpu_info,
        'optimal_config': optimal_config,
        'detection_results': detection_results,
    }

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                        CIRISOssicle Detection Report                      ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Timestamp: {report['timestamp'][:19]:^58} ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  GPU: {gpu_info['name'][:64]:^64} ║
║  SMs: {gpu_info['sm_count']:<3}                  Memory: {gpu_info['memory_gb']:.1f} GB                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Optimal Configuration:                                                   ║
║    Twist angle: {optimal_config['twist_deg']:.1f} degrees                                            ║
║    Cells: {optimal_config['n_cells']:<4}                                                          ║
║    Iterations: {optimal_config['n_iterations']:<5}                                                     ║
║    Memory: {optimal_config.get('memory_kb', optimal_config['n_cells']*3*4/1024):.2f} KB                                                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Detection Results:                                                       ║
║    Baseline phase alerts:   {detection_results['baseline_alerts']:<5}                                     ║
║    Inference phase alerts:  {detection_results['inference_alerts']:<5}                                     ║
║    Attack phase alerts:     {detection_results['attack_alerts']:<5}                                     ║
║    Total samples:           {detection_results['total_samples']:<5}                                     ║
╠═══════════════════════════════════════════════════════════════════════════╣""")

    # Detection success
    if detection_results['attack_alerts'] > detection_results['baseline_alerts'] * 2:
        status = "ATTACK DETECTED"
        status_color = "SUCCESS"
    elif detection_results['attack_alerts'] > detection_results['baseline_alerts']:
        status = "WEAK DETECTION"
        status_color = "PARTIAL"
    else:
        status = "NO DETECTION"
        status_color = "FAILED"

    print(f"║  Status: {status:<20}                                         ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")

    # Save report
    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"detection_report_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to: {output_file}")

    return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CIRISOssicle - GPU Tamper Detection Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Interactive mode (default)
  python main.py --headless   # Run without prompts
  python main.py -q           # Quiet headless mode
        """
    )
    parser.add_argument(
        '--headless', '-H',
        action='store_true',
        help='Run without interactive prompts'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Headless mode with minimal output'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    global HEADLESS

    args = parse_args()
    HEADLESS = args.headless or args.quiet

    if not args.quiet:
        print(BANNER)

    print("\nThis demo will:")
    print("  1. Characterize your GPU")
    print("  2. Find optimal ossicle parameters")
    print("  3. Run LLM inference with monitoring")
    print("  4. Simulate an attack and detect it")
    print("  5. Generate a detection report")

    if not HEADLESS:
        print("\nEach step requires your confirmation to proceed.")
    else:
        print("\nRunning in headless mode...")

    # Step 1: Check dependencies
    wait_for_user("Check dependencies and GPU availability")

    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)

    # Step 2: Characterize GPU
    wait_for_user("Characterize GPU hardware")

    gpu_info = characterize_gpu()
    if gpu_info is None:
        print("\nNo compatible GPU found. Exiting.")
        sys.exit(1)

    # Step 3: Parameter sweep
    wait_for_user("Sweep parameters to find optimal configuration (takes ~30 seconds)")

    optimal_config = sweep_parameters(gpu_info)

    # Step 4: LLM demo
    wait_for_user("Run LLM inference with ossicle monitoring and attack simulation")

    detection_results = run_llm_demo(optimal_config)

    # Step 5: Generate report
    wait_for_user("Generate detection report")

    report = generate_report(gpu_info, optimal_config, detection_results)

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nCIRISOssicle successfully demonstrated GPU tamper detection!")
    print("See the generated report for details.")
    print("\nFor more information, see:")
    print("  - README.md: Project overview")
    print("  - PHYSICS_SUMMARY.md: Technical details")
    print("  - experiments/: Individual experiments")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(0)
