#!/usr/bin/env python3
"""
Detailed analysis of first 5 minutes - looking for movement signal at sub-minute resolution.
"""

import json
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

events_file = "/home/emoore/coherence_gradient_experiment/events/waves/events_lapbuntu2_20260108_232932.jsonl"

events = []
with open(events_file, 'r') as f:
    for line in f:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

# Get first 5 minutes of events
timestamps = [e.get('timestamp_unix', 0) for e in events if e.get('timestamp_unix', 0) > 0]
min_ts = min(timestamps)

first_5min_events = [e for e in events
                     if min_ts <= e.get('timestamp_unix', 0) < min_ts + 300]

print(f"Events in first 5 minutes: {len(first_5min_events)}")
print(f"Start time (Chicago): {datetime.fromtimestamp(min_ts - 6*3600, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Group by 10-second windows
window_size = 10
ten_sec_windows = defaultdict(list)
for e in first_5min_events:
    ts = e.get('timestamp_unix', 0)
    window = int((ts - min_ts) // window_size)
    ten_sec_windows[window].append(e)

print("="*70)
print("EVENT RATE PER 10-SECOND WINDOW (First 5 minutes)")
print("="*70)
print()
print(f"{'Window':<8} {'Chicago Time':<20} {'Events':<8} {'Gradient':<10} {'σ_values':<15}")
print("-"*65)

event_counts = []
gradient_counts = []
sigma_values = []

for window in range(30):  # 30 x 10s = 5 min
    window_events = ten_sec_windows[window]
    count = len(window_events)
    gradients = sum(1 for e in window_events if e.get('event_type') == 'gradient_anomaly')

    # Get sigma values
    sigmas = [e.get('sigma', 0) for e in window_events]
    avg_sigma = np.mean(sigmas) if sigmas else 0
    max_sigma = max(sigmas) if sigmas else 0

    event_counts.append(count)
    gradient_counts.append(gradients)
    sigma_values.append(avg_sigma)

    chicago_time = datetime.fromtimestamp(min_ts + window * window_size - 6*3600, tz=timezone.utc)

    # Highlight first 2 minutes (windows 0-11)
    marker = "**" if window < 12 else "  "

    print(f"{window:<8} {chicago_time.strftime('%H:%M:%S'):<20} {count:<8} {gradients:<10} {avg_sigma:.2f} (max {max_sigma:.2f}) {marker}")

print()
print("** = First 2 minutes (when movement occurred)")
print()

# Compare first 2 min vs next 3 min
first_2min_events = sum(event_counts[:12])
next_3min_events = sum(event_counts[12:])
first_2min_gradients = sum(gradient_counts[:12])
next_3min_gradients = sum(gradient_counts[12:])

print("="*70)
print("COMPARISON: First 2 min vs Next 3 min")
print("="*70)
print()
print(f"{'Metric':<30} {'First 2 min (moving)':<20} {'Next 3 min (still)':<20}")
print("-"*70)
print(f"{'Total events':<30} {first_2min_events:<20} {next_3min_events:<20}")
print(f"{'Events per 10s (avg)':<30} {first_2min_events/12:.1f}{'':<15} {next_3min_events/18:.1f}")
print(f"{'Gradient anomalies':<30} {first_2min_gradients:<20} {next_3min_gradients:<20}")
print(f"{'Gradient rate per 10s':<30} {first_2min_gradients/12:.2f}{'':<15} {next_3min_gradients/18:.2f}")
print()

# Look at polarization during movement vs still
print("="*70)
print("POLARIZATION DURING MOVEMENT VS STILL")
print("="*70)
print()

first_2min_all = []
next_3min_all = []

for window in range(12):
    first_2min_all.extend(ten_sec_windows[window])
for window in range(12, 30):
    next_3min_all.extend(ten_sec_windows[window])

def analyze_polarization(events_list, name):
    s_plus = sum(1 for e in events_list if e.get('polarization_sign') == 'S+')
    s_minus = sum(1 for e in events_list if e.get('polarization_sign') == 'S-')
    total = s_plus + s_minus
    if total > 0:
        asymmetry = (s_plus - s_minus) / total
    else:
        asymmetry = 0
    print(f"{name}: S+={s_plus}, S-={s_minus}, asymmetry={asymmetry:+.3f}")

analyze_polarization(first_2min_all, "First 2 min (moving)")
analyze_polarization(next_3min_all, "Next 3 min (still)")
print()

# Statistical comparison
print("="*70)
print("STATISTICAL SIGNIFICANCE")
print("="*70)
print()

# Event rate comparison (t-test)
first_2_rates = event_counts[:12]
next_3_rates = event_counts[12:]

mean1, std1 = np.mean(first_2_rates), np.std(first_2_rates)
mean2, std2 = np.mean(next_3_rates), np.std(next_3_rates)

# Simple t-test
pooled_std = np.sqrt((std1**2 / 12 + std2**2 / 18))
t_stat = (mean1 - mean2) / (pooled_std + 1e-10)

print(f"Event rate (per 10s):")
print(f"  First 2 min: {mean1:.2f} ± {std1:.2f}")
print(f"  Next 3 min:  {mean2:.2f} ± {std2:.2f}")
print(f"  Difference:  {mean1 - mean2:.2f}")
print(f"  T-statistic: {t_stat:.2f}")
print()

if abs(t_stat) > 2.0:
    print("*** SIGNIFICANT DIFFERENCE DETECTED ***")
else:
    print("No significant difference in event rate")
print()

# Look at sigma values (event intensity)
print("="*70)
print("EVENT INTENSITY (Sigma values)")
print("="*70)
print()

first_2_sigmas = [e.get('sigma', 0) for e in first_2min_all]
next_3_sigmas = [e.get('sigma', 0) for e in next_3min_all]

print(f"Mean sigma (first 2 min): {np.mean(first_2_sigmas):.3f}")
print(f"Mean sigma (next 3 min):  {np.mean(next_3_sigmas):.3f}")
print(f"Max sigma (first 2 min):  {max(first_2_sigmas):.3f}")
print(f"Max sigma (next 3 min):   {max(next_3_sigmas):.3f}")
