#!/usr/bin/env python3
"""
Analyze wave detector events for correlations with physical movement.

User reported moving laptop around 11PM Chicago time (05:00 UTC).
"""

import json
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

# Load the wave events
events_file = "/home/emoore/coherence_gradient_experiment/events/waves/events_lapbuntu2_20260108_232932.jsonl"

events = []
with open(events_file, 'r') as f:
    for line in f:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"Loaded {len(events)} events")
print()

# Group events by 5-minute windows
window_size_sec = 300  # 5 minutes
event_windows = defaultdict(list)

# Get all timestamps
timestamps = []
for e in events:
    ts = e.get('timestamp_unix', 0)
    if ts > 0:
        timestamps.append(ts)
        window_idx = int(ts // window_size_sec)
        event_windows[window_idx].append(e)

if not timestamps:
    print("No timestamps found!")
    exit(1)

min_ts = min(timestamps)
max_ts = max(timestamps)

print(f"Time range (UTC):")
print(f"  Start: {datetime.fromtimestamp(min_ts, tz=timezone.utc)}")
print(f"  End:   {datetime.fromtimestamp(max_ts, tz=timezone.utc)}")
print()

# Chicago is UTC-6
print(f"Time range (Chicago CST/UTC-6):")
print(f"  Start: {datetime.fromtimestamp(min_ts - 6*3600, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  End:   {datetime.fromtimestamp(max_ts - 6*3600, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Analyze event rate per window
window_indices = sorted(event_windows.keys())
event_counts = [len(event_windows[w]) for w in window_indices]

print(f"Window size: {window_size_sec}s ({window_size_sec/60:.0f} min)")
print(f"Number of windows: {len(window_indices)}")
print()

# Find baseline event rate (median)
baseline_rate = np.median(event_counts)
rate_std = np.std(event_counts)
print(f"Event rate statistics:")
print(f"  Median: {baseline_rate:.1f} events/{window_size_sec}s")
print(f"  Mean:   {np.mean(event_counts):.1f} events/{window_size_sec}s")
print(f"  Std:    {rate_std:.1f}")
print(f"  Min:    {min(event_counts)}")
print(f"  Max:    {max(event_counts)}")
print()

# Find significant spikes (>2σ from median)
print("="*70)
print("SIGNIFICANT RATE SPIKES (>2σ from median)")
print("="*70)
print()
print(f"{'Time (Chicago)':<25} {'Events':<8} {'Rate':<10} {'Z-score':<10}")
print("-"*55)

spikes = []
for i, w in enumerate(window_indices):
    count = event_counts[i]
    z_score = (count - baseline_rate) / (rate_std + 1e-10)

    if abs(z_score) > 2.0:
        ts = w * window_size_sec
        chicago_time = datetime.fromtimestamp(ts - 6*3600, tz=timezone.utc)
        print(f"{chicago_time.strftime('%Y-%m-%d %H:%M:%S'):<25} {count:<8} {count/(window_size_sec/60):.1f}/min   {z_score:>+.2f}σ")
        spikes.append({
            'chicago_time': chicago_time,
            'utc_timestamp': ts,
            'count': count,
            'z_score': z_score,
            'events': event_windows[w]
        })

print()

# Look specifically at the first hour (11 PM - 12 AM Chicago = 05:00-06:00 UTC)
print("="*70)
print("FIRST HOUR DETAIL (11 PM - 12 AM Chicago)")
print("="*70)
print()

# Filter events from first hour
first_hour_start = 5 * 3600 + (min_ts // 86400) * 86400  # 05:00 UTC of that day
first_hour_events = [e for e in events if first_hour_start <= e.get('timestamp_unix', 0) < first_hour_start + 3600]

if first_hour_events:
    # Group by minute
    minute_counts = defaultdict(int)
    for e in first_hour_events:
        ts = e.get('timestamp_unix', 0)
        minute = int(ts // 60) * 60
        minute_counts[minute] += 1

    print(f"{'Minute (Chicago)':<25} {'Events':<8}")
    print("-"*35)

    for minute in sorted(minute_counts.keys())[:30]:  # First 30 minutes
        chicago_time = datetime.fromtimestamp(minute - 6*3600, tz=timezone.utc)
        print(f"{chicago_time.strftime('%Y-%m-%d %H:%M'):<25} {minute_counts[minute]:<8}")

print()

# Analyze event types during spikes
print("="*70)
print("EVENT TYPE DISTRIBUTION DURING SPIKES")
print("="*70)
print()

if spikes:
    for spike in spikes[:5]:  # First 5 spikes
        type_counts = defaultdict(int)
        for e in spike['events']:
            type_counts[e.get('event_type', 'unknown')] += 1

        print(f"{spike['chicago_time'].strftime('%H:%M')}: {dict(type_counts)}")
else:
    print("No significant spikes detected")

print()

# Look at polarization balance during different periods
print("="*70)
print("POLARIZATION BALANCE ANALYSIS")
print("="*70)
print()

# First 30 min vs rest
first_30_min = [e for e in events if e.get('timestamp_unix', 0) < min_ts + 1800]
rest = [e for e in events if e.get('timestamp_unix', 0) >= min_ts + 1800]

def get_polarization_balance(event_list):
    s_plus = sum(1 for e in event_list if e.get('polarization_sign') == 'S+')
    s_minus = sum(1 for e in event_list if e.get('polarization_sign') == 'S-')
    total = s_plus + s_minus
    if total == 0:
        return 0, 0, 0
    return s_plus, s_minus, (s_plus - s_minus) / total

sp1, sm1, bal1 = get_polarization_balance(first_30_min)
sp2, sm2, bal2 = get_polarization_balance(rest)

print(f"First 30 minutes: S+={sp1}, S-={sm1}, balance={bal1:+.3f}")
print(f"Rest of run:      S+={sp2}, S-={sm2}, balance={bal2:+.3f}")
print()

# Look for gradient anomalies (may indicate physical perturbation)
print("="*70)
print("GRADIENT ANOMALIES (may indicate physical movement)")
print("="*70)
print()

gradient_events = [e for e in events if e.get('event_type') == 'gradient_anomaly']
print(f"Total gradient anomalies: {len(gradient_events)}")

# Group by 15-minute windows
grad_windows = defaultdict(int)
for e in gradient_events:
    ts = e.get('timestamp_unix', 0)
    window = int(ts // 900) * 900  # 15-min windows
    grad_windows[window] += 1

if grad_windows:
    print(f"\nGradient anomalies per 15-min window:")
    print(f"{'Time (Chicago)':<25} {'Count':<8}")
    print("-"*35)

    for window in sorted(grad_windows.keys())[:20]:
        chicago_time = datetime.fromtimestamp(window - 6*3600, tz=timezone.utc)
        print(f"{chicago_time.strftime('%Y-%m-%d %H:%M'):<25} {grad_windows[window]:<8}")

print()
print("="*70)
print("SUMMARY")
print("="*70)
print()
print(f"Total events: {len(events)}")
print(f"Duration: {(max_ts - min_ts)/3600:.1f} hours")
print(f"Average rate: {len(events) / ((max_ts - min_ts)/3600):.0f} events/hour")
print(f"Significant spikes: {len(spikes)}")
print(f"Gradient anomalies: {len(gradient_events)}")
