#!/usr/bin/env python3
"""Analyze Clanet scaling from sweep results."""
import numpy as np

# All sweep results (G_ads=0.0 and G_ads=-1.0)
# Also including previous runs at U=0.10 and U=0.20

# Format: (G_ads, U, We, D_eq, D_max, D_spread, Clanet_pred, err%)
data = {
    # G_ads = 0.0 (baseline)
    "G0_U0.05":  (0.0, 0.05, 0.0162, 0.750, 0.850, 0.100, 0.321, 164.9),
    "G0_U0.10":  (0.0, 0.10, 0.0646, 0.750, 0.925, 0.175, 0.454, 103.9),
    "G0_U0.15":  (0.0, 0.15, 0.1454, 0.750, 1.100, 0.350, 0.556,  97.9),
    "G0_U0.20":  (0.0, 0.20, 0.2584, 0.750, 0.800, 0.050, 0.642,  24.7),
    "G0_U0.25":  (0.0, 0.25, 0.4038, 0.750, 1.200, 0.450, 0.717,  67.3),
    "G0_U0.30":  (0.0, 0.30, 0.5815, 0.750, 1.425, 0.675, 0.786,  81.3),
    # G_ads = -1.0 (hydrophilic)
    "G-1_U0.05": (-1.0, 0.05, 0.0162, 0.725, 0.850, 0.125, 0.321, 164.9),
    "G-1_U0.10": (-1.0, 0.10, 0.0647, 0.725, 0.750, 0.025, 0.454,  65.2),
    "G-1_U0.15": (-1.0, 0.15, 0.1455, 0.725, 1.050, 0.325, 0.556,  88.9),
    "G-1_U0.20": (-1.0, 0.20, 0.2587, 0.725, 0.750, 0.025, 0.642,  16.8),
    "G-1_U0.25": (-1.0, 0.25, 0.4042, 0.725, 1.050, 0.325, 0.718,  46.3),
    "G-1_U0.30": (-1.0, 0.30, 0.5821, 0.725, 1.300, 0.575, 0.786,  65.4),
}

print("=" * 80)
print("  CLANET SCALING ANALYSIS")
print("=" * 80)

# Extract by G_ads
for g_label, g_val in [("G_ads=0.0 (baseline)", 0.0), ("G_ads=-1.0 (hydrophilic)", -1.0)]:
    subset = {k: v for k, v in data.items() if v[0] == g_val}
    we_vals = np.array([v[2] for v in subset.values()])
    d_spread = np.array([v[5] for v in subset.values()])
    clanet = np.array([v[6] for v in subset.values()])
    err = np.array([v[7] for v in subset.values()])

    print(f"\n--- {g_label} ---")
    print(f"  We range: [{we_vals.min():.4f}, {we_vals.max():.4f}]")
    print(f"  D_spread range: [{d_spread.min():.3f}, {d_spread.max():.3f}]")
    print(f"  Clanet err range: [{err.min():.1f}%, {err.max():.1f}%]")
    print(f"  Best Clanet err: {err.min():.1f}% at We={we_vals[err.argmin()]:.4f}")

# D_spread vs We power law fit
print(f"\n--- Power Law Fit: D_spread = A * We^B ---")
for g_val, g_name in [(0.0, "G=0"), (-1.0, "G=-1")]:
    subset = {k: v for k, v in data.items() if v[0] == g_val}
    we_vals = np.array([v[2] for v in subset.values()])
    d_spread = np.array([v[5] for v in subset.values()])

    # Filter out zero spread values
    mask = d_spread > 0
    if mask.sum() < 2:
        print(f"  {g_name}: not enough data points")
        continue

    log_we = np.log10(we_vals[mask])
    log_ds = np.log10(d_spread[mask])

    # Linear fit in log space
    coeffs = np.polyfit(log_we, log_ds, 1)
    B = coeffs[0]
    A = 10**coeffs[1]

    print(f"  {g_name}: D_spread = {A:.4f} * We^{B:.2f}")
    print(f"    Clanet scaling predicts: D_spread ~ We^0.25")

# Compare D_max/D0 vs Clanet prediction
print(f"\n--- D_max/D0 vs Clanet: D_max = 0.9 * We^0.25 ---")
print(f"{'G_ads':>6} {'U':>5} {'We':>7} {'D_max':>6} {'Clanet':>6} {'err%':>6}")
print("-" * 40)
for name in sorted(data.keys()):
    v = data[name]
    print(f"{v[0]:6.1f} {v[1]:5.2f} {v[2]:7.4f} {v[4]:6.3f} {v[6]:6.3f} {v[7]:6.1f}%")

# Check if D_spread follows scaling
print(f"\n--- D_spread vs We^0.25 (normalized) ---")
for g_val in [0.0, -1.0]:
    subset = {k: v for k, v in data.items() if v[0] == g_val}
    spreads = []
    for v in subset.values():
        we = v[2]
        ds = v[5]
        clanet_spread = 0.9 * we**0.25  # Expected from Clanet
        spreads.append((we, ds, clanet_spread, ds/clanet_spread if clanet_spread > 0 else 0))

    g_name = f"G_ads={g_val:.1f}"
    print(f"\n  {g_name}:")
    for we, ds, cs, ratio in sorted(spreads):
        print(f"    We={we:.4f}: D_spread={ds:.3f}, expected={cs:.3f}, ratio={ratio:.2f}")
