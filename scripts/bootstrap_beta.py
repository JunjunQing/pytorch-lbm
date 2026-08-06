#!/usr/bin/env python3
"""Bootstrap uncertainty analysis for β calibration constant.

β relates α_geo to physical parameters through: 
  α_geo - 1 = β · g(θ_eq, R*, ξ/D₀)
  
We estimate β from the grid convergence data and compute confidence intervals.
"""
import numpy as np
import json, os, sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

# ============================================================
# Calibration data
# ============================================================
# Grid convergence at R*=1.0, We=7.9, θ=162°, α=1.5
# k_max values from converged resolutions
N_vals = np.array([80, 100, 120, 150, 180])
k_vals = np.array([2.355, 2.586, 2.586, 2.655, 2.655])

# Only converged resolutions (N >= 100)
N_conv = N_vals[N_vals >= 100]
k_conv = k_vals[N_vals >= 100]

# R* sweep at We=7.9, α=1.5 (N=150)
R_star_vals = np.array([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
k_R = np.array([3.000, 3.148, 2.655, 2.355, 2.161, 1.462])

# Contact angle sweep at R*=1.0, α=1.5 (N=150)
theta_vals = np.array([90, 120, 140, 162])
k_theta = np.array([1.540, 1.540, 2.586, 2.655])

# α sensitivity at R*=1.0, We=7.9 (N=150)
amp_vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
k_amp = np.array([1.541, 2.290, 2.290, 2.655, 4.556])

# ============================================================
# Bootstrap: estimate β from grid convergence data
# ============================================================
# β is calibrated so that the scaling law predicts α=1.5 at baseline
# Baseline: θ=162°, R*=1.0, ξ=4 lu, D₀=45 lu, α=1.5
# Form: α_geo = 1 + β * cos(θ_eq)²/R* * ξ/D₀

theta_baseline = np.radians(162)
R_star_baseline = 1.0
xi = 4.0
D0 = 45.0

def compute_beta_from_k(k, theta=theta_baseline):
    """Infer β from k_max measurement using the AC-wetting balance formula.
    
    Original formula from beta_derivation.tex:
      α_geo = 1 + β × R_AC / |g_target|
      where R_AC = c_s²/2 = 1/6, |g_target| = 2/ξ × |cos(θ)|/sin(θ)
    
    β = (α_geo - 1) × |g_target| / R_AC
    
    α_geo is inferred from k via linear interpolation:
      k = k₀ at α=1.0, k = k_ref at α=1.5
    """
    k0 = 1.541  # k at α_geo=1.0 (no amplification)
    k_ref = 2.655  # k at α_geo=1.5 (calibrated)
    
    # Linear interpolation for α from k
    alpha = 1.0 + (k - k0) * (0.5) / (k_ref - k0)
    alpha = max(alpha, 1.0)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    g_target = (2.0 / xi) * abs(cos_t) / sin_t
    R_AC = 1.0 / 6.0
    beta = (alpha - 1.0) * g_target / R_AC
    return beta

# Bootstrap from grid convergence (all at R*=1.0, θ=162°)
n_bootstrap = 100000
np.random.seed(42)
betas = []
for _ in range(n_bootstrap):
    idx = np.random.choice(len(k_conv), size=len(k_conv), replace=True)
    k_sample = k_conv[idx]
    k_mean = np.mean(k_sample)
    beta = compute_beta_from_k(k_mean)
    betas.append(beta)

betas = np.array(betas)
beta_mean = np.mean(betas)
beta_std = np.std(betas)
beta_ci = np.percentile(betas, [2.5, 97.5])

print("=" * 60)
print("β BOOTSTRAP ANALYSIS")
print("=" * 60)
print(f"Converged resolutions: {N_conv.tolist()}")
print(f"  k values: {k_conv.tolist()}")
print(f"  Mean k: {np.mean(k_conv):.4f}")
print(f"  k uncertainty: ±{np.std(k_conv):.4f} (±{np.std(k_conv)/np.mean(k_conv)*100:.2f}%)")
print()
print(f"Grid parameters: θ=162°, R*=1.0, ξ={xi}, D₀={D0}")
print(f"  |g_target| = 2/{xi} × |cos(162°)|/sin(162°) = 1.539")
print(f"  R_AC = c_s²/2 = 1/6 = 0.1667")
print()
print(f"β from grid convergence bootstrap ({n_bootstrap} samples):")
print(f"  Mean: {beta_mean:.4f}")
print(f"  Std:  {beta_std:.4f}")
print(f"  95% CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]")
print(f"  Full range: [{betas.min():.4f}, {betas.max():.4f}]")
print()

# ============================================================
# Direct computation at each converged resolution
# ============================================================
print("-" * 60)
print("β from each converged resolution:")
print("-" * 60)
betas_direct = []
for N, k in zip(N_conv, k_conv):
    b = compute_beta_from_k(k)
    betas_direct.append(b)
    print(f"  N={N}: k={k:.3f} → β={b:.3f}")

betas_direct = np.array(betas_direct)
print(f"  Mean: {np.mean(betas_direct):.3f}")
print(f"  Range: [{betas_direct.min():.3f}, {betas_direct.max():.3f}]")
print(f"  Half-range uncertainty: ±{(betas_direct.max() - betas_direct.min()) / 2:.3f}")
print()

# ============================================================
# Overall estimate
# ============================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
beta_recommended = np.mean(betas_direct)
beta_uncertainty = (betas_direct.max() - betas_direct.min()) / 2
print(f"  β = {beta_recommended:.2f} ± {beta_uncertainty:.2f}")
print(f"  Match with paper: β = 4.6 ± 0.3")
print(f"  Deviation: Δβ = {beta_recommended - 4.6:.2f}")
print()

# ============================================================
# Save results
# ============================================================
results = {
    'beta_mean': float(beta_recommended),
    'beta_uncertainty': float(beta_uncertainty),
    'beta_std_bootstrap': float(beta_std),
    'beta_ci_95': [float(beta_ci[0]), float(beta_ci[1])],
    'beta_per_resolution': {f'N={int(N)}': float(b) for N, b in zip(N_conv, betas_direct)},
    'method': 'Bootstrap from grid convergence using AC-wetting balance formula',
    'formula': 'β = (α_geo - 1) × |g_target| / R_AC',
    'g_target': 1.539,
    'R_AC': 1.0/6.0,
}
out_path = os.path.join(_project_root, 'results', 'beta_analysis.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {out_path}")
