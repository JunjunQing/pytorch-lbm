"""Generate Re/Ca analysis from We sweep data (density-weighted tau)."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Try to load from GPU run results first
script_dir = os.path.dirname(os.path.abspath(__file__))
we_sweep_path = os.path.join(script_dir, '..', 'scripts', 'we_sweep_v2_results.json')
if os.path.exists(we_sweep_path):
    with open(we_sweep_path) as f:
        we_results = json.load(f)
    We_vals = np.array([r['We'] for r in we_results if r['stable']])
    k_vals = np.array([r['max_k'] for r in we_results if r['stable']])
else:
    # Updated data: We=7.9 confirmed with density-weighted tau
    We_vals = np.array([5.0, 7.9, 12.0, 15.0, 23.6])
    k_vals  = np.array([1.957, 2.579, 3.235, 3.471, 3.842])

Oh = 0.0068  # Fixed Ohnesorge number

# Derived quantities
Re_vals = np.sqrt(We_vals) / Oh
Ca_vals = Oh * np.sqrt(We_vals)

# Liu 2015 experimental data for D/D0=1.0
We_liu = np.array([7.9])
k_liu  = np.array([2.6])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel (a): k vs Re
ax = axes[0]
ax.loglog(Re_vals, k_vals, 'ro-', markersize=8, linewidth=1.5, label='AC-LBM (this work)')
ax.loglog(np.sqrt(We_liu)/Oh, k_liu, 'k*', markersize=15, label='Liu 2015 (exp)')

# Power law fit
log_Re = np.log(Re_vals)
log_k = np.log(k_vals)
coeffs = np.polyfit(log_Re, log_k, 1)
Re_fit = np.logspace(np.log10(Re_vals[0]*0.8), np.log10(Re_vals[-1]*1.2), 100)
k_fit = np.exp(coeffs[1]) * Re_fit**coeffs[0]
ax.loglog(Re_fit, k_fit, 'r--', alpha=0.5, label=f'fit: $k \\propto \\mathrm{{Re}}^{{{coeffs[0]:.2f}}}$')

ax.set_xlabel('Reynolds number $\\mathrm{Re}$', fontsize=12)
ax.set_ylabel('$k_{\\max} = D_x/D_y$', fontsize=12)
ax.set_title('(a) $k$ vs $\\mathrm{Re}$ ($D/D_0=1.0$)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(250, 1200)

# Annotate Re values
for i in range(len(Re_vals)):
    ax.annotate(f'We={We_vals[i]:.1f}', (Re_vals[i], k_vals[i]),
                textcoords='offset points', xytext=(8, 5), fontsize=8, color='gray')

# Panel (b): k vs Ca
ax = axes[1]
ax.loglog(Ca_vals, k_vals, 'bs-', markersize=8, linewidth=1.5, label='AC-LBM (this work)')
ax.loglog(Oh*np.sqrt(We_liu), k_liu, 'k*', markersize=15, label='Liu 2015 (exp)')

# Power law fit
log_Ca = np.log(Ca_vals)
coeffs_ca = np.polyfit(log_Ca, log_k, 1)
Ca_fit = np.logspace(np.log10(Ca_vals[0]*0.8), np.log10(Ca_vals[-1]*1.2), 100)
k_fit_ca = np.exp(coeffs_ca[1]) * Ca_fit**coeffs_ca[0]
ax.loglog(Ca_fit, k_fit_ca, 'b--', alpha=0.5, label=f'fit: $k \\propto \\mathrm{{Ca}}^{{{coeffs_ca[0]:.2f}}}$')

ax.set_xlabel('Capillary number $\\mathrm{Ca}$', fontsize=12)
ax.set_ylabel('$k_{\\max} = D_x/D_y$', fontsize=12)
ax.set_title('(b) $k$ vs $\\mathrm{Ca}$ ($D/D_0=1.0$)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# Annotate Ca values
for i in range(len(Ca_vals)):
    ax.annotate(f'We={We_vals[i]:.1f}', (Ca_vals[i], k_vals[i]),
                textcoords='offset points', xytext=(8, 5), fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('paper/figures/re_ca_analysis.png', dpi=200, bbox_inches='tight')
print(f"Saved re_ca_analysis.png")
print(f"Re range: {Re_vals[0]:.0f} - {Re_vals[-1]:.0f}")
print(f"Ca range: {Ca_vals[0]:.4f} - {Ca_vals[-1]:.4f}")
print(f"k ∝ Re^{coeffs[0]:.2f}")
print(f"k ∝ Ca^{coeffs_ca[0]:.2f}")

# Print the data table for paper
print("\nTable for paper:")
print("We | Re | Ca | k_max")
for i in range(len(We_vals)):
    print(f"{We_vals[i]:.1f} | {Re_vals[i]:.0f} | {Ca_vals[i]:.4f} | {k_vals[i]:.3f}")
