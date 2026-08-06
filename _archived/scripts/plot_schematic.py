#!/usr/bin/env python3
"""Physical problem schematic for the paper (English labels for matplotlib compatibility)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.font_manager as fm

# Use Noto Sans CJK SC for Chinese text
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Noto Sans CJK SC', 'DejaVu Sans'],
    'font.size': 11,
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
})

fig = plt.figure(figsize=(16, 10))

# ===================================================================
# (a) Side view (x-z plane)
# ===================================================================
ax1 = fig.add_axes([0.05, 0.52, 0.42, 0.42])

R_ridge = 1.0
theta_ridge = np.linspace(0, 2*np.pi, 200)
ax1.fill(R_ridge*np.cos(theta_ridge), R_ridge*np.sin(theta_ridge),
         color='#8B7355', alpha=0.6)
ax1.plot(R_ridge*np.cos(theta_ridge), R_ridge*np.sin(theta_ridge), 'k-', lw=2)

# Droplet
R_drop = 1.0
cx_d, cz_d = 0, R_ridge + R_drop + 0.3
theta_drop = np.linspace(0, 2*np.pi, 200)
ax1.fill(cx_d + R_drop*np.cos(theta_drop), cz_d + R_drop*np.sin(theta_drop),
         color='#4A90D9', alpha=0.4)
ax1.plot(cx_d + R_drop*np.cos(theta_drop), cz_d + R_drop*np.sin(theta_drop),
         'b-', lw=2)

# Velocity arrow
ax1.annotate('', xy=(cx_d, cz_d - R_drop*0.8),
             xytext=(cx_d, cz_d + R_drop*0.3),
             arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
ax1.text(cx_d + 0.15, cz_d - R_drop*0.3, r'$U_0$', fontsize=14, color='red',
         fontweight='bold')

# D0 dimension
ax1.annotate('', xy=(cx_d - R_drop, cz_d + R_drop + 0.3),
             xytext=(cx_d + R_drop, cz_d + R_drop + 0.3),
             arrowprops=dict(arrowstyle='<->', color='navy', lw=1.5))
ax1.text(cx_d, cz_d + R_drop + 0.45, r'$D_0$', fontsize=13, ha='center',
         color='navy', fontweight='bold')

# D ridge dimension
ax1.annotate('', xy=(-R_ridge, -R_ridge - 0.3),
             xytext=(R_ridge, -R_ridge - 0.3),
             arrowprops=dict(arrowstyle='<->', color='#8B4513', lw=1.5))
ax1.text(0, -R_ridge - 0.5, '$D$ (ridge diameter)', fontsize=11, ha='center',
         color='#8B4513')

ax1.text(cx_d, cz_d + 0.1, 'Droplet', fontsize=12, ha='center', color='blue')
ax1.text(0, -0.3, 'Ridge', fontsize=12, ha='center', color='#5B3A1A')

for i in range(-3, 4):
    ax1.axvline(x=i, color='gray', alpha=0.08, lw=0.5)
for j in range(-2, 5):
    ax1.axhline(y=j, color='gray', alpha=0.08, lw=0.5)

ax1.set_xlabel(r'$x$ (perpendicular to ridge)', fontsize=12)
ax1.set_ylabel(r'$z$ (vertical)', fontsize=12)
ax1.set_title(r'(a) Side view ($x$-$z$ cross-section)', fontsize=13, fontweight='bold')
ax1.set_xlim(-3, 3); ax1.set_ylim(-2, 4); ax1.set_aspect('equal')

# ===================================================================
# (b) Top view showing D_x and D_y
# ===================================================================
ax2 = fig.add_axes([0.55, 0.52, 0.42, 0.42])

ridge_w = 2*R_ridge
ridge_len = 5
ax2.fill([-ridge_w/2, ridge_w/2, ridge_w/2, -ridge_w/2],
         [-ridge_len/2, -ridge_len/2, ridge_len/2, ridge_len/2],
         color='#8B7355', alpha=0.5)
ax2.plot([-ridge_w/2, ridge_w/2, ridge_w/2, -ridge_w/2, -ridge_w/2],
         [-ridge_len/2, -ridge_len/2, ridge_len/2, ridge_len/2, -ridge_len/2],
         'k-', lw=2)

spread_a = 2.2
spread_b = 0.9
theta_sp = np.linspace(0, 2*np.pi, 200)
ax2.fill(spread_a*np.cos(theta_sp), spread_b*np.sin(theta_sp),
         color='#4A90D9', alpha=0.4)
ax2.plot(spread_a*np.cos(theta_sp), spread_b*np.sin(theta_sp), 'b-', lw=2)

# D_x
y_m = spread_b + 0.3
ax2.annotate('', xy=(spread_a, y_m), xytext=(-spread_a, y_m),
             arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax2.text(0, y_m + 0.2, r'$D_x$ (perp. to ridge)', fontsize=12,
         ha='center', color='red', fontweight='bold')

# D_y
x_m = spread_a + 0.3
ax2.annotate('', xy=(x_m, spread_b), xytext=(x_m, -spread_b),
             arrowprops=dict(arrowstyle='<->', color='#228B22', lw=2))
ax2.text(x_m + 0.2, 0, r'$D_y$' + '\n(along ridge)', fontsize=11,
         ha='left', va='center', color='#228B22', fontweight='bold')

ax2.text(0, -0.15, r'$k = D_x / D_y$', fontsize=14, ha='center',
         bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='black', lw=1.5),
         fontweight='bold')

ax2.annotate('', xy=(3.0, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax2.annotate('', xy=(0, 3.0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax2.text(3.2, 0, '$x$', fontsize=13, fontweight='bold', va='center')
ax2.text(0, 3.2, '$y$', fontsize=13, fontweight='bold', ha='center')

ax2.text(-ridge_w/2 - 0.3, 0, 'Ridge\n(cylinder)', fontsize=10, ha='right',
         va='center', color='#5B3A1A', rotation=90)

ax2.set_title(r'(b) Top view ($x$-$y$ plane at surface)', fontsize=13, fontweight='bold')
ax2.set_xlim(-3.5, 3.5); ax2.set_ylim(-3.5, 3.5); ax2.set_aspect('equal')
ax2.set_xticks([]); ax2.set_yticks([])

# ===================================================================
# (c) Asymmetry mechanism
# ===================================================================
ax3 = fig.add_axes([0.05, 0.05, 0.42, 0.38])

R = 1.5
theta_half = np.linspace(-np.pi/2, np.pi/2, 100)
x_r = R * np.cos(theta_half)
y_r = R * np.sin(theta_half)

ax3.fill_between(x_r, -2, y_r, color='#8B7355', alpha=0.4)
ax3.fill_between(x_r, y_r, y_r + 0.3, color='#4A90D9', alpha=0.3)
ax3.plot(x_r, y_r, 'k-', lw=2.5)

for angle in [0, np.pi/6, np.pi/3, np.pi/4]:
    for sign in [1, -1]:
        a = sign * angle
        px, py = R*np.cos(a), R*np.sin(a)
        nx, ny = 0.8*np.cos(a), 0.8*np.sin(a)
        ax3.annotate('', xy=(px+nx, py+ny), xytext=(px, py),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax3.text(0, R+1.0, 'Normal divergence\ndrives $x$-spreading', fontsize=11,
         ha='center', color='red', fontweight='bold',
         bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

ax3.annotate('', xy=(R+1.5, -0.5), xytext=(R+0.5, -0.5),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax3.text(R+1.0, -0.9, r'$x$: $D_x$ increases', fontsize=10, ha='center', color='blue')

ax3.set_xlim(-3, 3); ax3.set_ylim(-1.5, 3); ax3.set_aspect('equal')
ax3.set_title('(c) Mechanism: curvature-driven normal divergence', fontsize=13, fontweight='bold')
ax3.set_xlabel(r'$x$ (perpendicular to ridge)')
ax3.set_ylabel('$z$')
ax3.set_xticks([]); ax3.set_yticks([])

# ===================================================================
# (d) Parameter space
# ===================================================================
ax4 = fig.add_axes([0.55, 0.05, 0.42, 0.38])

dd0 = [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.76, 3.5]
k_dd0 = [5.26, 3.21, 3.35, 2.68, 2.58, 1.91, 1.54, 1.31, 1.15]

ax4.semilogy(dd0, k_dd0, 'ro-', ms=10, lw=2, zorder=5, label=r'$\mathrm{We}=7.9$ (this work)')
ax4.axhline(y=1.0, color='gray', ls=':', alpha=0.4, lw=1.5)
ax4.text(3.2, 1.05, r'Flat ($k=1$)', fontsize=10, color='gray')

# We inset
inset = ax4.inset_axes([0.50, 0.55, 0.42, 0.38])
we_vals = [5.0, 7.9, 12.0, 15.0, 23.6]
k_we = [2.04, 2.79, 4.47, 5.13, 6.67]
inset.plot(we_vals, k_we, 'bs-', ms=6, lw=1.5)
inset.set_xlabel(r'$\mathrm{We}$', fontsize=9)
inset.set_ylabel(r'$k_{\max}$', fontsize=9)
inset.set_title(r'$D/D_0=1.0$', fontsize=9)
inset.grid(True, alpha=0.2)
inset.tick_params(labelsize=8)

# Paper targets
ax4.semilogy([1.0], [2.6], 'k*', ms=15, zorder=6, label=r'Liu 2015 ($\mathrm{We}=7.9$)')
ax4.semilogy([2.76], [1.33], 'k*', ms=15, zorder=6)

ax4.annotate(r'$k\approx2.6$', xy=(1.0, 2.6), xytext=(1.6, 3.8),
             fontsize=11, arrowprops=dict(arrowstyle='->', lw=1.5), fontweight='bold')
ax4.annotate(r'$k\approx1.33$', xy=(2.76, 1.33), xytext=(3.2, 1.9),
             fontsize=11, arrowprops=dict(arrowstyle='->', lw=1.5), fontweight='bold')

ax4.set_xlabel(r'$D/D_0$ (ridge-to-droplet diameter ratio)', fontsize=12)
ax4.set_ylabel(r'$k_{\max} = D_x / D_y$', fontsize=12)
ax4.set_title(r'(d) Asymmetry vs ridge curvature', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10, loc='upper right')
ax4.set_xlim(0.3, 4.0); ax4.set_ylim(0.8, 7)
ax4.grid(True, alpha=0.15)

save_path = '/mnt/simulation-projects/pytorch_lbm/paper/figures/schematic.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {save_path}')
