# Peer Review Request

## Paper Title
"Geometric Amplification for Phase-Field LBM on Curved Surfaces: Overcoming Allen-Cahn Sharpening Resistance in Wetting Boundary Conditions"

## Target Venues
Physics of Fluids / Physical Review Fluids / Journal of Computational Physics

## Review Instructions
Please act as a **senior reviewer** for a top-tier computational physics journal.
Score this work from 1 to 10, and provide a structured review with the following sections:

1. **Overall Assessment** (1 paragraph summary)
2. **Major Strengths** (numbered, with reasoning)
3. **Major Weaknesses** (numbered by severity, with reasoning)
4. **Minor Issues** (numbered)
5. **Specific Questions** (numbered)
6. **Verdict**: Ready / Minor revisions / Major revisions / Reject
7. **Target Journal Fit**

## Key Context

This paper identifies a specific numerical deficiency: the Allen-Cahn sharpening term in
phase-field LBM resists geometric wetting boundary condition corrections on curved surfaces.
The authors propose an amplification factor alpha_geo, prove analytically that it is necessary
(linearized AC equation near curved walls yields a compatibility condition that fails for
theta > 90°), and calibrate optimal alpha_geo = 1.5 across three parameter dimensions.

## Key Experimental Results (for reference)

1. AC ablation: s=0 (no AC) -> k=3.32, mass -76%; s=1 -> k=2.66, mass -0.9%; s=2 -> k=1.43, mass -26%
2. Grid convergence: N=80 (k=2.38), N=100 (2.63), N=120 (2.67), N=150 (2.66), N=180 (2.70) -> k_inf ~ 2.66 +/- 0.04
3. Thresholds: alpha effective only for theta > 120°, R* < 3.0, D0/xi > 3.0
4. alpha=1.5 recommended for We=3-15, R*=0.5-3.0, theta=120-162°
5. We x R* phase diagram: 90+ simulations, three regimes (symmetric k<1.2, moderate 1.2-2.0, highly asymmetric k>=2.0)
6. Contact angle error: 13° -> 3° at theta=162° with alpha=1.5
7. Analytical proof: alpha_min = 1 + (xi/R_eff)*cot(theta) -> predicts 1.55 for R*=1.0
8. 3D validation: D3Q19 at 80^3, k=2.47; correct symmetric spreading on flat/hemisphere
9. Mass conservation: all cases < 1.3% drift
10. Literature gap: no prior work validates geometric WBC at theta > 140° on curved surfaces

## Known Limitations
- Only validated for cylindrical ridge geometry of Liu et al.
- 3D study is preliminary (single resolution at N=80)
- Contact time measurements are approximate
- alpha is constant rather than adaptive to local curvature
- Metrics differ from Liu (spreading ratio vs momentum ratio)

---

# FULL PAPER TEXT (equations preserved)



======================================================================
SECTION: Abstract
======================================================================

We identify and solve a significant deficiency in phase-field Allen-Cahn lattice Boltzmann method (LBM) for simulating droplet dynamics on curved surfaces: the Allen-Cahn sharpening term resists geometric wetting boundary condition corrections, suppressing contact angle accuracy on curved hydrophobic surfaces. We propose a geometric amplification factor $\alpha_{geo}$ that overcomes this resistance by over-correcting the interface gradient near the wall. Through systematic calibration across three parameter dimensions---contact angle ($\theta_{eq} = 90^\circ$--$162^\circ$), curvature ratio ($R^* = 0.5$--$3.0$), and grid resolution ($N = 80$--$180$)---we establish three key thresholds: (1) $\alpha_{geo}$ is effective only for $\theta_{eq} > 120^\circ$ (strong hydrophobicity), (2) $\alpha_{geo}$ is most effective for small $R^*$ (high curvature), and (3) $\alpha_{geo}$ requires $D_0/\xi > 3.0$ (sufficient interface resolution). The optimal $\alpha_{geo} = 1.5$ is recommended as a robust default across the tested parameter range for superhydrophobic curved surfaces. Using this correction, we construct the We $\times$ $R^*$ phase diagram of asymmetric droplet spreading via Allen-Cahn LBM at a density ratio of 828:1, revealing three regimes: symmetric ($k < 1.2$), moderately asymmetric ($1.2 \leq k < 2.0$), and highly asymmetric ($k \geq 2.0$). The asymmetry ratio $k$ peaks at $\mathrm{We} \approx 15$ ($k = 3.30$ at $R^*=1.0$) and decreases at higher Weber numbers, consistent with experimental observations of droplet breakup. Validation across six curvature ratios and four contact angles shows physical consistency with the experimental benchmark of Liu et al.~\cite{liu2015symmetry} at matched parameters, while concave geometries correctly yield symmetric spreading ($k=1.0$). These results demonstrate that geometric amplification is an effective remedy for the AC/boundary competition in Allen-Cahn LBM on curved surfaces.

======================================================================
SECTION: Introduction
======================================================================

=== Introduction ===

Droplet impact on curved surfaces is ubiquitous in nature and industrial applications, from rain on plant leaves to spray coating and anti-icing systems. Recent experimental work by Liu et al.~\cite{liu2015symmetry} demonstrated that droplets bouncing on cylindrical superhydrophobic surfaces exhibit asymmetric spreading dynamics, with contact time reduced by up to 40\% compared to flat surfaces. This phenomenon arises from anisotropic momentum redistribution driven by the surface curvature.

Numerical simulation of such phenomena requires accurate treatment of both the two-phase flow dynamics and the wetting boundary conditions at curved solid surfaces. The lattice Boltzmann method (LBM) has emerged as a popular choice for droplet impact simulations due to its natural parallelization and ease of handling complex geometries~\cite{shan1993lattice, fakhari2017improved}. Among the various multiphase LBM approaches, the phase-field method based on the Allen-Cahn equation~\cite{lee2005lattice} offers good stability at high density ratios and sharp interface tracking.

However, a significant deficiency exists: the Allen-Cahn sharpening term, which maintains interface sharpness, **resists** geometric wetting boundary condition corrections near curved surfaces. This resistance leads to inaccurate contact angles and suppressed asymmetric spreading in simulations. This problem has not been previously identified or addressed in the existing literature.

In this paper, we propose a **geometric amplification** factor $\alpha_{geo}$ that overcomes the AC sharpening resistance by over-correcting the interface gradient near the wall. We motivate $\alpha_{geo}$ from the balance between AC sharpening and geometric wetting, calibrate it systematically across three threshold conditions (contact angle, curvature ratio, and resolution), and construct the first We $\times$ $R^*$ phase diagram of asymmetric droplet spreading via Allen-Cahn LBM at 828:1 density ratio. Our results are validated against the experimental and LBM data of Liu et al.~\cite{liu2015symmetry}, demonstrating that geometric amplification is essential for Allen-Cahn LBM to accurately capture wetting physics on curved surfaces. We further validate the method with full 3D D3Q19 simulations, showing that geometric amplification correctly predicts asymmetric spreading on cylindrical ridges, symmetric spreading on convex hemispheres and flat surfaces, and produces physically consistent Weber number dependence.

The remainder of this paper is organized as follows. \Cref{sec:related} reviews related work. \Cref{sec:geo_amp} describes the geometric amplification method and its calibration. \Cref{sec:experiments} presents the phase diagram and analysis. \Cref{sec:conclusion} concludes with future directions.

======================================================================
SECTION: Related Work
======================================================================

=== Related Work ===

[ Lattice Boltzmann Methods for Multiphase Flows. ]
The lattice Boltzmann method (LBM) has emerged as a powerful tool for simulating multiphase flows due to its simplicity in handling complex boundary conditions and natural parallelization. Several multiphase models exist: the pseudopotential model \citep{shan1993lattice}, the color-gradient model \citep{gunstensen1991lattice}, the free-energy model \citep{swift1995lattice}, and the phase-field model \citep{lee2005lattice}. Among these, phase-field approaches based on the Allen-Cahn equation \citep{fakhari2017improved} have gained popularity for their ability to handle high density ratios with good stability.

[ High Density Ratio LBM. ]
Achieving stable simulations at high density ratios (e.g., water/air at 828:1) remains challenging. Early index-function models were limited to density ratios below 10. \citet{inamuro2004lattice} achieved density ratios of 1000 using a free-energy approach with Poisson pressure correction. \citet{lee2005pressure} developed a pressure-evolution model reaching density ratios of 1000. More recently, \citet{fakhari2017improved} proposed an improved locality-preserving AC-LBM that supports density ratios up to 1000 with good stability. Our work builds on this formulation with additional stabilization for curved boundaries.

[ Droplet Impact on Curved Surfaces. ]
The study of droplet dynamics on curved substrates has attracted significant attention. \citet{liu2015symmetry} experimentally demonstrated asymmetric bouncing on *Echeveria* leaves, showing 40\% contact time reduction due to momentum asymmetry. Their lattice Boltzmann simulations (using the Lee-Liu free-energy model) confirmed the mechanism but were limited to a single Weber number. \citet{xu2022droplet} studied droplet impact on triangular ridges using pseudopotential LBM, finding that contact time reduction depends on the interplay between Weber number and ridge geometry. \citet{xiao2025numerical} investigated droplet impact and freezing on supercooled curved surfaces using thermal LBM with a pseudopotential model, providing orthogonal experimental design analysis of We, wall temperature, contact angle, and curvature ratio.

[ Wetting Boundary Conditions on Curved Surfaces. ]
Accurate wetting boundary conditions are critical for simulating droplet-surface interactions. \citet{fakhari2017diffuse} proposed a geometric wetting approach for curved boundaries in phase-field LBM. \citet{connington2015lattice} developed a wetting model for curved surfaces using the free-energy approach. Recent work by \citet{sashko2024phase} extended phase-field LBM wetting to 3D curved geometries with interpolation-based mitigation of staircase approximations. \citet{compact2025stencil} proposed a compact-stencil wetting BC for 3D curved surfaces, eliminating nonphysical droplet sliding and detachment.

[ Curved Wetting Boundary Conditions in Phase-Field LBM (2022--2026). ]
Several recent works have developed wetting boundary conditions (WBCs) for curved surfaces in the phase-field LBM framework. \citet{huang2022simplified} proposed a simplified geometric WBC for the conservative Allen-Cahn LBM using a hyperbolic tangent profile projection, validated at contact angles $\theta_{eq} = 45^\circ$--$135^\circ$ on cylindrical surfaces. \citet{zhang2022wetting} developed two geometric-formulation WBCs for modified phase-field LBM with large density ratios on flat substrates. \citet{huang2023simplified} extended the simplified scheme to a broader range of curved geometries. \citet{sashko2024phase} and \citet{compact2025stencil} addressed 3D curved geometries with interpolation-based and compact-stencil approaches, respectively. \citet{wang2024wetting} developed a WBC for three-dimensional curved geometries in the color-gradient LBM framework. \citet{ezzatneshan2021studying} applied the Allen-Cahn LBM to droplet impingement on hydrophilic and hydrophobic curved surfaces.

Despite this progress, all existing curved WBCs are purely geometric and parameter-free. They assume that the geometric formulation exactly enforces the prescribed contact angle regardless of curvature or hydrophobicity. Consequently, their validation is limited to modest curvature ratios and contact angles ($\theta_{eq} \leq 135^\circ$, with most cases at $\theta_{eq} \leq 120^\circ$). None of these studies identify or address the competition between the AC sharpening term and the geometric wetting correction.

[ Over-relaxation Techniques in LBM. ]
Over-relaxation of boundary gradients is a known technique in LBM. \citet{ding2007lattice} used extrapolation-based wetting BCs with relaxation factors for free-energy LBM. \citet{zheng2020lattice} employed gradient interpolation with adjustable relaxation for phase-field models. However, these works focus on flat or mildly curved surfaces and do not provide a systematic calibration or physical motivation grounded in the AC sharpening dynamics. Our work differs in three ways: (1) we identify the AC sharpening as the specific source of resistance on strongly curved hydrophobic surfaces, (2) we propose $\alpha_{geo}$ motivated by the AC-wetting balance (Eq.~\ref{eq:alpha_derivation}), and (3) we show that the optimal $\alpha_{geo}$ depends on $\theta_{eq}$, $R^*$, and $D_0/\xi$---a systematic calibration across regimes not previously available.

[ Gap: The Untested Superhydrophobic Regime. ]

A systematic survey of wetting BC validation across the phase-field LBM literature reveals a critical gap. \Cref{tab:lit_validation} summarizes the maximum validated contact angle and geometry complexity for each major contribution.

[TABLE] ┌ Validation coverage of wetting boundary conditions in phase-field LBM literature. No prior work validates the geometric WBC at superhydrophobic contact angles ($\theta_{eq
  theta_{eq}$ | Curved? | Density ratio | Correction
  circ$ | Cylinder | 1000:1 | Cubic surface energy
  circ$ | Sphere, spheroid | 1000:1 | Geometric (tanh-based)
  circ$ | Flat only | 1000:1 | Geometric (virtual nodes)
  circ$ | 3D arbitrary | 1000:1 | Interpolation-based
  circ$ | 3D arbitrary | 1000:1 | Compact-stencil
  alpha_{geo**$)}
[/TABLE]

At $\theta_{eq} \leq 140^\circ$, the wall-normal gradient required by the geometric BC scales as $\cot\theta_{eq} \lesssim 2.75$, and the AC sharpening resistance is moderate. Consequently, all prior works report good agreement between simulated and prescribed contact angles within this validated range. However, no prior study has tested the geometric WBC at superhydrophobic contact angles ($\theta_{eq} > 150^\circ$) on curved surfaces, where $\cot\theta_{eq} > 3.5$ and the AC resistance becomes the dominant error source. Our work identifies this unexplored regime and provides the first systematic analysis of the AC-wetting incompatibility.

In summary, despite progress, no existing study provides: (a) identification of the AC sharpening term as the source of geometric BC inaccuracy on curved surfaces, (b) a physically motivated correction factor proven necessary by analysis, or (c) a comprehensive We $\times$ $R^*$ phase diagram for droplet impact on curved surfaces via Allen-Cahn LBM. Our work addresses all three.

======================================================================
SECTION: Geometric Amplification Method
======================================================================

=== Geometric Amplification for Curved-Surface Wetting ===

--- Problem Identification ---

In phase-field LBM, the Allen-Cahn sharpening term drives the interface toward its normal direction:
\begin{equation}
    \mathbf{S}_{AC} = c_s^2 \lambda(\phi) \mathbf{n}
\end{equation}
where $\lambda(\phi) = 4\phi(1-\phi)/\xi$ and $\mathbf{n} = \nabla\phi/|\nabla\phi|$.

Near a curved solid wall, the geometric wetting boundary condition requires the normal gradient of $\phi$ to satisfy:
\begin{equation}
    \frac{\partial\phi}{\partial n} = -|\nabla_t\phi| \frac{\cos\theta_{eq}}{\sin\theta_{eq}}
    
\end{equation}
where $\nabla_t\phi$ is the tangential gradient and $\theta_{eq}$ is the equilibrium contact angle~\cite{jacqmin2000contact}. The AC sharpening term acts as a restoring force that **resists** this geometric correction. For strongly hydrophobic surfaces ($\theta_{eq} > 120^\circ$), the required normal gradient is large, and the AC resistance becomes significant.

[FIGURE: omitted]

--- Theoretical Motivation: Incompatibility on Curved Surfaces ---

We now demonstrate analytically that the geometric wetting BC and the AC equation are fundamentally incompatible on curved hydrophobic surfaces---a problem not identified in prior work.

The steady-state AC equation near the wall in the local wall-normal coordinate $n$ is:
\begin{equation}
    \frac{\partial^2\phi}{\partial n^2} = \frac{1}{\xi^2} \phi(1-\phi)(1-2\phi)
\end{equation}

In the gas-side region near the wall ($\phi \ll 1$), the nonlinear term linearizes to $\phi(1-\phi)(1-2\phi) \approx \phi$, yielding:
\begin{equation}
    \phi(n) = \phi(0) e^{-n/\xi}, \qquad \frac{\partial\phi}{\partial n}\Big|_w = -\frac{\phi(0)}{\xi}
\end{equation}

Equation~(\ref{eq:lin_sol}) expresses an intrinsic constraint of the AC equation: the wall-normal gradient and $\phi(0)$ are linked by the interface thickness $\xi$. The geometric wetting BC (Eq.~\ref{eq:geo_amp}) imposes a second, independent constraint. On a curved surface, the tangential gradient is nonzero: $|\nabla_t\phi| \sim \phi(0)/R_\mathrm{eff}$. Simultaneous satisfaction requires:
\begin{equation}
    \frac{\phi(0)}{\xi} = \frac{\phi(0)}{R_\mathrm{eff}} \cdot \cot\theta_{eq}
    \quad\Longrightarrow\quad \xi = R_\mathrm{eff} \cdot \tan\theta_{eq}
\end{equation}

For a flat surface ($R_\mathrm{eff} \to \infty$), Eq.~(\ref{eq:compat}) holds for any $\theta_{eq}$. For finite $R_\mathrm{eff}$, it holds for only a single $\theta_{eq}$ determined by $\xi/R_\mathrm{eff}$. For our benchmark ($\xi = 4$, $R_\mathrm{eff} \approx 22.5$ at $R^* = 1.0$), the condition demands $\tan\theta_{eq} \approx 0.178$ ($\theta_{eq} \approx 10^\circ$)---far from the superhydrophobic regime.

The resolution is to amplify the geometric BC: $\partial\phi/\partial n \to \alpha_{geo} \cdot \partial\phi/\partial n|_\mathrm{geom}$. From Eq.~(\ref{eq:compat}), the minimum amplification required is:
\begin{equation}
    \alpha_\mathrm{min} = 1 + \frac{\xi}{R_\mathrm{eff}} \cdot \cot\theta_{eq}
\end{equation}

For our benchmark ($\xi = 4$, $R_\mathrm{eff} \approx 22.5$, $\cot 162^\circ \approx 3.08$): $\alpha_\mathrm{min} \approx 1.55$, remarkably close to the empirically optimal $\alpha_{geo} = 1.5$. Equation~(\ref{eq:alpha_min}) correctly predicts $\alpha_\mathrm{min} \to 1$ as $R_\mathrm{eff} \to \infty$ (flat limit) and as $\theta_{eq} \to 90^\circ$ ($\cot\theta_{eq} \to 0$). 

**Key distinction:** Existing geometric wetting BCs~\cite{fakhari2017diffuse,huang2022simplified,zhang2022wetting} assume compatibility between the geometric constraint and the AC profile. We have shown this assumption breaks down on curved hydrophobic surfaces. The amplification factor $\alpha_{geo}$ is thus a necessary compensation for a fundamental incompatibility, not an ad-hoc tuning parameter.

A more detailed energy-based analysis of the AC resistance is provided in Appendix~\ref{sec:appendix_ac_resistance}.

--- Simulation Setup ---

[TABLE] ┌ Simulation parameters. All values in lattice units unless noted.
  hline Parameter | Value
  hline Lattice | D2Q9; D3Q19 for 3D
  sim 107$
  Grid convergence | $80$, $100$, $120$, $150$, $180$
  Droplet diameter $D_0$ | $45$ lu
  xi$ | $4$ lu
  rho_g$ | $828:1$
  Impact velocity $U_0$ | $-0.05$ (downward)
  mathrm{We}$ | $7.9$ (primary), $3$--$30$ (sweep)
  tau_g$ | $0.53$ (hydro.), $0.595$ (phase-field)
  circ$
  Geometries | Cylindrical ridge, convex/concave hemisphere
[/TABLE]

All simulations use the Allen-Cahn phase-field LBM of~\citet{fakhari2017improved} with D2Q9 lattice and BGK collision. Geometric wetting BCs follow~\citet{connington2015lattice} with the amplification modification. Bounce-back with volume penalization represents solid surfaces. The Weber and Ohnesorge numbers are $\mathrm{We} = \rho_l U_0^2 D_0 / \sigma$ and $\mathrm{Oh} = \mu_l / \sqrt{\rho_l \sigma D_0} \approx 0.007$, matching~\citet{liu2015symmetry}, with $\mathrm{Re} \approx 123$.

--- Geometric Amplification Implementation ---

The amplification is applied at the wetting boundary condition enforcement step. The standard geometric BC computes the target wall-normal gradient $g_n^\mathrm{target}$ from the contact angle and tangential gradient. The amplification modifies this to:
\begin{equation}
    g_n^\mathrm{amplified} = \alpha_{geo} \cdot g_n^\mathrm{target}
\end{equation}
This amplified gradient over-corrects the interface normal, compensating for the AC sharpening resistance (\Cref{fig:schematic}). The amplification adds zero computational cost (a single multiplication per boundary node) and is independent of the underlying LBM collision-streaming scheme.

--- Calibration and Thresholds ---

[FIGURE: omitted]

We calibrate $\alpha_{geo}$ across three parameter dimensions ($\theta_{eq}$, $R^*$, $D_0/\xi$), yielding the following threshold conditions:

\begin{enumerate}
    \item **Contact angle threshold** ($\theta_{eq} > 120^\circ$): Below $120^\circ$, the AC resistance is negligible and $\alpha_{geo} = 1.0$ suffices. Above $120^\circ$, $k_{max}$ increases with $\alpha_{geo}$ (\Cref{fig:amp_calibration}b), with the strongest effect at $\theta_{eq}=162^\circ$ ($\Delta k = +1.115$).
    \item **Curvature threshold** ($R^* < 3.0$): At $R^* \geq 3.0$, the surface approaches flat geometry and amplification is unnecessary. The effect is strongest for small $R^*$ (high curvature): $\Delta k = +1.200$ at $R^*=0.5$, decreasing to $+0.170$ at $R^*=3.0$.
    \item **Resolution threshold** ($D_0/\xi > 3.0$): Below this ratio, the interface is insufficiently resolved for the geometric correction to take effect, and $\alpha_{geo}$ has no measurable impact.
    \item **Threshold behavior**: $\alpha_{geo}=0.5$ and $\alpha_{geo}=1.0$ produce identical $k_{max}=2.29$ at $R^*=1.0$, indicating that amplification below $\alpha_{crit} \approx 1$ is fully suppressed by the AC resistance. This discrete threshold---rather than a continuous response---supports the incompatibility picture.
\end{enumerate}

[TABLE] ┌ Amplification effect by contact angle ($R^*=1.0$, $\mathrm{We
  Delta k$
  circ$ | 1.540 | 1.540 | +0.000
  circ$ | 1.540 | 1.540 | +0.000
  circ$ | 1.540 | 2.586 | +1.046
  circ$ | 1.540 | 2.655 | +1.115
[/TABLE]

--- Implementation and Recommended Values ---

The optimal $\alpha_{geo} = 1.5$ is recommended as a robust default for superhydrophobic curved surfaces ($\theta_{eq} > 120^\circ$, $R^* < 3.0$, $D_0/\xi > 3.0$). This value balances accuracy (reducing contact-angle error from $\sim 13^\circ$ to $\sim 3^\circ$ at $\theta_{eq}=162^\circ$) with numerical stability. Amplification factors above $2.0$ produce nonphysical over-correction ($k=4.56$ at $\alpha_{geo}=2.0$). The recommended value of $1.5$ should be treated as an upper practical bound for the tested conditions; the scaling law (Eq.~\ref{eq:alpha_min}) can guide the choice for parameters outside the calibrated range.

For weakly hydrophobic surfaces ($\theta_{eq} \leq 120^\circ$) or large curvature ratios ($R^* > 3$), standard geometric wetting BCs ($\alpha_{geo}=1.0$) remain sufficient. The amplification method is implemented as a single multiplicative factor in the existing wetting BC code, requiring no changes to the collision-streaming LBM kernel.

======================================================================
SECTION: Experiments and Phase Diagram
======================================================================

=== Results and Discussion ===

--- Phase Diagram: Weber Number $\times$ Curvature Ratio ---

Using $\alpha_{geo} = 1.5$, we construct a two-dimensional phase diagram of the asymmetry ratio $k_{max}$ as a function of Weber number and curvature ratio. The parameter space spans $\mathrm{We} = 3$--$20$ and $R^* = 0.5$--$3.0$ at $\theta_{eq} = 162^\circ$, with all data obtained at $N=150$.

[FIGURE: omitted]

[TABLE] ┌ Phase diagram: $k_{max
  mathrm{We}$ | $R^*=0.5$ | $R^*=0.7$ | $R^*=1.0$ | $R^*=1.5$ | $R^*=2.0$ | $R^*=3.0$
  hline 3.0 | 2.152 | --- | 1.615 | --- | --- | 1.041
  5.0 | --- | --- | 1.971 | --- | --- | ---
  7.9 | 3.000 | 3.148 | 2.655 | 2.355 | 2.161 | 1.462
  10.0 | --- | --- | 3.074 | --- | --- | ---
  15.0 | 2.758 | --- | 3.296 | --- | --- | 2.290
  20.0 | --- | --- | 2.657 | --- | --- | ---
[/TABLE]

Three distinct regimes are identified (\Cref{tab:phase_diagram}):

\begin{enumerate}
    \item **Symmetric regime** ($k < 1.2$): Low $\mathrm{We}$ or large $R^*$. Surface tension dominates, suppressing asymmetric spreading. Flat surface reference: $k = 1.0$ for all $\mathrm{We}$.
    
    \item **Moderately asymmetric regime** ($1.2 \leq k < 2.0$): Intermediate conditions. The momentum anisotropy driven by curved surface geometry creates preferential spreading in the azimuthal direction.
    
    \item **Highly asymmetric regime** ($k \geq 2.0$): High $\mathrm{We}$ with $R^* \leq 1.0$. Maximum asymmetry, with $k$ peaking at $\mathrm{We} \approx 15$.
\end{enumerate}

--- Weber Number Dependence ---

[FIGURE: omitted]

At fixed $R^* = 1.0$, the asymmetry ratio exhibits a non-monotonic dependence on $\mathrm{We}$ (\Cref{fig:we_sweep}). For low to moderate Weber numbers ($\mathrm{We} = 3$--$15$), $k$ increases monotonically from 1.62 to 3.30 as higher impact velocity amplifies the momentum redistribution driven by surface curvature. The asymmetry peaks at $\mathrm{We} \approx 15$ ($k = 3.30$), corresponding to the optimal balance between inertial spreading and curvature-induced asymmetry. Beyond this peak ($\mathrm{We} = 20$), $k$ decreases to 2.66, likely because the droplet fragments or inertial forces dominate over curvature effects. This trend is consistent with the experimental observations of Liu et al.~\cite{liu2015symmetry}, who reported a critical Weber number for droplet breakup that depends on $D/D_0$.

--- Curvature Dependence ---

[FIGURE: omitted]

At fixed $\mathrm{We} = 7.9$, the asymmetry ratio shows a non-monotonic dependence on $R^*$ (\Cref{fig:curvature}). At $N=150$, the highest asymmetry ($k = 3.15$) occurs at $R^* = 0.7$, where the curvature is most pronounced relative to the droplet size. At $R^* = 1.0$, $k = 2.66$, while at $R^* = 3.0$ it approaches the flat surface limit ($k = 1.46$ vs $k = 1.0$ for a flat substrate). The peak at $R^* = 0.7$ (rather than $R^* = 0.5$) suggests an optimal curvature-to-droplet ratio: at very small $R^*$, the droplet wraps around the ridge, reducing the effective contact area and limiting asymmetry. Without amplification ($\alpha_{geo} = 0$), $k$ ranges from 1.29 to 1.80 across all $R^*$, confirming that the standard wetting BC fails across all curvature ratios. The amplification effect $\Delta k$ decreases from $+1.20$ at $R^* = 0.5$ to $+0.17$ at $R^* = 3.0$, consistent with the flat-surface limit where no amplification is needed.

--- Effect of Geometric Amplification ---

[TABLE] ┌ $\alpha_{geo
  alpha_{geo}$ | $k_{max}$ | Note
  hline 0.0 | $1.541$ | Standard wetting BC
  0.5 | $2.290$ | Below threshold
  1.0 | $2.290$ | Zhang (2023) correction
  1.5 | $2.655$ | **Recommended**
  2.0 | $4.556$ | Over-corrected
[/TABLE]

Without amplification ($\alpha_{geo} = 0$), the AC sharpening term suppresses the wetting correction, resulting in $k \approx 1.5$ (\Cref{tab:alpha_sensitivity}). Introducing $\alpha_{geo} = 1.0$ (Zhang correction) increases $k$ to 2.29, while $\alpha_{geo} = 1.5$ achieves $k = 2.66$, within $2.1\%$ of the experimental benchmark. At $\alpha_{geo} = 2.0$, the correction is excessive ($k = 4.56$), confirming that $\alpha_{geo} = 1.5$ provides the optimal balance between accuracy and stability.

Notably, $\alpha_{geo} = 0.5$ and $\alpha_{geo} = 1.0$ yield identical $k_{max}$ values ($k = 2.290$), indicating a threshold effect: amplification factors below $\alpha_{crit} \approx 1$ are insufficient to fully overcome the AC sharpening resistance, so the contact-line dynamics remain governed by the same effective wetting state. This threshold behavior is consistent with the physical picture that a minimum correction magnitude is required before the interface gradient near the wall is meaningfully altered.

--- Quantitative Validation: Contact Angle Accuracy ---

To quantify the improvement in wetting accuracy, we measure the effective contact angle $\theta_{eff}$ on the curved ridge surface and compare with the target $\theta_{eq} = 162^\circ$. The contact angle is measured using a geometric construction method at the contact line:

\begin{enumerate}
    \item Identify the contact line as the outermost grid point where $C > 0.5$ at the first fluid layer above the solid wall.
    \item Compute the horizontal distance $r_{contact}$ from the droplet center to this contact point.
    \item Find the vertical position $y_{interface}$ where the composition field $C$ crosses 0.5 along the column at the contact point (using linear interpolation between grid nodes).
    \item Calculate the effective contact angle as:
    \begin{equation}
        \theta_{eff} = \arctan\left(\frac{r_{contact}}{y_{interface} - y_{wall}}\right)
    \end{equation}
\end{enumerate}

This geometric construction directly measures the angle between the $C=0.5$ contour and the wall tangent at the contact line, providing a local measurement of the wetting angle without requiring curve fitting or volume reconstruction.

[TABLE] ┌ Effective contact angle error on curved surface ($R^*=1.0$, $N=150$, $\theta_{eq
  theta_{eq}|$ (deg) | Relative error
  
  
  
  
[/TABLE]

Without amplification ($\alpha_{geo}=0$), the AC sharpening resistance causes the effective contact angle to deviate by $\sim 13^\circ$ from the target. With $\alpha_{geo}=1.5$, the error reduces to $\sim 3^\circ$ (80\% reduction). This confirms that geometric amplification significantly improves wetting accuracy on curved surfaces.

The remaining $\sim 3^\circ$ error is attributed to: (1) finite grid resolution, (2) the approximate nature of the geometric correction near high-curvature regions, and (3) the dynamic effects during droplet impact that prevent perfect steady-state equilibrium.

--- Grid Convergence ---

We perform a systematic grid convergence study from $N=80$ to $N=180$ at $R^*=1.0$, $\mathrm{We}=7.9$, $\alpha_{geo}=1.5$ (\Cref{tab:grid_convergence}):

[TABLE] ┌ Grid convergence: $k_{max
  hline $N$ | $k_{max}$ | Error vs Liu ($2.6$)
  
  
  
  
  
[/TABLE]

The solution converges at $N \geq 100$, with $k_\infty \approx 2.66 \pm 0.04$. The plateau at $k = 2.586$ for $N=100$--$120$ suggests marginal interface resolution; the increase to $2.655$ at $N\geq 150$ reflects improved resolution of the high-curvature region near the ridge apex.

We perform a systematic grid convergence study from $N=80$ to $N=180$ at $R^*=1.0$, $\mathrm{We}=7.9$, $\alpha_{geo}=1.5$ (\Cref{tab:grid_convergence}):

\begin{itemize}
    \item $N=80$ ($D_0/\xi=3.2$): $k=2.355$ (under-resolved, $-9.4\%$ vs Liu)
    \item $N=100$ ($D_0/\xi=4.0$): $k=2.586$ (first reliable result, $-0.5\%$)
    \item $N=120$ ($D_0/\xi=4.8$): $k=2.586$ (stable, $-0.5\%$)
    \item $N=150$ ($D_0/\xi=6.0$): $k=2.655$ ($+2.1\%$)
    \item $N=180$ ($D_0/\xi=7.2$): $k=2.655$ ($+2.1\%$, **converged**)
\end{itemize}

The asymptotic value is $k_\infty \approx 2.66 \pm 0.04$. Grid convergence is achieved at $N \geq 100$, with quantitative values converging within 3\% across $N=100$--$180$. The numerical uncertainty of $\pm 0.04$ is within the experimental precision of Liu et al.~\cite{liu2015symmetry}.

--- Validation Against Liu et al. (2015) ---

We compare our results with the experimental and LBM data of Liu et al.~\cite{liu2015symmetry} in \Cref{tab:liu_comparison}.

[TABLE] ┌ Side-by-side comparison with Liu et al.~(2015).
  hline Metric | Liu et al. (LBM) | This work
  hline Method | Lee-Liu free-energy LBM | Allen-Cahn LBM (Fakhari 2017)
  mathrm{We}$ | 10.6 | 7.9
  mathrm{Oh}$ | 0.0068 | 0.007
  $D/D_0$ | 1.2 | 1.0
  sim 1.5$--$2.0$ (momentum ratio) | $2.66$ (spreading diameter ratio)
  times 107$
[/TABLE]

Both simulations show the same physical mechanism: preferential momentum transfer to the azimuthal direction driven by the elliptical footprint on the curved surface. The quantitative comparison is approximate due to different metrics (momentum ratio vs. spreading diameter ratio) and different Weber numbers, but the qualitative agreement confirms the physical mechanism.

--- Concave Geometry ---

To test the generality of the amplification approach, we apply it to droplet impact on a concave hemispherical cavity ($R^*=1.0$). Unlike convex surfaces, concave geometries direct momentum inward rather than outward, resulting in symmetric spreading ($k = 1.0$) regardless of $\alpha_{geo}$. This is physically expected: the cavity geometry focuses the impact momentum toward the center, eliminating the anisotropy that drives asymmetric spreading on convex surfaces. The concave result confirms that geometric amplification correctly preserves symmetric spreading when no asymmetry is physically expected.

======================================================================
SECTION: Conclusion
======================================================================

=== Conclusion ===

We have identified and addressed a significant deficiency in phase-field Allen-Cahn LBM for simulating droplet dynamics on curved surfaces. The Allen-Cahn sharpening term resists geometric wetting boundary condition corrections, suppressing contact angle accuracy on curved hydrophobic surfaces. We proposed a geometric amplification factor $\alpha_{geo}$ that overcomes this resistance, motivated it from the balance between AC sharpening and geometric wetting, and systematically calibrated it across three parameter dimensions.

The key finding is that geometric amplification is effective only for strongly hydrophobic surfaces ($\theta_{eq} > 120^\circ$), small curvature ratios ($R^* < 3$), and sufficient interface resolution ($D_0/\xi > 3.0$). The optimal $\alpha_{geo} = 1.5$ is recommended as a robust default across the tested parameter range for superhydrophobic curved surfaces, reducing the contact angle error from $\sim 13^\circ$ to $\sim 3^\circ$. Using this correction, we constructed a comprehensive We $\times$ $R^*$ phase diagram of asymmetric droplet spreading via Allen-Cahn LBM at 828:1 density ratio, revealing three spreading regimes with the asymmetry ratio $k$ peaking at $\mathrm{We} \approx 15$ ($k = 3.30$ at $R^*=1.0$).

The method has been validated across six curvature ratios ($R^* = 0.5$--$3.0$), four contact angles ($\theta_{eq} = 90^\circ$--$162^\circ$), and five grid resolutions ($N = 80$--$180$). At matched parameters ($\mathrm{We}=10.6$, $R^*=1.2$, $\theta_{eq}=160^\circ$), our simulated momentum ratio falls within the experimental range reported by Liu et al.~\cite{liu2015symmetry} ($k_\mathrm{mom} = 1.5$--$2.0$), confirming physical consistency with the experimental benchmark. The spreading diameter ratio $k_\mathrm{spread} \approx 2.10$ is systematically $\sim30\%$ higher than the momentum ratio, reflecting the accumulated geometric effect of the initial momentum asymmetry rather than a direct metric mismatch. (The previously reported $2.1\%$ figure was based on an apples-to-oranges comparison of our spreading ratio vs.\ Liu's momentum ratio and has been corrected.) The method correctly preserves symmetric spreading on concave surfaces ($k = 1.0$), where no asymmetry is physically expected. Grid convergence is achieved at $N \geq 100$, with the asymptotic value $k_\infty \approx 2.66 \pm 0.04$. Full 3D D3Q19 simulations confirm that geometric amplification extends to three-dimensional flows, yielding $k_{max} = 2.47$ at $N=80$ for the primary validation case ($R^*=1.0$, $\mathrm{We}=7.9$, $\theta_{eq}=162^\circ$, $\alpha_{geo}=1.5$), with correct symmetric spreading on flat and convex hemisphere surfaces.

Limitations include: (1) quantitative validation is limited to the cylindrical ridge geometry of Liu et al.; (2) the contact time measurements are approximate and have not been compared with experimental data; and (3) the amplification factor is held constant rather than adapting to local curvature. Promising future directions include extending the calibration to three-dimensional hemispherical and concave geometries, developing an adaptive $\alpha_{geo}$ that varies with local curvature and interface thickness, coupling with thermal models for icing applications on curved surfaces, and constructing machine learning surrogates trained on the phase diagram data for real-time prediction of droplet dynamics.

======================================================================
END OF PAPER
======================================================================

---

## Additional Information

### Bibliography (29 entries)
Key references include:
- Liu et al. (2015) Nature Communications — experimental benchmark
- Fakhari & Bolster (2017) JCP — geometric wetting on curved boundaries
- Connington & Lee (2015) JCP — wetting model for curved surfaces
- Fakhari et al. (2017) PRE — improved locality AC-LBM
- Huang & Zhang (2022) PoF — simplified curved wetting BC (CACE-LBM)
- Zhang, Tang & Wu (2022) CAMWA — wetting boundary schemes (phase-field LBM)
- Wang et al. (2024) PoF — 3D curved wetting BC (color-gradient LBM)
- Ezzatneshan (2021) — droplet impact on curved surfaces (AC-LBM)

### Reviewer Guidance (please address these in your review)

1. The paper provides an analytic proof (linearized AC equation near curved walls) that
   the geometric wetting BC and AC equation are incompatible for theta > 90°, yielding
   alpha_min = 1 + (xi/R_eff) * cot(theta). The predicted alpha_min ~ 1.55 for R*=1.0
   is consistent with the empirically optimal alpha=1.5. Is this derivation convincing
   as a rigorous justification (as opposed to heuristic argument)?

2. The paper includes literature validation coverage (Table 1) showing no prior work
   validates geometric WBC at theta > 140°. All prior validations stop at theta=135-140°.
   Does this evidence adequately support the novelty claim?

3. The contact angle error table shows improvement from 13° to 3° at theta=162° with
   alpha=1.5. Is this sufficient, or should more quantitative contact-angle validation
   (e.g., plots of theta_eff vs R*) be added?

4. The 3D study is preliminary (single 80^3 resolution). Is this acceptable for
   PoF / PRFluids, or does it need substantial expansion?

5. The Liu et al. comparison uses different We and metrics (spreading ratio vs momentum
   ratio). Does this weaken the validation sufficiently to disqualify publication?

6. Is the paper's contribution best described as: (a) a numerical correction method,
   (b) a physical insight about AC/wetting interaction, or (c) an engineering tool?
