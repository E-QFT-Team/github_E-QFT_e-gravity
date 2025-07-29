#!/usr/bin/env python3
"""
Example analysis script for E-QFT E-Gravity results
Shows how to load and analyze simulation outputs
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_scaling_results(L, sigma, expected_alpha=2.0, expected_beta=-1.0):
    """
    Analyze scaling results from a simulation run
    
    Parameters:
    -----------
    L : int
        Lattice size
    sigma : float
        Localization width
    expected_alpha : float
        Expected distance scaling exponent (should be ~2)
    expected_beta : float
        Expected gravity scaling exponent (should be ~-1)
    """
    
    print(f"\nAnalyzing results for L={L}, σ={sigma}")
    print("-" * 50)
    
    # In a real analysis, you would load the results from saved files
    # For this example, we'll use placeholder values
    
    # Simulated results (replace with actual data loading)
    alpha_measured = 1.95 + 0.1 * np.random.randn()
    alpha_error = 0.05
    beta_measured = -1.02 + 0.1 * np.random.randn()
    beta_error = 0.08
    plateau = 1.98 + 0.02 * np.random.randn()
    
    # Print results
    print(f"Distance scaling α = {alpha_measured:.3f} ± {alpha_error:.3f}")
    print(f"  Deviation from expected: {abs(alpha_measured - expected_alpha)/expected_alpha * 100:.1f}%")
    
    print(f"Gravity scaling β = {beta_measured:.3f} ± {beta_error:.3f}")
    print(f"  Deviation from expected: {abs(beta_measured - expected_beta)/abs(expected_beta) * 100:.1f}%")
    
    print(f"Topology signature c₁ = {plateau:.3f}")
    
    # Newton's constant calibration
    G_eff_lattice = 0.175  # From simulation
    a = 9.229e-35  # m (lattice spacing)
    c = 2.99792458e8  # m/s
    m_Pl = 2.176434e-8  # kg
    
    G_eff_SI = G_eff_lattice * (a * c**2) / m_Pl
    G_Newton = 6.67430e-11  # CODATA value
    
    print(f"\nNewton's constant:")
    print(f"  G_eff = {G_eff_SI:.2e} m³ kg⁻¹ s⁻²")
    print(f"  G_Newton = {G_Newton:.2e} m³ kg⁻¹ s⁻²")
    print(f"  Relative error: {abs(G_eff_SI - G_Newton)/G_Newton * 100:.1f}%")
    
    return {
        'alpha': (alpha_measured, alpha_error),
        'beta': (beta_measured, beta_error),
        'plateau': plateau,
        'G_eff': G_eff_SI
    }

def plot_scaling_comparison():
    """
    Create a comparison plot of scaling results for different parameters
    """
    # Simulation parameters from the paper
    params = [
        (16, 1.5),
        (32, 2.0),
        (40, 2.0),
        (64, 3.0)
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    L_values = []
    alpha_values = []
    beta_values = []
    
    for L, sigma in params:
        results = analyze_scaling_results(L, sigma)
        L_values.append(L)
        alpha_values.append(results['alpha'][0])
        beta_values.append(results['beta'][0])
    
    # Plot distance scaling
    ax1.plot(L_values, alpha_values, 'bo-', markersize=8)
    ax1.axhline(y=2.0, color='r', linestyle='--', label='Expected α=2')
    ax1.set_xlabel('Lattice size L')
    ax1.set_ylabel('Distance scaling exponent α')
    ax1.set_title('Distance Scaling vs Lattice Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot gravity scaling
    ax2.plot(L_values, beta_values, 'ro-', markersize=8)
    ax2.axhline(y=-1.0, color='b', linestyle='--', label='Expected β=-1')
    ax2.set_xlabel('Lattice size L')
    ax2.set_ylabel('Gravity scaling exponent β')
    ax2.set_title('Gravity Scaling vs Lattice Size')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('scaling_comparison.png', dpi=150)
    print("\nScaling comparison plot saved as 'scaling_comparison.png'")

if __name__ == "__main__":
    print("E-QFT E-Gravity Analysis Example")
    print("=" * 50)
    
    # Analyze individual runs
    analyze_scaling_results(L=40, sigma=2.0)
    
    # Create comparison plots
    plot_scaling_comparison()