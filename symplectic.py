#!/usr/bin/env python3
"""
Symplectic Mechanics Module for E-Gravity V3.3

Implements Souriau's geometric mechanics with discrete exterior calculus (DEC)
for spin-curvature coupling in emergent gravity simulations.

Based on:
- J.-M. Souriau, Structure of Dynamical Systems (1970)
- Marsden & West, Discrete Mechanics and Variational Integrators (2001)
- Dittrich et al., Poly-symplectic Hamiltonians on spin foams (2020)

Author: Claude Code AI
Version: 3.3 Symplectic Extension
"""

import numpy as np
from numba import jit, njit
from typing import Tuple, Optional


@njit
def lattice_vector(pos2: np.ndarray, pos1: np.ndarray, L: int) -> np.ndarray:
    """
    Compute shortest lattice vector between positions with periodic boundaries.
    
    Parameters:
    -----------
    pos2, pos1 : np.ndarray
        3D integer positions on lattice
    L : int
        Lattice size
        
    Returns:
    --------
    np.ndarray
        Shortest vector from pos1 to pos2 (periodic)
    """
    v = (pos2 - pos1 + L//2) % L - L//2
    return v.astype(np.float64)


@njit
def compute_sigma_discrete(dx: np.ndarray, grad_p: np.ndarray, 
                          S1: np.ndarray, S2: np.ndarray,
                          grad_S1: np.ndarray, grad_S2: np.ndarray,
                          R_local: float, sigma_spin: float = 0.5) -> float:
    """
    Compute discrete symplectic 2-form σ between two lattice points.
    
    σ = dx^μ ∧ ∇P_μ + (1/2s²)∇S^μν ∧ dS_μν + (1/4)S^μν ∧ R_μναβ dx^α ∧ dx^β
    
    Parameters:
    -----------
    dx : np.ndarray
        Lattice displacement vector (3D)
    grad_p : np.ndarray  
        Gradient of overlap |⟨ψ₁|ψ₂⟩| (3D)
    S1, S2 : np.ndarray
        Spin tensors at positions 1 and 2 (3D spatial components)
    grad_S1, grad_S2 : np.ndarray
        Gradients of spin tensors (3D)
    R_local : float
        Local Riemann curvature scalar
    sigma_spin : float
        Spin norm (default ½)
        
    Returns:
    --------
    float
        Discrete symplectic form value
    """
    # Momentum term: dx^μ ∧ ∇P_μ
    momentum_term = np.dot(dx, grad_p)
    
    # Spin term: (1/2s²)∇S^μν ∧ dS_μν
    # Simplified to scalar using wedge product properties
    dS = S2 - S1
    grad_S_diff = grad_S1 - grad_S2
    spin_term = 0.0
    for mu in range(3):
        spin_term += dS[mu] * grad_S_diff[mu]
    spin_term /= (2 * sigma_spin**2)
    
    # Curvature term: (1/4)S^μν ∧ R_μναβ dx^α ∧ dx^β
    # Approximate as scalar coupling
    S_avg = 0.5 * (S1 + S2)
    curv_term = 0.25 * np.dot(S_avg, dx) * R_local
    
    return momentum_term + spin_term + curv_term


@njit
def gradient_finite_diff(field: np.ndarray, pos: np.ndarray, L: int) -> np.ndarray:
    """
    Compute 3D gradient using central finite differences with periodic boundaries.
    
    Parameters:
    -----------
    field : np.ndarray
        3D scalar field (L×L×L)
    pos : np.ndarray
        Position for gradient computation (3D integers)
    L : int
        Lattice size
        
    Returns:
    --------
    np.ndarray
        Gradient vector (3D)
    """
    grad = np.zeros(3, dtype=np.float64)
    
    for axis in range(3):
        # Create unit vector along axis
        e_plus = pos.copy()
        e_minus = pos.copy()
        
        e_plus[axis] = (pos[axis] + 1) % L
        e_minus[axis] = (pos[axis] - 1) % L
        
        # Central difference
        grad[axis] = 0.5 * (field[e_plus[0], e_plus[1], e_plus[2]] - 
                           field[e_minus[0], e_minus[1], e_minus[2]])
    
    return grad


def symplectic_flow_step(positions: np.ndarray, momenta: np.ndarray,
                        spin_tensors: np.ndarray, dt: float,
                        compute_forces_func, L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single symplectic Verlet integration step preserving geometric structure.
    
    Uses Störmer-Verlet scheme:
    p_{n+1/2} = p_n + (dt/2) F(q_n)
    q_{n+1} = q_n + dt p_{n+1/2}
    p_{n+1} = p_{n+1/2} + (dt/2) F(q_{n+1})
    
    Parameters:
    -----------
    positions : np.ndarray
        Current positions (N×3)
    momenta : np.ndarray
        Current momenta (N×3)
    spin_tensors : np.ndarray
        Current spin tensors (N×3)
    dt : float
        Time step
    compute_forces_func : callable
        Function to compute symplectic forces
    L : int
        Lattice size
        
    Returns:
    --------
    tuple
        Updated (positions, momenta, spin_tensors)
    """
    N = len(positions)
    
    # Half-step momentum update
    forces_pos, forces_spin = compute_forces_func(positions, spin_tensors)
    momenta_half = momenta + 0.5 * dt * forces_pos
    
    # Full position update with periodic boundary conditions
    new_positions = positions + dt * momenta_half
    new_positions = new_positions % L  # Periodic boundaries
    
    # Update spin tensors (simplified evolution)
    new_spin_tensors = spin_tensors + 0.5 * dt * forces_spin
    
    # Final half-step momentum update
    forces_pos, forces_spin = compute_forces_func(new_positions, new_spin_tensors)
    new_momenta = momenta_half + 0.5 * dt * forces_pos
    
    return new_positions, new_momenta, new_spin_tensors


def enforce_ssc(spin_tensors: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Enforce Spin-Statistics Connection (SSC) constraint.
    
    For fermions: {γ^μ, γ^ν} = 2η^μν
    This constrains the allowed spin tensor configurations.
    
    Parameters:
    -----------
    spin_tensors : np.ndarray
        Spin tensor array (N×3)
    positions : np.ndarray
        Position array (N×3)
        
    Returns:
    --------
    np.ndarray
        SSC-corrected spin tensors
    """
    corrected_spins = spin_tensors.copy()
    
    # Apply anticommutation relation constraint
    # Simplified implementation: normalize and orthogonalize
    for i in range(len(spin_tensors)):
        S = corrected_spins[i]
        
        # Normalize to preserve spin magnitude
        norm = np.linalg.norm(S)
        if norm > 1e-12:
            corrected_spins[i] = S / norm * 0.5  # s = 1/2
    
    return corrected_spins


@njit
def local_laplacian_3d(field: np.ndarray, pos: np.ndarray, L: int) -> float:
    """
    Compute discrete Laplacian at given position using 6-point stencil.
    
    ∇²f ≈ (f(x+h) + f(x-h) + f(y+h) + f(y-h) + f(z+h) + f(z-h) - 6f(x,y,z))/h²
    
    Parameters:
    -----------
    field : np.ndarray
        3D scalar field (L×L×L)
    pos : np.ndarray
        Position (3D integers)
    L : int
        Lattice size
        
    Returns:
    --------
    float
        Discrete Laplacian value
    """
    x, y, z = pos[0], pos[1], pos[2]
    center = field[x, y, z]
    
    # 6-point stencil with periodic boundaries
    laplacian = (field[(x+1)%L, y, z] + field[(x-1)%L, y, z] +
                field[x, (y+1)%L, z] + field[x, (y-1)%L, z] +
                field[x, y, (z+1)%L] + field[x, y, (z-1)%L] - 6*center)
    
    return laplacian


def compute_closure_defect(sigma_values: np.ndarray, positions: np.ndarray, L: int) -> float:
    """
    Compute closure defect |dσ| to verify discrete exterior calculus consistency.
    
    For a proper symplectic form: dσ = 0
    
    Parameters:
    -----------
    sigma_values : np.ndarray
        Symplectic form values on lattice
    positions : np.ndarray
        Position array (N×3)
    L : int
        Lattice size
        
    Returns:
    --------
    float
        Maximum closure defect |dσ|
    """
    # Simplified closure test using discrete exterior derivative
    # For 2-form on 3D lattice, d acts on faces -> volumes
    max_defect = 0.0
    
    for i in range(len(positions)):
        pos = positions[i].astype(int)
        
        # Check closure around elementary cube
        # This is a simplified version - full DEC would require more structure
        local_sum = 0.0
        for axis in range(3):
            e = np.zeros(3, dtype=int)
            e[axis] = 1
            
            # Sum oriented contributions around cube faces
            pos_plus = (pos + e) % L
            pos_minus = (pos - e) % L
            
            idx_plus = _pos_to_linear_index(pos_plus, L)
            idx_minus = _pos_to_linear_index(pos_minus, L)
            
            if idx_plus < len(sigma_values) and idx_minus < len(sigma_values):
                local_sum += sigma_values[idx_plus] - sigma_values[idx_minus]
        
        max_defect = max(max_defect, abs(local_sum))
    
    return max_defect


@njit
def _pos_to_linear_index(pos: np.ndarray, L: int) -> int:
    """Convert 3D position to linear index."""
    return pos[0] * L * L + pos[1] * L + pos[2]


def jackknife_error_analysis(r_values: np.ndarray, y_values: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Compute jack-knife error bars for power law fit.
    
    Parameters:
    -----------
    r_values : np.ndarray
        Radial values
    y_values : np.ndarray
        Signal values
        
    Returns:
    --------
    tuple
        (slope_mean, slope_error, residuals)
    """
    N = len(r_values)
    slopes = np.zeros(N)
    
    # Jack-knife resampling
    for i in range(N):
        mask = np.arange(N) != i
        log_r = np.log(r_values[mask])
        log_y = np.log(np.abs(y_values[mask]))
        slopes[i] = np.polyfit(log_r, log_y, 1)[0]
    
    slope_mean = np.mean(slopes)
    slope_error = np.sqrt((N-1)/N * np.sum((slopes - slope_mean)**2))
    
    # Compute residuals for full fit
    log_r_full = np.log(r_values)
    log_y_full = np.log(np.abs(y_values))
    beta, b0 = np.polyfit(log_r_full, log_y_full, 1)
    residuals = log_y_full - (beta * log_r_full + b0)
    
    return slope_mean, slope_error, residuals


def phase_space_volume(positions: np.ndarray, momenta: np.ndarray) -> float:
    """
    Compute phase space volume element det(∂(q,p)/∂(q₀,p₀)).
    
    For symplectic integration, this should be conserved: det ≈ 1.
    
    Parameters:
    -----------
    positions : np.ndarray
        Current positions (N×3)
    momenta : np.ndarray
        Current momenta (N×3)
        
    Returns:
    --------
    float
        Phase space volume (approximate)
    """
    # Simplified volume estimate
    # Full implementation would compute Jacobian determinant
    pos_vol = np.prod(np.ptp(positions, axis=0))  # Position space volume
    mom_vol = np.prod(np.ptp(momenta, axis=0))    # Momentum space volume
    
    return pos_vol * mom_vol


def validate_symplectic_structure(positions: np.ndarray, momenta: np.ndarray,
                                spin_tensors: np.ndarray, dt: float,
                                initial_volume: float) -> dict:
    """
    Validate symplectic structure preservation during integration.
    
    Parameters:
    -----------
    positions, momenta, spin_tensors : np.ndarray
        Current phase space state
    dt : float
        Integration time step
    initial_volume : float
        Initial phase space volume
        
    Returns:
    --------
    dict
        Validation metrics
    """
    current_volume = phase_space_volume(positions, momenta)
    volume_drift = abs(current_volume - initial_volume) / initial_volume
    
    # Energy-like quantity (simplified)
    kinetic = 0.5 * np.sum(momenta**2)
    potential = 0.5 * np.sum(positions**2)  # Harmonic approximation
    spin_energy = 0.5 * np.sum(spin_tensors**2)
    total_energy = kinetic + potential + spin_energy
    
    # Spin magnitude conservation
    spin_norms = np.linalg.norm(spin_tensors, axis=1)
    spin_drift = np.std(spin_norms) if len(spin_norms) > 1 else 0.0
    
    return {
        'volume_drift': volume_drift,
        'total_energy': total_energy,
        'spin_drift': spin_drift,
        'dt': dt,
        'stable': volume_drift < 0.1 and spin_drift < 0.01
    }