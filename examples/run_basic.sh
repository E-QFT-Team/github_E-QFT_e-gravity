#!/bin/bash
# Basic example run script for E-QFT E-Gravity simulations

echo "Running E-QFT E-Gravity basic example..."
echo "========================================"

# Small lattice test (quick check)
echo -e "\n1. Small lattice test (L=16):"
python ../advanced_eqft.py --L 16 --dim 256 --sigma 1.5 --symplectic-mode

# Medium lattice with wavelet filtering
echo -e "\n2. Medium lattice with wavelet filtering (L=32):"
python ../advanced_eqft.py --L 32 --dim 1024 --sigma 2.0 --symplectic-mode --wavelet-filter bayes

# Production run matching paper results
echo -e "\n3. Production run (L=40):"
python ../advanced_eqft.py --L 40 --dim 1024 --sigma 2.0 --symplectic-mode --debug-symplectic --diag-plots

echo -e "\nAll runs completed! Check the generated PNG files for results."