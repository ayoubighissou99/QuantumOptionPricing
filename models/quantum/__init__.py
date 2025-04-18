"""
Quantum models for option pricing.

This package contains various quantum algorithms for pricing financial derivatives.
This is a simplified version that avoids problematic imports.
"""

from models.quantum.qae_pricing import (
    QuantumAmplitudeEstimationPricer,
    QuantumMonteCarloPricer
)
from models.quantum.qmc_advanced import (
    PennyLaneQuantumMonteCarlo,
    QuantumKitaevPricer
)

__all__ = [
    'QuantumAmplitudeEstimationPricer',
    'QuantumMonteCarloPricer',
    'PennyLaneQuantumMonteCarlo',
    'QuantumKitaevPricer'
]