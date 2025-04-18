"""
Quantum models for option pricing.

This package contains various quantum algorithms for pricing financial derivatives.
"""

from quantum.qae_pricing import (
    QuantumAmplitudeEstimationPricer,
    QuantumMonteCarloPricer
)
from quantum.qmc_advanced import (
    PennyLaneQuantumMonteCarlo,
    QuantumKitaevPricer
)

__all__ = [
    'QuantumAmplitudeEstimationPricer',
    'QuantumMonteCarloPricer',
    'PennyLaneQuantumMonteCarlo',
    'QuantumKitaevPricer'
]