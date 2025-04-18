"""
Utility functions for quantum option pricing.

This package contains various utilities for data loading, visualization,
and performance evaluation.
"""

from utils.data_loader import (
    load_option_chain,
    calculate_implied_volatility,
    extract_risk_free_rate,
    prepare_volatility_surface,
    interpolate_volatility,
    generate_synthetic_option_data,
    save_synthetic_data
)

from utils.plotting import (
    set_plotting_style,
    plot_option_price_comparison,
    plot_option_price_vs_strike,
    plot_convergence,
    plot_volatility_surface,
    plot_option_greeks,
    plot_quantum_vs_classical_performance,
    plot_monte_carlo_paths,
    interactive_option_price_comparison,
    interactive_volatility_surface,
    interactive_circuit_visualization,
    interactive_greek_visualization
)

from utils.metrics import (
    PerformanceTracker,
    calculate_price_difference,
    calculate_speedup,
    calculate_quantum_advantage_threshold,
    calculate_confidence_interval,
    compare_model_results
)

__all__ = [
    # Data loader functions
    'load_option_chain',
    'calculate_implied_volatility',
    'extract_risk_free_rate',
    'prepare_volatility_surface',
    'interpolate_volatility',
    'generate_synthetic_option_data',
    'save_synthetic_data',
    
    # Plotting functions
    'set_plotting_style',
    'plot_option_price_comparison',
    'plot_option_price_vs_strike',
    'plot_convergence',
    'plot_volatility_surface',
    'plot_option_greeks',
    'plot_quantum_vs_classical_performance',
    'plot_monte_carlo_paths',
    'interactive_option_price_comparison',
    'interactive_volatility_surface',
    'interactive_circuit_visualization',
    'interactive_greek_visualization',
    
    # Metrics and evaluation functions
    'PerformanceTracker',
    'calculate_price_difference',
    'calculate_speedup',
    'calculate_quantum_advantage_threshold',
    'calculate_confidence_interval',
    'compare_model_results'
]