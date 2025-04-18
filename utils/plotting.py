"""
Visualization utilities for option pricing models.

This module provides functions for creating various plots and visualizations
to analyze and compare option pricing models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def set_plotting_style():
    """
    Set the style for matplotlib plots.
    """
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def plot_option_price_comparison(parameter_values, classical_prices, quantum_prices, 
                                parameter_name, title=None, figsize=(10, 6)):
    """
    Plot a comparison of option prices from classical and quantum models.
    
    Parameters:
    -----------
    parameter_values : list or numpy.ndarray
        Values of the parameter being varied
    classical_prices : list or numpy.ndarray
        Option prices from the classical model
    quantum_prices : list or numpy.ndarray
        Option prices from the quantum model
    parameter_name : str
        Name of the parameter being varied
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Set style
    set_plotting_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    ax.plot(parameter_values, classical_prices, 'o-', label='Classical Model', color='#1f77b4')
    ax.plot(parameter_values, quantum_prices, 's-', label='Quantum Model', color='#ff7f0e')
    
    # Add error bars if data is available
    if isinstance(quantum_prices, tuple) and len(quantum_prices) == 3:
        # Unpack mean and error bounds
        mean_prices, lower_bounds, upper_bounds = quantum_prices
        ax.fill_between(parameter_values, lower_bounds, upper_bounds, color='#ff7f0e', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel(parameter_name)
    ax.set_ylabel('Option Price')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Option Price Comparison: Classical vs. Quantum')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_option_price_vs_strike(strikes, call_prices, put_prices, spot_price=None, title=None, figsize=(10, 6)):
    """
    Plot option prices against strike prices.
    
    Parameters:
    -----------
    strikes : list or numpy.ndarray
        Strike prices
    call_prices : list or numpy.ndarray
        Call option prices
    put_prices : list or numpy.ndarray
        Put option prices
    spot_price : float, optional
        Current price of the underlying asset
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Set style
    set_plotting_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    ax.plot(strikes, call_prices, 'o-', label='Call Option', color='#2ca02c')
    ax.plot(strikes, put_prices, 's-', label='Put Option', color='#d62728')
    
    # Add vertical line for spot price
    if spot_price is not None:
        ax.axvline(x=spot_price, color='k', linestyle='--', alpha=0.7, label=f'Spot Price ({spot_price})')
    
    # Set labels and title
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Option Price')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Option Prices vs. Strike Price')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_convergence(num_iterations, classical_errors, quantum_errors, title=None, figsize=(10, 6)):
    """
    Plot convergence of error over iterations for classical and quantum models.
    
    Parameters:
    -----------
    num_iterations : list or numpy.ndarray
        Number of iterations or samples
    classical_errors : list or numpy.ndarray
        Errors for the classical model
    quantum_errors : list or numpy.ndarray
        Errors for the quantum model
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Set style
    set_plotting_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    ax.plot(num_iterations, classical_errors, 'o-', label='Classical Model', color='#1f77b4')
    ax.plot(num_iterations, quantum_errors, 's-', label='Quantum Model', color='#ff7f0e')
    
    # Set log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set labels and title
    ax.set_xlabel('Number of Iterations / Samples')
    ax.set_ylabel('Error')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Convergence Analysis: Classical vs. Quantum')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_volatility_surface(volatility_surface_df, figsize=(12, 8)):
    """
    Plot a 3D volatility surface.
    
    Parameters:
    -----------
    volatility_surface_df : pandas.DataFrame
        DataFrame containing volatility surface data with columns:
        'strike', 'time_to_maturity', and 'implied_volatility'
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Set style
    set_plotting_style()
    
    # Create pivot table
    pivot_df = volatility_surface_df.pivot(
        index='time_to_maturity',
        columns='strike',
        values='implied_volatility'
    )
    
    # Create meshgrid
    X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
    Z = pivot_df.values
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    
    # Set labels and title
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Maturity (years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Implied Volatility Surface')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Adjust view angle
    ax.view_init(elev=30, azim=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_option_greeks(spot_prices, greeks_dict, title=None, figsize=(12, 8)):
    """
    Plot option Greeks against spot prices.
    
    Parameters:
    -----------
    spot_prices : list or numpy.ndarray
        Spot prices of the underlying asset
    greeks_dict : dict
        Dictionary containing Greek values with keys:
        'delta', 'gamma', 'theta', 'vega', 'rho'
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Set style
    set_plotting_style()
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Colors for each Greek
    colors = {
        'delta': '#1f77b4',
        'gamma': '#ff7f0e',
        'theta': '#2ca02c',
        'vega': '#d62728',
        'rho': '#9467bd'
    }
    
    # Plot each Greek
    for i, (greek, values) in enumerate(greeks_dict.items()):
        if i < 5:  # We only have 5 subplots
            ax = axes[i]
            ax.plot(spot_prices, values, '-', label=greek.capitalize(), color=colors[greek])
            ax.set_xlabel('Spot Price')
            ax.set_ylabel(greek.capitalize())
            ax.set_title(f'{greek.capitalize()} vs. Spot Price')
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Remove the last unused subplot
    if len(greeks_dict) < 6:
        fig.delaxes(axes[5])
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Option Greeks', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig


def plot_quantum_vs_classical_performance(quantum_time, classical_time, quantum_accuracy, classical_accuracy, num_qubits, figsize=(12, 6)):
    """
    Plot performance comparison between quantum and classical models.
    
    Parameters:
    -----------
    quantum_time : list or numpy.ndarray
        Execution time for quantum model
    classical_time : list or numpy.ndarray
        Execution time for classical model
    quantum_accuracy : list or numpy.ndarray
        Accuracy for quantum model
    classical_accuracy : list or numpy.ndarray
        Accuracy for classical model
    num_qubits : list or numpy.ndarray
        Number of qubits used
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Set style
    set_plotting_style()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot execution time
    ax1.plot(num_qubits, quantum_time, 's-', label='Quantum Model', color='#ff7f0e')
    ax1.plot(num_qubits, classical_time, 'o-', label='Classical Model', color='#1f77b4')
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot accuracy
    ax2.plot(num_qubits, quantum_accuracy, 's-', label='Quantum Model', color='#ff7f0e')
    ax2.plot(num_qubits, classical_accuracy, 'o-', label='Classical Model', color='#1f77b4')
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


# Plotly versions for interactive visualizations

def interactive_option_price_comparison(parameter_values, classical_prices, quantum_prices, 
                                      parameter_name, option_type='Call', model_names=None):
    """
    Create an interactive plot of option prices from different models.
    
    Parameters:
    -----------
    parameter_values : list or numpy.ndarray
        Values of the parameter being varied
    classical_prices : list or numpy.ndarray
        Option prices from the classical model
    quantum_prices : list or numpy.ndarray
        Option prices from the quantum model
    parameter_name : str
        Name of the parameter being varied
    option_type : str, optional
        Type of option ('Call' or 'Put')
    model_names : list, optional
        Names of the models
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created interactive figure
    """
    # Default model names
    if model_names is None:
        model_names = ['Classical Model', 'Quantum Model']
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=parameter_values,
        y=classical_prices,
        mode='lines+markers',
        name=model_names[0],
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8, symbol='circle')
    ))
    
    # Handle confidence intervals if provided
    if isinstance(quantum_prices, tuple) and len(quantum_prices) == 3:
        # Unpack mean and error bounds
        mean_prices, lower_bounds, upper_bounds = quantum_prices
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=parameter_values,
            y=mean_prices,
            mode='lines+markers',
            name=model_names[1],
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8, symbol='square')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=parameter_values,
            y=upper_bounds,
            mode='lines',
            name='Upper Bound',
            line=dict(color='#ff7f0e', width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=parameter_values,
            y=lower_bounds,
            mode='lines',
            name='Lower Bound',
            line=dict(color='#ff7f0e', width=0),
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.3)',
            showlegend=False
        ))
    else:
        # Add simple line
        fig.add_trace(go.Scatter(
            x=parameter_values,
            y=quantum_prices,
            mode='lines+markers',
            name=model_names[1],
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8, symbol='square')
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{option_type} Option Price Comparison',
        xaxis_title=parameter_name,
        yaxis_title='Option Price',
        legend_title='Model',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def interactive_volatility_surface(volatility_surface_df):
    """
    Create an interactive 3D volatility surface.
    
    Parameters:
    -----------
    volatility_surface_df : pandas.DataFrame
        DataFrame containing volatility surface data with columns:
        'strike', 'time_to_maturity', and 'implied_volatility'
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created interactive figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add 3D surface
    fig.add_trace(go.Surface(
        x=volatility_surface_df['strike'].values,
        y=volatility_surface_df['time_to_maturity'].values,
        z=volatility_surface_df['implied_volatility'].values,
        colorscale='Viridis',
        colorbar=dict(title='Implied Volatility')
    ))
    
    # Update layout
    fig.update_layout(
        title='Implied Volatility Surface',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Time to Maturity (years)',
            zaxis_title='Implied Volatility',
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        template='plotly_white'
    )
    
    return fig


def interactive_circuit_visualization(circuit_data):
    """
    Create an interactive visualization of quantum circuit statistics.
    
    Parameters:
    -----------
    circuit_data : dict
        Dictionary containing circuit data with keys:
        'depth', 'width', 'num_operations', 'num_qubits', etc.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created interactive figure
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Circuit Depth vs. Number of Qubits',
            'Number of Operations vs. Number of Qubits',
            'Execution Time vs. Number of Qubits',
            'Circuit Statistics'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'table'}]
        ]
    )
    
    # Add traces for circuit depth
    fig.add_trace(
        go.Scatter(
            x=circuit_data['num_qubits'],
            y=circuit_data['depth'],
            mode='lines+markers',
            name='Circuit Depth',
            marker=dict(color='#1f77b4')
        ),
        row=1, col=1
    )
    
    # Add traces for number of operations
    fig.add_trace(
        go.Scatter(
            x=circuit_data['num_qubits'],
            y=circuit_data['num_operations'],
            mode='lines+markers',
            name='Number of Operations',
            marker=dict(color='#ff7f0e')
        ),
        row=1, col=2
    )
    
    # Add traces for execution time
    fig.add_trace(
        go.Scatter(
            x=circuit_data['num_qubits'],
            y=circuit_data['execution_time'],
            mode='lines+markers',
            name='Execution Time',
            marker=dict(color='#2ca02c')
        ),
        row=2, col=1
    )
    
    # Add table for circuit statistics
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[
                    ['Max Depth', 'Max Qubits', 'Total Gates', 'Single-Qubit Gates', 'Two-Qubit Gates'],
                    [
                        circuit_data['max_depth'],
                        circuit_data['max_qubits'],
                        circuit_data['total_gates'],
                        circuit_data['single_qubit_gates'],
                        circuit_data['two_qubit_gates']
                    ]
                ],
                fill_color='lavender',
                align='left'
            )
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Quantum Circuit Analysis',
        showlegend=False,
        height=800,
        width=1000,
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text='Number of Qubits', row=1, col=1)
    fig.update_yaxes(title_text='Circuit Depth', row=1, col=1)
    
    fig.update_xaxes(title_text='Number of Qubits', row=1, col=2)
    fig.update_yaxes(title_text='Number of Operations', row=1, col=2)
    
    fig.update_xaxes(title_text='Number of Qubits', row=2, col=1)
    fig.update_yaxes(title_text='Execution Time (s)', row=2, col=1)
    
    return fig


def interactive_greek_visualization(spot_prices, greeks_dict):
    """
    Create an interactive visualization of option Greeks.
    
    Parameters:
    -----------
    spot_prices : list or numpy.ndarray
        Spot prices of the underlying asset
    greeks_dict : dict
        Dictionary containing Greek values with keys:
        'delta', 'gamma', 'theta', 'vega', 'rho'
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created interactive figure
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Delta vs. Spot Price',
            'Gamma vs. Spot Price',
            'Theta vs. Spot Price',
            'Vega vs. Spot Price',
            'Rho vs. Spot Price'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, None]
        ]
    )
    
    # Add traces for each Greek
    greek_colors = {
        'delta': '#1f77b4',
        'gamma': '#ff7f0e',
        'theta': '#2ca02c',
        'vega': '#d62728',
        'rho': '#9467bd'
    }
    
    # Add traces in specific positions
    greek_positions = {
        'delta': (1, 1),
        'gamma': (1, 2),
        'theta': (2, 1),
        'vega': (2, 2),
        'rho': (3, 1)
    }
    
    for greek, values in greeks_dict.items():
        if greek in greek_positions:
            row, col = greek_positions[greek]
            
            fig.add_trace(
                go.Scatter(
                    x=spot_prices,
                    y=values,
                    mode='lines',
                    name=greek.capitalize(),
                    line=dict(color=greek_colors.get(greek, '#1f77b4'), width=2)
                ),
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        title='Option Greeks Analysis',
        showlegend=True,
        height=800,
        width=1000,
        legend_title='Greek',
        template='plotly_white'
    )
    
    # Update axes
    for row, col in greek_positions.values():
        fig.update_xaxes(title_text='Spot Price', row=row, col=col)
    
    # Update specific y-axes
    fig.update_yaxes(title_text='Delta', row=1, col=1)
    fig.update_yaxes(title_text='Gamma', row=1, col=2)
    fig.update_yaxes(title_text='Theta', row=2, col=1)
    fig.update_yaxes(title_text='Vega', row=2, col=2)
    fig.update_yaxes(title_text='Rho', row=3, col=1)
    
    return fig


def interactive_quantum_circuit_diagram(circuit_data):
    """
    Create an interactive visualization of quantum circuit.
    
    Parameters:
    -----------
    circuit_data : dict
        Dictionary containing circuit visualization data
        
    Returns:
    --------
    str
        HTML string for the circuit visualization
    """
    # For interactive circuit visualization, we would typically use Qiskit's
    # built-in visualizations, which return HTML. This function is a placeholder
    # that would be implemented using Qiskit's visualization tools.
    
    # Example usage would be:
    # from qiskit.visualization import circuit_drawer
    # circuit_html = circuit_drawer(circuit, output='mpl', interactive=True)
    
    # For now, return a placeholder message
    return """
    <div style="text-align: center; padding: 20px;">
        <h3>Quantum Circuit Visualization</h3>
        <p>The actual interactive circuit would be rendered here using Qiskit's visualization tools.</p>
    </div>
    """


def plot_monte_carlo_paths(paths, spot_price, strike_price, time_to_maturity, num_paths_to_show=10, figsize=(10, 6)):
    """
    Plot Monte Carlo simulation paths for asset price evolution.
    
    Parameters:
    -----------
    paths : numpy.ndarray
        Array of shape (num_simulations, num_steps+1) containing simulated price paths
    spot_price : float
        Initial price of the underlying asset
    strike_price : float
        Strike price of the option
    time_to_maturity : float
        Time to option expiration in years
    num_paths_to_show : int, optional
        Number of paths to display
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Set style
    set_plotting_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate time points
    num_steps = paths.shape[1] - 1
    time_points = np.linspace(0, time_to_maturity, num_steps + 1)
    
    # Randomly select paths to show
    np.random.seed(42)  # For reproducibility
    num_paths = min(num_paths_to_show, paths.shape[0])
    path_indices = np.random.choice(paths.shape[0], num_paths, replace=False)
    
    # Plot selected paths
    for idx in path_indices:
        ax.plot(time_points, paths[idx], alpha=0.7, linewidth=1)
    
    # Add horizontal line for strike price
    ax.axhline(y=strike_price, color='r', linestyle='--', label=f'Strike Price (K={strike_price})')
    
    # Add horizontal line for initial price
    ax.axhline(y=spot_price, color='g', linestyle='--', label=f'Initial Price (S={spot_price})')
    
    # Set labels and title
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Asset Price')
    ax.set_title('Monte Carlo Simulation: Asset Price Paths')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig