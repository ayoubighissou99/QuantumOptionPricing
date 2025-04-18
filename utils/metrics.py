"""
Metrics and evaluation utilities for option pricing models.

This module provides functions for evaluating and comparing the performance
of different option pricing models.
"""

import time
import numpy as np
import pandas as pd
from collections import defaultdict


class PerformanceTracker:
    """
    Track and compare performance metrics of different option pricing models.
    """
    
    def __init__(self):
        """
        Initialize the performance tracker.
        """
        self.metrics = defaultdict(dict)
        self.execution_times = defaultdict(list)
        self.accuracy_metrics = defaultdict(list)
        self.last_reference_price = None
    
    def time_execution(self, model_name, func, *args, **kwargs):
        """
        Time the execution of a function and store the result.
        
        Parameters:
        -----------
        model_name : str
            Name of the model being timed
        func : callable
            Function to time
        *args, **kwargs
            Arguments to pass to the function
            
        Returns:
        --------
        result
            Result of the function call
        float
            Execution time in seconds
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.execution_times[model_name].append(execution_time)
        
        return result, execution_time
    
    def calculate_accuracy(self, model_name, predicted_price, true_price=None):
        """
        Calculate accuracy metrics for a model prediction.
        
        Parameters:
        -----------
        model_name : str
            Name of the model being evaluated
        predicted_price : float
            Predicted option price
        true_price : float, optional
            True option price (if available)
            
        Returns:
        --------
        dict
            Dictionary containing accuracy metrics
        """
        # If true price is not provided, use the last reference price
        if true_price is None:
            if self.last_reference_price is None:
                raise ValueError("No reference price available")
            true_price = self.last_reference_price
        else:
            # Store this as the reference price for future comparisons
            self.last_reference_price = true_price
        
        # Calculate relative error
        if true_price != 0:
            relative_error = abs(predicted_price - true_price) / true_price
        else:
            relative_error = float('inf') if predicted_price != 0 else 0.0
        
        # Calculate absolute error
        absolute_error = abs(predicted_price - true_price)
        
        # Store metrics
        metrics = {
            'predicted_price': predicted_price,
            'true_price': true_price,
            'absolute_error': absolute_error,
            'relative_error': relative_error
        }
        
        self.accuracy_metrics[model_name].append(metrics)
        
        return metrics
    
    def add_custom_metric(self, model_name, metric_name, value):
        """
        Add a custom metric for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        metric_name : str
            Name of the metric
        value : any
            Value of the metric
        """
        if model_name not in self.metrics:
            self.metrics[model_name] = {}
        
        if metric_name not in self.metrics[model_name]:
            self.metrics[model_name][metric_name] = []
        
        self.metrics[model_name][metric_name].append(value)
    
    def get_average_execution_time(self, model_name=None):
        """
        Get the average execution time for a model.
        
        Parameters:
        -----------
        model_name : str, optional
            Name of the model. If None, returns for all models.
            
        Returns:
        --------
        dict or float
            Average execution time for the specified model(s)
        """
        if model_name is not None:
            if model_name in self.execution_times:
                return np.mean(self.execution_times[model_name])
            else:
                return None
        else:
            return {model: np.mean(times) for model, times in self.execution_times.items()}
    
    def get_average_accuracy(self, model_name=None, metric='relative_error'):
        """
        Get the average accuracy for a model.
        
        Parameters:
        -----------
        model_name : str, optional
            Name of the model. If None, returns for all models.
        metric : str, optional
            Name of the accuracy metric to use
            
        Returns:
        --------
        dict or float
            Average accuracy for the specified model(s)
        """
        if model_name is not None:
            if model_name in self.accuracy_metrics:
                return np.mean([m[metric] for m in self.accuracy_metrics[model_name]])
            else:
                return None
        else:
            return {model: np.mean([m[metric] for m in metrics]) 
                   for model, metrics in self.accuracy_metrics.items()}
    
    def get_summary_statistics(self):
        """
        Get summary statistics for all models.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing summary statistics
        """
        models = list(set(list(self.execution_times.keys()) + 
                         list(self.accuracy_metrics.keys()) +
                         list(self.metrics.keys())))
        
        stats = []
        for model in models:
            model_stats = {'model': model}
            
            # Add execution time statistics
            if model in self.execution_times and self.execution_times[model]:
                times = self.execution_times[model]
                model_stats['avg_execution_time'] = np.mean(times)
                model_stats['std_execution_time'] = np.std(times)
                model_stats['min_execution_time'] = np.min(times)
                model_stats['max_execution_time'] = np.max(times)
            
            # Add accuracy statistics
            if model in self.accuracy_metrics and self.accuracy_metrics[model]:
                abs_errors = [m['absolute_error'] for m in self.accuracy_metrics[model]]
                rel_errors = [m['relative_error'] for m in self.accuracy_metrics[model]]
                
                model_stats['avg_absolute_error'] = np.mean(abs_errors)
                model_stats['std_absolute_error'] = np.std(abs_errors)
                model_stats['avg_relative_error'] = np.mean(rel_errors)
                model_stats['std_relative_error'] = np.std(rel_errors)
            
            # Add custom metrics
            if model in self.metrics:
                for metric_name, values in self.metrics[model].items():
                    if values:
                        model_stats[f'avg_{metric_name}'] = np.mean(values)
                        model_stats[f'std_{metric_name}'] = np.std(values)
            
            stats.append(model_stats)
        
        return pd.DataFrame(stats)
    
    def reset(self):
        """
        Reset all tracked metrics.
        """
        self.metrics = defaultdict(dict)
        self.execution_times = defaultdict(list)
        self.accuracy_metrics = defaultdict(list)
        self.last_reference_price = None


def calculate_price_difference(price1, price2):
    """
    Calculate the absolute and relative difference between two prices.
    
    Parameters:
    -----------
    price1 : float
        First price
    price2 : float
        Second price
        
    Returns:
    --------
    tuple
        (absolute_difference, relative_difference)
    """
    absolute_diff = abs(price1 - price2)
    
    # Calculate relative difference
    if price2 != 0:
        relative_diff = absolute_diff / price2
    else:
        relative_diff = float('inf') if price1 != 0 else 0.0
    
    return absolute_diff, relative_diff


def calculate_speedup(classical_time, quantum_time):
    """
    Calculate the speedup factor of quantum vs classical computation.
    
    Parameters:
    -----------
    classical_time : float
        Execution time for the classical algorithm
    quantum_time : float
        Execution time for the quantum algorithm
        
    Returns:
    --------
    float
        Speedup factor (classical_time / quantum_time)
    """
    if quantum_time == 0:
        return float('inf')
    
    return classical_time / quantum_time


def calculate_quantum_advantage_threshold(num_qubits, classical_complexity_factor=2.0):
    """
    Calculate the theoretical threshold where quantum advantage should occur.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits
    classical_complexity_factor : float, optional
        Factor for classical complexity (typically 2.0 for exponential speedup)
        
    Returns:
    --------
    float
        Threshold value
    """
    # For typical Monte Carlo, quantum advantage is O(2^n) vs O(2^(n/2))
    # This is a simplified model and actual thresholds would depend on many factors
    quantum_complexity = 2 ** (num_qubits / 2)
    classical_complexity = 2 ** num_qubits
    
    return classical_complexity / quantum_complexity * classical_complexity_factor


def calculate_confidence_interval(values, confidence=0.95):
    """
    Calculate confidence interval for a set of values.
    
    Parameters:
    -----------
    values : list or numpy.ndarray
        Values to calculate confidence interval for
    confidence : float, optional
        Confidence level (0 to 1)
        
    Returns:
    --------
    tuple
        (lower_bound, upper_bound)
    """
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std_err = np.std(values, ddof=1) / np.sqrt(n)
    
    # For small samples, use t-distribution
    from scipy import stats
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    lower_bound = mean - t_val * std_err
    upper_bound = mean + t_val * std_err
    
    return lower_bound, upper_bound


def compare_model_results(models_results, reference_model=None):
    """
    Compare results from multiple models.
    
    Parameters:
    -----------
    models_results : dict
        Dictionary mapping model names to their results
    reference_model : str, optional
        Name of the model to use as reference for comparison
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing comparison metrics
    """
    if reference_model is None:
        # Use the first model as reference
        reference_model = list(models_results.keys())[0]
    
    if reference_model not in models_results:
        raise ValueError(f"Reference model '{reference_model}' not found in results")
    
    reference_price = models_results[reference_model]['price']
    
    comparison = []
    for model_name, results in models_results.items():
        price = results['price']
        abs_diff, rel_diff = calculate_price_difference(price, reference_price)
        
        model_comparison = {
            'model': model_name,
            'price': price,
            'abs_diff_from_ref': abs_diff,
            'rel_diff_from_ref': rel_diff,
            'is_reference': model_name == reference_model
        }
        
        # Add confidence interval if available
        if 'confidence_interval' in results:
            lower, upper = results['confidence_interval']
            model_comparison['ci_lower'] = lower
            model_comparison['ci_upper'] = upper
            model_comparison['ci_width'] = upper - lower
        
        # Add execution info if available
        for key in ['execution_time', 'circuit_depth', 'num_qubits', 'shots']:
            if key in results:
                model_comparison[key] = results[key]
        
        comparison.append(model_comparison)
    
    return pd.DataFrame(comparison)