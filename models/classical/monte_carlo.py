"""
Monte Carlo simulation for option pricing.

This module implements advanced Monte Carlo methods for pricing various types of options.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


class MonteCarloSimulator:
    """
    Monte Carlo simulator for option pricing.
    
    This class implements various Monte Carlo methods for pricing options,
    including European, American, and exotic options.
    """
    
    def __init__(self, num_simulations=10000, num_time_steps=252, random_seed=None):
        """
        Initialize the Monte Carlo simulator.
        
        Parameters:
        -----------
        num_simulations : int, optional
            Number of Monte Carlo simulations
        num_time_steps : int, optional
            Number of time steps per year (e.g., 252 trading days)
        random_seed : int, optional
            Seed for random number generation
        """
        self.num_simulations = num_simulations
        self.num_time_steps = num_time_steps
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_paths(self, spot_price, time_to_maturity, risk_free_rate, volatility, dividend_yield=0.0):
        """
        Simulate price paths for the underlying asset.
        
        Parameters:
        -----------
        spot_price : float
            Current price of the underlying asset
        time_to_maturity : float
            Time to option expiration in years
        risk_free_rate : float
            Annual risk-free interest rate
        volatility : float
            Annualized volatility of the underlying asset
        dividend_yield : float, optional
            Annual dividend yield of the underlying asset
            
        Returns:
        --------
        numpy.ndarray
            Array of shape (num_simulations, num_steps+1) containing simulated price paths
        """
        # Calculate number of time steps
        num_steps = int(self.num_time_steps * time_to_maturity)
        
        # Calculate time step size
        dt = time_to_maturity / num_steps
        
        # Calculate drift and diffusion terms
        drift = (risk_free_rate - dividend_yield - 0.5 * volatility**2) * dt
        diffusion = volatility * np.sqrt(dt)
        
        # Generate random normal variables
        # Shape: (num_simulations, num_steps)
        random_normals = np.random.normal(0, 1, size=(self.num_simulations, num_steps))
        
        # Initialize price paths array
        # Shape: (num_simulations, num_steps+1)
        paths = np.zeros((self.num_simulations, num_steps + 1))
        paths[:, 0] = spot_price
        
        # Simulate price paths using geometric Brownian motion
        for t in range(1, num_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * random_normals[:, t-1])
        
        return paths
    
    def price_european_option(self, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, is_call=True, dividend_yield=0.0):
        """
        Price a European option using Monte Carlo simulation.
        
        Parameters:
        -----------
        spot_price : float
            Current price of the underlying asset
        strike_price : float
            Strike price of the option
        time_to_maturity : float
            Time to option expiration in years
        risk_free_rate : float
            Annual risk-free interest rate
        volatility : float
            Annualized volatility of the underlying asset
        is_call : bool, optional
            True for a call option, False for a put option
        dividend_yield : float, optional
            Annual dividend yield of the underlying asset
            
        Returns:
        --------
        dict
            Dictionary containing option price and standard error
        """
        # Simulate price paths
        paths = self.simulate_paths(spot_price, time_to_maturity, risk_free_rate, volatility, dividend_yield)
        
        # Extract terminal prices (last column of the paths array)
        terminal_prices = paths[:, -1]
        
        # Calculate payoffs at maturity
        if is_call:
            payoffs = np.maximum(terminal_prices - strike_price, 0)
        else:
            payoffs = np.maximum(strike_price - terminal_prices, 0)
        
        # Calculate present value of payoffs
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        present_values = discount_factor * payoffs
        
        # Calculate option price and standard error
        option_price = np.mean(present_values)
        std_error = np.std(present_values) / np.sqrt(self.num_simulations)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval_95': [option_price - 1.96 * std_error, option_price + 1.96 * std_error]
        }
    
    def price_american_option(self, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, is_call=True, dividend_yield=0.0):
        """
        Price an American option using the Least Squares Monte Carlo (LSM) method.
        
        Parameters:
        -----------
        spot_price : float
            Current price of the underlying asset
        strike_price : float
            Strike price of the option
        time_to_maturity : float
            Time to option expiration in years
        risk_free_rate : float
            Annual risk-free interest rate
        volatility : float
            Annualized volatility of the underlying asset
        is_call : bool, optional
            True for a call option, False for a put option
        dividend_yield : float, optional
            Annual dividend yield of the underlying asset
            
        Returns:
        --------
        dict
            Dictionary containing option price and standard error
        """
        # Calculate number of time steps
        num_steps = int(self.num_time_steps * time_to_maturity)
        
        # Simulate price paths
        paths = self.simulate_paths(spot_price, time_to_maturity, risk_free_rate, volatility, dividend_yield)
        
        # Calculate time step size
        dt = time_to_maturity / num_steps
        
        # Initialize array to store cash flows for each path
        cash_flows = np.zeros((self.num_simulations, num_steps + 1))
        
        # Calculate immediate exercise values at maturity (time step n)
        if is_call:
            cash_flows[:, -1] = np.maximum(paths[:, -1] - strike_price, 0)
        else:
            cash_flows[:, -1] = np.maximum(strike_price - paths[:, -1], 0)
        
        # Iterate backwards through time steps
        for t in range(num_steps - 1, 0, -1):
            # Identify in-the-money paths
            if is_call:
                itm_mask = paths[:, t] > strike_price
            else:
                itm_mask = paths[:, t] < strike_price
            
            # Only consider in-the-money paths for regression
            if np.sum(itm_mask) > 0:
                # Extract in-the-money paths and corresponding future cash flows
                itm_paths = paths[itm_mask, t]
                future_cf = cash_flows[itm_mask, t+1] * np.exp(-risk_free_rate * dt)
                
                # Perform polynomial regression (order 3)
                x = itm_paths
                X = np.column_stack([np.ones(len(x)), x, x**2, x**3])
                beta = np.linalg.lstsq(X, future_cf, rcond=None)[0]
                
                # Calculate continuation values for all in-the-money paths
                x_all = paths[:, t]
                X_all = np.column_stack([np.ones(len(x_all)), x_all, x_all**2, x_all**3])
                continuation_values = X_all.dot(beta)
                
                # Calculate immediate exercise values
                if is_call:
                    immediate_values = np.maximum(paths[:, t] - strike_price, 0)
                else:
                    immediate_values = np.maximum(strike_price - paths[:, t], 0)
                
                # Exercise if immediate value > continuation value
                exercise_mask = immediate_values > continuation_values
                cash_flows[exercise_mask, t] = immediate_values[exercise_mask]
                # If exercised, set future cash flows to zero
                cash_flows[exercise_mask, (t+1):] = 0
            
            # For out-of-the-money paths, no early exercise
            else:
                # Just carry forward the future cash flows
                cash_flows[:, t] = cash_flows[:, t+1] * np.exp(-risk_free_rate * dt)
        
        # Discount remaining cash flows to present value
        present_values = np.zeros(self.num_simulations)
        for i in range(self.num_simulations):
            # Find the first non-zero cash flow for each path
            exercise_times = np.nonzero(cash_flows[i])[0]
            if len(exercise_times) > 0:
                first_exercise = exercise_times[0]
                present_values[i] = cash_flows[i, first_exercise] * np.exp(-risk_free_rate * (first_exercise * dt))
        
        # Calculate option price and standard error
        option_price = np.mean(present_values)
        std_error = np.std(present_values) / np.sqrt(self.num_simulations)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval_95': [option_price - 1.96 * std_error, option_price + 1.96 * std_error]
        }
    
    def price_barrier_option(self, spot_price, strike_price, barrier, time_to_maturity, risk_free_rate, volatility, 
                             option_type='down-and-out', is_call=True, dividend_yield=0.0):
        """
        Price a barrier option using Monte Carlo simulation.
        
        Parameters:
        -----------
        spot_price : float
            Current price of the underlying asset
        strike_price : float
            Strike price of the option
        barrier : float
            Barrier level
        time_to_maturity : float
            Time to option expiration in years
        risk_free_rate : float
            Annual risk-free interest rate
        volatility : float
            Annualized volatility of the underlying asset
        option_type : str, optional
            Type of barrier option: 'down-and-out', 'down-and-in', 'up-and-out', or 'up-and-in'
        is_call : bool, optional
            True for a call option, False for a put option
        dividend_yield : float, optional
            Annual dividend yield of the underlying asset
            
        Returns:
        --------
        dict
            Dictionary containing option price and standard error
        """
        # Simulate price paths
        paths = self.simulate_paths(spot_price, time_to_maturity, risk_free_rate, volatility, dividend_yield)
        
        # Determine if barrier is crossed for each path
        if option_type.startswith('down'):
            # For 'down' options, check if price goes below barrier
            barrier_crossed = np.min(paths, axis=1) <= barrier
        else:  # 'up' options
            # For 'up' options, check if price goes above barrier
            barrier_crossed = np.max(paths, axis=1) >= barrier
        
        # Determine which paths lead to valid payoffs based on barrier type
        if option_type.endswith('in'):
            # For 'in' options, barrier must be crossed
            valid_paths = barrier_crossed
        else:  # 'out' options
            # For 'out' options, barrier must not be crossed
            valid_paths = ~barrier_crossed
        
        # Calculate terminal payoffs for valid paths
        terminal_prices = paths[:, -1]
        
        if is_call:
            payoffs = np.maximum(terminal_prices - strike_price, 0)
        else:
            payoffs = np.maximum(strike_price - terminal_prices, 0)
        
        # Zero out payoffs for invalid paths
        payoffs[~valid_paths] = 0
        
        # Calculate present value of payoffs
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        present_values = discount_factor * payoffs
        
        # Calculate option price and standard error
        option_price = np.mean(present_values)
        std_error = np.std(present_values) / np.sqrt(self.num_simulations)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval_95': [option_price - 1.96 * std_error, option_price + 1.96 * std_error]
        }
    
    def price_asian_option(self, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, 
                           averaging_type='arithmetic', is_call=True, dividend_yield=0.0):
        """
        Price an Asian option using Monte Carlo simulation.
        
        Parameters:
        -----------
        spot_price : float
            Current price of the underlying asset
        strike_price : float
            Strike price of the option
        time_to_maturity : float
            Time to option expiration in years
        risk_free_rate : float
            Annual risk-free interest rate
        volatility : float
            Annualized volatility of the underlying asset
        averaging_type : str, optional
            Type of averaging to use: 'arithmetic' or 'geometric'
        is_call : bool, optional
            True for a call option, False for a put option
        dividend_yield : float, optional
            Annual dividend yield of the underlying asset
            
        Returns:
        --------
        dict
            Dictionary containing option price and standard error
        """
        # Simulate price paths
        paths = self.simulate_paths(spot_price, time_to_maturity, risk_free_rate, volatility, dividend_yield)
        
        # Calculate average prices along each path
        if averaging_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        else:  # geometric averaging
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))
        
        # Calculate payoffs at maturity
        if is_call:
            payoffs = np.maximum(avg_prices - strike_price, 0)
        else:
            payoffs = np.maximum(strike_price - avg_prices, 0)
        
        # Calculate present value of payoffs
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        present_values = discount_factor * payoffs
        
        # Calculate option price and standard error
        option_price = np.mean(present_values)
        std_error = np.std(present_values) / np.sqrt(self.num_simulations)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval_95': [option_price - 1.96 * std_error, option_price + 1.96 * std_error]
        }