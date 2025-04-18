"""
Black-Scholes model for option pricing.

This module implements the classical Black-Scholes model for pricing European options.
"""

import numpy as np
from scipy.stats import norm


def black_scholes_price(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, is_call=True):
    """
    Calculate the price of a European option using the Black-Scholes model.
    
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
        
    Returns:
    --------
    float
        Option price
    """
    # Calculate d1 and d2 parameters
    d1 = (np.log(spot_price / strike_price) + 
          (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)
    
    # Calculate option price based on option type
    if is_call:
        # Call option price formula
        price = spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
    else:
        # Put option price formula
        price = strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
    
    return price


def calculate_greeks(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, is_call=True):
    """
    Calculate the Greeks (Delta, Gamma, Theta, Vega, Rho) for a European option.
    
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
        
    Returns:
    --------
    dict
        Dictionary containing the Greeks
    """
    # Calculate d1 and d2 parameters
    d1 = (np.log(spot_price / strike_price) + 
          (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)
    
    # Calculate common terms
    sqrt_t = np.sqrt(time_to_maturity)
    n_d1 = norm.pdf(d1)  # Standard normal probability density function at d1
    
    # Calculate Delta (1st derivative with respect to spot price)
    if is_call:
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Calculate Gamma (2nd derivative with respect to spot price)
    gamma = n_d1 / (spot_price * volatility * sqrt_t)
    
    # Calculate Theta (1st derivative with respect to time)
    theta_term1 = -(spot_price * n_d1 * volatility) / (2 * sqrt_t)
    
    if is_call:
        theta_term2 = -risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
        theta = theta_term1 + theta_term2
    else:
        theta_term2 = risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)
        theta = theta_term1 + theta_term2
    
    # Calculate Vega (1st derivative with respect to volatility)
    vega = spot_price * sqrt_t * n_d1
    
    # Calculate Rho (1st derivative with respect to risk-free rate)
    if is_call:
        rho = strike_price * time_to_maturity * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
    else:
        rho = -strike_price * time_to_maturity * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta / 365,  # Convert to daily theta
        'vega': vega / 100,  # Convert to 1% change in volatility
        'rho': rho / 100,  # Convert to 1% change in interest rate
    }


def monte_carlo_price(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, is_call=True, num_simulations=10000, random_seed=None):
    """
    Calculate the price of a European option using Monte Carlo simulation.
    
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
    num_simulations : int, optional
        Number of Monte Carlo simulations
    random_seed : int, optional
        Seed for random number generation
        
    Returns:
    --------
    float
        Option price
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Calculate parameters for the geometric Brownian motion
    dt = time_to_maturity
    drift = (risk_free_rate - 0.5 * volatility**2) * dt
    diffusion = volatility * np.sqrt(dt)
    
    # Generate random normal variables
    random_normals = np.random.normal(0, 1, num_simulations)
    
    # Simulate terminal stock prices
    terminal_prices = spot_price * np.exp(drift + diffusion * random_normals)
    
    # Calculate payoffs at maturity
    if is_call:
        payoffs = np.maximum(terminal_prices - strike_price, 0)
    else:
        payoffs = np.maximum(strike_price - terminal_prices, 0)
    
    # Calculate present value of payoffs
    discount_factor = np.exp(-risk_free_rate * time_to_maturity)
    option_price = discount_factor * np.mean(payoffs)
    
    return option_price