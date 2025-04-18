"""
Data loading and preprocessing utilities.

This module provides functions for loading and preprocessing financial data
for option pricing models.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def load_option_chain(file_path):
    """
    Load an option chain dataset from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing option chain data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the option chain data
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Option chain file not found: {file_path}")
    
    # Load the data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading option chain data: {str(e)}")
    
    # Validate required columns
    required_columns = ['strike', 'expiration', 'call_price', 'put_price', 'underlying_price']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in option chain data: {', '.join(missing_columns)}")
    
    # Convert date columns to datetime
    if 'expiration' in df.columns:
        df['expiration'] = pd.to_datetime(df['expiration'])
    
    # Calculate time to maturity in years
    if 'expiration' in df.columns:
        current_date = df['expiration'].min()  # Use the earliest date as reference
        df['time_to_maturity'] = (df['expiration'] - current_date).dt.days / 365.25
    
    return df


def calculate_implied_volatility(option_price, spot_price, strike_price, time_to_maturity, risk_free_rate, is_call=True, precision=0.0001, max_iterations=100):
    """
    Calculate implied volatility using the Newton-Raphson method.
    
    Parameters:
    -----------
    option_price : float
        Market price of the option
    spot_price : float
        Current price of the underlying asset
    strike_price : float
        Strike price of the option
    time_to_maturity : float
        Time to option expiration in years
    risk_free_rate : float
        Annual risk-free interest rate
    is_call : bool, optional
        True for a call option, False for a put option
    precision : float, optional
        Desired precision for the result
    max_iterations : int, optional
        Maximum number of iterations
        
    Returns:
    --------
    float
        Implied volatility
    """
    # Import here to avoid circular imports
    from models.classical.black_scholes import black_scholes_price
    
    # Initial guess for volatility
    volatility = 0.2  # Start with 20% volatility
    
    for _ in range(max_iterations):
        # Calculate option price using current volatility estimate
        price = black_scholes_price(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, is_call)
        
        # Calculate price difference
        price_diff = option_price - price
        
        # If difference is within precision, return current volatility
        if abs(price_diff) < precision:
            return volatility
        
        # Calculate vega (sensitivity of option price to volatility)
        vega = spot_price * np.sqrt(time_to_maturity) * np.exp(-0.5 * volatility**2 * time_to_maturity) / np.sqrt(2 * np.pi)
        
        # Update volatility estimate using Newton-Raphson method
        volatility += price_diff / vega
        
        # Ensure volatility stays within reasonable bounds
        volatility = max(0.001, min(volatility, 5.0))
    
    # If no convergence after max iterations, return the last estimate
    return volatility


def extract_risk_free_rate(option_chain_df, method='put_call_parity'):
    """
    Extract the implied risk-free rate from option chain data.
    
    Parameters:
    -----------
    option_chain_df : pandas.DataFrame
        DataFrame containing option chain data
    method : str, optional
        Method to use: 'put_call_parity' or 'zero_curve'
        
    Returns:
    --------
    float
        Estimated risk-free rate
    """
    if method == 'put_call_parity':
        # Use put-call parity to estimate risk-free rate
        # For European options: call - put = S - K*exp(-r*T)
        
        # Filter for at-the-money options
        spot_price = option_chain_df['underlying_price'].iloc[0]
        atm_options = option_chain_df.copy()
        atm_options['strike_diff'] = abs(atm_options['strike'] - spot_price)
        atm_options = atm_options.sort_values('strike_diff').head(5)
        
        # Calculate implied risk-free rates
        rates = []
        for _, row in atm_options.iterrows():
            call_price = row['call_price']
            put_price = row['put_price']
            strike = row['strike']
            time_to_maturity = row['time_to_maturity']
            
            # Solve for r: call - put = S - K*exp(-r*T)
            # Therefore: r = -ln((S - (call - put))/K) / T
            if time_to_maturity > 0:
                rate = -np.log((spot_price - (call_price - put_price)) / strike) / time_to_maturity
                if 0 < rate < 0.2:  # Filter out unreasonable rates
                    rates.append(rate)
        
        if rates:
            return np.median(rates)
        else:
            return 0.02  # Default to 2% if calculation fails
    
    elif method == 'zero_curve':
        # Use zero-coupon yield curve method
        # This is a simplified version, in reality would use treasury yield curve
        return 0.02  # Default to 2% for now
    
    else:
        raise ValueError(f"Unknown method: {method}")


def prepare_volatility_surface(option_chain_df):
    """
    Prepare a volatility surface from option chain data.
    
    Parameters:
    -----------
    option_chain_df : pandas.DataFrame
        DataFrame containing option chain data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing volatility surface data
    """
    # Calculate implied volatility for all options
    volatility_data = []
    
    # Get unique combinations of expiration and strike
    grouped_data = option_chain_df.groupby(['expiration', 'strike'])
    
    for (expiration, strike), group in grouped_data:
        # Get the first row for this strike and expiration
        row = group.iloc[0]
        
        # Extract necessary values
        spot_price = row['underlying_price']
        time_to_maturity = row['time_to_maturity']
        risk_free_rate = extract_risk_free_rate(option_chain_df)
        
        # Calculate implied volatility for call option
        try:
            call_iv = calculate_implied_volatility(
                row['call_price'], 
                spot_price, 
                strike, 
                time_to_maturity, 
                risk_free_rate, 
                is_call=True
            )
        except:
            call_iv = np.nan
        
        # Calculate implied volatility for put option
        try:
            put_iv = calculate_implied_volatility(
                row['put_price'], 
                spot_price, 
                strike, 
                time_to_maturity, 
                risk_free_rate, 
                is_call=False
            )
        except:
            put_iv = np.nan
        
        # Average the call and put IVs if both are available
        if not np.isnan(call_iv) and not np.isnan(put_iv):
            avg_iv = (call_iv + put_iv) / 2
        elif not np.isnan(call_iv):
            avg_iv = call_iv
        elif not np.isnan(put_iv):
            avg_iv = put_iv
        else:
            avg_iv = np.nan
        
        # Append to volatility data
        volatility_data.append({
            'expiration': expiration,
            'strike': strike,
            'time_to_maturity': time_to_maturity,
            'moneyness': strike / spot_price,
            'implied_volatility': avg_iv,
            'call_iv': call_iv,
            'put_iv': put_iv
        })
    
    # Create DataFrame from volatility data
    volatility_df = pd.DataFrame(volatility_data)
    
    return volatility_df


def interpolate_volatility(volatility_surface_df, strike, time_to_maturity):
    """
    Interpolate volatility from a volatility surface.
    
    Parameters:
    -----------
    volatility_surface_df : pandas.DataFrame
        DataFrame containing volatility surface data
    strike : float
        Strike price for which to interpolate volatility
    time_to_maturity : float
        Time to maturity for which to interpolate volatility
        
    Returns:
    --------
    float
        Interpolated implied volatility
    """
    # Convert to pivot table for interpolation
    pivot_df = volatility_surface_df.pivot(
        index='time_to_maturity',
        columns='strike',
        values='implied_volatility'
    )
    
    # Get unique strikes and times to maturity
    unique_strikes = pivot_df.columns.tolist()
    unique_ttm = pivot_df.index.tolist()
    
    # Check if strike and time_to_maturity are within range
    if strike < min(unique_strikes) or strike > max(unique_strikes):
        raise ValueError(f"Strike {strike} is outside the range of available data")
    
    if time_to_maturity < min(unique_ttm) or time_to_maturity > max(unique_ttm):
        raise ValueError(f"Time to maturity {time_to_maturity} is outside the range of available data")
    
    # Interpolate along the time dimension for the given strikes
    time_interp_values = []
    for s in unique_strikes:
        # Extract non-NaN values for this strike
        time_values = pivot_df[s].dropna()
        if len(time_values) > 1:
            # Create interpolation function
            interp_func = interp1d(
                time_values.index.values,
                time_values.values,
                bounds_error=False,
                fill_value='extrapolate'
            )
            # Interpolate at the desired time to maturity
            time_interp_values.append((s, float(interp_func(time_to_maturity))))
    
    # Create a new DataFrame for strike interpolation
    strike_df = pd.DataFrame(time_interp_values, columns=['strike', 'iv'])
    
    # Interpolate along the strike dimension
    if len(strike_df) > 1:
        strike_interp_func = interp1d(
            strike_df['strike'].values,
            strike_df['iv'].values,
            bounds_error=False,
            fill_value='extrapolate'
        )
        # Interpolate at the desired strike
        volatility = float(strike_interp_func(strike))
        return volatility
    else:
        return np.nan


def generate_synthetic_option_data(num_samples=100, seed=42):
    """
    Generate synthetic option data for testing and demonstration.
    
    Parameters:
    -----------
    num_samples : int, optional
        Number of option contracts to generate
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic option data
    """
    np.random.seed(seed)
    
    # Base parameters
    base_spot_price = 100.0
    risk_free_rate = 0.02
    base_volatility = 0.2
    
    # Generate data
    data = []
    current_date = pd.Timestamp.now().normalize()
    
    # Generate different maturities
    maturities = [30, 60, 90, 180, 270, 365]  # Days
    
    # Generate different strikes
    moneyness_levels = np.linspace(0.8, 1.2, 9)  # 80% to 120% of spot price
    
    for days in maturities:
        # Calculate expiration date
        expiration = current_date + pd.Timedelta(days=days)
        time_to_maturity = days / 365.0
        
        # Vary volatility by maturity (volatility term structure)
        maturity_volatility = base_volatility * (1 + 0.1 * np.log(time_to_maturity + 0.5))
        
        for moneyness in moneyness_levels:
            # Calculate strike price
            strike = base_spot_price * moneyness
            
            # Vary volatility by strike (volatility smile)
            strike_adjustment = 0.05 * (moneyness - 1.0)**2
            volatility = maturity_volatility + strike_adjustment
            
            # Calculate prices using Black-Scholes
            from models.classical.black_scholes import black_scholes_price
            
            call_price = black_scholes_price(
                base_spot_price,
                strike,
                time_to_maturity,
                risk_free_rate,
                volatility,
                is_call=True
            )
            
            put_price = black_scholes_price(
                base_spot_price,
                strike,
                time_to_maturity,
                risk_free_rate,
                volatility,
                is_call=False
            )
            
            # Add some random noise to prices (market inefficiency)
            noise_factor = 0.02  # 2%
            call_price *= (1 + np.random.uniform(-noise_factor, noise_factor))
            put_price *= (1 + np.random.uniform(-noise_factor, noise_factor))
            
            # Create data entry
            data.append({
                'underlying_price': base_spot_price,
                'strike': strike,
                'expiration': expiration,
                'time_to_maturity': time_to_maturity,
                'call_price': call_price,
                'put_price': put_price,
                'true_volatility': volatility,
                'risk_free_rate': risk_free_rate
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


def save_synthetic_data(df, file_path):
    """
    Save synthetic option data to a CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing option data
    file_path : str
        Path to save the CSV file
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    
    print(f"Saved synthetic option data to {file_path}")


if __name__ == "__main__":
    # Generate and save synthetic data when module is run directly
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             "data", "synthetic_option_data.csv")
    
    df = generate_synthetic_option_data(num_samples=100)
    save_synthetic_data(df, data_path)