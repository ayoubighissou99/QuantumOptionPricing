"""
Quantum Amplitude Estimation (QAE) for option pricing.

This module implements simplified quantum algorithms for pricing European options
using Quantum Amplitude Estimation (QAE) and Quantum Monte Carlo.
"""

import numpy as np
from qiskit import  QuantumCircuit, execute, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import QFT


class QuantumAmplitudeEstimationPricer:
    """
    Option pricer using Quantum Amplitude Estimation (QAE).
    
    This is a simplified implementation that focuses on demonstration rather than
    optimal implementation.
    """
    
    def __init__(self, num_uncertainty_qubits=3, num_evaluation_qubits=3, shots=1024, backend_name='qasm_simulator'):
        """
        Initialize the QAE pricer.
        
        Parameters:
        -----------
        num_uncertainty_qubits : int, optional
            Number of qubits to represent the uncertainty model
        num_evaluation_qubits : int, optional
            Number of qubits for the QAE algorithm
        shots : int, optional
            Number of shots for the quantum simulation
        backend_name : str, optional
            Name of the Qiskit backend to use
        """
        self.num_uncertainty_qubits = num_uncertainty_qubits
        self.num_evaluation_qubits = num_evaluation_qubits
        self.shots = shots
        
        # Set up quantum backend
        self.backend = Aer.get_backend(backend_name)
    
    def _create_uncertainty_circuit(self, spot_price, volatility, time_to_maturity, risk_free_rate):
        """
        Create a quantum circuit for encoding the uncertainty distribution.
        
        Parameters:
        -----------
        spot_price : float
            Current price of the underlying asset
        volatility : float
            Annualized volatility of the underlying asset
        time_to_maturity : float
            Time to option expiration in years
        risk_free_rate : float
            Annual risk-free interest rate
            
        Returns:
        --------
        qiskit.QuantumCircuit
            Quantum circuit encoding the uncertainty model
        tuple
            Lower and upper bounds for the asset price distribution
        """
        # Calculate parameters for the log-normal distribution
        mu = (risk_free_rate - 0.5 * volatility**2) * time_to_maturity
        sigma = volatility * np.sqrt(time_to_maturity)
        
        # Determine bounds for the asset price distribution
        low = np.maximum(0, spot_price * np.exp(mu - 6 * sigma))
        high = spot_price * np.exp(mu + 6 * sigma)
        
        # Create a simple uncertainty circuit with Hadamard gates
        circuit = QuantumCircuit(self.num_uncertainty_qubits)
        
        # Apply Hadamard gates to create a uniform superposition
        for qubit in range(self.num_uncertainty_qubits):
            circuit.h(qubit)
        
        return circuit, (low, high)
    
    def _create_payoff_circuit(self, uncertainty_circuit, strike_price, bounds, is_call=True):
        """
        Create a quantum circuit for computing the option payoff.
        
        Parameters:
        -----------
        uncertainty_circuit : qiskit.QuantumCircuit
            Quantum circuit encoding the uncertainty model
        strike_price : float
            Strike price of the option
        bounds : tuple
            Lower and upper bounds for the asset price distribution
        is_call : bool, optional
            True for a call option, False for a put option
            
        Returns:
        --------
        qiskit.QuantumCircuit
            Quantum circuit encoding the payoff calculation
        """
        # Extract uncertainty circuit parameters
        num_qubits = uncertainty_circuit.num_qubits
        
        # Create a new circuit that includes an ancilla qubit for the payoff
        circuit = QuantumCircuit(num_qubits + 1, 1)
        
        # Add the uncertainty model
        circuit.compose(uncertainty_circuit, qubits=range(num_qubits), inplace=True)
        
        # Add a simple rotation on the payoff qubit based on the strike price
        # This is a very simplified approach - in a real implementation, we would
        # encode the actual payoff function more precisely
        
        # For demonstration, we'll just use a controlled rotation from each qubit
        # to approximate the payoff function
        low, high = bounds
        
        # Calculate a scale factor for the rotations
        if is_call:
            scale = np.pi / num_qubits if high > strike_price else 0
        else:
            scale = np.pi / num_qubits if strike_price > low else 0
        
        # Apply controlled rotations from each uncertainty qubit to the payoff qubit
        for qubit in range(num_qubits):
            weight = 2**(qubit) / (2**num_qubits - 1)
            angle = scale * weight
            circuit.cry(angle, qubit, num_qubits)
        
        # Measure the payoff qubit
        circuit.measure(num_qubits, 0)
        
        return circuit
    
    def price_option(self, spot_price, strike_price, volatility, risk_free_rate, time_to_maturity, is_call=True):
        """
        Price a European option using a simplified Quantum Amplitude Estimation approach.
        
        Parameters:
        -----------
        spot_price : float
            Current price of the underlying asset
        strike_price : float
            Strike price of the option
        volatility : float
            Annualized volatility of the underlying asset
        risk_free_rate : float
            Annual risk-free interest rate
        time_to_maturity : float
            Time to option expiration in years
        is_call : bool, optional
            True for a call option, False for a put option
            
        Returns:
        --------
        dict
            Dictionary containing option price and related information
        """
        # Create the uncertainty model circuit
        uncertainty_circuit, bounds = self._create_uncertainty_circuit(
            spot_price,
            volatility,
            time_to_maturity,
            risk_free_rate
        )
        
        # Create the payoff circuit
        payoff_circuit = self._create_payoff_circuit(
            uncertainty_circuit,
            strike_price,
            bounds,
            is_call
        )
        
        # Execute the circuit
        transpiled_circuit = transpile(payoff_circuit, self.backend)
        job = execute(transpiled_circuit, self.backend, shots=self.shots)
        counts = job.result().get_counts()
        
        # Extract the estimated amplitude
        if '1' in counts:
            ones_count = counts['1']
        else:
            ones_count = 0
        
        estimated_amplitude = ones_count / self.shots
        
        # Calculate the option price
        low, high = bounds
        price_range = high - low
        expected_payoff = estimated_amplitude * price_range
        
        # Apply discount
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        option_price = discount_factor * expected_payoff
        
        # Calculate confidence interval using binomial distribution
        std_dev = np.sqrt((estimated_amplitude * (1 - estimated_amplitude)) / self.shots)
        z_score = 1.96  # 95% confidence interval
        ci_half_width = z_score * std_dev * price_range * discount_factor
        
        # Prepare the result
        result = {
            'price': option_price,
            'confidence_interval': [
                max(0, option_price - ci_half_width),
                option_price + ci_half_width
            ],
            'circuit_depth': payoff_circuit.depth(),
            'num_qubits': payoff_circuit.num_qubits,
            'shots': self.shots,
            'estimated_amplitude': estimated_amplitude
        }
        
        return result


class QuantumMonteCarloPricer:
    """
    Option pricer using Quantum Monte Carlo.
    
    This is a simplified implementation for demonstration purposes.
    """
    
    def __init__(self, num_qubits=6, shots=1024, backend_name='qasm_simulator'):
        """
        Initialize the Quantum Monte Carlo pricer.
        
        Parameters:
        -----------
        num_qubits : int, optional
            Number of qubits for the quantum circuit
        shots : int, optional
            Number of shots for the quantum simulation
        backend_name : str, optional
            Name of the Qiskit backend to use
        """
        self.num_qubits = num_qubits
        self.shots = shots
        self.backend = Aer.get_backend(backend_name)
    
    def _create_sampling_circuit(self):
        """
        Create a quantum circuit for sampling.
        
        Returns:
        --------
        qiskit.QuantumCircuit
            Quantum circuit for sampling
        """
        circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Apply Hadamard gates to create a uniform superposition
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        
        # Measure all qubits
        circuit.measure(range(self.num_qubits), range(self.num_qubits))
        
        return circuit
    
    def price_option(self, spot_price, strike_price, volatility, risk_free_rate, time_to_maturity, is_call=True):
        """
        Price a European option using Quantum Monte Carlo.
        
        Parameters:
        -----------
        spot_price : float
            Current price of the underlying asset
        strike_price : float
            Strike price of the option
        volatility : float
            Annualized volatility of the underlying asset
        risk_free_rate : float
            Annual risk-free interest rate
        time_to_maturity : float
            Time to option expiration in years
        is_call : bool, optional
            True for a call option, False for a put option
            
        Returns:
        --------
        dict
            Dictionary containing option price and related information
        """
        # Calculate parameters for the log-normal distribution
        mu = (risk_free_rate - 0.5 * volatility**2) * time_to_maturity
        sigma = volatility * np.sqrt(time_to_maturity)
        
        # Determine bounds for the asset price distribution
        low = np.maximum(0, spot_price * np.exp(mu - 6 * sigma))
        high = spot_price * np.exp(mu + 6 * sigma)
        
        # Create the sampling circuit
        circuit = self._create_sampling_circuit()
        
        # Execute the circuit
        transpiled_circuit = transpile(circuit, self.backend)
        job = execute(transpiled_circuit, self.backend, shots=self.shots)
        counts = job.result().get_counts()
        
        # Convert binary measurement outcomes to stock prices
        payoffs = []
        total_range = high - low
        max_int = 2**self.num_qubits - 1
        
        for bitstring, count in counts.items():
            # Convert bitstring to integer
            int_value = int(bitstring, 2)
            
            # Map to price range
            price = low + (int_value / max_int) * total_range
            
            # Calculate payoff
            if is_call:
                payoff = max(0, price - strike_price)
            else:
                payoff = max(0, strike_price - price)
            
            # Add to payoffs list with appropriate weight
            payoffs.extend([payoff] * count)
        
        # Calculate expected payoff
        expected_payoff = np.mean(payoffs)
        
        # Apply discount
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        option_price = discount_factor * expected_payoff
        
        # Calculate standard error
        std_error = np.std(payoffs) / np.sqrt(self.shots)
        
        # Prepare the result
        result = {
            'price': option_price,
            'confidence_interval': [
                max(0, option_price - 1.96 * discount_factor * std_error),
                option_price + 1.96 * discount_factor * std_error
            ],
            'circuit_depth': circuit.depth(),
            'num_qubits': circuit.num_qubits,
            'shots': self.shots,
            'expected_payoff': expected_payoff,
            'std_error': discount_factor * std_error
        }
        
        return result