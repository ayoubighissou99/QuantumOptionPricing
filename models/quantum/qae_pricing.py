"""
Quantum Amplitude Estimation (QAE) for option pricing.

This module implements quantum algorithms for pricing European options
using Quantum Amplitude Estimation (QAE).
"""

import numpy as np
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_algorithms import EstimationProblem, AmplitudeEstimation
from qiskit.utils import QuantumInstance


class QuantumAmplitudeEstimationPricer:
    """
    Option pricer using Quantum Amplitude Estimation (QAE).
    
    This class implements an option pricing model based on QAE, which provides
    a quadratic speedup compared to classical Monte Carlo simulation.
    """
    
    def __init__(self, num_uncertainty_qubits=3, num_evaluation_qubits=5, shots=1024, backend_name='qasm_simulator'):
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
        self.quantum_instance = QuantumInstance(
            backend=self.backend,
            shots=self.shots
        )
    
    def _create_uncertainty_model(self, num_qubits, spot_price, volatility, time_to_maturity, risk_free_rate, bounds=None):
        """
        Create a quantum circuit that encodes the log-normal distribution of the stock price.
        
        Parameters:
        -----------
        num_qubits : int
            Number of qubits for the uncertainty model
        spot_price : float
            Current price of the underlying asset
        volatility : float
            Annualized volatility of the underlying asset
        time_to_maturity : float
            Time to option expiration in years
        risk_free_rate : float
            Annual risk-free interest rate
        bounds : tuple, optional
            Lower and upper bounds for the asset price distribution
            
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
        if bounds is None:
            # Set bounds to cover most of the log-normal distribution
            # Typically, we want to cover ±6 standard deviations
            low = np.maximum(0, spot_price * np.exp(mu - 6 * sigma))
            high = spot_price * np.exp(mu + 6 * sigma)
            bounds = (low, high)
        
        # Create a simple circuit with Hadamard gates for now
        # In a real application, this would be replaced with a proper
        # log-normal distribution encoding
        uncertainty_model = QuantumCircuit(num_qubits, name='Uncertainty Model')
        
        # Apply Hadamard gates to create a uniform superposition
        # This is a simplification - a real implementation would use
        # quantum circuits to approximate the log-normal distribution
        for qubit in range(num_qubits):
            uncertainty_model.h(qubit)
        
        return uncertainty_model, bounds
    
    def _create_payoff_operator(self, num_uncertainty_qubits, strike_price, bounds, is_call=True):
        """
        Create a quantum circuit that encodes the European option payoff function.
        
        Parameters:
        -----------
        num_uncertainty_qubits : int
            Number of qubits for the uncertainty model
        strike_price : float
            Strike price of the option
        bounds : tuple
            Lower and upper bounds for the asset price distribution
        is_call : bool, optional
            True for a call option, False for a put option
            
        Returns:
        --------
        qiskit.circuit.library.LinearAmplitudeFunction
            Quantum circuit encoding the payoff function
        """
        # Extract bounds
        low, high = bounds
        
        # Define the payoff function
        if is_call:
            # Call option payoff: max(S - K, 0)
            def payoff(x):
                return np.maximum(0, x - strike_price)
        else:
            # Put option payoff: max(K - S, 0)
            def payoff(x):
                return np.maximum(0, strike_price - x)
        
        # Scale the payoff to fit within [0, 1]
        # Compute the maximum possible payoff within the given bounds
        if is_call:
            max_payoff = max(0, high - strike_price)
        else:
            max_payoff = max(0, strike_price - low)
        
        # Avoid division by zero
        if max_payoff == 0:
            max_payoff = 1
        
        # Create the slope and offset for the linear amplitude function
        # This is a simplified approach
        slope = 1.0 / max_payoff
        offset = 0.0
        
        # Create the linear amplitude function circuit
        # This maps the payoff function to qubit amplitudes
        payoff_circuit = LinearAmplitudeFunction(
            num_state_qubits=num_uncertainty_qubits,
            slope=slope,
            offset=offset,
            domain=(low, high),
            image=(0, 1),
            breakpoints=[strike_price],
            name="Payoff Function"
        )
        
        return payoff_circuit
    
    def price_option(self, spot_price, strike_price, volatility, risk_free_rate, time_to_maturity, is_call=True):
        """
        Price a European option using Quantum Amplitude Estimation.
        
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
        uncertainty_circuit, bounds = self._create_uncertainty_model(
            self.num_uncertainty_qubits,
            spot_price,
            volatility,
            time_to_maturity,
            risk_free_rate
        )
        
        # Create the payoff operator circuit
        payoff_circuit = self._create_payoff_operator(
            self.num_uncertainty_qubits,
            strike_price,
            bounds,
            is_call
        )
        
        # Combine the uncertainty model and payoff operator
        # The payoff circuit operates on the uncertainty circuit output
        option_pricing_circuit = payoff_circuit.compose(uncertainty_circuit)
        
        # Set up the estimation problem for QAE
        # We want to estimate the expected payoff
        estimation_problem = EstimationProblem(
            state_preparation=option_pricing_circuit,
            objective_qubits=[self.num_uncertainty_qubits],  # index of the objective qubit
            post_processing=lambda x: x * (bounds[1] - bounds[0])  # Scale the result back to the original domain
        )
        
        # Create the QAE algorithm
        qae = AmplitudeEstimation(
            num_eval_qubits=self.num_evaluation_qubits,
            quantum_instance=self.quantum_instance
        )
        
        # Run the QAE algorithm
        qae_result = qae.estimate(estimation_problem)
        
        # Extract the estimated amplitude (expected payoff)
        estimated_amplitude = qae_result.estimation
        
        # Calculate the option price by discounting the expected payoff
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        option_price = discount_factor * estimated_amplitude
        
        # Prepare the result
        result = {
            'price': option_price,
            'confidence_interval': [
                discount_factor * qae_result.confidence_interval_68[0],
                discount_factor * qae_result.confidence_interval_68[1]
            ],
            'circuit_depth': option_pricing_circuit.depth(),
            'num_qubits': option_pricing_circuit.num_qubits,
            'shots': self.shots,
            'estimated_amplitude': estimated_amplitude,
            'estimated_precision': qae_result.estimation_processed_std_dev,
            'qae_iterations': 2 ** self.num_evaluation_qubits
        }
        
        return result


class QuantumMonteCarloPricer:
    """
    Option pricer using Quantum Monte Carlo.
    
    This class implements an option pricing model based on quantum circuits
    for sampling from distributions.
    """
    
    def __init__(self, num_qubits=6, shots=1024, backend_name='qasm_simulator'):
        """
        Initialize the Quantum Monte Carlo pricer.
        
        Parameters:
        -----------
        num_qubits : int, optional
            Number of qubits to represent the uncertainty model
        shots : int, optional
            Number of shots for the quantum simulation
        backend_name : str, optional
            Name of the Qiskit backend to use
        """
        self.num_qubits = num_qubits
        self.shots = shots
        
        # Set up quantum backend
        self.backend = Aer.get_backend(backend_name)
    
    def _create_distribution_circuit(self, spot_price, volatility, time_to_maturity, risk_free_rate):
        """
        Create a quantum circuit that can sample from a log-normal distribution.
        
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
            Quantum circuit for sampling
        float
            Lower bound of the distribution
        float
            Upper bound of the distribution
        """
        # Calculate parameters for the log-normal distribution
        mu = (risk_free_rate - 0.5 * volatility**2) * time_to_maturity
        sigma = volatility * np.sqrt(time_to_maturity)
        
        # Determine bounds for the asset price distribution
        low = np.maximum(0, spot_price * np.exp(mu - 6 * sigma))
        high = spot_price * np.exp(mu + 6 * sigma)
        
        # Create a circuit for sampling
        # This is a simplified approach - we'll just create a uniform distribution
        circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Apply Hadamard gates to create a uniform superposition
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        
        # Measure all qubits
        circuit.measure(range(self.num_qubits), range(self.num_qubits))
        
        return circuit, low, high
    
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
        # Create the sampling circuit
        circuit, low, high = self._create_distribution_circuit(
            spot_price,
            volatility,
            time_to_maturity,
            risk_free_rate
        )
        
        # Transpile the circuit for the backend
        transpiled_circuit = transpile(circuit, self.backend)
        
        # Run the circuit to generate samples
        job = self.backend.run(transpiled_circuit, shots=self.shots)
        counts = job.result().get_counts()
        
        # Convert binary measurement outcomes to stock prices
        total_range = high - low
        payoffs = []
        
        for bitstring, count in counts.items():
            # Convert bitstring to a number between 0 and 1
            fraction = int(bitstring, 2) / (2**self.num_qubits - 1)
            
            # Map to the price range
            price = low + fraction * total_range
            
            # Calculate the payoff
            if is_call:
                payoff = max(0, price - strike_price)
            else:
                payoff = max(0, strike_price - price)
            
            # Add to the list of payoffs with the appropriate weight
            payoffs.extend([payoff] * count)
        
        # Calculate the average payoff
        expected_payoff = np.mean(payoffs)
        
        # Calculate the option price by discounting the expected payoff
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        option_price = discount_factor * expected_payoff
        
        # Calculate the standard error
        std_error = np.std(payoffs) / np.sqrt(self.shots)
        
        # Prepare the result
        result = {
            'price': option_price,
            'confidence_interval': [
                discount_factor * (expected_payoff - 1.96 * std_error),
                discount_factor * (expected_payoff + 1.96 * std_error)
            ],
            'circuit_depth': circuit.depth(),
            'num_qubits': circuit.num_qubits,
            'shots': self.shots,
            'expected_payoff': expected_payoff,
            'std_error': discount_factor * std_error
        }
        
        return result