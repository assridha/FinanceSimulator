"""Geometric Brownian Motion (GBM) model for price simulation."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
import time
import logging

from .base import BaseModel, ModelFactory


class GeometricBrownianMotion(BaseModel):
    """
    Geometric Brownian Motion model for simulating asset prices.
    
    The model assumes that asset returns follow a normal distribution and
    uses the formula:
    dS = μ*S*dt + σ*S*dW
    
    where:
    - S is the asset price
    - μ is the drift (expected return)
    - σ is the volatility
    - dW is a Wiener process increment
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the GBM model.
        
        Args:
            params: Dictionary containing model parameters:
                - drift: Expected return (annualized)
                - volatility: Volatility (annualized)
                - random_seed: Random seed for reproducibility (optional)
        """
        super().__init__(params)
        
        # Set random seed if provided
        if 'random_seed' in self.params:
            np.random.seed(self.params['random_seed'])
    
    def _validate_params(self) -> None:
        """
        Validate GBM parameters.
        
        Raises:
            ValueError: If parameters are invalid
        """
        required_params = ['drift', 'volatility']
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Check that volatility is positive
        if self.params['volatility'] <= 0:
            raise ValueError("Volatility must be positive")
    
    def fit(self, data: pd.DataFrame) -> 'GeometricBrownianMotion':
        """
        Fit the GBM model to historical data.
        
        For GBM, fitting involves calculating drift and volatility from historical returns.
        
        Args:
            data: DataFrame with a 'Close' column
            
        Returns:
            Self for method chaining
        """
        # Calculate log returns
        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        
        # Calculate volatility (annualized)
        self.params['volatility'] = returns.std() * np.sqrt(252)
        
        # Calculate drift (annualized)
        # Using the risk-neutral formula: drift = risk_free_rate - 0.5 * volatility^2
        # If risk-free rate is not provided, use historical mean return
        if 'risk_free_rate' in self.params:
            self.params['drift'] = (self.params['risk_free_rate'] - 
                                   0.5 * self.params['volatility']**2)
        else:
            self.params['drift'] = returns.mean() * 252
        
        return self
    
    def simulate(
        self, 
        starting_price: float, 
        horizon: int, 
        paths: int
    ) -> np.ndarray:
        """
        Simulate price paths using Geometric Brownian Motion.
        
        Args:
            starting_price: Initial price
            horizon: Number of steps to simulate
            paths: Number of paths to generate
            
        Returns:
            Array of shape (paths, horizon+1) containing the simulated price paths
        """
        start_time = time.time()
        logging.info(f"GBM: Simulating {paths} paths for {horizon} days horizon")
        
        # Extract parameters
        drift = self.params['drift']
        volatility = self.params['volatility']
        
        # Convert annual parameters to daily
        daily_drift = drift / 252
        daily_volatility = volatility / np.sqrt(252)
        
        # Set up time grid
        dt = 1  # 1 day
        
        # Precompute the drift term for efficiency
        drift_term = (daily_drift - 0.5 * daily_volatility**2) * dt
        
        # Generate all random shocks at once for vectorization
        setup_time = time.time()
        np.random.seed(self.params.get('random_seed', None))
        random_shocks = np.random.normal(0, 1, size=(paths, horizon)) * daily_volatility * np.sqrt(dt)
        
        # Initialize price paths array
        price_paths = np.zeros((paths, horizon + 1))
        price_paths[:, 0] = starting_price
        
        setup_elapsed = time.time() - setup_time
        logging.info(f"GBM: Setup completed in {setup_elapsed:.2f} seconds")
        
        # Vectorized computation of all paths
        calc_time = time.time()
        for t in range(1, horizon + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(drift_term + random_shocks[:, t-1])
            
            # Log progress for very large simulations
            if t % 50 == 0 and paths > 5000:
                logging.debug(f"GBM: Processed {t}/{horizon} steps")
        
        calc_elapsed = time.time() - calc_time
        logging.info(f"GBM: Path calculation completed in {calc_elapsed:.2f} seconds")
        
        total_elapsed = time.time() - start_time
        logging.info(f"GBM: Total simulation time: {total_elapsed:.2f} seconds")
        
        return price_paths
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return super().get_params()


# Register the model with the factory
ModelFactory.register_model('gbm', GeometricBrownianMotion)