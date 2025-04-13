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
            logging.info(f"GBM: Initialized with random seed {self.params['random_seed']}")
        else:
            logging.info("GBM: Initialized with no random seed (results will vary between runs)")
        
        logging.info(f"GBM: Model parameters - Drift: {self.params['drift']:.4f}, Volatility: {self.params['volatility']:.4f}")
    
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
        logging.info(f"GBM: Fitting model to historical data with {len(data)} data points")
        fit_start_time = time.time()
        
        # Calculate log returns
        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        
        # Calculate volatility (annualized)
        old_volatility = self.params.get('volatility', None)
        self.params['volatility'] = returns.std() * np.sqrt(252)
        
        # Calculate drift (annualized)
        # Using the risk-neutral formula: drift = risk_free_rate - 0.5 * volatility^2
        # If risk-free rate is not provided, use historical mean return
        old_drift = self.params.get('drift', None)
        if 'risk_free_rate' in self.params:
            self.params['drift'] = (self.params['risk_free_rate'] - 
                                   0.5 * self.params['volatility']**2)
            logging.info(f"GBM: Using risk-neutral drift calculation with risk-free rate {self.params['risk_free_rate']:.4f}")
        else:
            self.params['drift'] = returns.mean() * 252
            logging.info(f"GBM: Using historical mean return for drift calculation")
        
        fit_elapsed = time.time() - fit_start_time
        
        # Log parameter changes
        if old_volatility is not None:
            vol_change = (self.params['volatility'] - old_volatility) / old_volatility * 100
            logging.info(f"GBM: Volatility changed from {old_volatility:.4f} to {self.params['volatility']:.4f} ({vol_change:+.2f}%)")
        else:
            logging.info(f"GBM: Fitted volatility: {self.params['volatility']:.4f}")
            
        if old_drift is not None:
            drift_change = (self.params['drift'] - old_drift) / old_drift * 100 if old_drift != 0 else float('inf')
            logging.info(f"GBM: Drift changed from {old_drift:.4f} to {self.params['drift']:.4f} ({drift_change:+.2f}%)")
        else:
            logging.info(f"GBM: Fitted drift: {self.params['drift']:.4f}")
        
        logging.info(f"GBM: Model fitting completed in {fit_elapsed:.4f} seconds")
        
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
        logging.info(f"\n=== GBM SIMULATION STARTED ===")
        logging.info(f"GBM: Simulating price paths with the following parameters:")
        logging.info(f"GBM: - Starting price: ${starting_price:.2f}")
        logging.info(f"GBM: - Time horizon: {horizon} days")
        logging.info(f"GBM: - Number of paths: {paths}")
        logging.info(f"GBM: - Annual drift: {self.params['drift']:.4f}")
        logging.info(f"GBM: - Annual volatility: {self.params['volatility']:.4f}")
        
        # Extract parameters
        drift = self.params['drift']
        volatility = self.params['volatility']
        
        # Convert annual parameters to daily
        daily_drift = drift / 252
        daily_volatility = volatility / np.sqrt(252)
        
        logging.info(f"GBM: - Daily drift: {daily_drift:.6f}")
        logging.info(f"GBM: - Daily volatility: {daily_volatility:.6f}")
        
        # Set up time grid
        dt = 1  # 1 day
        
        # Precompute the drift term for efficiency
        drift_term = (daily_drift - 0.5 * daily_volatility**2) * dt
        
        # Generate all random shocks at once for vectorization
        setup_time = time.time()
        logging.info(f"GBM: Generating random shocks for all paths...")
        np.random.seed(self.params.get('random_seed', None))
        random_shocks = np.random.normal(0, 1, size=(paths, horizon)) * daily_volatility * np.sqrt(dt)
        
        # Initialize price paths array
        price_paths = np.zeros((paths, horizon + 1))
        price_paths[:, 0] = starting_price
        
        setup_elapsed = time.time() - setup_time
        logging.info(f"GBM: Setup completed in {setup_elapsed:.2f} seconds")
        
        # Vectorized computation of all paths
        calc_time = time.time()
        logging.info(f"GBM: Calculating price paths...")
        
        # Log progress more frequently for large simulations
        log_frequency = max(1, min(horizon // 10, 50))  # Log at most 10 times, but at least once
        
        for t in range(1, horizon + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(drift_term + random_shocks[:, t-1])
            
            # Log progress at regular intervals
            if t % log_frequency == 0 or t == horizon:
                progress_pct = (t / horizon) * 100
                elapsed_so_far = time.time() - calc_time
                time_per_step = elapsed_so_far / t
                est_remaining = time_per_step * (horizon - t)
                
                logging.info(f"GBM: Progress {progress_pct:.1f}% - {t}/{horizon} steps - Est. remaining: {est_remaining:.1f}s")
        
        calc_elapsed = time.time() - calc_time
        logging.info(f"GBM: Path calculation completed in {calc_elapsed:.2f} seconds")
        
        # Calculate some statistics on the final prices
        final_prices = price_paths[:, -1]
        mean_final = final_prices.mean()
        median_final = np.median(final_prices)
        min_final = final_prices.min()
        max_final = final_prices.max()
        
        logging.info(f"GBM: Final price statistics:")
        logging.info(f"GBM: - Mean final price: ${mean_final:.2f}")
        logging.info(f"GBM: - Median final price: ${median_final:.2f}")
        logging.info(f"GBM: - Min final price: ${min_final:.2f}")
        logging.info(f"GBM: - Max final price: ${max_final:.2f}")
        
        # Log overall performance
        total_elapsed = time.time() - start_time
        paths_per_second = paths / total_elapsed
        logging.info(f"GBM: Total simulation time: {total_elapsed:.2f} seconds ({paths_per_second:.1f} paths/second)")
        logging.info(f"=== GBM SIMULATION COMPLETED ===\n")
        
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