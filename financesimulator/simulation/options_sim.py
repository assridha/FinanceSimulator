"""Options simulation module."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import os

from ..data.fetcher import MarketDataFetcher
from ..models.base import BaseModel, ModelFactory
from ..options.black_scholes import BlackScholes
from ..options.strategy import OptionsStrategy, Action, InstrumentType


class OptionsPricer:
    """Class for pricing options along simulated price paths."""
    
    def __init__(
        self, 
        pricing_model: str = 'black_scholes',
        risk_free_rate: Optional[float] = None,
        dividend_yield: float = 0.0
    ):
        """
        Initialize options pricer.
        
        Args:
            pricing_model: Model to use for option pricing ('black_scholes', 'binomial', etc.)
            risk_free_rate: Risk-free interest rate (decimal). If None, will be fetched.
            dividend_yield: Dividend yield (decimal)
        """
        self.pricing_model = pricing_model.lower()
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.data_fetcher = MarketDataFetcher()
        
        # Initialize risk-free rate if needed
        if self.risk_free_rate is None:
            self.risk_free_rate = self.data_fetcher.get_risk_free_rate()
    
    def price_option(
        self,
        option_type: str,
        S: float,  # Stock price
        K: float,  # Strike price
        T: float,  # Time to expiration (years)
        sigma: float,  # Implied volatility
        is_american: bool = False  # American or European style
    ) -> float:
        """
        Price an option using the specified model.
        
        Args:
            option_type: 'call' or 'put'
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility
            is_american: Whether the option is American style (can be exercised early)
            
        Returns:
            Option price
        """
        if self.pricing_model == 'black_scholes':
            # Use Black-Scholes (only valid for European options)
            if option_type.lower() == 'call':
                return BlackScholes.call_price(S, K, self.risk_free_rate, sigma, T)
            else:  # put
                return BlackScholes.put_price(S, K, self.risk_free_rate, sigma, T)
        elif self.pricing_model == 'binomial':
            # TODO: Implement binomial tree model (supports American options)
            raise NotImplementedError("Binomial tree model not yet implemented")
        elif self.pricing_model == 'heston':
            # TODO: Implement Heston stochastic volatility model
            raise NotImplementedError("Heston model not yet implemented")
        else:
            raise ValueError(f"Unknown pricing model: {self.pricing_model}")
    
    def calculate_greeks(
        self,
        option_type: str,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> Dict[str, float]:
        """
        Calculate option greeks.
        
        Args:
            option_type: 'call' or 'put'
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility
            
        Returns:
            Dictionary containing the option greeks
        """
        if self.pricing_model == 'black_scholes':
            return BlackScholes.calculate_greeks(
                option_type=option_type,
                S=S,
                K=K,
                r=self.risk_free_rate,
                sigma=sigma,
                T=T
            )
        else:
            # Other models would use numerical methods
            raise NotImplementedError(f"Greeks calculation for {self.pricing_model} not implemented")
    
    def price_along_path(
        self,
        option_type: str,
        strike: float,
        days_to_expiration: int,
        price_path: np.ndarray,
        volatility: float,
        time_points: np.ndarray
    ) -> np.ndarray:
        """
        Calculate option prices along a simulated price path.
        
        Args:
            option_type: 'call' or 'put'
            strike: Option strike price
            days_to_expiration: Days until option expiration
            price_path: Array of stock prices along the path
            volatility: Implied volatility
            time_points: Array of time points (days from start)
            
        Returns:
            Array of option prices along the path
        """
        # Calculate option price at each point on the path
        option_prices = np.zeros_like(price_path)
        
        for i, (S, t) in enumerate(zip(price_path, time_points)):
            # Time to expiration in years
            T = max(0, (days_to_expiration - t) / 365)
            
            if T <= 0:
                # Option at or past expiration, intrinsic value only
                if option_type.lower() == 'call':
                    option_prices[i] = max(0, S - strike)
                else:  # put
                    option_prices[i] = max(0, strike - S)
            else:
                # Price using the chosen model
                option_prices[i] = self.price_option(
                    option_type=option_type,
                    S=S,
                    K=strike,
                    T=T,
                    sigma=volatility
                )
                
        return option_prices


class OptionsSimulation:
    """
    Class for simulating options prices and strategies.
    """
    
    def __init__(
        self,
        ticker: str,
        starting_price: Optional[float] = None,
        volatility: Optional[float] = None,
        drift: Optional[float] = None,
        risk_free_rate: Optional[float] = None,
        dividend_yield: float = 0.0,
        pricing_model: str = 'black_scholes',
        stock_model: str = 'gbm',
        random_seed: Optional[int] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize options simulation.
        
        Args:
            ticker: Stock ticker symbol
            starting_price: Starting stock price. If None, will be fetched.
            volatility: Stock volatility. If None, will be calculated from historical data.
            drift: Stock drift (mean return). If None, will be calculated.
            risk_free_rate: Risk-free interest rate. If None, will be fetched.
            dividend_yield: Dividend yield
            pricing_model: Model to use for option pricing
            stock_model: Model to use for stock price simulation
            random_seed: Random seed for reproducibility
            model_config: Additional model-specific configuration parameters
        """
        self.ticker = ticker
        self.data_fetcher = MarketDataFetcher()
        
        # Fetch or use provided values
        self.starting_price = starting_price if starting_price is not None else self.data_fetcher.get_current_price(ticker)
        
        if volatility is None:
            self.volatility = self.data_fetcher.calculate_historical_volatility(ticker, lookback_period='1y')
        else:
            self.volatility = volatility
            
        if drift is None:
            self.drift = self.data_fetcher.calculate_mean_return(ticker, lookback_period='1y')
        else:
            self.drift = drift
            
        if risk_free_rate is None:
            self.risk_free_rate = self.data_fetcher.get_risk_free_rate()
        else:
            self.risk_free_rate = risk_free_rate
            
        self.dividend_yield = dividend_yield
        self.stock_model_name = stock_model
        self.random_seed = random_seed
        self.model_config = model_config or {}
        
        # Initialize stock price model
        self.stock_model = self._create_stock_model()
        
        # Initialize options pricer
        self.options_pricer = OptionsPricer(
            pricing_model=pricing_model,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield
        )
        
        # Simulation results
        self.stock_paths = None
        self.time_points = None
        self.option_prices = None
        self.strategy_values = None
    
    def _create_stock_model(self) -> BaseModel:
        """Create the stock price model based on configuration."""
        # Create a comprehensive parameters dict combining standard and model-specific params
        # Basic parameters that most models might need
        model_params = {
            'drift': self.drift,
            'volatility': self.volatility,
            'random_seed': self.random_seed
        }
        
        # Add any model-specific parameters from the model_config
        if self.model_config:
            # The model-specific params can override basic params if needed
            model_specific_params = self.model_config.get(self.stock_model_name, {})
            model_params.update(model_specific_params)
        
        # Use the ModelFactory to create the appropriate model
        # This decouples the model selection from the simulator
        try:
            return ModelFactory.create_model(self.stock_model_name, model_params)
        except ValueError as e:
            # Re-raise with specific error message for better diagnostics
            raise ValueError(f"Error creating stock model '{self.stock_model_name}': {str(e)}")
    
    def simulate_stock_paths(self, num_paths: int, horizon: int) -> np.ndarray:
        """
        Simulate stock price paths.
        
        Args:
            num_paths: Number of paths to simulate
            horizon: Time horizon in days
            
        Returns:
            Array of simulated stock price paths with shape (num_paths, horizon+1)
        """
        self.stock_paths = self.stock_model.simulate(self.starting_price, horizon, num_paths)
        self.time_points = np.arange(horizon + 1)
        return self.stock_paths
    
    def simulate_option(
        self,
        option_type: str,
        strike: float,
        days_to_expiration: int,
        num_paths: int = 1000,
        horizon: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate option prices along stock price paths.
        
        Args:
            option_type: 'call' or 'put'
            strike: Option strike price
            days_to_expiration: Days until option expiration
            num_paths: Number of paths to simulate
            horizon: Time horizon in days (defaults to days_to_expiration)
            
        Returns:
            Tuple of (stock_paths, option_prices) arrays
        """
        if horizon is None:
            horizon = days_to_expiration
            
        # Make sure we have stock paths
        if self.stock_paths is None or self.stock_paths.shape[0] != num_paths or self.stock_paths.shape[1] != horizon + 1:
            self.simulate_stock_paths(num_paths, horizon)
            
        # Simulate option prices along each path
        self.option_prices = np.zeros_like(self.stock_paths)
        
        for path_idx in range(num_paths):
            self.option_prices[path_idx, :] = self.options_pricer.price_along_path(
                option_type=option_type,
                strike=strike,
                days_to_expiration=days_to_expiration,
                price_path=self.stock_paths[path_idx, :],
                volatility=self.volatility,
                time_points=self.time_points
            )
            
        return self.stock_paths, self.option_prices
    
    def simulate_strategy(
        self,
        strategy: OptionsStrategy,
        num_paths: int = 1000,
        horizon: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Simulate an options strategy over time.
        
        Args:
            strategy: OptionsStrategy instance
            num_paths: Number of paths to simulate
            horizon: Time horizon in days (defaults to longest component expiration)
            
        Returns:
            Tuple of (stock_paths, component_prices, strategy_values) arrays
        """
        # Evaluate the strategy to ensure all components are resolved
        strategy_details = strategy.evaluate()
        
        # Find the longest expiration among option components
        max_expiration = 0
        for component in strategy.components:
            if component.instrument_type != InstrumentType.STOCK:
                days_to_expiry = component.option_spec.resolve(
                    strategy.ticker,
                    strategy.current_price,
                    self.data_fetcher
                )['days_to_expiration']
                max_expiration = max(max_expiration, days_to_expiry)
        
        # Set horizon to longest expiration if not provided
        if horizon is None:
            horizon = max_expiration
        
        # Make sure we have stock paths
        if self.stock_paths is None or self.stock_paths.shape[0] != num_paths or self.stock_paths.shape[1] != horizon + 1:
            self.simulate_stock_paths(num_paths, horizon)
        
        # Initialize arrays to store component prices
        component_prices = {}
        
        # Calculate initial values for positions
        initial_values = {}
        total_initial_value = 0
        
        for i, component in enumerate(strategy.components):
            if component.instrument_type == InstrumentType.STOCK:
                # For stock components, calculate the full position value
                sign = 1 if component.action == Action.BUY else -1
                component_prices[f"component_{i}"] = sign * self.stock_paths * component.quantity
                
                # Track initial value (sign indicates buy/sell direction)
                initial_value = self.starting_price * component.quantity
                if component.action == Action.SELL:
                    initial_value = -initial_value
                initial_values[f"component_{i}"] = initial_value
                total_initial_value += initial_value  # Add actual value (positive for cost, negative for credit)
            else:
                # For option components, simulate option prices
                option_type = 'call' if component.instrument_type == InstrumentType.CALL else 'put'
                strike = component.option_spec.resolved_strike
                days_to_expiry = component.option_spec.resolve(
                    strategy.ticker,
                    strategy.current_price,
                    self.data_fetcher
                )['days_to_expiration']
                
                # Calculate option prices along each path
                component_prices[f"component_{i}"] = np.zeros_like(self.stock_paths)
                
                # Get initial option price
                initial_option_price = self.options_pricer.price_option(
                    option_type=option_type,
                    S=self.starting_price,
                    K=strike,
                    T=days_to_expiry / 365,  # Convert to years
                    sigma=self.volatility
                )
                
                # For option components, calculate option price and apply sign based on action
                sign = 1 if component.action == Action.BUY else -1
                initial_position_value = sign * initial_option_price * component.quantity
                initial_values[f"component_{i}"] = initial_position_value  # Keep sign
                total_initial_value += initial_position_value  # Add actual value (positive for cost, negative for credit)
                
                for path_idx in range(num_paths):
                    component_prices[f"component_{i}"][path_idx, :] = self.options_pricer.price_along_path(
                        option_type=option_type,
                        strike=strike,
                        days_to_expiration=days_to_expiry,
                        price_path=self.stock_paths[path_idx, :],
                        volatility=self.volatility,
                        time_points=self.time_points
                    ) * component.quantity
                    
                    # Apply sign based on action (buy/sell)
                    if component.action == Action.SELL:
                        component_prices[f"component_{i}"][path_idx, :] *= -1
        
        # Calculate strategy values by combining component prices
        self.strategy_values = np.zeros_like(self.stock_paths)
        
        for i, component in enumerate(strategy.components):
            # Add the component value to the strategy value
            self.strategy_values += component_prices[f"component_{i}"]
        
        # Store total initial investment for proper return calculations
        self.total_initial_value = total_initial_value
        
        return self.stock_paths, component_prices, self.strategy_values
    
    def calculate_statistics(self, values: np.ndarray, initial_investment: float) -> Dict[str, float]:
        """
        Calculate statistics for strategy values.
        
        Args:
            values: Array of strategy values
            initial_investment: Initial investment amount (can be negative for credit trades)
            
        Returns:
            Dictionary of statistics
        """
        final_values = values[:, -1]
        
        # For proper returns calculation, we need to handle negative initial investments (credits)
        # For credits, a positive final value represents a profit
        if initial_investment > 0:
            # For debits: Return = (Final Value / Initial Investment) - 1
            final_returns = final_values / initial_investment - 1
        else:
            # For credits: Return = (Final Value - Initial Investment) / abs(Initial Investment)
            # This gives positive returns when final value is better than initial credit
            final_returns = (final_values - initial_investment) / abs(initial_investment)
        
        stats = {
            'mean_value': np.mean(final_values),
            'median_value': np.median(final_values),
            'min_value': np.min(final_values),
            'max_value': np.max(final_values),
            'std_value': np.std(final_values),
            'mean_return': np.mean(final_returns),
            'median_return': np.median(final_returns),
            'min_return': np.min(final_returns),
            'max_return': np.max(final_returns),
            'std_return': np.std(final_returns),
            'sharpe_ratio': np.mean(final_returns) / np.std(final_returns) if np.std(final_returns) > 0 else 0,
            'win_rate': np.mean(final_returns > 0),
            'profit_factor': abs(np.sum(final_returns[final_returns > 0]) / np.sum(final_returns[final_returns < 0])) if np.sum(final_returns[final_returns < 0]) != 0 else float('inf'),
            'percentile_10': np.percentile(final_values, 10),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_50': np.percentile(final_values, 50),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_90': np.percentile(final_values, 90),
        }
        
        return stats 