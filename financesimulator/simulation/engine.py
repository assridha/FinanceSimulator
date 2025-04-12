"""Core simulation engine for running financial simulations."""

import numpy as np
import pandas as pd
import yaml
import os
from typing import Dict, Any, Optional, Union, List, Tuple

from financesimulator.data.fetcher import MarketDataFetcher
from financesimulator.models.base import ModelFactory


class SimulationEngine:
    """Core engine for running financial simulations."""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None, 
                 offline_mode: bool = False, prefetch: bool = True):
        """
        Initialize the simulation engine.
        
        Args:
            config_path: Path to configuration YAML file
            config_dict: Configuration dictionary (alternative to config_path)
            offline_mode: Whether to run in offline mode (no API calls)
            prefetch: Whether to prefetch common data in background
        
        Raises:
            ValueError: If neither config_path nor config_dict is provided
        """
        if config_path is None and config_dict is None:
            raise ValueError("Either config_path or config_dict must be provided")
        
        # Load configuration
        if config_path is not None:
            self.config = self._load_config_from_file(config_path)
        else:
            self.config = config_dict
        
        # Validate configuration
        self._validate_config()
        
        # Get cache settings from config
        cache_settings = self._get_cache_settings()
        cache_enabled = cache_settings.get('enabled', True)
        cache_max_age = cache_settings.get('max_age', 86400)
        force_refresh = cache_settings.get('force_refresh', False)
        
        # Initialize components with cache settings
        self.data_fetcher = MarketDataFetcher(
            offline_mode=offline_mode, 
            prefetch=prefetch,
            cache_enabled=cache_enabled,
            cache_max_age=cache_max_age,
            force_refresh=force_refresh
        )
        self.offline_mode = offline_mode
        
        # Initialize simulation results
        self.results = None
    
    def _load_config_from_file(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If the config file does not exist
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def _validate_config(self) -> None:
        """
        Validate the configuration.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        # Check required sections
        required_sections = ['simulation']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Check simulation type
        simulation_type = self.config['simulation'].get('type')
        if simulation_type not in ['stock', 'options', 'options_strategy', 'bitcoin']:
            raise ValueError(f"Invalid simulation type: {simulation_type}")
        
        # Check if model is specified
        if 'model' not in self.config['simulation'] and 'model' not in self.config.get('stock', {}):
            raise ValueError("Model must be specified in the configuration")
    
    def _resolve_auto_params(self) -> None:
        """
        Resolve automatic parameters in the configuration.
        
        This method handles parameters marked as 'auto' in the configuration, 
        such as starting_price, volatility, etc.
        """
        simulation_type = self.config['simulation']['type']
        
        # Process stock or bitcoin simulation
        if simulation_type in ['stock', 'bitcoin']:
            ticker = self.config['simulation']['ticker']
            
            # Resolve starting price if auto
            if self.config['simulation'].get('starting_price') == 'auto':
                self.config['simulation']['starting_price'] = self.data_fetcher.get_current_price(ticker)
            
            # Resolve model parameters if auto
            if 'model_params' in self.config:
                if self.config['model_params'].get('volatility') == 'auto':
                    lookback = self.config.get('data', {}).get('lookback_period', '1y')
                    self.config['model_params']['volatility'] = self.data_fetcher.calculate_historical_volatility(
                        ticker, lookback_period=lookback
                    )
                
                if self.config['model_params'].get('drift') == 'auto':
                    lookback = self.config.get('data', {}).get('lookback_period', '1y')
                    self.config['model_params']['drift'] = self.data_fetcher.calculate_mean_return(
                        ticker, lookback_period=lookback
                    )
        
        # Process options or options strategy simulation
        elif simulation_type in ['options', 'options_strategy']:
            # Handle stock component
            if 'stock' in self.config:
                ticker = self.config['stock']['ticker']
                
                # Resolve starting price if auto
                if self.config['stock'].get('starting_price') == 'auto':
                    self.config['stock']['starting_price'] = self.data_fetcher.get_current_price(ticker)
                
                # Resolve model parameters if auto
                if 'model_params' in self.config:
                    if self.config['model_params'].get('volatility') == 'auto':
                        lookback = self.config.get('data', {}).get('lookback_period', '1y')
                        self.config['model_params']['volatility'] = self.data_fetcher.calculate_historical_volatility(
                            ticker, lookback_period=lookback
                        )
                    
                    if self.config['model_params'].get('drift') == 'auto':
                        lookback = self.config.get('data', {}).get('lookback_period', '1y')
                        self.config['model_params']['drift'] = self.data_fetcher.calculate_mean_return(
                            ticker, lookback_period=lookback
                        )
            
            # Handle risk-free rate if auto
            if self.config.get('data', {}).get('risk_free_rate') == 'auto':
                self.config['data']['risk_free_rate'] = self.data_fetcher.get_risk_free_rate()
    
    def run(self) -> Dict[str, Any]:
        """
        Run the simulation based on the configuration.
        
        Returns:
            Dictionary containing simulation results
        """
        # Resolve automatic parameters
        self._resolve_auto_params()
        
        # Get simulation type
        simulation_type = self.config['simulation']['type']
        
        # Run appropriate simulation
        if simulation_type == 'stock':
            return self._run_stock_simulation()
        elif simulation_type == 'bitcoin':
            return self._run_bitcoin_simulation()
        elif simulation_type == 'options':
            return self._run_options_simulation()
        elif simulation_type == 'options_strategy':
            return self._run_options_strategy_simulation()
        else:
            raise ValueError(f"Unsupported simulation type: {simulation_type}")
    
    def _run_stock_simulation(self) -> Dict[str, Any]:
        """
        Run a stock simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        # Extract configuration
        ticker = self.config['simulation']['ticker']
        model_name = self.config['simulation']['model']
        paths = self.config['simulation']['paths']
        horizon = self.config['simulation']['horizon']
        starting_price = self.config['simulation']['starting_price']
        model_params = self.config.get('model_params', {})
        
        # Create model
        model = ModelFactory.create_model(model_name, model_params)
        
        # Simulate price paths
        price_paths = model.simulate(starting_price, horizon, paths)
        
        # Store results
        self.results = {
            'ticker': ticker,
            'model': model_name,
            'price_paths': price_paths,
            'horizon': horizon,
            'paths': paths,
            'model_params': model.get_params()
        }
        
        return self.results
    
    def _run_bitcoin_simulation(self) -> Dict[str, Any]:
        """
        Run a Bitcoin simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        # Bitcoin simulation is essentially the same as stock simulation
        return self._run_stock_simulation()
    
    def _run_options_simulation(self) -> Dict[str, Any]:
        """
        Run an options simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        # To be implemented in options_sim.py
        raise NotImplementedError("Options simulation not yet implemented")
    
    def _run_options_strategy_simulation(self) -> Dict[str, Any]:
        """
        Run an options strategy simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        # Extract configuration
        if 'stock' not in self.config:
            raise ValueError("Options strategy simulation requires stock configuration")
            
        ticker = self.config['stock']['ticker']
        model_name = self.config['stock'].get('model', 'gbm')
        paths = self.config['simulation']['paths']
        horizon = self.config['simulation']['horizon']
        starting_price = self.config['stock'].get('starting_price', 'auto')
        model_params = self.config.get('model_params', {})
        strategy = self.config.get('strategy', [])
        
        # If starting price is 'auto', fetch it
        if starting_price == 'auto':
            starting_price = self.data_fetcher.get_current_price(ticker)
            
        # Create model for stock price simulation
        model = ModelFactory.create_model(model_name, model_params)
        
        # Simulate stock price paths
        price_paths = model.simulate(starting_price, horizon, paths)
        
        # For now, we're just returning the stock simulation results
        # In a real implementation, we would apply the options strategy to calculate payoffs
        self.results = {
            'ticker': ticker,
            'model': model_name,
            'price_paths': price_paths,
            'horizon': horizon,
            'paths': paths,
            'model_params': model.get_params(),
            'strategy': strategy,
            'simulation_type': 'options_strategy'
        }
        
        return self.results
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get simulation results.
        
        Returns:
            Dictionary containing simulation results
            
        Raises:
            RuntimeError: If run() has not been called yet
        """
        if self.results is None:
            raise RuntimeError("No results available. Call run() first.")
        
        return self.results

    def _get_cache_settings(self) -> Dict[str, Any]:
        """Get cache settings from config."""
        # Default cache settings
        default_settings = {
            'enabled': True,
            'max_age': 86400,  # 24 hours
            'force_refresh': False
        }
        
        # Get settings from config if they exist
        cache_settings = self.config.get('data', {}).get('cache', {})
        
        # Return merged settings (with defaults for missing values)
        return {**default_settings, **cache_settings}