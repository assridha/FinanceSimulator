"""Core simulation engine for running financial simulations."""

import os
import logging
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple

from financesimulator.data.fetcher import MarketDataFetcher
from financesimulator.models.base import ModelFactory
from ..options.strategy import OptionsStrategy, StrategyComponent, InstrumentType, Action, OptionSpecification
from ..visualization.plots import SimulationVisualizer, OptionsStrategyVisualizer
from .options_sim import OptionsSimulation


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
        Run a stock price simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        # Extract configuration
        ticker = self.config['simulation']['ticker']
        model_name = self.config['simulation']['model']
        paths = self.config['simulation']['paths']
        horizon = self.config['simulation']['horizon']
        starting_price = self.config['simulation']['starting_price']
        
        # Handle model parameters more flexibly
        # Extract both common parameters and model-specific parameters
        model_params = {}
        
        # First add common parameters
        if 'model_params' in self.config:
            model_params.update(self.config['model_params'])
        
        # Then overlay model-specific parameters if they exist
        if 'models' in self.config and model_name in self.config['models']:
            model_specific_params = self.config['models'][model_name]
            model_params.update(model_specific_params)
        
        # Create model
        try:
            model = ModelFactory.create_model(model_name, model_params)
        except ValueError as e:
            raise ValueError(f"Error creating model '{model_name}': {str(e)}")
        
        # Simulate price paths
        price_paths = model.simulate(starting_price, horizon, paths)
        
        # Store results
        self.results = {
            'ticker': ticker,
            'model': model_name,
            'price_paths': price_paths,
            'horizon': horizon,
            'paths': paths,
            'model_params': model.get_params(),
            'simulation_type': 'stock'  # Add simulation type
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
            Dictionary of results
        """
        # Similar setup as stock simulation
        ticker = self.config.get('stock', {}).get('ticker', 'SPY')
        model_type = self.config.get('stock', {}).get('model', 'gbm')
        num_paths = self.config.get('simulation', {}).get('paths', 1000)
        horizon = self.config.get('simulation', {}).get('horizon', 252)
        output_dir = self.config.get('visualization', {}).get('output_dir', 'outputs')
        
        # Get model-specific configurations from config
        # This enables passing model-specific parameters to the simulation
        model_config = {}
        if 'model_params' in self.config:
            # Extract common parameters first
            common_params = {
                k: v for k, v in self.config.get('model_params', {}).items()
                if k in ['volatility', 'drift', 'random_seed']
            }
            
            # Structure the model config to have model-specific params
            # This allows different models to have their own sections
            for model_name, model_section in self.config.get('models', {}).items():
                model_config[model_name] = model_section
                
            # Add any model-specific configs from the main model_params section
            # This is for backward compatibility or when the config is flat
            if model_type in model_config:
                # Add common params to model-specific section only if not already there
                for k, v in common_params.items():
                    if k not in model_config[model_type]:
                        model_config[model_type][k] = v
            else:
                # If no specific section exists, create one with common params
                model_config[model_type] = common_params
        
        # Create options simulation
        options_sim = OptionsSimulation(
            ticker=ticker,
            starting_price=self.config.get('stock', {}).get('starting_price', None),
            volatility=self.config.get('model_params', {}).get('volatility', None),
            drift=self.config.get('model_params', {}).get('drift', None),
            risk_free_rate=self.config.get('data', {}).get('risk_free_rate', None),
            dividend_yield=self.config.get('data', {}).get('dividend_yield', 0.0),
            pricing_model=self.config.get('options', {}).get('pricing_model', 'black_scholes'),
            stock_model=model_type,
            random_seed=self.config.get('model_params', {}).get('random_seed', None),
            model_config=model_config
        )
        
        # Generate stock price paths
        logging.info(f"Simulating {num_paths} price paths for {ticker} using {model_type} model")
        stock_paths = options_sim.simulate_stock_paths(num_paths, horizon)
        starting_price = options_sim.starting_price
        
        # Parse strategy configuration
        strategy_config = self.config.get('strategy', [])
        expiration_days = horizon  # Default to simulation horizon
        
        # Create strategy
        strategy = OptionsStrategy(ticker=ticker)
        
        # Add components to strategy based on configuration
        for component in strategy_config:
            action = component.get('action', '').lower()
            instrument = component.get('instrument', '').lower()
            quantity = component.get('quantity', 1)
            
            if instrument == 'stock':
                # Add stock position
                position = StrategyComponent(
                    instrument_type=InstrumentType.STOCK,
                    action=Action(action.upper()),
                    quantity=quantity
                )
                strategy.add_component(position)
                
            elif instrument in ['call', 'put']:
                # Parse strike price
                strike_str = str(component.get('strike', '0%ATM'))
                if '%' in strike_str:
                    # Percentage based strike (e.g., 10%OTM, 5%ITM, 0%ATM)
                    parts = strike_str.split('%')
                    percent = float(parts[0]) / 100
                    moneyness = parts[1].upper()
                    
                    if moneyness == 'ATM':
                        strike = starting_price
                    elif moneyness == 'OTM':
                        if instrument == 'call':
                            strike = starting_price * (1 + percent)
                        else:  # put
                            strike = starting_price * (1 - percent)
                    elif moneyness == 'ITM':
                        if instrument == 'call':
                            strike = starting_price * (1 - percent)
                        else:  # put
                            strike = starting_price * (1 + percent)
                    else:
                        raise ValueError(f"Invalid moneyness in strike: {strike_str}")
                else:
                    # Absolute strike price
                    strike = float(strike_str)
                
                # Parse expiration
                expiration = component.get('expiration', f"{horizon}d")
                if isinstance(expiration, str) and expiration.endswith('d'):
                    expiration_days = int(expiration[:-1])
                else:
                    expiration_days = int(expiration)
                
                # Create option specification
                option_spec = OptionSpecification(
                    option_type=instrument,
                    strike_spec=strike,
                    expiration_spec=expiration_days,
                    quantity=quantity,
                    action=Action(action.upper())
                )
                
                # Add option position
                position = StrategyComponent(
                    instrument_type=InstrumentType.CALL if instrument == 'call' else InstrumentType.PUT,
                    action=Action(action.upper()),
                    quantity=quantity,
                    option_spec=option_spec
                )
                strategy.add_component(position)
                
            else:
                logging.warning(f"Unknown instrument type: {instrument}")
        
        # Simulate strategy
        logging.info(f"Simulating {ticker} options strategy: {strategy.name}")
        stock_paths, component_values, strategy_values = options_sim.simulate_strategy(
            strategy=strategy,
            num_paths=num_paths,
            horizon=horizon
        )
        
        # Calculate statistics
        strategy_details = strategy.evaluate()
        initial_investment = options_sim.total_initial_value
        stats = options_sim.calculate_statistics(strategy_values, initial_investment)
        
        # Create visualizations
        if self.config.get('visualization', {}).get('plot_paths', True):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Create visualizer instance
            visualizer = OptionsStrategyVisualizer(
                results={
                    'ticker': ticker,
                    'stock_paths': stock_paths,
                    'strategy_values': strategy_values,
                    'strategy': {
                        'total_cost': initial_investment,
                        'components': strategy_details['components']
                    },
                    'starting_price': starting_price,
                    'stats': stats
                }
            )
            
            # Create paths plot
            paths_output_file = os.path.join(output_dir, f"{ticker}_options_strategy_paths.png")
            logging.info(f"Creating options strategy paths plot at: {paths_output_file}")
            visualizer.visualize_paths(
                values=strategy_values,
                initial_investment=initial_investment,
                title=f"{ticker} Options Strategy Paths",
                save_path=paths_output_file,
                show_plot=False
            )
            logging.info(f"Saved paths plot to: {paths_output_file}")
            
            # Create distribution plot
            dist_path = os.path.join(output_dir, f"{ticker}_options_strategy_distribution.png")
            visualizer.visualize_distribution(
                title=f"{ticker} Options Strategy Distribution of Final Values",
                save_path=dist_path,
                show_plot=False
            )
            
            # Create returns distribution plot
            returns_path = os.path.join(output_dir, f"{ticker}_options_strategy_returns.png")
            visualizer.visualize_returns_distribution(
                title=f"{ticker} Options Strategy Returns Distribution",
                save_path=returns_path,
                show_plot=False
            )
            
            # Create histogram plot
            hist_path = os.path.join(output_dir, f"{ticker}_options_strategy_histogram.png")
            visualizer.visualize_histogram(
                title=f"{ticker} Options Strategy Final Values",
                save_path=hist_path,
                show_plot=False
            )
            
            # Plot payoff curve
            if self.config.get('visualization', {}).get('plot_payoff', False):
                payoff_path = os.path.join(output_dir, f"{ticker}_options_strategy_payoff.png")
                payoff_data = strategy.calculate_payoff_curve()
                visualizer.visualize_payoff_curve(
                    payoff_data=payoff_data,
                    title=f"{ticker} Options Strategy Payoff at Expiration",
                    save_path=payoff_path,
                    show_plot=False
                )
                
            # Plot strategy vs stock returns
            if self.config.get('visualization', {}).get('plot_strategy_vs_stock', False):
                comparison_path = os.path.join(output_dir, f"{ticker}_strategy_vs_stock.png")
                
                # Scale stock paths according to the first strategy component's quantity
                stock_quantity = 100
                for component in strategy.components:
                    if component.instrument_type.value == "STOCK":
                        stock_quantity = component.quantity
                        break
                
                stock_values = stock_paths * stock_quantity
                initial_stock_cost = starting_price * stock_quantity
                
                visualizer.visualize_strategy_vs_stock(
                    stock_values=stock_values,
                    initial_stock_cost=initial_stock_cost,
                    save_path=comparison_path,
                    show_plot=False
                )
            
            # Create summary dashboard
            summary_path = os.path.join(output_dir, f"{ticker}_options_strategy_summary.png")
            payoff_data = strategy.calculate_payoff_curve()
            visualizer.generate_summary_dashboard(
                payoff_data=payoff_data,
                title=f"{ticker} Options Strategy Summary",
                save_path=summary_path,
                show_plot=False
            )
            
        # Prepare results
        results = {
            'ticker': ticker,
            'stock_paths': stock_paths.tolist(),  # Convert to list for JSON serialization
            'strategy_values': strategy_values.tolist(),
            'component_values': {k: v.tolist() for k, v in component_values.items()},
            'time_points': stock_paths.shape[1],
            'strategy': {
                'name': strategy.name,
                'total_cost': initial_investment,
                'components': strategy_details['components']
            },
            'starting_price': starting_price,
            'stats': stats
        }
        
        return results
    
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