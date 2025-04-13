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
        logging.info("\n=== RESOLVING SIMULATION PARAMETERS ===")
        simulation_type = self.config['simulation']['type']
        
        # Process stock or bitcoin simulation
        if simulation_type in ['stock', 'bitcoin']:
            ticker = self.config['simulation']['ticker']
            logging.info(f"Resolving parameters for {simulation_type.upper()} simulation of {ticker}")
            
            # Resolve starting price if auto
            if self.config['simulation'].get('starting_price') == 'auto':
                starting_price = self.data_fetcher.get_current_price(ticker)
                self.config['simulation']['starting_price'] = starting_price
                logging.info(f"Resolved starting price: ${starting_price:.2f}")
            else:
                logging.info(f"Using provided starting price: ${self.config['simulation']['starting_price']:.2f}")
            
            # Resolve model parameters if auto
            if 'model_params' in self.config:
                if self.config['model_params'].get('volatility') == 'auto':
                    lookback = self.config.get('data', {}).get('lookback_period', '1y')
                    volatility = self.data_fetcher.calculate_historical_volatility(
                        ticker, lookback_period=lookback
                    )
                    self.config['model_params']['volatility'] = volatility
                    logging.info(f"Resolved historical volatility: {volatility:.4f} (lookback: {lookback})")
                else:
                    logging.info(f"Using provided volatility: {self.config['model_params']['volatility']:.4f}")
                
                if self.config['model_params'].get('drift') == 'auto':
                    lookback = self.config.get('data', {}).get('lookback_period', '1y')
                    drift = self.data_fetcher.calculate_mean_return(
                        ticker, lookback_period=lookback
                    )
                    self.config['model_params']['drift'] = drift
                    logging.info(f"Resolved mean return (drift): {drift:.4f} (lookback: {lookback})")
                else:
                    logging.info(f"Using provided drift: {self.config['model_params']['drift']:.4f}")
        
        # Process options or options strategy simulation
        elif simulation_type in ['options', 'options_strategy']:
            logging.info(f"Resolving parameters for {simulation_type.upper()} simulation")
            # Handle stock component
            if 'stock' in self.config:
                ticker = self.config['stock']['ticker']
                logging.info(f"Resolving underlying stock parameters for {ticker}")
                
                # Resolve starting price if auto
                if self.config['stock'].get('starting_price') == 'auto':
                    starting_price = self.data_fetcher.get_current_price(ticker)
                    self.config['stock']['starting_price'] = starting_price
                    logging.info(f"Resolved stock starting price: ${starting_price:.2f}")
                else:
                    logging.info(f"Using provided stock starting price: ${self.config['stock']['starting_price']:.2f}")
                
                # Resolve model parameters if auto
                if 'model_params' in self.config:
                    if self.config['model_params'].get('volatility') == 'auto':
                        lookback = self.config.get('data', {}).get('lookback_period', '1y')
                        volatility = self.data_fetcher.calculate_historical_volatility(
                            ticker, lookback_period=lookback
                        )
                        self.config['model_params']['volatility'] = volatility
                        logging.info(f"Resolved historical volatility: {volatility:.4f} (lookback: {lookback})")
                    else:
                        logging.info(f"Using provided volatility: {self.config['model_params']['volatility']:.4f}")
                    
                    if self.config['model_params'].get('drift') == 'auto':
                        lookback = self.config.get('data', {}).get('lookback_period', '1y')
                        drift = self.data_fetcher.calculate_mean_return(
                            ticker, lookback_period=lookback
                        )
                        self.config['model_params']['drift'] = drift
                        logging.info(f"Resolved mean return (drift): {drift:.4f} (lookback: {lookback})")
                    else:
                        logging.info(f"Using provided drift: {self.config['model_params']['drift']:.4f}")
            
            # Handle risk-free rate if auto
            if self.config.get('data', {}).get('risk_free_rate') == 'auto':
                risk_free_rate = self.data_fetcher.get_risk_free_rate()
                self.config['data']['risk_free_rate'] = risk_free_rate
                logging.info(f"Resolved risk-free rate: {risk_free_rate:.4f}")
            else:
                if 'risk_free_rate' in self.config.get('data', {}):
                    logging.info(f"Using provided risk-free rate: {self.config['data']['risk_free_rate']:.4f}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the simulation based on the configuration.
        
        Returns:
            Dictionary containing simulation results
        """
        logging.info("\n=== STARTING SIMULATION ===")
        
        # Resolve automatic parameters
        self._resolve_auto_params()
        
        # Get simulation type
        simulation_type = self.config['simulation']['type']
        logging.info(f"Running {simulation_type.upper()} simulation")
        
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
        logging.info("\n=== STOCK SIMULATION SETUP ===")
        
        # Extract configuration
        ticker = self.config['simulation']['ticker']
        model_name = self.config['simulation']['model']
        paths = self.config['simulation']['paths']
        horizon = self.config['simulation']['horizon']
        starting_price = self.config['simulation']['starting_price']
        
        logging.info(f"Simulation details:")
        logging.info(f"- Ticker: {ticker}")
        logging.info(f"- Model: {model_name}")
        logging.info(f"- Starting price: ${starting_price:.2f}")
        logging.info(f"- Simulation horizon: {horizon} days")
        logging.info(f"- Number of paths: {paths}")
        
        # Handle model parameters more flexibly
        # Extract both common parameters and model-specific parameters
        model_params = {}
        
        # First add common parameters
        if 'model_params' in self.config:
            model_params.update(self.config['model_params'])
            if 'volatility' in model_params:
                logging.info(f"- Volatility: {model_params['volatility']:.4f}")
            if 'drift' in model_params:
                logging.info(f"- Drift (mean return): {model_params['drift']:.4f}")
        
        # Then overlay model-specific parameters if they exist
        if 'models' in self.config and model_name in self.config['models']:
            model_specific_params = self.config['models'][model_name]
            model_params.update(model_specific_params)
            logging.info(f"- Additional {model_name} parameters: {model_specific_params}")
        
        # Create model
        logging.info("\n=== INITIALIZING MODEL ===")
        try:
            model = ModelFactory.create_model(model_name, model_params)
            logging.info(f"Successfully initialized {model_name} model")
        except ValueError as e:
            error_msg = f"Error creating model '{model_name}': {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # Simulate price paths
        logging.info("\n=== RUNNING SIMULATION ===")
        logging.info(f"Simulating {paths} price paths for {ticker} over {horizon} days")
        price_paths = model.simulate(starting_price, horizon, paths)
        logging.info(f"Simulation complete, generated {paths} price paths")
        
        # Calculate quick statistics for logging
        final_prices = price_paths[:, -1]
        mean_final = final_prices.mean()
        median_final = float(sorted(final_prices)[len(final_prices) // 2])
        min_final = final_prices.min()
        max_final = final_prices.max()
        
        logging.info("\n=== SIMULATION RESULTS SUMMARY ===")
        logging.info(f"Final price statistics after {horizon} days:")
        logging.info(f"- Mean: ${mean_final:.2f}")
        logging.info(f"- Median: ${median_final:.2f}")
        logging.info(f"- Min: ${min_final:.2f}")
        logging.info(f"- Max: ${max_final:.2f}")
        logging.info(f"- Range: ${max_final - min_final:.2f}")
        
        # Store results
        self.results = {
            'ticker': ticker,
            'model': model_name,
            'price_paths': price_paths,
            'horizon': horizon,
            'paths': paths,
            'model_params': model.get_params(),
            'simulation_type': 'stock',  # Add simulation type
            'starting_price': starting_price
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
        logging.info("\n=== OPTIONS SIMULATION SETUP ===")
        
        # Extract common configuration 
        ticker = self.config['stock']['ticker']
        model_name = self.config['stock'].get('model', 'gbm')
        paths = self.config['simulation']['paths']
        horizon = self.config['simulation']['horizon']
        starting_price = self.config['stock']['starting_price']
        
        # Extract option specific configuration
        option_type = self.config['option']['type'].lower()  # call or put
        strike = self.config['option']['strike']
        days_to_expiration = self.config['option']['days_to_expiration']
        
        logging.info(f"Options simulation details:")
        logging.info(f"- Underlying ticker: {ticker}")
        logging.info(f"- Stock model: {model_name}")
        logging.info(f"- Starting stock price: ${starting_price:.2f}")
        logging.info(f"- Option type: {option_type.upper()}")
        logging.info(f"- Strike price: ${strike:.2f}")
        logging.info(f"- Days to expiration: {days_to_expiration}")
        logging.info(f"- Simulation horizon: {horizon} days")
        logging.info(f"- Number of paths: {paths}")
        
        # Get volatility and risk-free rate for option pricing
        volatility = self.config['model_params'].get('volatility')
        drift = self.config['model_params'].get('drift')
        risk_free_rate = self.config['data'].get('risk_free_rate')
        
        logging.info(f"- Volatility: {volatility:.4f}")
        logging.info(f"- Drift (mean return): {drift:.4f}")
        logging.info(f"- Risk-free rate: {risk_free_rate:.4f}")
        
        # Calculate initial option price
        from financesimulator.options.black_scholes import black_scholes_price
        initial_option_price = black_scholes_price(
            option_type=option_type,
            S=starting_price,
            K=strike,
            T=days_to_expiration/252,  # Convert days to years
            r=risk_free_rate,
            sigma=volatility
        )
        
        logging.info(f"Initial option pricing:")
        logging.info(f"- Black-Scholes price: ${initial_option_price:.2f}")
        logging.info(f"- Stock price / Strike ratio: {starting_price/strike:.4f}")
        
        # Try to get real market option price if available
        try:
            option_market_data = self.data_fetcher.get_option_price(
                ticker=ticker,
                option_type=option_type,
                strike=strike,
                days_to_expiration=days_to_expiration
            )
            if option_market_data:
                logging.info(f"- Market option price: ${option_market_data['price']:.2f}")
                logging.info(f"- Market implied volatility: {option_market_data['implied_volatility']:.4f}")
        except Exception as e:
            logging.info(f"- Market option data not available: {str(e)}")
    
    def _run_options_strategy_simulation(self) -> Dict[str, Any]:
        """
        Run an options strategy simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        logging.info("\n=== OPTIONS STRATEGY SIMULATION SETUP ===")
        
        # Extract configuration
        ticker = self.config['stock']['ticker']
        starting_price = self.config['stock']['starting_price']
        paths = self.config['simulation']['paths']
        horizon = self.config.get('simulation', {}).get('horizon')
        
        logging.info(f"Strategy simulation details:")
        logging.info(f"- Underlying ticker: {ticker}")
        logging.info(f"- Starting stock price: ${starting_price:.2f}")
        logging.info(f"- Number of paths: {paths}")
        if horizon:
            logging.info(f"- Specific horizon: {horizon} days")
        else:
            logging.info(f"- Horizon: Auto (based on strategy expirations)")
            
        # Get model parameters
        model_type = self.config['stock'].get('model', 'gbm')
        volatility = self.config['model_params'].get('volatility')
        drift = self.config['model_params'].get('drift')
        risk_free_rate = self.config['data'].get('risk_free_rate')
        
        logging.info(f"Model parameters:")
        logging.info(f"- Stock model: {model_type}")
        logging.info(f"- Volatility: {volatility:.4f}")
        logging.info(f"- Drift (mean return): {drift:.4f}")
        logging.info(f"- Risk-free rate: {risk_free_rate:.4f}")
        
        # Set up strategy components
        strategy_config = self.config.get('strategy', [])
        
        # Check if strategy is a list or dict and handle accordingly
        if isinstance(strategy_config, list):
            # Handle legacy format where strategy is a list of components
            strategy_name = f"{ticker} Strategy"
            components = strategy_config
        elif isinstance(strategy_config, dict):
            # Handle newer format where strategy is a dict with 'name' and 'components' keys
            strategy_name = strategy_config.get('name', f"{ticker} Strategy")
            components = strategy_config.get('components', [])
        else:
            # Handle unexpected format
            logging.error(f"Unexpected strategy configuration format: {type(strategy_config)}")
            strategy_name = f"{ticker} Strategy"
            components = []
        
        logging.info(f"\n=== STRATEGY: {strategy_name.upper()} ===")
        logging.info(f"Number of components: {len(components)}")
        
        # Create strategy instance
        logging.info("Creating strategy components:")
        strategy = OptionsStrategy(ticker=ticker, name=strategy_name)
        
        # Add each component to the strategy
        for i, component_config in enumerate(components):
            instrument = component_config.get('instrument', 'stock')
            action = component_config.get('action', 'buy')
            quantity = component_config.get('quantity', 1)
            
            logging.info(f"Component {i+1}: {action.upper()} {quantity}x {instrument.upper()}")
            
            # Process stock component
            if instrument.lower() == 'stock':
                logging.info(f"  - Adding stock position: {action} {quantity} shares")
                logging.info(f"  - Current stock price: ${starting_price:.2f}")
                strategy.add_stock(
                    action=action,
                    quantity=quantity
                )
            
            # Process option component
            elif instrument.lower() in ['call', 'put']:
                strike = component_config.get('strike')
                expiration = component_config.get('expiration')
                strike_pct = component_config.get('strike_pct')
                days_to_expiration = component_config.get('days_to_expiration')
                strike_type = 'ATM'
                
                # Parse strike if it's a string with format like "10%OTM" or "5%ITM"
                if isinstance(strike, str) and '%' in strike:
                    parts = strike.split('%')
                    pct_value = float(parts[0])
                    moneyness = parts[1].upper()
                    
                    if moneyness == 'ATM':
                        calculated_strike = starting_price
                        strike_type = 'ATM'
                    elif moneyness == 'OTM':
                        if instrument.lower() == 'call':
                            calculated_strike = starting_price * (1 + pct_value/100)
                        else:  # put
                            calculated_strike = starting_price * (1 - pct_value/100)
                        strike_type = f"{pct_value}% OTM"
                    elif moneyness == 'ITM':
                        if instrument.lower() == 'call':
                            calculated_strike = starting_price * (1 - pct_value/100)
                        else:  # put
                            calculated_strike = starting_price * (1 + pct_value/100)
                        strike_type = f"{pct_value}% ITM"
                    
                    logging.info(f"  - Parsed strike {strike} to ${calculated_strike:.2f} ({strike_type})")
                    strike = calculated_strike
                    
                # Parse expiration if it's a string with format like "30d" or "3m"
                if isinstance(expiration, str):
                    if expiration.endswith('d'):
                        days_to_expiration = int(expiration[:-1])
                        logging.info(f"  - Parsed expiration {expiration} to {days_to_expiration} days")
                    elif expiration.endswith('m'):
                        # Convert months to days (approximate)
                        months = int(expiration[:-1])
                        days_to_expiration = months * 30
                        logging.info(f"  - Parsed expiration {expiration} to {days_to_expiration} days (approx {months} months)")
                
                if strike:
                    logging.info(f"  - Strike price: ${strike:.2f}")
                    strike_type = 'custom'
                elif strike_pct:
                    logging.info(f"  - Strike percentage: {strike_pct:.1f}% of current price")
                    calculated_strike = starting_price * (1 + strike_pct/100)
                    logging.info(f"  - Calculated strike: ${calculated_strike:.2f}")
                    strike_type = f"{strike_pct:+.1f}%"
                
                if expiration:
                    logging.info(f"  - Expiration date: {expiration}")
                elif days_to_expiration:
                    logging.info(f"  - Days to expiration: {days_to_expiration}")
                
                # Set option spec fields
                option_spec = {}
                if strike:
                    option_spec['strike'] = strike
                elif strike_pct:
                    option_spec['strike_pct'] = strike_pct
                
                if expiration:
                    option_spec['expiration'] = expiration
                elif days_to_expiration:
                    option_spec['days_to_expiration'] = days_to_expiration
                
                delta = component_config.get('delta')
                if delta:
                    option_spec['delta'] = delta
                    logging.info(f"  - Target delta: {delta:.2f}")
                    strike_type = f"delta-{delta}"
                
                logging.info(f"  - Adding {strike_type} {instrument} option: {action} {quantity} contracts")
                
                # Determine the strike specification to use
                strike_spec = None
                if strike:
                    strike_spec = strike
                elif strike_pct:
                    # Format as percentage OTM/ITM based on whether strike is above or below current price
                    if strike_pct > 0:
                        strike_spec = f"{strike_pct}%OTM"
                    elif strike_pct < 0:
                        strike_spec = f"{abs(strike_pct)}%ITM"
                    else:
                        strike_spec = "ATM"
                elif delta:
                    strike_spec = f"delta:{delta}"
                else:
                    strike_spec = "ATM"  # Default to ATM if no strike specified

                # Determine the expiration specification to use
                expiration_spec = None
                if expiration:
                    expiration_spec = expiration
                elif days_to_expiration:
                    expiration_spec = days_to_expiration
                else:
                    # Default to horizon days if available, otherwise 30 days
                    expiration_spec = horizon if horizon else 30
                
                strategy.add_option(
                    option_type=instrument,
                    action=action,
                    quantity=quantity,
                    strike_spec=strike_spec,
                    expiration_spec=expiration_spec
                )
            # Unknown instrument type
            else:
                logging.warning(f"Unknown instrument type: {instrument}")
        
        logging.info(f"\n=== EVALUATING STRATEGY ===")
        logging.info(f"Simulating {ticker} options strategy: {strategy.name}")
        
        # Create options simulation instance
        options_sim = OptionsSimulation(
            ticker=ticker,
            starting_price=starting_price,
            volatility=volatility,
            drift=drift,
            risk_free_rate=risk_free_rate,
            stock_model=model_type,
            random_seed=self.config['model_params'].get('random_seed')
        )
        
        # Run strategy simulation
        logging.info("\n=== RUNNING STRATEGY SIMULATION ===")
        stock_paths, component_prices, strategy_values = options_sim.simulate_strategy(
            strategy=strategy, 
            num_paths=paths,
            horizon=horizon
        )
        
        # Calculate statistics
        logging.info("\n=== STRATEGY RESULTS SUMMARY ===")
        
        # Get initial investment
        strategy_details = strategy.evaluate()
        initial_investment = abs(strategy_details['total_cost'])
        initial_value = strategy_details['value'] if 'value' in strategy_details else strategy_details['total_cost']
        
        logging.info(f"Strategy initial details:")
        logging.info(f"- Initial cost: ${initial_investment:.2f}")
        logging.info(f"- Initial value: ${initial_value:.2f}")
        
        # Calculate final values
        final_values = strategy_values[:, -1]
        mean_final = final_values.mean()
        median_final = float(sorted(final_values)[len(final_values) // 2])
        min_final = final_values.min()
        max_final = final_values.max()
        
        # Log summary statistics
        logging.info(f"Final strategy value statistics:")
        logging.info(f"- Mean: ${mean_final:.2f}")
        logging.info(f"- Median: ${median_final:.2f}")
        logging.info(f"- Min: ${min_final:.2f}")
        logging.info(f"- Max: ${max_final:.2f}")
        logging.info(f"- Range: ${max_final - min_final:.2f}")
        
        # Calculate returns
        if initial_investment > 0:
            # Check if initial_value is different from initial_investment
            if initial_value != initial_investment and 'value' in strategy_details:
                mean_return = (mean_final - initial_value) / initial_investment
                median_return = (median_final - initial_value) / initial_investment
                min_return = (min_final - initial_value) / initial_investment
                max_return = (max_final - initial_value) / initial_investment
            else:
                # If there's no separate 'value' key, or it equals the investment, use simpler calculation
                mean_return = (mean_final / initial_investment) - 1
                median_return = (median_final / initial_investment) - 1
                min_return = (min_final / initial_investment) - 1
                max_return = (max_final / initial_investment) - 1
            
            logging.info(f"Return statistics:")
            logging.info(f"- Mean return: {mean_return:.2%}")
            logging.info(f"- Median return: {median_return:.2%}")
            logging.info(f"- Min return: {min_return:.2%}")
            logging.info(f"- Max return: {max_return:.2%}")
        
        # Store detailed component data for results dictionary, without additional logging
        option_details = {}
        for i, component in enumerate(strategy.components):
            if component.instrument_type != InstrumentType.STOCK:
                if component.option_spec.resolved:
                    option_type = component.option_spec.option_type
                    strike = component.option_spec.resolved_strike
                    days_to_expiry = component.option_spec.resolved_days_to_expiration
                    initial_price = component.option_spec.resolved_price
                    
                    option_details[f"component_{i}"] = {
                        'type': option_type,
                        'strike': strike,
                        'days_to_expiry': days_to_expiry,
                        'initial_price': initial_price
                    }
        
        # Create visualization if requested
        if self.config.get('visualization', {}).get('strategy_paths', True):
            logging.info("\n=== GENERATING STRATEGY VISUALIZATION ===")
            visualizer = OptionsStrategyVisualizer(
                results={
                    'ticker': ticker,
                    'strategy': strategy_name,
                    'stock_paths': stock_paths,
                    'strategy_values': strategy_values,
                    'time_points': options_sim.time_points,
                    'starting_price': starting_price,
                    'initial_investment': initial_investment,
                    'strategy': {
                        'total_cost': initial_investment
                    }
                }
            )
            
            # Determine output directory and file
            output_dir = self.config.get('visualization', {}).get('output_dir', 'outputs')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            strategy_name_slug = strategy_name.lower().replace(' ', '_')
            
            # Create paths plot
            paths_output_file = os.path.join(output_dir, f"{ticker}_{strategy_name_slug}_paths.png")
            logging.info(f"Creating options strategy paths plot at: {paths_output_file}")
            visualizer.visualize_paths(
                title=f"{ticker} {strategy_name} Strategy Simulation",
                save_path=paths_output_file,
                show_plot=False
            )
            logging.info(f"Saved paths plot to: {paths_output_file}")
            
            # Create distribution plot
            distribution_output_file = os.path.join(output_dir, f"{ticker}_{strategy_name_slug}_distribution.png")
            logging.info(f"Creating options strategy distribution plot at: {distribution_output_file}")
            visualizer.visualize_distribution(
                title=f"{ticker} {strategy_name} Distribution of Final Values",
                save_path=distribution_output_file,
                show_plot=False
            )
            logging.info(f"Saved distribution plot to: {distribution_output_file}")
            
            # Create payoff curve
            payoff_data = strategy.calculate_payoff_curve()
            payoff_output_file = os.path.join(output_dir, f"{ticker}_{strategy_name_slug}_payoff.png")
            logging.info(f"Creating options strategy payoff curve at: {payoff_output_file}")
            visualizer.visualize_payoff_curve(
                payoff_data=payoff_data,
                title=f"{ticker} {strategy_name} Payoff at Expiration",
                save_path=payoff_output_file,
                show_plot=False
            )
            logging.info(f"Saved payoff curve to: {payoff_output_file}")
            
            # Create strategy vs stock plot
            stock_values = stock_paths * 100  # Assuming 100 shares for comparison
            stock_cost = starting_price * 100
            vs_stock_output_file = os.path.join(output_dir, f"{ticker}_{strategy_name_slug}_vs_stock.png")
            logging.info(f"Creating options strategy vs stock plot at: {vs_stock_output_file}")
            visualizer.visualize_strategy_vs_stock(
                strategy_values=strategy_values,
                initial_strategy_cost=initial_investment,
                stock_values=stock_values,
                initial_stock_cost=stock_cost,
                title=f"{ticker} {strategy_name} vs Stock Returns",
                save_path=vs_stock_output_file,
                show_plot=False
            )
            logging.info(f"Saved strategy vs stock plot to: {vs_stock_output_file}")
        
        # Store results
        self.results = {
            'ticker': ticker,
            'strategy': strategy_name,
            'stock_paths': stock_paths,
            'strategy_values': strategy_values,
            'component_prices': component_prices,
            'paths': paths,
            'horizon': options_sim.time_points[-1],
            'starting_price': starting_price,
            'initial_investment': initial_investment,
            'option_details': option_details,
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