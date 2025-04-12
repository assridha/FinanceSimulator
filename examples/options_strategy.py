"""Example of using Finance Simulator for options strategy simulation."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# Add parent directory to Python path to allow importing financesimulator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financesimulator.data.fetcher import MarketDataFetcher
from financesimulator.models.gbm import GeometricBrownianMotion
from financesimulator.options.black_scholes import BlackScholes


def simulate_covered_call_strategy():
    """
    Simulate a covered call options strategy.
    
    This example demonstrates how to simulate a covered call strategy (long stock + short call)
    using the GBM model for stock price simulation and the Black-Scholes model for option pricing.
    """
    # Initialize data fetcher
    fetcher = MarketDataFetcher()
    
    # Set up parameters
    ticker = 'AAPL'
    option_expiry_days = 30
    num_contracts = 1
    shares_per_contract = 100
    otm_percentage = 0.05  # 5% OTM
    
    # Get current stock price
    current_price = fetcher.get_current_price(ticker)
    
    # Calculate option strike price (5% OTM)
    strike_price = current_price * (1 + otm_percentage)
    
    # Get risk-free rate
    risk_free_rate = fetcher.get_risk_free_rate()
    
    # Calculate volatility from historical data
    volatility = fetcher.calculate_historical_volatility(ticker, lookback_period='1y')
    
    # Set up simulation parameters
    num_paths = 1000
    days_to_expiry = option_expiry_days
    trading_days_per_year = 252
    
    # Initialize GBM model
    gbm_params = {
        'drift': fetcher.calculate_mean_return(ticker, lookback_period='1y'),
        'volatility': volatility,
        'random_seed': 42
    }
    stock_model = GeometricBrownianMotion(gbm_params)
    
    # Simulate stock price paths
    stock_paths = stock_model.simulate(current_price, days_to_expiry, num_paths)
    
    # Calculate initial option premium
    option_premium = BlackScholes.call_price(
        S=current_price,
        K=strike_price,
        r=risk_free_rate,
        sigma=volatility,
        T=days_to_expiry / trading_days_per_year
    )
    
    # Calculate option prices at each time step for each path
    option_prices = np.zeros_like(stock_paths)
    for t in range(days_to_expiry + 1):
        time_to_expiry = (days_to_expiry - t) / trading_days_per_year
        if t == days_to_expiry:  # At expiration
            # Option value at expiration is max(0, stock_price - strike)
            option_prices[:, t] = np.maximum(0, stock_paths[:, t] - strike_price)
        else:
            for p in range(num_paths):
                option_prices[p, t] = BlackScholes.call_price(
                    S=stock_paths[p, t],
                    K=strike_price,
                    r=risk_free_rate,
                    sigma=volatility,
                    T=time_to_expiry
                )
    
    # Calculate strategy value at each time step
    # Strategy: Long 100 shares of stock + Short 1 call option
    strategy_values = (stock_paths * shares_per_contract) - (option_prices * num_contracts * shares_per_contract)
    
    # Add initial option premium received
    strategy_values += option_premium * num_contracts * shares_per_contract
    
    # Print strategy details
    print(f"=== Covered Call Strategy Simulation ===")
    print(f"Stock: {ticker}")
    print(f"Current stock price: ${current_price:.2f}")
    print(f"Call option strike: ${strike_price:.2f} ({otm_percentage:.1%} OTM)")
    print(f"Days to expiration: {days_to_expiry}")
    print(f"Option premium received: ${option_premium:.2f} per share (${option_premium * shares_per_contract:.2f} per contract)")
    print(f"Volatility: {volatility:.2%}")
    print(f"Risk-free rate: {risk_free_rate:.2%}")
    
    # Calculate strategy statistics
    initial_investment = current_price * shares_per_contract
    initial_strategy_value = initial_investment + (option_premium * shares_per_contract)
    final_strategy_values = strategy_values[:, -1]
    
    mean_final_value = np.mean(final_strategy_values)
    median_final_value = np.median(final_strategy_values)
    min_final_value = np.min(final_strategy_values)
    max_final_value = np.max(final_strategy_values)
    
    mean_return = (mean_final_value / initial_investment) - 1
    median_return = (median_final_value / initial_investment) - 1
    min_return = (min_final_value / initial_investment) - 1
    max_return = (max_final_value / initial_investment) - 1
    
    # Print strategy statistics
    print("\n=== Strategy Statistics ===")
    print(f"Initial investment: ${initial_investment:.2f}")
    print(f"Mean final value: ${mean_final_value:.2f} (Return: {mean_return:.2%})")
    print(f"Median final value: ${median_final_value:.2f} (Return: {median_return:.2%})")
    print(f"Min final value: ${min_final_value:.2f} (Return: {min_return:.2%})")
    print(f"Max final value: ${max_final_value:.2f} (Return: {max_return:.2%})")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot a subset of strategy value paths
    time_steps = np.arange(days_to_expiry + 1)
    num_paths_to_plot = min(100, num_paths)
    for i in range(num_paths_to_plot):
        plt.plot(time_steps, strategy_values[i, :], color='skyblue', alpha=0.2, linewidth=0.8)
    
    # Plot statistics
    percentiles = np.percentile(strategy_values, [10, 50, 90], axis=0)
    plt.plot(time_steps, percentiles[0], color='blue', linestyle='--', linewidth=2, label='10th percentile')
    plt.plot(time_steps, percentiles[1], color='navy', linewidth=2.5, label='Median')
    plt.plot(time_steps, percentiles[2], color='darkblue', linestyle='--', linewidth=2, label='90th percentile')
    
    # Plot initial investment line
    plt.axhline(y=initial_investment, color='red', linestyle=':', linewidth=1.5, label='Initial Investment')
    
    plt.xlabel('Days to Expiration')
    plt.ylabel('Strategy Value ($)')
    plt.title(f'Covered Call Strategy: {ticker} with ${strike_price:.2f} Strike Call')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'{ticker}_covered_call_simulation.png'), dpi=300, bbox_inches='tight')
    
    # Show distribution of final strategy values
    plt.figure(figsize=(12, 8))
    plt.hist(final_strategy_values, bins=50, alpha=0.7, color='skyblue')
    plt.axvline(x=initial_investment, color='red', linestyle=':', linewidth=1.5, label='Initial Investment')
    plt.axvline(x=mean_final_value, color='navy', linewidth=2, label=f'Mean: ${mean_final_value:.2f}')
    
    plt.xlabel('Strategy Value at Expiration ($)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Final Strategy Values: {ticker} Covered Call')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'{ticker}_covered_call_distribution.png'), dpi=300, bbox_inches='tight')
    
    # Calculate max profit (premium received) and max loss scenarios
    max_profit = option_premium * shares_per_contract
    breakeven_price = current_price - option_premium
    
    print("\n=== Strategy Characteristics ===")
    print(f"Max profit: ${max_profit:.2f} (achieved if stock price at expiry is >= ${strike_price:.2f})")
    print(f"Breakeven price: ${breakeven_price:.2f}")
    print(f"Probability of profit: {np.mean(final_strategy_values > initial_investment):.2%}")
    
    return {
        'stock_paths': stock_paths,
        'option_prices': option_prices,
        'strategy_values': strategy_values,
        'current_price': current_price,
        'strike_price': strike_price,
        'option_premium': option_premium,
        'initial_investment': initial_investment,
        'days_to_expiry': days_to_expiry
    }


if __name__ == "__main__":
    results = simulate_covered_call_strategy()
    plt.show()  # Show all plots