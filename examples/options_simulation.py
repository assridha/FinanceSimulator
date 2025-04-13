"""Example of using Finance Simulator for options simulation with strategy returns visualization."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# Add parent directory to Python path to allow importing financesimulator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financesimulator.data.fetcher import MarketDataFetcher
from financesimulator.options.strategy import OptionsStrategy, Action
from financesimulator.simulation.options_sim import OptionsSimulation


def run_options_strategy_simulation():
    """
    Run an options strategy simulation with visualizations.
    
    This example demonstrates how to create and simulate an options strategy
    (in this case, a bull call spread) with visualization of strategy returns.
    """
    # Set up parameters
    ticker = 'AAPL'
    expiry_days = 45
    num_paths = 1000
    
    # Create a bull call spread strategy
    strategy = OptionsStrategy(ticker=ticker, name=f"{ticker} Bull Call Spread")
    
    # Add components: buy a lower strike call and sell a higher strike call
    strategy.add_option(
        option_type='call',
        action=Action.BUY,
        quantity=1,
        strike_spec='5%OTM',  # 5% out of the money
        expiration_spec=expiry_days
    )
    
    strategy.add_option(
        option_type='call',
        action=Action.SELL,
        quantity=1,
        strike_spec='10%OTM',  # 10% out of the money
        expiration_spec=expiry_days
    )
    
    # Evaluate the strategy
    evaluation = strategy.evaluate()
    
    print("==== Strategy Evaluation ====")
    print(f"Ticker: {evaluation['ticker']}")
    print(f"Current Price: ${evaluation['current_price']:.2f}")
    print(f"Strategy: {evaluation['strategy']}")
    print("\nComponents:")
    for i, component in enumerate(evaluation['components']):
        print(f"  Component {i+1}:")
        for key, value in component.items():
            if key == 'greeks':
                print(f"    Greeks:")
                for greek, g_value in value.items():
                    print(f"      {greek}: {g_value:.4f}")
            elif key == 'expiration':
                print(f"    {key}: {value.strftime('%Y-%m-%d')}")
            else:
                print(f"    {key}: {value}")
    
    print(f"\nTotal Cost: ${evaluation['total_cost']:.2f}")
    
    # Calculate payoff curve for visualization
    payoff_data = strategy.calculate_payoff_curve()
    
    print("\n==== Strategy Characteristics ====")
    print(f"Max Profit: ${payoff_data['max_profit']:.2f}")
    print(f"Max Loss: ${payoff_data['max_loss']:.2f}")
    print(f"Breakeven Points: {', '.join([f'${p:.2f}' for p in payoff_data['breakeven_points']])}")
    
    # Create options simulation
    options_sim = OptionsSimulation(
        ticker=ticker,
        random_seed=42
    )
    
    # Run simulation
    print("\n==== Running Simulation ====")
    stock_paths, component_prices, strategy_values = options_sim.simulate_strategy(
        strategy=strategy,
        num_paths=num_paths,
        horizon=expiry_days
    )
    
    # Calculate statistics
    initial_investment = abs(evaluation['total_cost'])
    statistics = options_sim.calculate_statistics(
        values=strategy_values,
        initial_investment=initial_investment
    )
    
    print("\n==== Simulation Results ====")
    print(f"Mean Final Value: ${statistics['mean_value']:.2f}")
    print(f"Mean Return: {statistics['mean_return']:.2%}")
    print(f"Probability of Profit: {statistics['win_rate']:.2%}")
    print(f"Value at Risk (10%): ${statistics['percentile_10']:.2f}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Visualize results
    print("\n==== Creating Visualizations ====")
    
    # Plot strategy paths
    options_sim.visualize_paths(
        values=strategy_values,
        initial_investment=initial_investment,
        title=f"{ticker} Bull Call Spread - Strategy Paths",
        save_path=os.path.join(output_dir, f"{ticker}_bull_call_spread_paths.png")
    )
    
    # Plot distribution of final values
    options_sim.visualize_distribution(
        values=strategy_values,
        initial_investment=initial_investment,
        title=f"{ticker} Bull Call Spread - Distribution of Final Values",
        save_path=os.path.join(output_dir, f"{ticker}_bull_call_spread_distribution.png")
    )
    
    # Plot payoff curve
    options_sim.visualize_payoff_curve(
        strategy=strategy,
        save_path=os.path.join(output_dir, f"{ticker}_bull_call_spread_payoff.png")
    )
    
    # Plot strategy vs stock returns
    stock_values = stock_paths * 100  # Assuming 100 shares equivalent for comparison
    stock_cost = evaluation['current_price'] * 100
    
    options_sim.visualize_strategy_vs_stock(
        strategy_values=strategy_values,
        initial_strategy_cost=initial_investment,
        stock_values=stock_values,
        initial_stock_cost=stock_cost,
        title=f"{ticker} Bull Call Spread vs Stock Returns",
        save_path=os.path.join(output_dir, f"{ticker}_strategy_vs_stock.png")
    )
    
    print("\nSimulation complete! Visualizations saved to outputs directory.")
    
    return {
        'strategy': strategy,
        'evaluation': evaluation,
        'stock_paths': stock_paths,
        'strategy_values': strategy_values,
        'statistics': statistics
    }


if __name__ == "__main__":
    results = run_options_strategy_simulation()
    plt.show()  # Show all plots 