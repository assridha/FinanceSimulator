"""Example of using Finance Simulator for stock price simulation."""

import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to Python path to allow importing financesimulator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financesimulator.simulation.engine import SimulationEngine
from financesimulator.visualization.plots import SimulationVisualizer
from financesimulator.data.fetcher import MarketDataFetcher


def run_stock_simulation():
    """Run a stock simulation and visualize the results."""
    
    # Create simulation configuration
    config = {
        'simulation': {
            'type': 'stock',
            'ticker': 'AAPL',
            'model': 'gbm',
            'paths': 1000,
            'horizon': 252,  # One year of trading days
        },
        'model_params': {
            'random_seed': 42
        },
        'data': {
            'lookback_period': '1y',
            'frequency': 'daily'
        },
        'visualization': {
            'plot_paths': True,
            'confidence_intervals': [0.1, 0.5, 0.9]
        }
    }
    
    # Create data fetcher to get market data
    data_fetcher = MarketDataFetcher()
    
    # Get current price
    current_price = data_fetcher.get_current_price('AAPL')
    config['simulation']['starting_price'] = current_price
    
    # Calculate historical volatility and mean return
    volatility = data_fetcher.calculate_historical_volatility('AAPL', lookback_period='1y')
    mean_return = data_fetcher.calculate_mean_return('AAPL', lookback_period='1y')
    
    # Set model parameters
    config['model_params']['volatility'] = volatility
    config['model_params']['drift'] = mean_return
    
    print(f"Running simulation for AAPL:")
    print(f"- Starting price: ${current_price:.2f}")
    print(f"- Historical volatility: {volatility:.2%}")
    print(f"- Historical mean return: {mean_return:.2%}")
    print(f"- Simulating {config['simulation']['paths']} paths over {config['simulation']['horizon']} trading days")
    
    # Create and run simulation
    engine = SimulationEngine(config_dict=config)
    results = engine.run()
    
    print("\nSimulation complete.")
    
    # Create visualizer
    visualizer = SimulationVisualizer(results)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate and save summary dashboard
    print("Generating visualizations...")
    dashboard_path = os.path.join(output_dir, 'aapl_simulation_summary.png')
    visualizer.generate_summary_dashboard(save_path=dashboard_path)
    print(f"Summary dashboard saved to: {dashboard_path}")
    
    # Calculate statistics of final prices
    final_prices = results['price_paths'][:, -1]
    mean_final = final_prices.mean()
    median_final = float(sorted(final_prices)[len(final_prices) // 2])
    min_final = final_prices.min()
    max_final = final_prices.max()
    
    print("\nSimulation Statistics:")
    print(f"- Mean final price: ${mean_final:.2f}")
    print(f"- Median final price: ${median_final:.2f}")
    print(f"- Min final price: ${min_final:.2f}")
    print(f"- Max final price: ${max_final:.2f}")
    print(f"- Range: ${max_final - min_final:.2f}")
    
    return results, visualizer


if __name__ == "__main__":
    results, visualizer = run_stock_simulation()
    
    # Show additional visualizations
    visualizer.plot_histogram(title="Distribution of Final Stock Prices")
    visualizer.plot_returns_distribution(title="Distribution of Annualized Returns")
    
    plt.show()  # Show all plots