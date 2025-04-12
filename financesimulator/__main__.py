"""Main module for the Finance Simulator package."""

import argparse
import sys
import os
import logging
import time
from typing import Dict, Any, Optional, List

from financesimulator.simulation.engine import SimulationEngine
from financesimulator.visualization.plots import SimulationVisualizer


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('financesimulator.log')
        ]
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Finance Simulator')
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--output', '-o', help='Output directory for plots')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--offline', action='store_true', help='Run in offline mode (use cached data only)')
    parser.add_argument('--no-prefetch', action='store_true', help='Disable automatic prefetching of common tickers')
    
    return parser.parse_args()


def run_simulation(config_path: str, offline_mode: bool = False, prefetch: bool = True) -> Dict[str, Any]:
    """
    Run a simulation using the given configuration file.
    
    Args:
        config_path: Path to configuration YAML file
        offline_mode: Whether to run in offline mode (no API calls)
        prefetch: Whether to prefetch common data
        
    Returns:
        Simulation results
    """
    # Load config to check cache settings (minimal load just for display)
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Get cache settings for logging
        cache_settings = config.get('data', {}).get('cache', {})
        cache_enabled = cache_settings.get('enabled', True)
        force_refresh = cache_settings.get('force_refresh', False)
        
        cache_status = "enabled"
        if not cache_enabled:
            cache_status = "disabled"
        elif force_refresh:
            cache_status = "bypassed (force refresh)"
            
        logging.info(f"Running simulation with config: {config_path}" + 
                    (", offline mode enabled" if offline_mode else "") +
                    f", data cache {cache_status}")
    except Exception:
        # Fall back to basic logging if config can't be read
        logging.info(f"Running simulation with config: {config_path}" + 
                    (", offline mode enabled" if offline_mode else ""))
    
    # Create simulation engine
    start_time = time.time()
    engine = SimulationEngine(config_path=config_path, offline_mode=offline_mode, prefetch=prefetch)
    
    # Run simulation
    results = engine.run()
    
    elapsed_time = time.time() - start_time
    logging.info(f"Simulation completed successfully in {elapsed_time:.2f} seconds")
    
    return results


def create_visualizations(results: Dict[str, Any], output_dir: Optional[str] = None) -> None:
    """
    Create visualizations for simulation results.
    
    Args:
        results: Simulation results
        output_dir: Output directory for plots
    """
    logging.info("Creating visualizations")
    
    # Get ticker and simulation type for filenames
    ticker = results.get('ticker', 'asset')
    simulation_type = results.get('simulation_type', 'simulation')
    
    # Skip standard visualizations for options strategy simulations as they'll have their own visualizations
    if simulation_type == 'options_strategy':
        logging.info("Options strategy simulation detected - standard visualizations skipped")
        return
    
    # Create visualizer
    visualizer = SimulationVisualizer(results)
    
    # Determine output directory
    if output_dir is None:
        output_dir = 'outputs'
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate summary dashboard
    dashboard_path = os.path.join(output_dir, f"{ticker}_{simulation_type}_summary.png")
    visualizer.generate_summary_dashboard(
        save_path=dashboard_path,
        show_plot=False
    )
    logging.info(f"Saved summary dashboard to {dashboard_path}")
    
    # Generate individual plots
    paths_plot_path = os.path.join(output_dir, f"{ticker}_{simulation_type}_paths.png")
    visualizer.plot_price_paths(
        confidence_intervals=[0.1, 0.5, 0.9],
        save_path=paths_plot_path,
        show_plot=False
    )
    logging.info(f"Saved price paths plot to {paths_plot_path}")
    
    hist_plot_path = os.path.join(output_dir, f"{ticker}_{simulation_type}_histogram.png")
    visualizer.plot_histogram(
        save_path=hist_plot_path,
        show_plot=False
    )
    logging.info(f"Saved histogram plot to {hist_plot_path}")
    
    returns_plot_path = os.path.join(output_dir, f"{ticker}_{simulation_type}_returns.png")
    visualizer.plot_returns_distribution(
        save_path=returns_plot_path,
        show_plot=False
    )
    logging.info(f"Saved returns distribution plot to {returns_plot_path}")


def main():
    """Main entry point for the Finance Simulator package."""
    # Set up logging
    setup_logging()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    start_time = time.time()
    
    try:
        # Run simulation
        results = run_simulation(args.config, 
                                offline_mode=args.offline,
                                prefetch=not args.no_prefetch)
        
        # Create visualizations if enabled
        if not args.no_plot:
            create_visualizations(results, args.output)
        
        elapsed_time = time.time() - start_time
        logging.info(f"Finance Simulator completed successfully in {elapsed_time:.2f} seconds")
        return 0
        
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())