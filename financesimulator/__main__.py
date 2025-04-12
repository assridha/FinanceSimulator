"""Main module for the Finance Simulator package."""

import argparse
import sys
import os
import logging
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
    
    return parser.parse_args()


def run_simulation(config_path: str) -> Dict[str, Any]:
    """
    Run a simulation using the given configuration file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Simulation results
    """
    logging.info(f"Running simulation with config: {config_path}")
    
    # Create simulation engine
    engine = SimulationEngine(config_path=config_path)
    
    # Run simulation
    results = engine.run()
    
    logging.info("Simulation completed successfully")
    
    return results


def create_visualizations(results: Dict[str, Any], output_dir: Optional[str] = None) -> None:
    """
    Create visualizations for simulation results.
    
    Args:
        results: Simulation results
        output_dir: Output directory for plots
    """
    logging.info("Creating visualizations")
    
    # Create visualizer
    visualizer = SimulationVisualizer(results)
    
    # Determine output directory
    if output_dir is None:
        output_dir = 'outputs'
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get ticker and simulation type for filenames
    ticker = results.get('ticker', 'asset')
    simulation_type = results.get('simulation_type', 'simulation')
    
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
    
    try:
        # Run simulation
        results = run_simulation(args.config)
        
        # Create visualizations if enabled
        if not args.no_plot:
            create_visualizations(results, args.output)
        
        logging.info("Finance Simulator completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())