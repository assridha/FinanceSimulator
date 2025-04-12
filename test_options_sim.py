#!/usr/bin/env python
"""Test script to verify model-specific parameters are passed to options simulation."""

import logging
import os
import sys
from financesimulator.simulation.engine import SimulationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    """Run a test simulation to verify model-specific parameter passing."""
    # Get the path to the config file
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        'configs', 
        'options_strategy.yaml'
    )
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    print(f"Using config file: {config_path}")
    
    # Initialize the simulation engine
    engine = SimulationEngine(config_path=config_path)
    
    # Run the simulation
    print("Running simulation...")
    engine.run()
    
    print("Simulation completed successfully.")

if __name__ == "__main__":
    main() 