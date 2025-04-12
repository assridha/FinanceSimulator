# Finance Simulator

A Python-based Monte Carlo simulator for stocks, options, and Bitcoin with modular design and multiple prediction models.

## Features

### Current Features

#### Stock and Bitcoin Simulator
- Modular architecture for easy extension
- Multiple prediction models:
  - Geometric Brownian Motion (GBM)
  - ARIMA
  - LSTM
- Configuration via YAML files
- Market data retrieval from yfinance
- Automatic calculation of statistics (volatility, mean return, etc.)
- Visualization and analysis tools

#### Options Simulator
- Integration with stock price simulations
- Black-Scholes model implementation
- Custom strategy simulation (e.g., BUY 100 Stock + SELL 10% OTM Call)
- Flexible option description (% OTM, ATM, Delta-based, specific strike)
- Options chain retrieval from yfinance
- Greeks calculation and analysis

### Planned Features
- Portfolio optimization
- Sensitivity analysis
- Backtesting capabilities
- Additional asset classes (Forex, Futures)
- Monte Carlo VaR calculations
- Additional option pricing models (Binomial, Heston)
- Web interface for simulation setup and visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/assridha/FinanceSimulator.git
cd FinanceSimulator

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run a stock simulation using default configuration
python -m financesimulator --config configs/stock_sim_default.yaml

# Run an options strategy simulation
python -m financesimulator --config configs/options_strategy.yaml

# Run a Bitcoin simulation with LSTM model
python -m financesimulator --config configs/bitcoin_lstm.yaml
```

## Project Structure

```
FinanceSimulator/
├── financesimulator/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py       # Market data retrieval
│   │   └── processor.py     # Data preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py          # Base model interface
│   │   ├── gbm.py           # Geometric Brownian Motion
│   │   ├── arima.py         # ARIMA model
│   │   └── lstm.py          # LSTM model
│   ├── options/
│   │   ├── __init__.py
│   │   ├── black_scholes.py # Black-Scholes implementation
│   │   ├── greeks.py        # Greeks calculation
│   │   └── strategy.py      # Options strategy builder
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── engine.py        # Core simulation engine
│   │   ├── stock_sim.py     # Stock simulation
│   │   ├── options_sim.py   # Options simulation
│   │   └── bitcoin_sim.py   # Bitcoin simulation
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py         # Standard visualization
│       └── analysis.py      # Analysis tools
├── configs/                 # Configuration files
│   ├── stock_sim_default.yaml
│   ├── options_strategy.yaml
│   └── bitcoin_lstm.yaml
├── examples/                # Example scripts
│   ├── stock_simulation.py
│   ├── options_strategy.py
│   └── bitcoin_forecast.py
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_options.py
│   └── test_simulation.py
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

## Configuration

Simulations are configured using YAML files. Example configuration:

```yaml
# Stock simulation with GBM model
simulation:
  type: stock
  ticker: AAPL
  model: gbm
  paths: 1000
  horizon: 252  # Trading days (1 year)
  starting_price: auto  # Fetch from market data

model_params:
  volatility: auto  # Calculate from historical data
  drift: auto       # Calculate from historical data
  random_seed: 42

data:
  lookback_period: 252  # Days of historical data to use
  frequency: daily

visualization:
  plot_paths: true
  confidence_intervals: [0.1, 0.5, 0.9]
  save_to: outputs/aapl_sim.png
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.