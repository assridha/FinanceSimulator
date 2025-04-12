"""Data fetcher module for retrieving market data."""

import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional, Union, Tuple, List

class MarketDataFetcher:
    """Class for fetching market data from various sources."""
    
    def __init__(self):
        """Initialize the market data fetcher."""
        self.cache = {}  # Simple cache to avoid repeated API calls
    
    def get_stock_data(
        self, 
        ticker: str, 
        period: str = "1y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock historical data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical price data
        """
        cache_key = f"{ticker}_{period}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        # Basic data validation
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        # Store in cache
        self.cache[cache_key] = data
        
        return data
    
    def get_current_price(self, ticker: str) -> float:
        """
        Get the current price of a stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current price
        """
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('regularMarketPrice')
        
        if current_price is None:
            # Fallback to last close if market price not available
            data = self.get_stock_data(ticker, period="1d")
            current_price = data['Close'].iloc[-1]
            
        return current_price
    
    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate (using 10-year Treasury yield as proxy).
        
        Returns:
            Current risk-free rate as a decimal (e.g., 0.03 for 3%)
        """
        # Use 10-year Treasury yield as a proxy for risk-free rate
        treasury_ticker = "^TNX"
        try:
            treasury = yf.Ticker(treasury_ticker)
            rate = treasury.info.get('regularMarketPrice')
            
            if rate is None:
                # Fallback to a default value if rate not available
                return 0.035  # 3.5% as fallback
                
            # Convert percentage to decimal
            return rate / 100.0
        except Exception:
            # Return a default value if there's an error
            return 0.035  # 3.5% as fallback
    
    def get_option_chain(
        self, 
        ticker: str, 
        expiration_date: Optional[Union[str, dt.date]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get options chain for a given stock.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Options expiration date. If None, get the nearest expiration.
            
        Returns:
            Dictionary containing 'calls' and 'puts' DataFrames
        """
        stock = yf.Ticker(ticker)
        
        # Get list of available expiration dates
        expiration_dates = stock.options
        
        if not expiration_dates:
            raise ValueError(f"No options data available for {ticker}")
            
        # If expiration date not specified, use the nearest one
        if expiration_date is None:
            expiration_date = expiration_dates[0]
        
        # Convert string date to datetime object if needed
        if isinstance(expiration_date, str):
            try:
                expiration_date = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
            except ValueError:
                # Try another format
                expiration_date = dt.datetime.strptime(expiration_date, '%m/%d/%Y').date()
                
        # Get options chain for given expiration
        options = stock.option_chain(expiration_date.strftime('%Y-%m-%d') 
                                    if isinstance(expiration_date, dt.date) 
                                    else expiration_date)
        
        return {
            'calls': options.calls,
            'puts': options.puts
        }
    
    def find_option_by_criteria(
        self,
        ticker: str,
        option_type: str,  # 'call' or 'put'
        criteria: str,     # 'strike', 'delta', 'otm_pct'
        value: float,
        expiration_date: Optional[Union[str, dt.date]] = None,
        tolerance: float = 0.05
    ) -> pd.Series:
        """
        Find an option that matches given criteria.
        
        Args:
            ticker: Stock ticker symbol
            option_type: 'call' or 'put'
            criteria: Criteria to search by ('strike', 'delta', 'otm_pct')
            value: Value to match for the given criteria
            expiration_date: Options expiration date
            tolerance: Tolerance for finding closest match
            
        Returns:
            Option data as pandas Series
        """
        # Get current stock price for OTM calculations
        current_price = self.get_current_price(ticker)
        
        # Get options chain
        options_chain = self.get_option_chain(ticker, expiration_date)
        
        # Select calls or puts based on option_type
        if option_type.lower() == 'call':
            options = options_chain['calls']
        elif option_type.lower() == 'put':
            options = options_chain['puts']
        else:
            raise ValueError("option_type must be either 'call' or 'put'")
        
        # Find option based on criteria
        if criteria == 'strike':
            # Find option with strike closest to the specified value
            options['diff'] = abs(options['strike'] - value)
            option = options.loc[options['diff'].idxmin()]
            
        elif criteria == 'delta':
            # Find option with delta closest to the specified value
            if 'delta' not in options.columns:
                # If greeks not in data, can't search by delta
                raise ValueError("Delta information not available in options chain")
                
            options['diff'] = abs(options['delta'] - value)
            option = options.loc[options['diff'].idxmin()]
            
        elif criteria == 'otm_pct':
            # For calls, OTM means strike > current price
            # For puts, OTM means strike < current price
            if option_type.lower() == 'call':
                target_strike = current_price * (1 + value)
                options['diff'] = abs(options['strike'] - target_strike)
                option = options.loc[options['diff'].idxmin()]
            else:
                target_strike = current_price * (1 - value)
                options['diff'] = abs(options['strike'] - target_strike)
                option = options.loc[options['diff'].idxmin()]
                
        elif criteria == 'atm':
            # Find at-the-money option (strike closest to current price)
            options['diff'] = abs(options['strike'] - current_price)
            option = options.loc[options['diff'].idxmin()]
            
        else:
            raise ValueError(f"Unknown criteria: {criteria}")
        
        return option
    
    def calculate_historical_volatility(
        self, 
        ticker: str, 
        lookback_period: str = "1y", 
        interval: str = "1d"
    ) -> float:
        """
        Calculate historical volatility for a stock.
        
        Args:
            ticker: Stock ticker symbol
            lookback_period: Period to use for calculation
            interval: Data interval
            
        Returns:
            Annualized volatility
        """
        # Get historical data
        data = self.get_stock_data(ticker, period=lookback_period, interval=interval)
        
        # Calculate daily returns
        data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Remove NaN values
        returns = data['Returns'].dropna()
        
        # Calculate volatility
        daily_volatility = returns.std()
        
        # Annualize volatility based on the interval
        if interval == '1d':
            # Assuming 252 trading days in a year
            annualized_volatility = daily_volatility * np.sqrt(252)
        elif interval == '1wk':
            # Assuming 52 trading weeks in a year
            annualized_volatility = daily_volatility * np.sqrt(52)
        elif interval == '1mo':
            # Assuming 12 trading months in a year
            annualized_volatility = daily_volatility * np.sqrt(12)
        else:
            # Default to daily
            annualized_volatility = daily_volatility * np.sqrt(252)
            
        return annualized_volatility
    
    def calculate_mean_return(
        self, 
        ticker: str, 
        lookback_period: str = "1y", 
        interval: str = "1d"
    ) -> float:
        """
        Calculate mean return for a stock.
        
        Args:
            ticker: Stock ticker symbol
            lookback_period: Period to use for calculation
            interval: Data interval
            
        Returns:
            Annualized mean return
        """
        # Get historical data
        data = self.get_stock_data(ticker, period=lookback_period, interval=interval)
        
        # Calculate daily returns
        data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Remove NaN values
        returns = data['Returns'].dropna()
        
        # Calculate mean daily return
        mean_daily_return = returns.mean()
        
        # Annualize mean return based on the interval
        if interval == '1d':
            # Assuming 252 trading days in a year
            annualized_return = mean_daily_return * 252
        elif interval == '1wk':
            # Assuming 52 trading weeks in a year
            annualized_return = mean_daily_return * 52
        elif interval == '1mo':
            # Assuming 12 trading months in a year
            annualized_return = mean_daily_return * 12
        else:
            # Default to daily
            annualized_return = mean_daily_return * 252
            
        return annualized_return