"""Data fetcher module for retrieving market data."""

import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import os
import pickle
import time
from typing import Dict, Optional, Union, Tuple, List, Any
import logging

# Global risk-free rate cache for quick access across instances
_GLOBAL_RISK_FREE_RATE = {
    'rate': None,
    'timestamp': 0,
    'max_age': 3600  # 1 hour default
}

class MarketDataFetcher:
    """Class for fetching market data from various sources."""
    
    def __init__(self, cache_dir: str = ".cache", offline_mode: bool = False, prefetch: bool = False,
                 cache_enabled: bool = True, cache_max_age: int = 86400, force_refresh: bool = False):
        """
        Initialize the market data fetcher.
        
        Args:
            cache_dir: Directory to store cached data
            offline_mode: If True, only use cached data and don't make API calls
            prefetch: If True, prefetch data for common tickers in background
            cache_enabled: Whether to use cached data (if False, always fetch fresh data)
            cache_max_age: Maximum age of cached data in seconds before refreshing
            force_refresh: If True, force refresh of data even if cached
        """
        self.cache = {}  # Memory cache for current session
        self.calculation_cache = {}  # Cache for calculated metrics
        self.cache_dir = cache_dir
        self.offline_mode = offline_mode
        self.cache_enabled = cache_enabled
        self.cache_max_age = cache_max_age
        self.force_refresh = force_refresh
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # If prefetch is enabled, start background prefetch
        if prefetch and not offline_mode and cache_enabled:
            self._prefetch_common_data()
    
    def _prefetch_common_data(self):
        """
        Prefetch data for common tickers to speed up first-time runs.
        This runs in the background to avoid blocking.
        """
        import threading
        
        def prefetch_worker():
            # List of common tickers to prefetch
            common_tickers = ['SPY', 'AAPL', 'MSFT', 'GOOG', 'NVDA', 'TSLA','IBIT','MSTR']
            period = '1y'
            interval = '1d'
            
            print("Starting background prefetch for common tickers...")
            for ticker in common_tickers:
                # Check if already cached
                cache_key = f"{ticker}_{period}_{interval}"
                if self._load_from_file_cache(cache_key) is None:
                    try:
                        print(f"Prefetching data for {ticker}...")
                        self.get_stock_data(ticker, period, interval)
                    except Exception as e:
                        print(f"Error prefetching {ticker}: {e}")
            
            print("Background prefetch complete.")
        
        # Start prefetch in background thread
        thread = threading.Thread(target=prefetch_worker)
        thread.daemon = True  # Allow program to exit even if thread is running
        thread.start()
    
    def _get_cache_path(self, key: str) -> str:
        """Get path for cache file."""
        # Replace any characters that might be invalid for filenames
        sanitized_key = key.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, f"{sanitized_key}.pkl")
    
    def _save_to_file_cache(self, key: str, data: Any) -> None:
        """Save data to file cache."""
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_from_file_cache(self, key: str) -> Optional[Any]:
        """Load data from file cache if it exists."""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def get_stock_data(
        self, 
        ticker: str, 
        period: str = "1y", 
        interval: str = "1d",
        max_cache_age: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch stock historical data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            max_cache_age: Maximum age of cached data in seconds before refreshing
            
        Returns:
            DataFrame with historical price data
        """
        # Use instance max_cache_age if not specified
        if max_cache_age is None:
            max_cache_age = self.cache_max_age
            
        cache_key = f"{ticker}_{period}_{interval}"
        
        # Check if caching is disabled or force refresh is enabled
        if not self.cache_enabled or self.force_refresh:
            # Skip cache unless in offline mode
            if not self.offline_mode:
                print(f"Fetching fresh data for {ticker} from API (caching {'disabled' if not self.cache_enabled else 'bypassed'})...")
                stock = yf.Ticker(ticker)
                data = stock.history(period=period, interval=interval)
                
                # Store in memory cache even when disk caching is disabled
                self.cache[cache_key] = data
                
                # If caching is still enabled, update the file cache even with force_refresh
                if self.cache_enabled:
                    self._save_to_file_cache(cache_key, (data, time.time()))
                
                return data
        
        # Check memory cache first
        if cache_key in self.cache:
            print(f"Using cached data for {ticker} (memory cache)")
            return self.cache[cache_key]
        
        # Check file cache
        file_cache = self._load_from_file_cache(cache_key)
        if file_cache is not None:
            cache_data, cache_timestamp = file_cache
            # Check if cache is still valid
            if time.time() - cache_timestamp < max_cache_age or self.offline_mode:
                self.cache[cache_key] = cache_data  # Store in memory cache
                print(f"Using cached data for {ticker} (file cache, age: {(time.time() - cache_timestamp)/3600:.1f} hours)")
                return cache_data
        
        # If in offline mode and no cache exists, raise error
        if self.offline_mode:
            raise ValueError(f"No cached data available for {ticker} and offline mode is enabled")
        
        # Fetch from API
        print(f"Fetching data for {ticker} from API...")
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        # Basic data validation
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        # Store in both memory and file cache if caching is enabled
        self.cache[cache_key] = data
        if self.cache_enabled:
            self._save_to_file_cache(cache_key, (data, time.time()))
        
        return data
    
    def get_current_price(self, ticker: str, max_cache_age: Optional[int] = None) -> float:
        """
        Get the current price of a stock.
        
        Args:
            ticker: Stock ticker symbol
            max_cache_age: Maximum age of cached data in seconds before refreshing
            
        Returns:
            Current price
        """
        # Use instance max_cache_age if not specified
        if max_cache_age is None:
            max_cache_age = self.cache_max_age
            
        cache_key = f"price_{ticker}"
        
        # Check if caching is disabled or force refresh is enabled
        if not self.cache_enabled or self.force_refresh:
            # Skip cache unless in offline mode
            if not self.offline_mode:
                print(f"Fetching fresh current price for {ticker} (caching {'disabled' if not self.cache_enabled else 'bypassed'})...")
                # Use the fast approach even when not using cache
                try:
                    # Get most recent day's data
                    data = self.get_stock_data(ticker, period="5d", interval="1d")
                    
                    if not data.empty:
                        # Get the most recent closing price
                        current_price = data['Close'].iloc[-1]
                        
                        # Store in memory cache even when disk caching is disabled
                        self.cache[cache_key] = current_price
                        
                        # If caching is still enabled, update the file cache
                        if self.cache_enabled:
                            self._save_to_file_cache(cache_key, (current_price, time.time()))
                        
                        return current_price
                except Exception as e:
                    print(f"Error getting historical data: {e}")
                    
                # Fall back to slower method if needed
                stock = yf.Ticker(ticker)
                current_price = stock.info.get('regularMarketPrice')
                
                if current_price is not None:
                    # Store in memory cache
                    self.cache[cache_key] = current_price
                    
                    # If caching is enabled, store in file cache
                    if self.cache_enabled:
                        self._save_to_file_cache(cache_key, (current_price, time.time()))
                    
                    return current_price
                
                # If couldn't get price through any method
                raise ValueError(f"Could not fetch current price for {ticker}")
        
        # Check memory cache
        if cache_key in self.cache:
            print(f"Using cached price for {ticker} (memory cache)")
            return self.cache[cache_key]
        
        # Check file cache
        file_cache = self._load_from_file_cache(cache_key)
        if file_cache is not None:
            price, cache_timestamp = file_cache
            # Check if cache is still valid
            if time.time() - cache_timestamp < max_cache_age or self.offline_mode:
                self.cache[cache_key] = price  # Store in memory cache
                print(f"Using cached price for {ticker} (file cache, age: {(time.time() - cache_timestamp)/3600:.1f} hours)")
                return price
        
        # FAST APPROACH: Get the last closing price from historical data (1d data)
        # This is much faster than using stock.info.get('regularMarketPrice')
        print(f"Getting current price for {ticker} using historical data...")
        try:
            # Get most recent day's data
            data = self.get_stock_data(ticker, period="5d", interval="1d")
            
            if not data.empty:
                # Get the most recent closing price
                current_price = data['Close'].iloc[-1]
                
                # Store in both memory and file cache if caching is enabled
                self.cache[cache_key] = current_price
                if self.cache_enabled:
                    self._save_to_file_cache(cache_key, (current_price, time.time()))
                
                return current_price
        except Exception as e:
            print(f"Error getting historical data: {e}")
        
        # Only fall back to the slow method if the fast method fails
        if not self.offline_mode:
            print(f"Falling back to slower API method for {ticker} price...")
            stock = yf.Ticker(ticker)
            current_price = stock.info.get('regularMarketPrice')
            
            if current_price is not None:
                # Store in both memory and file cache if caching is enabled
                self.cache[cache_key] = current_price
                if self.cache_enabled:
                    self._save_to_file_cache(cache_key, (current_price, time.time()))
                return current_price
        
        # If all else fails, use a reasonable default or raise an error
        raise ValueError(f"Could not fetch current price for {ticker}")
    
    def get_risk_free_rate(self, max_cache_age: Optional[int] = None) -> float:
        """
        Get current risk-free rate (using 10-year Treasury yield as proxy).
        
        Args:
            max_cache_age: Maximum age of cached data in seconds before refreshing
            
        Returns:
            Current risk-free rate as a decimal (e.g., 0.03 for 3%)
        """
        overall_start_time = time.time()
        print(f"[PERF] Starting risk-free rate fetch at {time.strftime('%H:%M:%S', time.localtime())}")
        
        # Use instance max_cache_age if not specified
        if max_cache_age is None:
            max_cache_age = self.cache_max_age
            
        # Check global memory cache first (fastest access)
        global _GLOBAL_RISK_FREE_RATE
        if (_GLOBAL_RISK_FREE_RATE['rate'] is not None and 
            time.time() - _GLOBAL_RISK_FREE_RATE['timestamp'] < _GLOBAL_RISK_FREE_RATE['max_age']):
            rate = _GLOBAL_RISK_FREE_RATE['rate']
            print(f"Using global cached risk-free rate: {rate:.4f}")
            print(f"[PERF] Global cache access took {time.time() - overall_start_time:.6f} seconds")
            return rate
            
        cache_key = "risk_free_rate"
        
        # Check if caching is disabled or force refresh is enabled
        if not self.cache_enabled or self.force_refresh:
            # Skip cache unless in offline mode
            if not self.offline_mode:
                fetch_start = time.time()
                print(f"Fetching fresh risk-free rate (caching {'disabled' if not self.cache_enabled else 'bypassed'})...")
                try:
                    # Get most recent day's data for the 10-year Treasury yield
                    try:
                        data_fetch_start = time.time()
                        data = self.get_stock_data("^TNX", period="5d", interval="1d")
                        print(f"[PERF] Treasury data fetch took {time.time() - data_fetch_start:.2f} seconds")
                    except Exception as e:
                        print(f"[ERROR] Treasury data fetch failed: {e}")
                        raise

                    if not data.empty:
                        # Get the most recent closing value and convert from percentage to decimal
                        rate = data['Close'].iloc[-1] / 100.0
                        
                        # Store in global, memory and file caches
                        _GLOBAL_RISK_FREE_RATE['rate'] = rate
                        _GLOBAL_RISK_FREE_RATE['timestamp'] = time.time()
                        _GLOBAL_RISK_FREE_RATE['max_age'] = max_cache_age
                        
                        self.cache[cache_key] = rate
                        
                        # If caching is enabled, store in file cache
                        if self.cache_enabled:
                            cache_start = time.time()
                            self._save_to_file_cache(cache_key, (rate, time.time()))
                            print(f"[PERF] File cache write took {time.time() - cache_start:.2f} seconds")
                        
                        print(f"[PERF] Total fresh fetch took {time.time() - fetch_start:.2f} seconds")
                        return rate
                except Exception as e:
                    print(f"Error getting treasury yield data: {e}")
                    # Fall back to default value
                    default_rate = 0.035  # 3.5% as fallback
                    _GLOBAL_RISK_FREE_RATE['rate'] = default_rate
                    _GLOBAL_RISK_FREE_RATE['timestamp'] = time.time()
                    return default_rate
        
        # Check memory cache
        if cache_key in self.cache:
            rate = self.cache[cache_key]
            print(f"Using cached risk-free rate (memory cache): {rate:.4f}")
            elapsed = time.time() - overall_start_time
            print(f"[PERF] Memory cache access took {elapsed:.4f} seconds")
            
            # Update global cache too
            _GLOBAL_RISK_FREE_RATE['rate'] = rate
            _GLOBAL_RISK_FREE_RATE['timestamp'] = time.time()
            _GLOBAL_RISK_FREE_RATE['max_age'] = max_cache_age
            
            return rate
        
        # Check file cache 
        file_cache_start = time.time()
        file_cache = self._load_from_file_cache(cache_key)
        file_cache_time = time.time() - file_cache_start
        print(f"[PERF] File cache check took {file_cache_time:.4f} seconds")
        
        if file_cache is not None:
            rate, cache_timestamp = file_cache
            # Check if cache is still valid
            if time.time() - cache_timestamp < max_cache_age or self.offline_mode:
                # Store in memory and global cache
                self.cache[cache_key] = rate
                _GLOBAL_RISK_FREE_RATE['rate'] = rate
                _GLOBAL_RISK_FREE_RATE['timestamp'] = time.time()
                _GLOBAL_RISK_FREE_RATE['max_age'] = max_cache_age
                
                cache_age_hours = (time.time() - cache_timestamp) / 3600
                print(f"Using cached risk-free rate (file cache, age: {cache_age_hours:.1f} hours): {rate:.4f}")
                elapsed = time.time() - overall_start_time
                print(f"[PERF] Total cached rate fetch took {elapsed:.4f} seconds")
                return rate
        
        # FAST APPROACH: Get the risk-free rate from historical data instead of info
        treasury_ticker = "^TNX"
        try:
            hist_start = time.time()
            print("Getting risk-free rate using historical data...")
            # Get the most recent day's data for the 10-year Treasury yield
            try:
                data = self.get_stock_data(treasury_ticker, period="5d", interval="1d")
                hist_time = time.time() - hist_start
                print(f"[PERF] Historical data fetch took {hist_time:.2f} seconds")
            except Exception as e:
                print(f"[ERROR] Historical data fetch failed: {e}")
                raise
            
            if not data.empty:
                # Get the most recent closing value and convert from percentage to decimal
                rate = data['Close'].iloc[-1] / 100.0
                
                # Store in both memory and file cache if caching is enabled
                self.cache[cache_key] = rate
                if self.cache_enabled:
                    cache_write_start = time.time()
                    self._save_to_file_cache(cache_key, (rate, time.time()))
                    print(f"[PERF] File cache write took {time.time() - cache_write_start:.4f} seconds")
                
                elapsed = time.time() - overall_start_time
                print(f"[PERF] Total risk-free rate fetch took {elapsed:.2f} seconds")
                return rate
        except Exception as e:
            print(f"Error getting treasury yield data: {e}")
        
        # If in offline mode or both approaches failed, use default rate
        default_rate = 0.035  # 3.5% as fallback
        
        # Only try the slow method if not in offline mode and the fast method failed
        if not self.offline_mode:
            slow_start = time.time()
            print("Falling back to slower API method for risk-free rate...")
            try:
                treasury = yf.Ticker(treasury_ticker)
                api_start = time.time()
                rate = treasury.info.get('regularMarketPrice')
                print(f"[PERF] Treasury API call took {time.time() - api_start:.2f} seconds")
                
                if rate is not None:
                    # Convert percentage to decimal
                    rate = rate / 100.0
                    
                    # Store in both memory and file cache if caching is enabled
                    self.cache[cache_key] = rate
                    if self.cache_enabled:
                        self._save_to_file_cache(cache_key, (rate, time.time()))
                    
                    print(f"[PERF] Slow method took {time.time() - slow_start:.2f} seconds")
                    return rate
            except Exception as e:
                print(f"[ERROR] Slow API method failed: {e}")
                # Fall back to default if there's an error
                pass
        
        # Return default value as last resort
        print(f"Using default risk-free rate: {default_rate}")
        self.cache[cache_key] = default_rate
        if self.cache_enabled:
            self._save_to_file_cache(cache_key, (default_rate, time.time()))
        
        elapsed = time.time() - overall_start_time
        print(f"[PERF] Total risk-free rate fetch (using default) took {elapsed:.2f} seconds")
        return default_rate
    
    def get_option_chain(
        self, ticker: str, expiration_date: Optional[Union[str, dt.date, dt.datetime]] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get the options chain for a stock ticker.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Expiration date (optional). If not provided, will use the nearest available date.
                Can be provided as string 'YYYY-MM-DD', date object, or datetime object.
                
        Returns:
            Dictionary with 'calls' and 'puts' as keys, containing DataFrames of option data
            
        Raises:
            ValueError: If the options data cannot be retrieved or if the expiration date is invalid
        """
        try:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            
            if not available_dates:
                error_msg = f"No options data available for {ticker}"
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            logging.info(f"Available expiration dates for {ticker}: {available_dates[:5]}{'...' if len(available_dates) > 5 else ''}")
            
            # Handle the case where expiration_date is provided
            if expiration_date is not None:
                # If expiration_date is a string, try to convert it to a date object
                if isinstance(expiration_date, str):
                    try:
                        # Try to parse as YYYY-MM-DD
                        expiration_date_obj = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
                        expiration_date_str = expiration_date
                        logging.info(f"Using provided expiration date: {expiration_date_str}")
                    except ValueError:
                        try:
                            # Try to parse as MM/DD/YYYY
                            expiration_date_obj = dt.datetime.strptime(expiration_date, '%m/%d/%Y').date()
                            expiration_date_str = expiration_date_obj.strftime('%Y-%m-%d')
                            logging.info(f"Converted provided date from MM/DD/YYYY to YYYY-MM-DD: {expiration_date_str}")
                        except ValueError:
                            error_msg = f"Invalid date format: '{expiration_date}'. Expected 'YYYY-MM-DD' or 'MM/DD/YYYY'"
                            logging.error(error_msg)
                            raise ValueError(error_msg)
                
                # If expiration_date is a datetime object, convert to date
                elif isinstance(expiration_date, dt.datetime):
                    expiration_date_obj = expiration_date.date()
                    expiration_date_str = expiration_date_obj.strftime('%Y-%m-%d')
                    logging.info(f"Converted datetime to date: {expiration_date_str}")
                
                # If expiration_date is a date object, convert to string
                elif isinstance(expiration_date, dt.date):
                    expiration_date_obj = expiration_date
                    expiration_date_str = expiration_date_obj.strftime('%Y-%m-%d')
                    logging.info(f"Using provided date object: {expiration_date_str}")
                
                else:
                    error_msg = f"Unsupported date type: {type(expiration_date).__name__}. Expected string, date, or datetime"
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                
                # Check if the expiration date is available
                if expiration_date_str not in available_dates:
                    logging.warning(f"Expiration date {expiration_date_str} not available for {ticker}")
                    logging.info(f"Finding nearest available expiration date...")
                    expiration_date_str = self.find_nearest_expiration_date(ticker, expiration_date_obj)
                    logging.info(f"Using nearest available expiration date: {expiration_date_str}")
            else:
                # If no expiration date is provided, use the nearest available date
                today = dt.date.today()
                expiration_date_str = self.find_nearest_expiration_date(ticker, today)
                logging.info(f"No expiration date provided. Using nearest available date: {expiration_date_str}")
            
            # Get the option chain for the specified expiration date
            try:
                option_chain = stock.option_chain(expiration_date_str)
                logging.info(f"Successfully retrieved option chain for {ticker} with expiration date {expiration_date_str}")
                
                # Check if the option chain contains data
                if option_chain.calls.empty and option_chain.puts.empty:
                    error_msg = f"Retrieved empty option chain for {ticker} with expiration date {expiration_date_str}"
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                
                # Log the number of calls and puts
                logging.info(f"Retrieved {len(option_chain.calls)} calls and {len(option_chain.puts)} puts")
                
                return {
                    'calls': option_chain.calls,
                    'puts': option_chain.puts
                }
            except Exception as e:
                error_msg = f"Error retrieving option chain for {ticker} with expiration date {expiration_date_str}: {str(e)}"
                logging.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            if isinstance(e, ValueError):
                # Re-raise ValueErrors since they're already properly formatted
                raise
            else:
                # Wrap other exceptions with more context
                error_msg = f"Error retrieving option chain for {ticker}: {str(e)}"
                logging.error(error_msg)
                raise ValueError(error_msg)
    
    def find_nearest_expiration_date(
        self,
        ticker: str,
        target_date: Union[str, dt.date, dt.datetime]
    ) -> str:
        """
        Find the nearest available expiration date to a target date.
        
        Args:
            ticker: Stock ticker symbol
            target_date: Target expiration date
            
        Returns:
            Nearest available expiration date in string format 'YYYY-MM-DD'
            
        Raises:
            ValueError: If no options data is available for the ticker or
                       if the target_date format is invalid
        """
        try:
            stock = yf.Ticker(ticker)
            available_dates = stock.options
            
            if not available_dates:
                error_msg = f"No options data available for {ticker}"
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            # Convert target_date to datetime.date object if it's a string or datetime
            if isinstance(target_date, str):
                try:
                    target_date = dt.datetime.strptime(target_date, '%Y-%m-%d').date()
                    logging.info(f"Parsed date string '{target_date}' using YYYY-MM-DD format")
                except ValueError:
                    try:
                        # Try another format
                        target_date = dt.datetime.strptime(target_date, '%m/%d/%Y').date()
                        logging.info(f"Parsed date string '{target_date}' using MM/DD/YYYY format")
                    except ValueError:
                        error_msg = f"Invalid date format: '{target_date}'. Expected 'YYYY-MM-DD' or 'MM/DD/YYYY'"
                        logging.error(error_msg)
                        raise ValueError(error_msg)
            elif isinstance(target_date, dt.datetime):
                target_date = target_date.date()
                logging.info(f"Converted datetime to date: {target_date}")
            elif not isinstance(target_date, dt.date):
                error_msg = f"Unsupported date type: {type(target_date).__name__}. Expected string, date, or datetime"
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            # Log available dates for debugging
            logging.info(f"Available expiration dates for {ticker}: {available_dates[:5]}{'...' if len(available_dates) > 5 else ''}")
            
            # Convert available dates to datetime objects
            try:
                available_dates_dt = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in available_dates]
            except ValueError as e:
                error_msg = f"Error parsing available dates for {ticker}: {str(e)}"
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            # Find the date with the smallest absolute difference
            nearest_date = min(available_dates_dt, key=lambda x: abs((x - target_date).days))
            
            # Convert back to string format
            nearest_date_str = nearest_date.strftime('%Y-%m-%d')
            
            # Log the difference in days for debugging
            days_diff = abs((nearest_date - target_date).days)
            logging.info(f"Found nearest expiration date to {target_date}: {nearest_date_str} (difference of {days_diff} days)")
            
            return nearest_date_str
            
        except Exception as e:
            if isinstance(e, ValueError):
                # Re-raise ValueErrors since they're already properly formatted
                raise
            else:
                # Wrap other exceptions with more context
                error_msg = f"Error finding nearest expiration date for {ticker}: {str(e)}"
                logging.error(error_msg)
                raise ValueError(error_msg)
    
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
            
        Raises:
            ValueError: If no option matching the criteria can be found
        """
        # Get current stock price for OTM calculations
        current_price = self.get_current_price(ticker)
        
        # Log detailed information about the expiration date being used
        logging.info(f"Finding option with criteria: {criteria}={value} for {ticker} {option_type}")
        logging.info(f"Expiration date input: {expiration_date} (type: {type(expiration_date).__name__})")
        
        # Standardize expiration date format
        if isinstance(expiration_date, dt.datetime):
            expiration_date = expiration_date.date()
            logging.info(f"Converted datetime to date: {expiration_date}")
        
        # Get options chain
        try:
            options_chain = self.get_option_chain(ticker, expiration_date)
            
            # Select calls or puts based on option_type
            if option_type.lower() == 'call':
                options = options_chain['calls']
            elif option_type.lower() == 'put':
                options = options_chain['puts']
            else:
                raise ValueError("option_type must be either 'call' or 'put'")
            
            # Log diagnostics about available options
            logging.info(f"Found {len(options)} {option_type} options for {ticker} with expiration {expiration_date}")
            if len(options) > 0:
                min_strike = options['strike'].min()
                max_strike = options['strike'].max()
                logging.info(f"Strike range: {min_strike} to {max_strike}")
                
                # Check for expiration field if needed later
                expiration_field_found = False
                expiration_field_name = None
                for field in ['expiration', 'expirationDate', 'expDate']:
                    if field in options.columns:
                        expiration_field_found = True
                        expiration_field_name = field
                        logging.info(f"Found expiration field in options data: '{field}'")
                        break
                
                if not expiration_field_found:
                    logging.warning(f"No standard expiration field found in options data. Available columns: {list(options.columns)}")
                    
                    # Don't trust lastTradeDate as the expiration date - it's not the same thing
                    if 'lastTradeDate' in options.columns:
                        logging.warning(f"Found 'lastTradeDate' field, but this is NOT the expiration date")
                    
                    # Add a synthetic expiration field using the provided expiration date
                    # This ensures we're using the correct date for expiration
                    expiration_str = None
                    if isinstance(expiration_date, dt.date):
                        expiration_str = expiration_date.strftime('%Y-%m-%d')
                    elif isinstance(expiration_date, str):
                        # Try to verify if it's a valid date format 
                        try:
                            date_obj = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
                            expiration_str = expiration_date
                        except ValueError:
                            # If we can't parse it, just use it as is
                            expiration_str = expiration_date
                    else:
                        # If we don't know the format, use the string representation
                        expiration_str = str(expiration_date)
                    
                    options['expiration'] = expiration_str
                    logging.info(f"Added synthetic expiration field with value: {expiration_str}")
                    
                    # Also log the lastTradeDate for comparison if available
                    if 'lastTradeDate' in options.columns:
                        sample_last_trade = options['lastTradeDate'].iloc[0] if not options['lastTradeDate'].empty else 'N/A'
                        logging.info(f"For reference, lastTradeDate example: {sample_last_trade}")
            else:
                logging.warning(f"No {option_type} options found for {ticker} with expiration {expiration_date}")
            
            # Find option based on criteria
            if criteria == 'strike':
                # Find option with strike closest to the specified value
                options['diff'] = abs(options['strike'] - value)
                option = options.loc[options['diff'].idxmin()]
                logging.info(f"Found option with strike {option['strike']} (requested: {value})")
                
            elif criteria == 'delta':
                # Find option with delta closest to the specified value
                if 'delta' not in options.columns:
                    # If greeks not in data, can't search by delta
                    raise ValueError("Delta information not available in options chain")
                    
                options['diff'] = abs(options['delta'] - value)
                option = options.loc[options['diff'].idxmin()]
                logging.info(f"Found option with delta {option['delta']} (requested: {value})")
                
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
                logging.info(f"Found option with strike {option['strike']} (target OTM: {target_strike})")
                    
            elif criteria == 'atm':
                # Find at-the-money option (strike closest to current price)
                options['diff'] = abs(options['strike'] - current_price)
                option = options.loc[options['diff'].idxmin()]
                logging.info(f"Found ATM option with strike {option['strike']} (current price: {current_price})")
                
            else:
                raise ValueError(f"Unknown criteria: {criteria}")
            
            # Log all available fields for debugging
            logging.info(f"Option data fields: {list(option.index)}")
            
            return option
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error finding option: {error_msg}")
            
            # Provide more context in the error message
            if "expiration" in error_msg.lower():
                stock = yf.Ticker(ticker)
                avail_expirations = stock.options
                logging.error(f"Available expiration dates for {ticker}: {avail_expirations}")
                raise ValueError(f"Expiration issue for {ticker} {option_type}: {error_msg}. Available dates: {avail_expirations[:5]}")
            else:
                raise
    
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
        # Check cache first
        cache_key = f"vol_{ticker}_{lookback_period}_{interval}"
        if cache_key in self.calculation_cache:
            return self.calculation_cache[cache_key]
            
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
            
        # Cache the result
        self.calculation_cache[cache_key] = annualized_volatility
            
        return annualized_volatility
    
    def calculate_mean_return(
        self, 
        ticker: str, 
        lookback_period: str = "1y", 
        interval: str = "1d"
    ) -> float:
        """
        Calculate mean historical return for a stock.
        
        Args:
            ticker: Stock ticker symbol
            lookback_period: Period to use for calculation
            interval: Data interval
            
        Returns:
            Annualized mean return
        """
        # Check cache first
        cache_key = f"mean_{ticker}_{lookback_period}_{interval}"
        if cache_key in self.calculation_cache:
            return self.calculation_cache[cache_key]
        
        # Get historical data
        data = self.get_stock_data(ticker, period=lookback_period, interval=interval)
        
        # Calculate daily returns
        data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Remove NaN values
        returns = data['Returns'].dropna()
        
        # Calculate mean return
        daily_mean_return = returns.mean()
        
        # Annualize mean return based on the interval
        if interval == '1d':
            # Assuming 252 trading days in a year
            annualized_mean_return = daily_mean_return * 252
        elif interval == '1wk':
            # Assuming 52 trading weeks in a year
            annualized_mean_return = daily_mean_return * 52
        elif interval == '1mo':
            # Assuming 12 trading months in a year
            annualized_mean_return = daily_mean_return * 12
        else:
            # Default to daily
            annualized_mean_return = daily_mean_return * 252
        
        # Cache the result
        self.calculation_cache[cache_key] = annualized_mean_return
        
        return annualized_mean_return