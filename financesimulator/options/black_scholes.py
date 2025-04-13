"""Black-Scholes option pricing model."""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, Optional
import logging
import time


class BlackScholes:
    """
    Black-Scholes option pricing model for European options.
    
    The model assumes:
    - No dividends
    - European-style options (exercise at expiration only)
    - Constant risk-free rate and volatility
    - Log-normal distribution of stock prices
    """
    
    @staticmethod
    def _calculate_d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters for Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Option strike price
            r: Risk-free interest rate (decimal)
            sigma: Volatility (decimal)
            T: Time to expiration (in years)
            
        Returns:
            Tuple containing d1 and d2 values
        """
        # Handle edge cases
        if T <= 0 or sigma <= 0:
            raise ValueError("Time to expiration and volatility must be positive")
            
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return d1, d2
    
    @classmethod
    def call_price(cls, S: float, K: float, r: float, sigma: float, T: float) -> float:
        """
        Calculate the price of a European call option.
        
        Args:
            S: Current stock price
            K: Option strike price
            r: Risk-free interest rate (decimal)
            sigma: Volatility (decimal)
            T: Time to expiration (in years)
            
        Returns:
            Call option price
        """
        d1, d2 = cls._calculate_d1_d2(S, K, r, sigma, T)
        
        # Black-Scholes formula for call
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        return call_price
    
    @classmethod
    def put_price(cls, S: float, K: float, r: float, sigma: float, T: float) -> float:
        """
        Calculate the price of a European put option.
        
        Args:
            S: Current stock price
            K: Option strike price
            r: Risk-free interest rate (decimal)
            sigma: Volatility (decimal)
            T: Time to expiration (in years)
            
        Returns:
            Put option price
        """
        d1, d2 = cls._calculate_d1_d2(S, K, r, sigma, T)
        
        # Black-Scholes formula for put
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return put_price
    
    @classmethod
    def calculate_greeks(
        cls, 
        option_type: str, 
        S: float, 
        K: float, 
        r: float, 
        sigma: float, 
        T: float
    ) -> Dict[str, float]:
        """
        Calculate option Greeks.
        
        Args:
            option_type: 'call' or 'put'
            S: Current stock price
            K: Option strike price
            r: Risk-free interest rate (decimal)
            sigma: Volatility (decimal)
            T: Time to expiration (in years)
            
        Returns:
            Dictionary containing greek values (delta, gamma, theta, vega, rho)
        """
        d1, d2 = cls._calculate_d1_d2(S, K, r, sigma, T)
        
        # Common calculations
        sqrt_T = np.sqrt(T)
        exp_rt = np.exp(-r * T)
        norm_pdf_d1 = norm.pdf(d1)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = norm_pdf_d1 / (S * sigma * sqrt_T)
        
        # Theta (per calendar day, assuming 365 days per year)
        if option_type.lower() == 'call':
            theta = (-S * sigma * norm_pdf_d1) / (2 * sqrt_T) - r * K * exp_rt * norm.cdf(d2)
        else:  # put
            theta = (-S * sigma * norm_pdf_d1) / (2 * sqrt_T) + r * K * exp_rt * norm.cdf(-d2)
        
        # Convert theta to daily
        theta = theta / 365
        
        # Vega (same for calls and puts)
        # Vega is expressed as change for 1% point change in volatility
        vega = S * sqrt_T * norm_pdf_d1 / 100
        
        # Rho (for 1% point change in interest rate)
        if option_type.lower() == 'call':
            rho = K * T * exp_rt * norm.cdf(d2) / 100
        else:  # put
            rho = -K * T * exp_rt * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    @classmethod
    def implied_volatility(
        cls, 
        option_type: str, 
        option_price: float, 
        S: float, 
        K: float, 
        r: float, 
        T: float, 
        precision: float = 0.00001, 
        max_iterations: int = 100
    ) -> float:
        """
        Calculate implied volatility using binary search.
        
        Args:
            option_type: 'call' or 'put'
            option_price: Market price of the option
            S: Current stock price
            K: Option strike price
            r: Risk-free interest rate (decimal)
            T: Time to expiration (in years)
            precision: Desired precision of the result
            max_iterations: Maximum number of iterations
            
        Returns:
            Implied volatility
            
        Raises:
            ValueError: If implied volatility cannot be found within constraints
        """
        # Input validation and logging
        if option_price <= 0:
            error_msg = f"Option price must be positive, got {option_price}"
            logging.error(f"Black-Scholes IV: {error_msg}")
            raise ValueError(error_msg)
        
        if T <= 0:
            error_msg = f"Time to expiration must be positive, got {T}"
            logging.error(f"Black-Scholes IV: {error_msg}")
            raise ValueError(error_msg)
        
        if S <= 0 or K <= 0:
            error_msg = f"Stock price and strike price must be positive, got S={S}, K={K}"
            logging.error(f"Black-Scholes IV: {error_msg}")
            raise ValueError(error_msg)
        
        logging.debug(f"Black-Scholes IV: Starting binary search for {option_type} option (market price: ${option_price:.4f})")
        
        # Set search bounds
        sigma_low = 0.001  # 0.1%
        sigma_high = 5.0   # 500%
        
        for i in range(max_iterations):
            sigma_mid = (sigma_low + sigma_high) / 2
            
            if option_type.lower() == 'call':
                price_mid = cls.call_price(S, K, r, sigma_mid, T)
            else:  # put
                price_mid = cls.put_price(S, K, r, sigma_mid, T)
            
            price_diff = price_mid - option_price
            
            # Log progress periodically
            if i % 10 == 0 or abs(price_diff) < precision:
                logging.debug(f"Black-Scholes IV: Iteration {i+1}: σ={sigma_mid:.6f}, price=${price_mid:.4f}, diff=${price_diff:.6f}")
            
            # Check if we're close enough
            if abs(price_diff) < precision:
                logging.debug(f"Black-Scholes IV: Converged after {i+1} iterations, implied volatility = {sigma_mid:.6f}")
                return sigma_mid
            
            # Adjust bounds based on price difference
            if price_diff > 0:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid
        
        # If we didn't converge, check if we're close enough
        final_sigma = (sigma_low + sigma_high) / 2
        
        if option_type.lower() == 'call':
            final_price = cls.call_price(S, K, r, final_sigma, T)
        else:  # put
            final_price = cls.put_price(S, K, r, final_sigma, T)
        
        final_diff = abs(final_price - option_price)
        
        if final_diff < precision * 10:  # Allow 10x the original precision
            logging.warning(f"Black-Scholes IV: Did not fully converge but close enough. IV={final_sigma:.6f}, error=${final_diff:.6f}")
            return final_sigma
        
        error_msg = f"Failed to find implied volatility within {max_iterations} iterations. Last σ={final_sigma:.6f}, error=${final_diff:.6f}"
        logging.error(f"Black-Scholes IV: {error_msg}")
        raise ValueError(error_msg)

    @staticmethod
    def _calculate_d1_d2_vectorized(S: np.ndarray, K: float, r: float, sigma: float, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized calculation of d1 and d2 parameters for Black-Scholes formula.
        
        Args:
            S: Array of stock prices
            K: Option strike price
            r: Risk-free interest rate (decimal)
            sigma: Volatility (decimal)
            T: Array of times to expiration (in years)
            
        Returns:
            Tuple containing arrays of d1 and d2 values
        """
        # Handle edge cases
        if np.any(T <= 0) or sigma <= 0:
            raise ValueError("Time to expiration and volatility must be positive")
            
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return d1, d2
    
    @classmethod
    def call_price_vectorized(cls, S: np.ndarray, K: float, r: float, sigma: float, T: np.ndarray) -> np.ndarray:
        """
        Vectorized calculation of European call option prices.
        
        Args:
            S: Array of stock prices
            K: Option strike price
            r: Risk-free interest rate (decimal)
            sigma: Volatility (decimal)
            T: Array of times to expiration (in years)
            
        Returns:
            Array of call option prices
        """
        d1, d2 = cls._calculate_d1_d2_vectorized(S, K, r, sigma, T)
        
        # Black-Scholes formula for call
        call_prices = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        return call_prices
    
    @classmethod
    def put_price_vectorized(cls, S: np.ndarray, K: float, r: float, sigma: float, T: np.ndarray) -> np.ndarray:
        """
        Vectorized calculation of European put option prices.
        
        Args:
            S: Array of stock prices
            K: Option strike price
            r: Risk-free interest rate (decimal)
            sigma: Volatility (decimal)
            T: Array of times to expiration (in years)
            
        Returns:
            Array of put option prices
        """
        d1, d2 = cls._calculate_d1_d2_vectorized(S, K, r, sigma, T)
        
        # Black-Scholes formula for put
        put_prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return put_prices

def black_scholes_price(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Calculate option price using Black-Scholes formula.
    
    Args:
        option_type: 'call' or 'put'
        S: Current stock price
        K: Option strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        
    Returns:
        Option price
    """
    start_time = time.time()
    logging.debug(f"Black-Scholes: Pricing {option_type} option - S=${S:.2f}, K=${K:.2f}, T={T:.4f}yr, σ={sigma:.4f}, r={r:.4f}")
    
    if option_type.lower() == 'call':
        price = BlackScholes.call_price(S, K, r, sigma, T)
    elif option_type.lower() == 'put':
        price = BlackScholes.put_price(S, K, r, sigma, T)
    else:
        raise ValueError(f"Invalid option type: {option_type}. Must be 'call' or 'put'.")
    
    elapsed = time.time() - start_time
    logging.debug(f"Black-Scholes: {option_type.capitalize()} price calculated: ${price:.4f} (in {elapsed:.6f}s)")
    
    return price

def calculate_implied_volatility(
    option_type: str,
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float
) -> float:
    """
    Calculate implied volatility based on the option's market price.
    
    Args:
        option_type: 'call' or 'put'
        option_price: Market price of the option
        S: Current stock price
        K: Option strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (decimal)
        
    Returns:
        Implied volatility
    """
    start_time = time.time()
    logging.debug(f"Black-Scholes: Calculating implied volatility for {option_type} option - price=${option_price:.2f}, S=${S:.2f}, K=${K:.2f}")
    
    try:
        iv = BlackScholes.implied_volatility(option_type, option_price, S, K, r, T)
        elapsed = time.time() - start_time
        logging.debug(f"Black-Scholes: Implied volatility calculated: {iv:.4f} (in {elapsed:.6f}s)")
        return iv
    except ValueError as e:
        elapsed = time.time() - start_time
        logging.warning(f"Black-Scholes: Failed to calculate implied volatility: {str(e)} (in {elapsed:.6f}s)")
        raise