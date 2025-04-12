"""Black-Scholes option pricing model."""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, Optional


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
        # Set search bounds
        sigma_low = 0.001  # 0.1%
        sigma_high = 5.0   # 500%
        
        for _ in range(max_iterations):
            sigma_mid = (sigma_low + sigma_high) / 2
            
            if option_type.lower() == 'call':
                price_mid = cls.call_price(S, K, r, sigma_mid, T)
            else:  # put
                price_mid = cls.put_price(S, K, r, sigma_mid, T)
            
            price_diff = price_mid - option_price
            
            # Check if we're close enough
            if abs(price_diff) < precision:
                return sigma_mid
            
            # Adjust bounds based on diff
            if price_diff > 0:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid
        
        # If we hit max iterations, return best guess
        return (sigma_low + sigma_high) / 2

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