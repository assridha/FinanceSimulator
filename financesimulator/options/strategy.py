"""Options strategy module for building and evaluating option strategies."""

from typing import List, Dict, Union, Optional, Tuple, Any
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

from ..data.fetcher import MarketDataFetcher
from .black_scholes import BlackScholes


class InstrumentType(Enum):
    """Type of financial instrument in a strategy."""
    STOCK = "STOCK"
    CALL = "CALL"
    PUT = "PUT"


class Action(Enum):
    """Action to take on an instrument (buy or sell)."""
    BUY = "BUY"
    SELL = "SELL"


class OptionSpecification:
    """Specification for an option in a strategy."""
    
    def __init__(
        self, 
        option_type: str,
        strike_spec: Union[float, str],
        expiration_spec: Union[int, str, datetime],
        quantity: int = 1,
        action: Action = Action.BUY
    ):
        """
        Initialize option specification.
        
        Args:
            option_type: 'call' or 'put'
            strike_spec: Strike price specification. Can be:
                - Numeric value (absolute strike price)
                - String with format '10%OTM', '5%ITM', 'ATM', etc.
                - String with format 'delta:0.30' (for delta-based selection)
            expiration_spec: Option expiration specification. Can be:
                - Number of days to expiration
                - String with format '30d', '3m', etc.
                - Specific datetime object
            quantity: Number of contracts
            action: BUY or SELL
        """
        self.option_type = option_type.lower()
        self.strike_spec = strike_spec
        self.expiration_spec = expiration_spec
        self.quantity = quantity
        self.action = action
        
        # These will be resolved when the option is evaluated
        self.resolved_strike: Optional[float] = None
        self.resolved_expiration: Optional[datetime] = None
        self.option_price: Optional[float] = None
        self.greeks: Dict[str, float] = {}
    
    def resolve(self, ticker: str, current_price: float, data_fetcher: MarketDataFetcher) -> Dict[str, Any]:
        """
        Resolve option specification to concrete values using market data.
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            data_fetcher: Market data fetcher instance
            
        Returns:
            Dictionary containing the resolved option details
        """
        # Resolve expiration
        if isinstance(self.expiration_spec, int):
            # Days to expiration
            self.resolved_expiration = datetime.now() + timedelta(days=self.expiration_spec)
        elif isinstance(self.expiration_spec, str):
            if self.expiration_spec.endswith('d'):
                # Format like '30d' for 30 days
                days = int(self.expiration_spec[:-1])
                self.resolved_expiration = datetime.now() + timedelta(days=days)
            elif self.expiration_spec.endswith('m'):
                # Format like '3m' for 3 months
                months = int(self.expiration_spec[:-1])
                self.resolved_expiration = datetime.now() + timedelta(days=months * 30)
            elif self.expiration_spec.endswith('y'):
                # Format like '1y' for 1 year
                years = int(self.expiration_spec[:-1])
                self.resolved_expiration = datetime.now() + timedelta(days=years * 365)
            else:
                # Try to parse as date string
                try:
                    self.resolved_expiration = datetime.strptime(self.expiration_spec, '%Y-%m-%d')
                except ValueError:
                    raise ValueError(f"Could not parse expiration specification: {self.expiration_spec}")
        elif isinstance(self.expiration_spec, datetime):
            self.resolved_expiration = self.expiration_spec
        else:
            raise ValueError(f"Invalid expiration specification: {self.expiration_spec}")
        
        # Calculate days to expiration
        days_to_expiration = (self.resolved_expiration - datetime.now()).days
        if days_to_expiration <= 0:
            raise ValueError("Option expiration must be in the future")
        
        # Resolve strike price
        if isinstance(self.strike_spec, (int, float)):
            # Absolute strike price
            self.resolved_strike = float(self.strike_spec)
        elif isinstance(self.strike_spec, str):
            if self.strike_spec.upper() == 'ATM':
                # At the money
                self.resolved_strike = current_price
            elif self.strike_spec.endswith('%OTM'):
                # Out of the money by percentage
                otm_pct = float(self.strike_spec.split('%')[0]) / 100
                if self.option_type == 'call':
                    self.resolved_strike = current_price * (1 + otm_pct)
                else:  # put
                    self.resolved_strike = current_price * (1 - otm_pct)
            elif self.strike_spec.endswith('%ITM'):
                # In the money by percentage
                itm_pct = float(self.strike_spec.split('%')[0]) / 100
                if self.option_type == 'call':
                    self.resolved_strike = current_price * (1 - itm_pct)
                else:  # put
                    self.resolved_strike = current_price * (1 + itm_pct)
            elif self.strike_spec.startswith('delta:'):
                # Delta-based strike selection
                target_delta = float(self.strike_spec.split(':')[1])
                # Use option chain to find option with closest delta
                option = data_fetcher.find_option_by_criteria(
                    ticker=ticker,
                    option_type=self.option_type,
                    criteria='delta',
                    value=target_delta,
                    expiration_date=self.resolved_expiration
                )
                self.resolved_strike = option['strike']
            else:
                # Try to parse as float
                try:
                    self.resolved_strike = float(self.strike_spec)
                except ValueError:
                    raise ValueError(f"Could not parse strike specification: {self.strike_spec}")
        else:
            raise ValueError(f"Invalid strike specification: {self.strike_spec}")
        
        # Calculate option price
        # First try to get market price from option chain
        try:
            option = data_fetcher.find_option_by_criteria(
                ticker=ticker,
                option_type=self.option_type,
                criteria='strike',
                value=self.resolved_strike,
                expiration_date=self.resolved_expiration
            )
            self.option_price = option['lastPrice']
            
            # Get greeks if available
            for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                if greek in option:
                    self.greeks[greek] = option[greek]
                    
        except (ValueError, KeyError):
            # Fall back to Black-Scholes model
            risk_free_rate = data_fetcher.get_risk_free_rate()
            volatility = data_fetcher.calculate_historical_volatility(ticker, lookback_period='1y')
            years_to_expiration = days_to_expiration / 365
            
            if self.option_type == 'call':
                self.option_price = BlackScholes.call_price(
                    S=current_price,
                    K=self.resolved_strike,
                    r=risk_free_rate,
                    sigma=volatility,
                    T=years_to_expiration
                )
            else:  # put
                self.option_price = BlackScholes.put_price(
                    S=current_price,
                    K=self.resolved_strike,
                    r=risk_free_rate,
                    sigma=volatility,
                    T=years_to_expiration
                )
                
            # Calculate greeks
            self.greeks = BlackScholes.calculate_greeks(
                option_type=self.option_type,
                S=current_price,
                K=self.resolved_strike,
                r=risk_free_rate,
                sigma=volatility,
                T=years_to_expiration
            )
        
        # Return resolved option details
        return {
            'type': self.option_type,
            'strike': self.resolved_strike,
            'expiration': self.resolved_expiration,
            'days_to_expiration': days_to_expiration,
            'price': self.option_price,
            'greeks': self.greeks
        }


class StrategyComponent:
    """Component of an options strategy (stock or option position)."""
    
    def __init__(
        self,
        instrument_type: InstrumentType,
        action: Action,
        quantity: int,
        option_spec: Optional[OptionSpecification] = None
    ):
        """
        Initialize strategy component.
        
        Args:
            instrument_type: Type of instrument (STOCK, CALL, PUT)
            action: BUY or SELL
            quantity: Number of shares or contracts
            option_spec: Option specification (required for CALL or PUT)
        """
        self.instrument_type = instrument_type
        self.action = action
        self.quantity = quantity
        self.option_spec = option_spec
        
        # Validate options specification
        if instrument_type in [InstrumentType.CALL, InstrumentType.PUT] and option_spec is None:
            raise ValueError("Option specification required for call or put")
            
        # Values to be calculated when evaluated
        self.initial_cost: Optional[float] = None
        self.initial_price: Optional[float] = None
        
    def evaluate(self, ticker: str, current_price: float, data_fetcher: MarketDataFetcher) -> Dict[str, Any]:
        """
        Evaluate the component based on current market conditions.
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            data_fetcher: Market data fetcher instance
            
        Returns:
            Dictionary containing evaluation results
        """
        sign = 1 if self.action == Action.BUY else -1
        
        if self.instrument_type == InstrumentType.STOCK:
            # Simple stock position
            self.initial_price = current_price
            self.initial_cost = sign * current_price * self.quantity
            
            return {
                'type': 'stock',
                'action': self.action.value,
                'quantity': self.quantity,
                'price': current_price,
                'cost': self.initial_cost
            }
        else:
            # Option position (call or put)
            option_type = 'call' if self.instrument_type == InstrumentType.CALL else 'put'
            
            # Make sure option type in the spec matches the instrument type
            if self.option_spec and self.option_spec.option_type != option_type:
                self.option_spec.option_type = option_type
                
            # Resolve option specification
            option_details = self.option_spec.resolve(ticker, current_price, data_fetcher)
            self.initial_price = option_details['price']
            
            # Calculate cost (negative for sell, positive for buy)
            self.initial_cost = sign * option_details['price'] * self.quantity
            
            # Return evaluation details
            return {
                'type': option_type,
                'action': self.action.value,
                'quantity': self.quantity,
                'strike': option_details['strike'],
                'expiration': option_details['expiration'],
                'days_to_expiration': option_details['days_to_expiration'],
                'price': option_details['price'],
                'cost': self.initial_cost,
                'greeks': option_details['greeks']
            }


class OptionsStrategy:
    """Class for building and evaluating options strategies."""
    
    def __init__(self, ticker: str, name: Optional[str] = None):
        """
        Initialize options strategy.
        
        Args:
            ticker: Stock ticker symbol
            name: Strategy name (optional)
        """
        self.ticker = ticker
        self.name = name or f"{ticker} Strategy"
        self.components: List[StrategyComponent] = []
        self.data_fetcher = MarketDataFetcher()
        self.current_price: Optional[float] = None
        
        # Evaluation results
        self.initial_cost: Optional[float] = None
        self.max_profit: Optional[float] = None
        self.max_loss: Optional[float] = None
        self.breakeven_points: List[float] = []
        
    def add_component(self, component: StrategyComponent) -> None:
        """
        Add a component to the strategy.
        
        Args:
            component: StrategyComponent to add
        """
        self.components.append(component)
    
    def add_stock(self, action: Union[Action, str], quantity: int) -> None:
        """
        Add a stock position to the strategy.
        
        Args:
            action: BUY or SELL
            quantity: Number of shares
        """
        if isinstance(action, str):
            action = Action(action.upper())
            
        component = StrategyComponent(
            instrument_type=InstrumentType.STOCK,
            action=action,
            quantity=quantity
        )
        
        self.add_component(component)
    
    def add_option(
        self,
        option_type: str,
        action: Union[Action, str],
        quantity: int,
        strike_spec: Union[float, str],
        expiration_spec: Union[int, str, datetime]
    ) -> None:
        """
        Add an option position to the strategy.
        
        Args:
            option_type: 'call' or 'put'
            action: BUY or SELL
            quantity: Number of contracts
            strike_spec: Strike price specification
            expiration_spec: Expiration specification
        """
        if isinstance(action, str):
            action = Action(action.upper())
            
        instrument_type = InstrumentType.CALL if option_type.lower() == 'call' else InstrumentType.PUT
        
        option_spec = OptionSpecification(
            option_type=option_type,
            strike_spec=strike_spec,
            expiration_spec=expiration_spec,
            quantity=quantity,
            action=action
        )
        
        component = StrategyComponent(
            instrument_type=instrument_type,
            action=action,
            quantity=quantity,
            option_spec=option_spec
        )
        
        self.add_component(component)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the strategy based on current market conditions.
        
        Returns:
            Dictionary containing evaluation results
        """
        if not self.components:
            raise ValueError("No components in strategy")
        
        # Get current stock price
        self.current_price = self.data_fetcher.get_current_price(self.ticker)
        
        # Evaluate each component
        results = []
        total_cost = 0
        
        for component in self.components:
            result = component.evaluate(self.ticker, self.current_price, self.data_fetcher)
            results.append(result)
            total_cost += result['cost']
            
        self.initial_cost = total_cost
        
        # Return evaluation summary
        return {
            'ticker': self.ticker,
            'strategy': self.name,
            'current_price': self.current_price,
            'components': results,
            'total_cost': self.initial_cost
        }
    
    def calculate_payoff(self, price_at_expiration: float) -> float:
        """
        Calculate strategy payoff at a specific stock price at expiration.
        
        Args:
            price_at_expiration: Stock price at expiration
            
        Returns:
            Strategy payoff
        """
        if not self.components:
            raise ValueError("No components in strategy")
            
        if self.current_price is None:
            self.evaluate()
            
        total_payoff = 0
        
        for component in self.components:
            if component.instrument_type == InstrumentType.STOCK:
                # Stock payoff is just the difference between future and current price
                sign = 1 if component.action == Action.BUY else -1
                stock_profit = sign * component.quantity * (price_at_expiration - self.current_price)
                total_payoff += stock_profit
            else:
                # Option payoff
                option_type = 'call' if component.instrument_type == InstrumentType.CALL else 'put'
                sign = 1 if component.action == Action.BUY else -1
                strike = component.option_spec.resolved_strike
                
                if option_type == 'call':
                    # Call payoff is max(0, stock_price - strike)
                    intrinsic_value = max(0, price_at_expiration - strike)
                else:
                    # Put payoff is max(0, strike - stock_price)
                    intrinsic_value = max(0, strike - price_at_expiration)
                    
                # Account for premium already paid/received
                # For short options, we gain the premium initially but lose the intrinsic value
                # For long options, we pay the premium initially but gain the intrinsic value
                if component.action == Action.BUY:
                    # Buying options: -premium + intrinsic_value
                    option_profit = (intrinsic_value - component.initial_price) * component.quantity
                else:
                    # Selling options: +premium - intrinsic_value
                    option_profit = (component.initial_price - intrinsic_value) * component.quantity
                
                total_payoff += option_profit
                
        return total_payoff
    
    def calculate_payoff_curve(self, price_range: Optional[Tuple[float, float]] = None, num_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate payoff curve across a range of prices.
        
        Args:
            price_range: Optional tuple of (min_price, max_price). If None, uses a range around current price.
            num_points: Number of price points to calculate
            
        Returns:
            Dictionary with 'prices' and 'payoffs' arrays
        """
        if not self.components:
            raise ValueError("No components in strategy")
            
        if self.current_price is None:
            self.evaluate()
            
        # Determine price range if not provided
        if price_range is None:
            # Default to ±50% of current price
            min_price = self.current_price * 0.5
            max_price = self.current_price * 1.5
        else:
            min_price, max_price = price_range
            
        # Generate price points
        prices = np.linspace(min_price, max_price, num_points)
        
        # Calculate payoff at each price
        payoffs = np.array([self.calculate_payoff(price) for price in prices])
        
        # Calculate max profit, max loss, and breakeven points
        self.max_profit = payoffs.max()
        self.max_loss = payoffs.min()
        
        # Find breakeven points (where payoff crosses zero)
        signs = np.sign(payoffs)
        sign_changes = np.where(np.diff(signs) != 0)[0]
        
        self.breakeven_points = []
        for idx in sign_changes:
            # Linear interpolation to find breakeven price
            p1, p2 = prices[idx], prices[idx + 1]
            v1, v2 = payoffs[idx], payoffs[idx + 1]
            
            # Only interpolate if one value is positive and one is negative
            if v1 * v2 < 0:
                ratio = abs(v1) / (abs(v1) + abs(v2))
                breakeven = p1 + ratio * (p2 - p1)
                self.breakeven_points.append(breakeven)
        
        return {
            'prices': prices,
            'payoffs': payoffs,
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'breakeven_points': self.breakeven_points
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OptionsStrategy':
        """
        Create a strategy from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            OptionsStrategy instance
        """
        if 'ticker' not in config:
            raise ValueError("Strategy configuration must include 'ticker'")
            
        ticker = config['ticker']
        name = config.get('name')
        
        strategy = cls(ticker=ticker, name=name)
        
        # Add components from config
        for component_config in config.get('components', []):
            instrument_type = component_config.get('instrument', '').upper()
            action = component_config.get('action', '').upper()
            quantity = component_config.get('quantity', 1)
            
            if instrument_type == 'STOCK':
                strategy.add_stock(
                    action=action,
                    quantity=quantity
                )
            elif instrument_type in ['CALL', 'PUT']:
                strategy.add_option(
                    option_type=instrument_type.lower(),
                    action=action,
                    quantity=quantity,
                    strike_spec=component_config.get('strike', 'ATM'),
                    expiration_spec=component_config.get('expiration', '30d')
                )
            else:
                raise ValueError(f"Unknown instrument type: {instrument_type}")
                
        return strategy 