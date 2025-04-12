"""Visualization module for simulation results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional, Union, Any

# Set up seaborn style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class SimulationVisualizer:
    """Class for visualizing Monte Carlo simulation results."""
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize the visualizer with simulation results.
        
        Args:
            results: Dictionary containing simulation results
        """
        self.results = results
        
        # Handle different result formats
        if 'price_paths' in self.results:
            paths = self.results['price_paths']
        elif 'stock_paths' in self.results:
            paths = self.results['stock_paths']
        else:
            raise ValueError("Results must contain either 'price_paths' or 'stock_paths'")
            
        # Convert to numpy array if it's a list
        if isinstance(paths, list):
            self.paths = np.array(paths)
        else:
            self.paths = paths
        
        # Ensure directory exists for saved visualizations
        output_dir = 'outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_price_paths(
        self, 
        num_paths: int = 100, 
        confidence_intervals: Optional[List[float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot simulated price paths.
        
        Args:
            num_paths: Number of individual paths to plot
            confidence_intervals: List of percentiles for confidence intervals (e.g., [0.1, 0.5, 0.9])
            title: Plot title
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        # Extract data
        price_paths = self.paths
        horizon = self.results.get('horizon', price_paths.shape[1] - 1)
        ticker = self.results.get('ticker', 'Asset')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create x-axis (time steps)
        time_steps = np.arange(horizon + 1)
        
        # Plot a subset of individual paths
        num_paths_to_plot = min(num_paths, price_paths.shape[0])
        for i in range(num_paths_to_plot):
            ax.plot(time_steps, price_paths[i, :], color='skyblue', alpha=0.3, linewidth=0.8)
        
        # Plot confidence intervals if specified
        if confidence_intervals:
            percentiles = np.percentile(price_paths, [p * 100 for p in confidence_intervals], axis=0)
            colors = sns.color_palette("flare", n_colors=len(confidence_intervals))
            
            for i, p in enumerate(confidence_intervals):
                ax.plot(time_steps, percentiles[i], color=colors[i], linewidth=2, 
                        label=f"{int(p * 100)}th percentile")
        
        # Plot median
        median = np.median(price_paths, axis=0)
        ax.plot(time_steps, median, color='darkblue', linewidth=2.5, label='Median')
        
        # Add labels and title
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price')
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Monte Carlo Simulation: {ticker} Price Paths')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure if specified
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show plot if specified
        if show_plot:
            plt.show()
            
        return fig
    
    def plot_histogram(
        self, 
        time_index: int = -1,
        bins: int = 50,
        kde: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot histogram of prices at a specific time step.
        
        Args:
            time_index: Index of time step to plot (default is the last step)
            bins: Number of histogram bins
            kde: Whether to overlay a kernel density estimate
            title: Plot title
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        # Extract data
        price_paths = self.paths
        ticker = self.results.get('ticker', 'Asset')
        
        # Get prices at specified time step
        prices = price_paths[:, time_index]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot histogram with KDE
        sns.histplot(prices, bins=bins, kde=kde, ax=ax, color='skyblue')
        
        # Add vertical line for mean
        mean_price = np.mean(prices)
        ax.axvline(mean_price, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_price:.2f}')
        
        # Add vertical line for median
        median_price = np.median(prices)
        ax.axvline(median_price, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_price:.2f}')
        
        # Add labels and title
        ax.set_xlabel('Price')
        ax.set_ylabel('Frequency')
        if title:
            ax.set_title(title)
        else:
            time_label = "Final" if time_index == -1 else f"Step {time_index}"
            ax.set_title(f'Distribution of {ticker} {time_label} Prices')
        
        # Add legend
        ax.legend()
        
        # Save figure if specified
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show plot if specified
        if show_plot:
            plt.show()
            
        return fig
    
    def plot_returns_distribution(
        self, 
        annualized: bool = True,
        bins: int = 50,
        kde: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot distribution of returns.
        
        Args:
            annualized: Whether to annualize returns
            bins: Number of histogram bins
            kde: Whether to overlay a kernel density estimate
            title: Plot title
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        # Extract data
        price_paths = self.paths
        horizon = self.results.get('horizon', price_paths.shape[1] - 1)
        ticker = self.results.get('ticker', 'Asset')
        
        # Calculate returns (from first to last step)
        initial_prices = price_paths[:, 0]
        final_prices = price_paths[:, -1]
        
        returns = (final_prices / initial_prices) - 1
        
        # Annualize returns if specified
        if annualized:
            # Assuming 252 trading days in a year
            annual_factor = 252 / horizon
            returns = (1 + returns) ** annual_factor - 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot histogram with KDE
        sns.histplot(returns, bins=bins, kde=kde, ax=ax, color='skyblue')
        
        # Add vertical line for mean
        mean_return = np.mean(returns)
        ax.axvline(mean_return, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_return:.2%}')
        
        # Add vertical line for median
        median_return = np.median(returns)
        ax.axvline(median_return, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_return:.2%}')
        
        # Add labels and title
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        
        # Format x-axis as percentage
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        if title:
            ax.set_title(title)
        else:
            return_type = "Annualized Returns" if annualized else "Returns"
            ax.set_title(f'Distribution of {ticker} {return_type}')
        
        # Add legend
        ax.legend()
        
        # Save figure if specified
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show plot if specified
        if show_plot:
            plt.show()
            
        return fig
    
    def plot_heatmap(
        self, 
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot heatmap of price paths over time.
        
        Args:
            title: Plot title
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        # Extract data
        price_paths = self.paths
        horizon = self.results.get('horizon', price_paths.shape[1] - 1)
        ticker = self.results.get('ticker', 'Asset')
        
        # Sample paths for better visualization (max 200)
        sample_size = min(200, price_paths.shape[0])
        sampled_paths = price_paths[np.random.choice(price_paths.shape[0], sample_size, replace=False)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(sampled_paths, cmap='viridis', ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Path')
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Heatmap of {ticker} Price Paths')
        
        # Save figure if specified
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show plot if specified
        if show_plot:
            plt.show()
            
        return fig
    
    def generate_summary_dashboard(
        self, 
        num_paths: int = 30,
        confidence_intervals: List[float] = [0.1, 0.5, 0.9],
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Generate a summary dashboard with multiple visualizations.
        
        Args:
            num_paths: Number of individual paths to plot
            confidence_intervals: List of percentiles for confidence intervals
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        # Extract data
        price_paths = self.paths
        horizon = self.results.get('horizon', price_paths.shape[1] - 1)
        ticker = self.results.get('ticker', 'Asset')
        
        # Create a 2x2 subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        
        # Flatten for easier indexing
        axs = axs.flatten()
        
        # 1. Price paths plot
        time_steps = np.arange(horizon + 1)
        
        # Plot a subset of individual paths
        num_paths_to_plot = min(num_paths, price_paths.shape[0])
        for i in range(num_paths_to_plot):
            axs[0].plot(time_steps, price_paths[i, :], color='skyblue', alpha=0.3, linewidth=0.8)
        
        # Plot confidence intervals
        percentiles = np.percentile(price_paths, [p * 100 for p in confidence_intervals], axis=0)
        colors = sns.color_palette("flare", n_colors=len(confidence_intervals))
        
        for i, p in enumerate(confidence_intervals):
            axs[0].plot(time_steps, percentiles[i], color=colors[i], linewidth=2, 
                    label=f"{int(p * 100)}th percentile")
        
        # Plot median
        median = np.median(price_paths, axis=0)
        axs[0].plot(time_steps, median, color='darkblue', linewidth=2.5, label='Median')
        
        axs[0].set_xlabel('Time Steps')
        axs[0].set_ylabel('Price')
        axs[0].set_title(f'{ticker} Price Paths')
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # 2. Final price histogram
        final_prices = price_paths[:, -1]
        
        sns.histplot(final_prices, bins=50, kde=True, ax=axs[1], color='skyblue')
        
        mean_price = np.mean(final_prices)
        axs[1].axvline(mean_price, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_price:.2f}')
        
        median_price = np.median(final_prices)
        axs[1].axvline(median_price, color='green', linestyle='--', linewidth=2, 
                      label=f'Median: {median_price:.2f}')
        
        axs[1].set_xlabel('Price')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title(f'Distribution of {ticker} Final Prices')
        axs[1].legend()
        
        # 3. Returns distribution
        initial_prices = price_paths[:, 0]
        returns = (final_prices / initial_prices) - 1
        
        # Annualize returns (assuming 252 trading days per year)
        annual_factor = 252 / horizon
        annualized_returns = (1 + returns) ** annual_factor - 1
        
        sns.histplot(annualized_returns, bins=50, kde=True, ax=axs[2], color='skyblue')
        
        mean_return = np.mean(annualized_returns)
        axs[2].axvline(mean_return, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_return:.2%}')
        
        median_return = np.median(annualized_returns)
        axs[2].axvline(median_return, color='green', linestyle='--', linewidth=2, 
                      label=f'Median: {median_return:.2%}')
        
        # Format x-axis as percentage
        from matplotlib.ticker import FuncFormatter
        axs[2].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        axs[2].set_xlabel('Annualized Return')
        axs[2].set_ylabel('Frequency')
        axs[2].set_title(f'Distribution of {ticker} Annualized Returns')
        axs[2].legend()
        
        # 4. Return statistics table
        axs[3].axis('off')
        
        # Calculate statistics
        mean_final = np.mean(final_prices)
        median_final = np.median(final_prices)
        std_final = np.std(final_prices)
        
        mean_ret = mean_return
        median_ret = median_return
        std_ret = np.std(annualized_returns)
        
        var_95 = np.percentile(final_prices, 5)
        var_99 = np.percentile(final_prices, 1)
        
        # Create statistics table
        stats_text = [
            f"Starting Price: ${price_paths[0, 0]:.2f}",
            f"Mean Final Price: ${mean_final:.2f}",
            f"Median Final Price: ${median_final:.2f}",
            f"Std Dev of Final Price: ${std_final:.2f}",
            f"",
            f"Mean Annualized Return: {mean_ret:.2%}",
            f"Median Annualized Return: {median_ret:.2%}",
            f"Std Dev of Ann. Return: {std_ret:.2%}",
            f"",
            f"Value at Risk (95%): ${var_95:.2f}",
            f"Value at Risk (99%): ${var_99:.2f}",
            f"",
            f"Sample Size: {price_paths.shape[0]} paths",
            f"Time Horizon: {horizon} steps"
        ]
        
        y_pos = 0.9
        for line in stats_text:
            if line == "":
                y_pos -= 0.03
            else:
                axs[3].text(0.1, y_pos, line, fontsize=12)
                y_pos -= 0.06
        
        axs[3].set_title('Summary Statistics')
        
        # Layout adjustments
        plt.tight_layout()
        
        # Main title
        model_name = self.results.get('model', self.results.get('stock_model', 'Model'))
        fig.suptitle(f"{ticker} Monte Carlo Simulation - {model_name.upper()}", fontsize=16, y=1.02)
        
        # Save figure if specified
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show plot if specified
        if show_plot:
            plt.show()
            
        return fig


class OptionsStrategyVisualizer:
    """Class for visualizing options strategy simulation results."""
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize the visualizer with options strategy simulation results.
        
        Args:
            results: Dictionary containing options strategy simulation results
        """
        self.results = results
        
        # Extract key data
        self.ticker = results.get('ticker', 'Asset')
        self.starting_price = results.get('starting_price', 100.0)
        
        # Get stock paths and strategy values
        if 'stock_paths' in results:
            self.stock_paths = np.array(results['stock_paths']) if isinstance(results['stock_paths'], list) else results['stock_paths']
        else:
            raise ValueError("Results must contain 'stock_paths'")
            
        if 'strategy_values' in results:
            self.strategy_values = np.array(results['strategy_values']) if isinstance(results['strategy_values'], list) else results['strategy_values']
        else:
            raise ValueError("Results must contain 'strategy_values'")
            
        # Get strategy details
        self.strategy = results.get('strategy', {})
        self.initial_investment = self.strategy.get('total_cost', 0.0)
        
        # Get time points (days)
        self.horizon = self.stock_paths.shape[1] - 1 if self.stock_paths.shape[1] > 1 else 0
        self.time_points = np.arange(self.horizon + 1)
        
        # Ensure output directory exists
        self.output_dir = 'outputs'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def visualize_paths(
        self,
        values: Optional[np.ndarray] = None,
        initial_investment: Optional[float] = None,
        title: Optional[str] = None,
        x_label: str = 'Days',
        y_label: str = 'Value ($)',
        num_paths_to_plot: int = 50,
        percentiles: List[float] = [0.1, 0.5, 0.9],
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Visualize strategy paths.
        
        Args:
            values: Array of values to plot (defaults to strategy_values)
            initial_investment: Initial investment amount (defaults to strategy total_cost)
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            num_paths_to_plot: Number of individual paths to plot
            percentiles: Percentiles to highlight
            figsize: Figure size
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        # Use default values if not provided - ensure we're using strategy values, not stock paths
        if values is None:
            values = self.strategy_values
        
        if initial_investment is None:
            initial_investment = self.initial_investment
            
        if title is None:
            title = f"{self.ticker} Options Strategy Paths"
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot a subset of individual paths
        num_paths = min(num_paths_to_plot, values.shape[0])
        for i in range(num_paths):
            ax.plot(self.time_points, values[i, :], color='skyblue', alpha=0.2, linewidth=0.8)
        
        # Plot percentiles
        percentile_values = np.percentile(values, [p * 100 for p in percentiles], axis=0)
        percentile_colors = ['blue', 'navy', 'darkblue']
        percentile_styles = ['--', '-', '--']
        
        for i, (p, color, style) in enumerate(zip(percentiles, percentile_colors, percentile_styles)):
            ax.plot(
                self.time_points, 
                percentile_values[i], 
                color=color, 
                linestyle=style, 
                linewidth=2, 
                label=f"{int(p * 100)}th percentile"
            )
        
        # Plot initial investment line
        ax.axhline(
            y=initial_investment, 
            color='red', 
            linestyle=':', 
            linewidth=1.5, 
            label='Initial Investment'
        )
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
            
        return fig
    
    def visualize_distribution(
        self,
        values: Optional[np.ndarray] = None,
        initial_investment: Optional[float] = None,
        title: Optional[str] = None,
        x_label: str = 'Value at Expiration ($)',
        y_label: str = 'Frequency',
        bins: int = 50,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Visualize distribution of final values.
        
        Args:
            values: Array of values (defaults to strategy_values)
            initial_investment: Initial investment amount (defaults to strategy total_cost)
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            bins: Number of histogram bins
            figsize: Figure size
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        # Use default values if not provided
        if values is None:
            values = self.strategy_values
        
        if initial_investment is None:
            initial_investment = self.initial_investment
            
        if title is None:
            title = f"{self.ticker} Options Strategy Distribution of Final Values"
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Final values
        final_values = values[:, -1]
        
        # Plot histogram
        ax.hist(final_values, bins=bins, alpha=0.7, color='skyblue')
        
        # Plot vertical lines
        mean_final = np.mean(final_values)
        ax.axvline(x=initial_investment, color='red', linestyle=':', linewidth=1.5, label='Initial Investment')
        ax.axvline(x=mean_final, color='navy', linewidth=2, label=f'Mean: ${mean_final:.2f}')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
            
        return fig
    
    def visualize_payoff_curve(
        self,
        payoff_data: Dict[str, Any],
        current_price: Optional[float] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Visualize strategy payoff curve at expiration.
        
        Args:
            payoff_data: Dictionary containing 'prices', 'payoffs', 'breakeven_points', 
                         'max_profit', and 'max_loss'
            current_price: Current stock price (defaults to starting_price)
            title: Plot title
            figsize: Figure size
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if current_price is None:
            current_price = self.starting_price
            
        if title is None:
            title = f'{self.ticker} Options Strategy Payoff at Expiration'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot payoff curve
        ax.plot(payoff_data['prices'], payoff_data['payoffs'], color='navy', linewidth=2.5)
        
        # Plot zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot breakeven points
        for point in payoff_data['breakeven_points']:
            ax.axvline(x=point, color='green', linestyle='--', linewidth=1.5)
            ax.annotate(
                f'BE: ${point:.2f}',
                xy=(point, 0),
                xytext=(point, payoff_data['max_profit'] * 0.25),
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green'
            )
        
        # Annotate max profit and loss
        max_profit = payoff_data['max_profit']
        max_loss = payoff_data['max_loss']
        
        if max_profit > 0:
            ax.axhline(y=max_profit, color='green', linestyle='--', linewidth=1, alpha=0.7)
            ax.annotate(
                f'Max Profit: ${max_profit:.2f}',
                xy=(payoff_data['prices'][0], max_profit),
                xytext=(payoff_data['prices'][0], max_profit),
                color='green',
                fontweight='bold'
            )
            
        if max_loss < 0:
            ax.axhline(y=max_loss, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.annotate(
                f'Max Loss: ${max_loss:.2f}',
                xy=(payoff_data['prices'][0], max_loss),
                xytext=(payoff_data['prices'][0], max_loss),
                color='red',
                fontweight='bold'
            )
        
        # Plot current stock price
        ax.axvline(x=current_price, color='blue', linestyle=':', linewidth=1.5)
        ax.annotate(
            f'Current: ${current_price:.2f}',
            xy=(current_price, 0),
            xytext=(current_price, max_loss * 0.5),
            arrowprops=dict(arrowstyle='->', color='blue'),
            color='blue'
        )
        
        ax.set_xlabel('Stock Price at Expiration ($)')
        ax.set_ylabel('Profit/Loss ($)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
            
        return fig
    
    def visualize_strategy_vs_stock(
        self,
        strategy_values: Optional[np.ndarray] = None,
        initial_strategy_cost: Optional[float] = None,
        stock_values: Optional[np.ndarray] = None,
        initial_stock_cost: Optional[float] = None,
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Visualize strategy vs stock-only returns.
        
        Args:
            strategy_values: Strategy values array (defaults to self.strategy_values)
            initial_strategy_cost: Initial strategy cost (defaults to self.initial_investment)
            stock_values: Stock values array (if None, uses scaled stock_paths)
            initial_stock_cost: Initial stock cost (if None, uses scaled starting_price)
            figsize: Figure size
            title: Plot title
            save_path: Path to save figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        # Use default values if not provided
        if strategy_values is None:
            strategy_values = self.strategy_values
        
        if initial_strategy_cost is None:
            initial_strategy_cost = self.initial_investment
            
        if stock_values is None:
            # Get stock quantity from strategy components
            stock_quantity = 100  # Default to 100 shares
            for component in self.results.get('strategy', {}).get('components', []):
                if component.get('type', '').upper() == 'STOCK':
                    stock_quantity = component.get('quantity', 100)
                    break
            
            stock_values = self.stock_paths * stock_quantity
            
        if initial_stock_cost is None:
            # If we have the stock quantity, use it
            if 'stock_quantity' in locals():
                initial_stock_cost = self.starting_price * stock_quantity
            else:
                # Otherwise try to estimate from the first value
                initial_stock_cost = stock_values[0, 0]
                
        if title is None:
            title = f"{self.ticker} Strategy vs Stock Returns"
            
        # Calculate returns (not values)
        strategy_returns = strategy_values / initial_strategy_cost - 1
        stock_returns = stock_values / initial_stock_cost - 1
        
        # Get final returns
        final_strategy_returns = strategy_returns[:, -1]
        final_stock_returns = stock_returns[:, -1]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histograms
        ax.hist(final_strategy_returns, bins=30, alpha=0.7, label='Strategy Returns', color='skyblue')
        ax.hist(final_stock_returns, bins=30, alpha=0.7, label='Stock Returns', color='salmon')
        
        # Plot vertical lines for means
        mean_strategy = np.mean(final_strategy_returns)
        mean_stock = np.mean(final_stock_returns)
        
        ax.axvline(x=mean_strategy, color='blue', linewidth=2, label=f'Mean Strategy: {mean_strategy:.2%}')
        ax.axvline(x=mean_stock, color='red', linewidth=2, label=f'Mean Stock: {mean_stock:.2%}')
        
        # Plot zero return line
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Break-even')
        
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis as percentage
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
            
        return fig
    
    def visualize_returns(
        self,
        values: Optional[np.ndarray] = None,
        initial_investment: Optional[float] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        bins: int = 30,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Visualize distribution of returns.
        
        Args:
            values: Array of values (defaults to strategy_values)
            initial_investment: Initial investment amount (defaults to strategy total_cost)
            title: Plot title
            figsize: Figure size
            bins: Number of histogram bins
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        # Use default values if not provided
        if values is None:
            values = self.strategy_values
            
        if initial_investment is None:
            initial_investment = self.initial_investment
            
        if title is None:
            title = f"{self.ticker} Options Strategy Returns Distribution"
            
        # Calculate returns
        final_returns = values[:, -1] / initial_investment - 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        ax.hist(final_returns, bins=bins, alpha=0.7, color='skyblue')
        
        # Plot vertical lines
        mean_return = np.mean(final_returns)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Break-even')
        ax.axvline(x=mean_return, color='blue', linewidth=2, label=f'Mean Return: {mean_return:.2%}')
        
        # Format axes
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        
        # Format x-axis as percentage
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
            
        return fig
    
    def generate_summary_dashboard(
        self,
        payoff_data: Dict[str, Any],
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 12),
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Generate a summary dashboard for the options strategy.
        
        Args:
            payoff_data: Dictionary containing payoff curve data
            title: Dashboard title
            figsize: Figure size
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if title is None:
            title = f"{self.ticker} Options Strategy Summary"
            
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Strategy paths
        num_paths = min(50, self.strategy_values.shape[0])
        for i in range(num_paths):
            axs[0, 0].plot(self.time_points, self.strategy_values[i, :], 
                           color='skyblue', alpha=0.2, linewidth=0.8)
            
        percentiles = [0.1, 0.5, 0.9]
        percentile_values = np.percentile(self.strategy_values, [p * 100 for p in percentiles], axis=0)
        percentile_colors = ['blue', 'navy', 'darkblue']
        percentile_styles = ['--', '-', '--']
        
        for i, (p, color, style) in enumerate(zip(percentiles, percentile_colors, percentile_styles)):
            axs[0, 0].plot(self.time_points, percentile_values[i], 
                           color=color, linestyle=style, linewidth=2, 
                           label=f"{int(p * 100)}th percentile")
            
        axs[0, 0].axhline(y=self.initial_investment, color='red', linestyle=':', 
                         linewidth=1.5, label='Initial Investment')
        axs[0, 0].set_title('Strategy Value Paths')
        axs[0, 0].set_xlabel('Days')
        axs[0, 0].set_ylabel('Value ($)')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # 2. Final values histogram
        axs[0, 1].hist(self.strategy_values[:, -1], bins=30, alpha=0.7, color='skyblue')
        axs[0, 1].axvline(x=self.initial_investment, color='red', linestyle=':', 
                          linewidth=1.5, label='Initial Investment')
        axs[0, 1].axvline(x=np.mean(self.strategy_values[:, -1]), color='blue', linewidth=2, 
                          label=f'Mean: ${np.mean(self.strategy_values[:, -1]):.2f}')
        axs[0, 1].set_title('Final Values Distribution')
        axs[0, 1].set_xlabel('Final Value ($)')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        # 3. Strategy payoff curve
        axs[1, 0].plot(payoff_data['prices'], payoff_data['payoffs'], color='navy', linewidth=2.5)
        axs[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        for point in payoff_data['breakeven_points']:
            axs[1, 0].axvline(x=point, color='green', linestyle='--', linewidth=1.5)
        axs[1, 0].axvline(x=self.starting_price, color='blue', linestyle=':', 
                          linewidth=1.5, label=f'Current: ${self.starting_price:.2f}')
        axs[1, 0].set_title('Strategy Payoff at Expiration')
        axs[1, 0].set_xlabel('Stock Price at Expiration ($)')
        axs[1, 0].set_ylabel('Profit/Loss ($)')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
        
        # 4. Returns vs Stock
        # Get stock quantity from strategy components
        stock_quantity = 100  # Default to 100 shares
        for component in self.results.get('strategy', {}).get('components', []):
            if component.get('type', '').upper() == 'STOCK':
                stock_quantity = component.get('quantity', 100)
                break
        
        stock_values = self.stock_paths * stock_quantity
        initial_stock_cost = self.starting_price * stock_quantity
        
        final_strategy_returns = self.strategy_values[:, -1] / self.initial_investment - 1
        final_stock_returns = stock_values[:, -1] / initial_stock_cost - 1
        
        axs[1, 1].hist(final_strategy_returns, bins=30, alpha=0.7, label='Strategy Returns', color='skyblue')
        axs[1, 1].hist(final_stock_returns, bins=30, alpha=0.7, label='Stock Returns', color='salmon')
        
        mean_strategy = np.mean(final_strategy_returns)
        mean_stock = np.mean(final_stock_returns)
        
        axs[1, 1].axvline(x=mean_strategy, color='blue', linewidth=2, label=f'Strategy: {mean_strategy:.2%}')
        axs[1, 1].axvline(x=mean_stock, color='red', linewidth=2, label=f'Stock: {mean_stock:.2%}')
        axs[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=1, label='Break-even')
        
        axs[1, 1].set_title('Strategy vs Stock Returns')
        axs[1, 1].set_xlabel('Return')
        axs[1, 1].set_ylabel('Frequency')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
        
        # Format x-axis as percentage for the returns plot
        from matplotlib.ticker import FuncFormatter
        axs[1, 1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # Overall title
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)  # Make room for the overall title
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
            
        return fig

    def visualize_returns_distribution(
        self,
        values: Optional[np.ndarray] = None,
        initial_investment: Optional[float] = None,
        title: Optional[str] = None,
        bins: int = 30,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Visualize returns distribution.
        
        Args:
            values: Array of values (defaults to strategy_values)
            initial_investment: Initial investment amount (defaults to strategy total_cost)
            title: Plot title
            bins: Number of histogram bins
            figsize: Figure size
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        # Use default values if not provided
        if values is None:
            values = self.strategy_values
        
        if initial_investment is None:
            initial_investment = self.initial_investment
            
        if title is None:
            title = f"{self.ticker} Options Strategy Returns Distribution"
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate returns
        final_returns = values[:, -1] / initial_investment - 1
        
        # Plot histogram
        ax.hist(final_returns, bins=bins, alpha=0.7, color='skyblue')
        
        # Plot vertical lines
        mean_return = np.mean(final_returns)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, label='Break-even')
        ax.axvline(x=mean_return, color='blue', linewidth=2, label=f'Mean Return: {mean_return:.2%}')
        
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis as percentage
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
            
        return fig
            
    def visualize_histogram(
        self,
        values: Optional[np.ndarray] = None,
        initial_investment: Optional[float] = None,
        title: Optional[str] = None,
        bins: int = 30,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Visualize histogram of final values.
        
        Args:
            values: Array of values (defaults to strategy_values)
            initial_investment: Initial investment amount (defaults to strategy total_cost)
            title: Plot title
            bins: Number of histogram bins
            figsize: Figure size
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        # Use default values if not provided
        if values is None:
            values = self.strategy_values
        
        if initial_investment is None:
            initial_investment = self.initial_investment
            
        if title is None:
            title = f"{self.ticker} Options Strategy Final Values"
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        ax.hist(values[:, -1], bins=bins, alpha=0.7, color='skyblue')
        
        # Plot vertical lines
        mean_value = np.mean(values[:, -1])
        ax.axvline(x=initial_investment, color='red', linestyle=':', linewidth=1.5, label='Initial Investment')
        ax.axvline(x=mean_value, color='blue', linewidth=2, label=f'Mean: ${mean_value:.2f}')
        
        ax.set_xlabel('Final Strategy Value ($)')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
            
        return fig