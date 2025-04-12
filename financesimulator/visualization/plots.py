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
        price_paths = self.results['price_paths']
        horizon = self.results['horizon']
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
        price_paths = self.results['price_paths']
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
        price_paths = self.results['price_paths']
        horizon = self.results['horizon']
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
        price_paths = self.results['price_paths']
        horizon = self.results['horizon']
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
        Generate a summary dashboard with multiple plots.
        
        Args:
            num_paths: Number of individual paths to plot
            confidence_intervals: List of percentiles for confidence intervals
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        # Extract data
        price_paths = self.results['price_paths']
        horizon = self.results['horizon']
        ticker = self.results.get('ticker', 'Asset')
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Price paths with confidence intervals
        ax1 = axs[0, 0]
        time_steps = np.arange(horizon + 1)
        
        # Plot a subset of individual paths
        num_paths_to_plot = min(num_paths, price_paths.shape[0])
        for i in range(num_paths_to_plot):
            ax1.plot(time_steps, price_paths[i, :], color='skyblue', alpha=0.3, linewidth=0.8)
        
        # Plot confidence intervals
        percentiles = np.percentile(price_paths, [p * 100 for p in confidence_intervals], axis=0)
        colors = sns.color_palette("flare", n_colors=len(confidence_intervals))
        
        for i, p in enumerate(confidence_intervals):
            ax1.plot(time_steps, percentiles[i], color=colors[i], linewidth=2, 
                    label=f"{int(p * 100)}th percentile")
        
        # Plot median
        median = np.median(price_paths, axis=0)
        ax1.plot(time_steps, median, color='darkblue', linewidth=2.5, label='Median')
        
        ax1.set_title(f'{ticker} Price Paths')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Final price histogram
        ax2 = axs[0, 1]
        final_prices = price_paths[:, -1]
        
        sns.histplot(final_prices, bins=50, kde=True, ax=ax2, color='skyblue')
        
        mean_price = np.mean(final_prices)
        ax2.axvline(mean_price, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_price:.2f}')
        
        median_price = np.median(final_prices)
        ax2.axvline(median_price, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_price:.2f}')
        
        ax2.set_title(f'Distribution of {ticker} Final Prices')
        ax2.set_xlabel('Price')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Plot 3: Returns distribution
        ax3 = axs[1, 0]
        
        initial_prices = price_paths[:, 0]
        returns = (final_prices / initial_prices) - 1
        
        # Annualize returns
        annual_factor = 252 / horizon
        annualized_returns = (1 + returns) ** annual_factor - 1
        
        sns.histplot(annualized_returns, bins=50, kde=True, ax=ax3, color='skyblue')
        
        mean_return = np.mean(annualized_returns)
        ax3.axvline(mean_return, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_return:.2%}')
        
        median_return = np.median(annualized_returns)
        ax3.axvline(median_return, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_return:.2%}')
        
        ax3.set_title(f'Distribution of {ticker} Annualized Returns')
        ax3.set_xlabel('Return')
        ax3.set_ylabel('Frequency')
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax3.legend()
        
        # Plot 4: Heatmap (sampled paths)
        ax4 = axs[1, 1]
        
        # Sample paths for better visualization
        sample_size = min(100, price_paths.shape[0])
        sampled_paths = price_paths[np.random.choice(price_paths.shape[0], sample_size, replace=False)]
        
        sns.heatmap(sampled_paths, cmap='viridis', ax=ax4)
        ax4.set_title(f'Heatmap of {ticker} Price Paths')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Path')
        
        # Adjust layout
        plt.tight_layout()
        
        # Add overall title
        fig.suptitle(f'Monte Carlo Simulation Summary: {ticker}', fontsize=16, y=1.02)
        
        # Save figure if specified
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show plot if specified
        if show_plot:
            plt.show()
            
        return fig