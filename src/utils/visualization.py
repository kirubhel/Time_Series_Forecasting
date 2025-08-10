"""
Visualization Utilities for Portfolio Analysis
Provides plotting functions for financial data analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set default style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_price_evolution(data: pd.DataFrame, symbols: List[str] = None, 
                        start_date: str = None, end_date: str = None):
    """Plot price evolution for multiple assets"""
    if symbols is None:
        symbols = data['Symbol'].unique()
    
    plt.figure(figsize=(15, 8))
    
    for symbol in symbols:
        symbol_data = data[data['Symbol'] == symbol]
        if start_date:
            symbol_data = symbol_data[symbol_data['Date'] >= start_date]
        if end_date:
            symbol_data = symbol_data[symbol_data['Date'] <= end_date]
        
        plt.plot(symbol_data['Date'], symbol_data['Close'], label=symbol, linewidth=2)
    
    plt.title('Asset Price Evolution', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_returns_analysis(data: pd.DataFrame, symbol: str):
    """Plot comprehensive returns analysis for a single asset"""
    asset_data = data[data['Symbol'] == symbol]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Daily returns over time
    axes[0, 0].plot(asset_data['Date'], asset_data['Daily_Return'], alpha=0.7)
    axes[0, 0].set_title(f'{symbol} Daily Returns', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Daily Return')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Returns histogram
    axes[0, 1].hist(asset_data['Daily_Return'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title(f'{symbol} Returns Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Daily Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Volatility over time
    axes[1, 0].plot(asset_data['Date'], asset_data['Volatility_20d'], label='20-day Volatility')
    axes[1, 0].plot(asset_data['Date'], asset_data['Volatility_60d'], label='60-day Volatility')
    axes[1, 0].set_title(f'{symbol} Rolling Volatility', fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Volatility')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Price with moving averages
    axes[1, 1].plot(asset_data['Date'], asset_data['Close'], label='Close Price', alpha=0.8)
    axes[1, 1].plot(asset_data['Date'], asset_data['MA_20'], label='20-day MA', alpha=0.8)
    axes[1, 1].plot(asset_data['Date'], asset_data['MA_60'], label='60-day MA', alpha=0.8)
    axes[1, 1].set_title(f'{symbol} Price with Moving Averages', fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(returns_data: pd.DataFrame):
    """Plot correlation matrix heatmap"""
    correlation_matrix = returns_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Asset Returns Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_efficient_frontier(frontier_data: pd.DataFrame, optimal_portfolio: Dict = None):
    """Plot efficient frontier with optimal portfolio points"""
    plt.figure(figsize=(12, 8))
    
    # Plot efficient frontier
    plt.scatter(frontier_data['Volatility'], frontier_data['Return'], 
               alpha=0.6, s=30, label='Efficient Frontier')
    
    # Mark optimal portfolios
    if optimal_portfolio:
        plt.scatter(optimal_portfolio['Volatility'], optimal_portfolio['Return'], 
                   color='red', s=200, marker='*', label='Optimal Portfolio')
    
    plt.xlabel('Portfolio Volatility', fontsize=12)
    plt.ylabel('Portfolio Return', fontsize=12)
    plt.title('Efficient Frontier', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_backtest_results(portfolio_values: List[Dict], benchmark_values: List[Dict] = None):
    """Plot backtest performance comparison"""
    portfolio_df = pd.DataFrame(portfolio_values)
    portfolio_df.set_index('date', inplace=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio value over time
    axes[0, 0].plot(portfolio_df.index, portfolio_df['portfolio_value'], label='Strategy')
    if benchmark_values:
        benchmark_df = pd.DataFrame(benchmark_values)
        benchmark_df.set_index('date', inplace=True)
        axes[0, 0].plot(benchmark_df.index, benchmark_df['portfolio_value'], label='Benchmark')
    axes[0, 0].set_title('Portfolio Value Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative returns
    cumulative_returns = portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0] - 1
    axes[0, 1].plot(portfolio_df.index, cumulative_returns, label='Strategy')
    if benchmark_values:
        benchmark_cumulative = benchmark_df['portfolio_value'] / benchmark_df['portfolio_value'].iloc[0] - 1
        axes[0, 1].plot(benchmark_df.index, benchmark_cumulative, label='Benchmark')
    axes[0, 1].set_title('Cumulative Returns', fontweight='bold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Cumulative Return')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Drawdown
    portfolio_df['peak'] = portfolio_df['portfolio_value'].expanding().max()
    portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
    axes[1, 0].fill_between(portfolio_df.index, portfolio_df['drawdown'], 0, alpha=0.3, color='red')
    axes[1, 0].set_title('Drawdown', fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Drawdown')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Daily returns distribution
    returns = portfolio_df['daily_return'].dropna()
    axes[1, 1].hist(returns, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Daily Returns Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Daily Return')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_forecast_comparison(historical_data: pd.Series, predictions: pd.Series, 
                           confidence_intervals: pd.DataFrame = None):
    """Plot forecast comparison with historical data"""
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(historical_data.index, historical_data.values, label='Historical', linewidth=2)
    
    # Plot predictions
    plt.plot(predictions.index, predictions.values, label='Forecast', linewidth=2, color='red')
    
    # Plot confidence intervals if available
    if confidence_intervals is not None:
        plt.fill_between(predictions.index, 
                        confidence_intervals.iloc[:, 0], 
                        confidence_intervals.iloc[:, 1], 
                        alpha=0.3, color='red', label='Confidence Interval')
    
    plt.title('Time Series Forecast', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def create_summary_table(metrics: Dict) -> pd.DataFrame:
    """Create a formatted summary table from metrics dictionary"""
    df = pd.DataFrame(metrics).T
    
    # Format percentages
    percentage_columns = ['Total_Return', 'Annualized_Return', 'Volatility', 'Max_Drawdown']
    for col in percentage_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
    
    # Format other metrics
    if 'Sharpe_Ratio' in df.columns:
        df['Sharpe_Ratio'] = df['Sharpe_Ratio'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
    
    return df 