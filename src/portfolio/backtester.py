"""
Backtesting Module for Portfolio Strategies
Implements backtesting framework for portfolio strategy validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PortfolioBacktester:
    """Backtesting framework for portfolio strategies"""
    
    def __init__(self, initial_capital: float = 100000, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.returns_data = None
        self.portfolio_values = []
        self.weights_history = []
        self.trades_history = []
        
    def set_data(self, returns_df: pd.DataFrame):
        """
        Set returns data for backtesting
        
        Args:
            returns_df: DataFrame with asset returns
        """
        self.returns_data = returns_df.dropna()
        logger.info(f"Data set for backtesting. Shape: {self.returns_data.shape}")
    
    def run_backtest(self, strategy_func: Callable, 
                    start_date: str = None, 
                    end_date: str = None,
                    rebalance_frequency: str = 'M') -> Dict:
        """
        Run backtest for a given strategy
        
        Args:
            strategy_func: Function that returns portfolio weights
            start_date: Start date for backtest
            end_date: End date for backtest
            rebalance_frequency: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            Dictionary with backtest results
        """
        if self.returns_data is None:
            raise ValueError("Data must be set before running backtest")
        
        logger.info("Starting portfolio backtest...")
        
        # Filter data by date range
        if start_date:
            self.returns_data = self.returns_data[self.returns_data.index >= start_date]
        if end_date:
            self.returns_data = self.returns_data[self.returns_data.index <= end_date]
        
        # Initialize tracking variables
        current_capital = self.initial_capital
        current_weights = None
        portfolio_value = self.initial_capital
        
        # Create date range for rebalancing
        rebalance_dates = self._get_rebalance_dates(rebalance_frequency)
        
        # Track performance
        self.portfolio_values = []
        self.weights_history = []
        self.trades_history = []
        
        for date in self.returns_data.index:
            # Check if rebalancing is needed
            if date in rebalance_dates or current_weights is None:
                # Get new weights from strategy
                try:
                    new_weights = strategy_func(self.returns_data.loc[:date])
                    if new_weights is None:
                        new_weights = current_weights if current_weights else self._equal_weight()
                except Exception as e:
                    logger.warning(f"Strategy failed on {date}: {str(e)}")
                    new_weights = current_weights if current_weights else self._equal_weight()
                
                # Calculate trades and transaction costs
                if current_weights is not None:
                    trades = self._calculate_trades(current_weights, new_weights, portfolio_value)
                    transaction_costs = sum(abs(trade) for trade in trades.values() if isinstance(trade, (int, float))) * self.transaction_cost
                    portfolio_value -= transaction_costs
                    
                    self.trades_history.append({
                        'date': date,
                        'trades': trades,
                        'transaction_costs': transaction_costs
                    })
                
                current_weights = new_weights
            
            # Calculate daily return
            if current_weights is not None:
                daily_return = (self.returns_data.loc[date] * pd.Series(current_weights)).sum()
                portfolio_value *= (1 + daily_return)
            
            # Record portfolio state
            self.portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'daily_return': daily_return if current_weights is not None else 0
            })
            
            if current_weights is not None:
                self.weights_history.append({
                    'date': date,
                    'weights': current_weights.copy()
                })
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics()
        
        logger.info("Backtest completed successfully")
        return results
    
    def _get_rebalance_dates(self, frequency: str) -> pd.DatetimeIndex:
        """Get rebalancing dates based on frequency"""
        if frequency == 'D':
            return self.returns_data.index
        elif frequency == 'W':
            return self.returns_data.resample('W').last().index
        elif frequency == 'M':
            return self.returns_data.resample('M').last().index
        elif frequency == 'Q':
            return self.returns_data.resample('Q').last().index
        elif frequency == 'Y':
            return self.returns_data.resample('Y').last().index
        else:
            raise ValueError("Invalid frequency. Use 'D', 'W', 'M', 'Q', or 'Y'")
    
    def _equal_weight(self) -> Dict[str, float]:
        """Generate equal weight portfolio"""
        n_assets = len(self.returns_data.columns)
        return {asset: 1/n_assets for asset in self.returns_data.columns}
    
    def _calculate_trades(self, current_weights: Dict[str, float], 
                         target_weights: Dict[str, float], 
                         portfolio_value: float) -> Dict[str, float]:
        """Calculate trades needed for rebalancing"""
        trades = {}
        
        for asset in target_weights.keys():
            current_weight = current_weights.get(asset, 0)
            target_weight = target_weights[asset]
            trade = target_weight - current_weight
            trades[asset] = trade
        
        return trades
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_values:
            return {}
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['cumulative_return'] = portfolio_df['portfolio_value'] / self.initial_capital - 1
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        
        # Calculate metrics
        total_return = portfolio_df['cumulative_return'].iloc[-1]
        annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
        volatility = portfolio_df['daily_return'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        portfolio_df['peak'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Calculate VaR and CVaR
        returns = portfolio_df['daily_return'].dropna()
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Calculate win rate
        positive_returns = (returns > 0).sum()
        total_returns = len(returns)
        win_rate = positive_returns / total_returns if total_returns > 0 else 0
        
        # Calculate benchmark comparison (equal weight)
        benchmark_returns = self.returns_data.mean(axis=1)
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        benchmark_total_return = benchmark_cumulative.iloc[-1] - 1
        
        # Information ratio
        excess_returns = portfolio_df['daily_return'] - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        metrics = {
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'Win_Rate': win_rate,
            'Benchmark_Return': benchmark_total_return,
            'Excess_Return': total_return - benchmark_total_return,
            'Information_Ratio': information_ratio,
            'Final_Portfolio_Value': portfolio_df['portfolio_value'].iloc[-1]
        }
        
        return metrics
    
    def plot_performance(self, benchmark_returns: pd.Series = None):
        """Plot backtest performance"""
        try:
            import matplotlib.pyplot as plt
            
            portfolio_df = pd.DataFrame(self.portfolio_values)
            portfolio_df.set_index('date', inplace=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Portfolio value over time
            axes[0, 0].plot(portfolio_df.index, portfolio_df['portfolio_value'])
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True)
            
            # Cumulative returns
            cumulative_returns = portfolio_df['portfolio_value'] / self.initial_capital - 1
            axes[0, 1].plot(portfolio_df.index, cumulative_returns, label='Strategy')
            
            if benchmark_returns is not None:
                benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
                axes[0, 1].plot(benchmark_returns.index, benchmark_cumulative, label='Benchmark')
            
            axes[0, 1].set_title('Cumulative Returns')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Cumulative Return')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Drawdown
            portfolio_df['peak'] = portfolio_df['portfolio_value'].expanding().max()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
            axes[1, 0].fill_between(portfolio_df.index, portfolio_df['drawdown'], 0, alpha=0.3, color='red')
            axes[1, 0].set_title('Drawdown')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Drawdown')
            axes[1, 0].grid(True)
            
            # Daily returns distribution
            returns = portfolio_df['daily_return'].dropna()
            axes[1, 1].hist(returns, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Daily Returns Distribution')
            axes[1, 1].set_xlabel('Daily Return')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def get_weights_summary(self) -> pd.DataFrame:
        """Get summary of portfolio weights over time"""
        if not self.weights_history:
            return pd.DataFrame()
        
        weights_df = pd.DataFrame([wh['weights'] for wh in self.weights_history])
        weights_df.index = [wh['date'] for wh in self.weights_history]
        
        return weights_df
    
    def compare_strategies(self, strategies: Dict[str, Callable], 
                          start_date: str = None, 
                          end_date: str = None) -> pd.DataFrame:
        """
        Compare multiple strategies
        
        Args:
            strategies: Dictionary of strategy functions
            start_date: Start date for comparison
            end_date: End date for comparison
            
        Returns:
            DataFrame with comparison results
        """
        results = {}
        
        for strategy_name, strategy_func in strategies.items():
            logger.info(f"Testing strategy: {strategy_name}")
            
            # Reset backtester
            self.portfolio_values = []
            self.weights_history = []
            self.trades_history = []
            
            # Run backtest
            try:
                strategy_results = self.run_backtest(strategy_func, start_date, end_date)
                results[strategy_name] = strategy_results
            except Exception as e:
                logger.error(f"Strategy {strategy_name} failed: {str(e)}")
                results[strategy_name] = {}
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results).T
        
        return results_df

def create_buy_and_hold_strategy(weights: Dict[str, float]) -> Callable:
    """Create a buy and hold strategy"""
    def strategy(data):
        return weights
    return strategy

def create_equal_weight_strategy() -> Callable:
    """Create an equal weight strategy"""
    def strategy(data):
        n_assets = len(data.columns)
        return {asset: 1/n_assets for asset in data.columns}
    return strategy

def create_momentum_strategy(lookback: int = 60) -> Callable:
    """Create a momentum-based strategy"""
    def strategy(data):
        if len(data) < lookback:
            return None
        
        # Calculate momentum (past returns)
        momentum = data.tail(lookback).mean()
        
        # Select top 2 assets
        top_assets = momentum.nlargest(2)
        
        # Equal weight among top assets
        weights = {asset: 0.5 for asset in top_assets.index}
        
        # Add cash for remaining assets
        for asset in data.columns:
            if asset not in weights:
                weights[asset] = 0
        
        return weights
    
    return strategy

def main():
    """Main function to demonstrate backtesting"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.collector import FinancialDataCollector
    from data.preprocessor import FinancialDataPreprocessor
    from portfolio.optimizer import PortfolioOptimizer
    
    # Collect and preprocess data
    collector = FinancialDataCollector()
    raw_data = collector.get_combined_data()
    
    if raw_data.empty:
        print("No data available")
        return
    
    preprocessor = FinancialDataPreprocessor()
    clean_data = preprocessor.clean_data(raw_data)
    returns_data = preprocessor.prepare_returns_data(clean_data)
    
    # Initialize backtester
    backtester = PortfolioBacktester(initial_capital=100000, transaction_cost=0.001)
    backtester.set_data(returns_data)
    
    # Create strategies
    strategies = {
        'Equal_Weight': create_equal_weight_strategy(),
        'Momentum': create_momentum_strategy(lookback=60),
        'Buy_and_Hold_TSLA': create_buy_and_hold_strategy({'TSLA': 1.0, 'BND': 0.0, 'SPY': 0.0})
    }
    
    # Run comparison
    results = backtester.compare_strategies(strategies, start_date='2020-01-01')
    
    print("Strategy Comparison Results:")
    print(results)
    
    # Run detailed backtest for one strategy
    print("\nDetailed Backtest for Equal Weight Strategy:")
    equal_weight_results = backtester.run_backtest(
        create_equal_weight_strategy(), 
        start_date='2020-01-01'
    )
    
    for metric, value in equal_weight_results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 