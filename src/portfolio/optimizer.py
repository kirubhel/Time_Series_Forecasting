"""
Portfolio Optimization Module
Implements Modern Portfolio Theory (MPT) for portfolio optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.objective_functions import negative_sharpe_ratio
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Portfolio optimizer using Modern Portfolio Theory"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.expected_returns = None
        self.covariance_matrix = None
        self.ef = None
        
    def prepare_data(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare returns data for optimization
        
        Args:
            returns_df: DataFrame with asset returns
            
        Returns:
            Prepared returns data
        """
        logger.info("Preparing data for portfolio optimization...")
        
        # Remove any missing values
        returns_clean = returns_df.dropna()
        
        # Convert to daily returns if needed
        if returns_clean.max().max() > 1:
            returns_clean = returns_clean / 100
        
        self.returns_data = returns_clean
        logger.info(f"Data prepared. Shape: {returns_clean.shape}")
        
        return returns_clean
    
    def calculate_expected_returns(self, method: str = 'mean') -> pd.Series:
        """
        Calculate expected returns for assets
        
        Args:
            method: Method for calculating expected returns ('mean', 'capm', 'ema')
            
        Returns:
            Expected returns series
        """
        if self.returns_data is None:
            raise ValueError("Data must be prepared before calculating expected returns")
        
        logger.info(f"Calculating expected returns using {method} method...")
        
        if method == 'mean':
            # Simple historical mean
            exp_returns = self.returns_data.mean() * 252  # Annualize
        elif method == 'ema':
            # Exponential moving average
            exp_returns = self.returns_data.ewm(span=252).mean().iloc[-1] * 252
        elif method == 'capm':
            # CAPM-based expected returns
            exp_returns = self._calculate_capm_returns()
        else:
            raise ValueError("Method must be 'mean', 'ema', or 'capm'")
        
        self.expected_returns = exp_returns
        logger.info("Expected returns calculated successfully")
        
        return exp_returns
    
    def _calculate_capm_returns(self) -> pd.Series:
        """Calculate CAPM-based expected returns"""
        # Use market return (SPY) as benchmark
        if 'SPY' not in self.returns_data.columns:
            raise ValueError("SPY data required for CAPM calculation")
        
        market_returns = self.returns_data['SPY']
        market_return = market_returns.mean() * 252
        
        capm_returns = pd.Series(index=self.returns_data.columns)
        
        for asset in self.returns_data.columns:
            if asset == 'SPY':
                capm_returns[asset] = market_return
            else:
                # Calculate beta
                asset_returns = self.returns_data[asset]
                covariance = np.cov(asset_returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance
                
                # CAPM formula: E(Ri) = Rf + βi(E(Rm) - Rf)
                capm_returns[asset] = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
        
        return capm_returns
    
    def calculate_covariance_matrix(self, method: str = 'sample') -> pd.DataFrame:
        """
        Calculate covariance matrix
        
        Args:
            method: Method for covariance estimation ('sample', 'ledoit_wolf', 'oas')
            
        Returns:
            Covariance matrix
        """
        if self.returns_data is None:
            raise ValueError("Data must be prepared before calculating covariance")
        
        logger.info(f"Calculating covariance matrix using {method} method...")
        
        if method == 'sample':
            # Sample covariance
            cov_matrix = self.returns_data.cov() * 252  # Annualize
        elif method == 'ledoit_wolf':
            # Ledoit-Wolf shrinkage
            cov_matrix = CovarianceShrinkage(self.returns_data).ledoit_wolf()
        elif method == 'oas':
            # Oracle Approximating Shrinkage
            cov_matrix = CovarianceShrinkage(self.returns_data).oracle_approximating()
        else:
            raise ValueError("Method must be 'sample', 'ledoit_wolf', or 'oas'")
        
        self.covariance_matrix = cov_matrix
        logger.info("Covariance matrix calculated successfully")
        
        return cov_matrix
    
    def create_efficient_frontier(self) -> EfficientFrontier:
        """
        Create efficient frontier object
        
        Returns:
            EfficientFrontier object
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Expected returns and covariance matrix must be calculated first")
        
        logger.info("Creating efficient frontier...")
        
        self.ef = EfficientFrontier(
            expected_returns=self.expected_returns,
            cov_matrix=self.covariance_matrix,
            weight_bounds=(0, 1)
        )
        
        logger.info("Efficient frontier created successfully")
        return self.ef
    
    def optimize_portfolio(self, objective: str = 'sharpe') -> Dict[str, float]:
        """
        Optimize portfolio weights
        
        Args:
            objective: Optimization objective ('sharpe', 'min_volatility', 'max_return')
            
        Returns:
            Dictionary with optimal weights
        """
        if self.ef is None:
            self.create_efficient_frontier()
        
        logger.info(f"Optimizing portfolio for {objective}...")
        
        if objective == 'sharpe':
            weights = self.ef.max_sharpe()
        elif objective == 'min_volatility':
            weights = self.ef.min_volatility()
        elif objective == 'max_return':
            weights = self.ef.max_quadratic_utility()
        else:
            raise ValueError("Objective must be 'sharpe', 'min_volatility', or 'max_return'")
        
        # Clean weights
        weights = self.ef.clean_weights()
        
        logger.info("Portfolio optimization completed")
        return weights
    
    def get_portfolio_performance(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary with performance metrics
        """
        if self.ef is None:
            raise ValueError("Efficient frontier must be created first")
        
        # Set weights
        self.ef.set_weights(weights)
        
        # Calculate metrics
        performance = self.ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
        
        metrics = {
            'Expected_Return': performance[0],
            'Volatility': performance[1],
            'Sharpe_Ratio': performance[2]
        }
        
        return metrics
    
    def generate_efficient_frontier_points(self, num_portfolios: int = 100) -> pd.DataFrame:
        """
        Generate points along the efficient frontier
        
        Args:
            num_portfolios: Number of portfolios to generate
            
        Returns:
            DataFrame with frontier points
        """
        if self.ef is None:
            self.create_efficient_frontier()
        
        logger.info(f"Generating {num_portfolios} efficient frontier points...")
        
        # Generate efficient frontier
        frontier = self.ef.efficient_frontier(num_portfolios=num_portfolios)
        
        # Convert to DataFrame
        frontier_df = pd.DataFrame(frontier)
        frontier_df.columns = ['Return', 'Volatility', 'Sharpe_Ratio'] + list(self.returns_data.columns)
        
        logger.info("Efficient frontier points generated")
        return frontier_df
    
    def calculate_value_at_risk(self, weights: Dict[str, float], confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            weights: Portfolio weights
            confidence_level: VaR confidence level
            
        Returns:
            VaR value
        """
        if self.returns_data is None:
            raise ValueError("Returns data must be available")
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns_data * pd.Series(weights)).sum(axis=1)
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, confidence_level * 100)
        
        return var
    
    def calculate_conditional_var(self, weights: Dict[str, float], confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR)
        
        Args:
            weights: Portfolio weights
            confidence_level: CVaR confidence level
            
        Returns:
            CVaR value
        """
        if self.returns_data is None:
            raise ValueError("Returns data must be available")
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns_data * pd.Series(weights)).sum(axis=1)
        
        # Calculate CVaR
        var = np.percentile(portfolio_returns, confidence_level * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        return cvar
    
    def rebalance_portfolio(self, current_weights: Dict[str, float], 
                           target_weights: Dict[str, float], 
                           transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Calculate rebalancing trades considering transaction costs
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            transaction_cost: Transaction cost as percentage
            
        Returns:
            Dictionary with rebalancing trades
        """
        trades = {}
        total_cost = 0
        
        for asset in target_weights.keys():
            current_weight = current_weights.get(asset, 0)
            target_weight = target_weights[asset]
            
            trade = target_weight - current_weight
            trades[asset] = trade
            
            if abs(trade) > 0.001:  # Minimum trade threshold
                total_cost += abs(trade) * transaction_cost
        
        trades['Total_Transaction_Cost'] = total_cost
        
        return trades
    
    def get_asset_allocation_summary(self, weights: Dict[str, float]) -> pd.DataFrame:
        """
        Get detailed asset allocation summary
        
        Args:
            weights: Portfolio weights
            
        Returns:
            DataFrame with allocation summary
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Expected returns and covariance must be calculated first")
        
        summary_data = []
        
        for asset, weight in weights.items():
            if weight > 0:
                expected_return = self.expected_returns[asset]
                volatility = np.sqrt(self.covariance_matrix.loc[asset, asset])
                
                summary_data.append({
                    'Asset': asset,
                    'Weight': weight,
                    'Expected_Return': expected_return,
                    'Volatility': volatility,
                    'Contribution_to_Return': weight * expected_return,
                    'Contribution_to_Risk': weight * volatility
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Weight', ascending=False)
        
        return summary_df

def main():
    """Main function to demonstrate portfolio optimization"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.collector import FinancialDataCollector
    from data.preprocessor import FinancialDataPreprocessor
    
    # Collect and preprocess data
    collector = FinancialDataCollector()
    raw_data = collector.get_combined_data()
    
    if raw_data.empty:
        print("No data available")
        return
    
    preprocessor = FinancialDataPreprocessor()
    clean_data = preprocessor.clean_data(raw_data)
    returns_data = preprocessor.prepare_returns_data(clean_data)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Prepare data
    optimizer.prepare_data(returns_data)
    
    # Calculate expected returns and covariance
    expected_returns = optimizer.calculate_expected_returns(method='mean')
    covariance_matrix = optimizer.calculate_covariance_matrix(method='sample')
    
    print("Expected Returns (Annualized):")
    print(expected_returns)
    print("\nCovariance Matrix (Annualized):")
    print(covariance_matrix)
    
    # Optimize portfolio
    optimal_weights = optimizer.optimize_portfolio(objective='sharpe')
    performance = optimizer.get_portfolio_performance(optimal_weights)
    
    print("\nOptimal Portfolio Weights:")
    for asset, weight in optimal_weights.items():
        print(f"{asset}: {weight:.4f}")
    
    print("\nPortfolio Performance:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate efficient frontier
    frontier_df = optimizer.generate_efficient_frontier_points(num_portfolios=50)
    print(f"\nEfficient Frontier generated with {len(frontier_df)} points")

if __name__ == "__main__":
    main() 