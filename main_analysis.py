#!/usr/bin/env python3
"""
Main Analysis Script for Portfolio Management Optimization
Runs the complete pipeline from data collection to portfolio optimization
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

# Import our modules
from data.collector import FinancialDataCollector
from data.preprocessor import FinancialDataPreprocessor
from models.arima_model import ARIMAModel
from models.lstm_model import LSTMModel
from portfolio.optimizer import PortfolioOptimizer
from portfolio.backtester import PortfolioBacktester
from utils.visualization import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PortfolioAnalysisPipeline:
    """Main pipeline for portfolio analysis and optimization"""
    
    def __init__(self):
        self.collector = FinancialDataCollector()
        self.preprocessor = FinancialDataPreprocessor()
        self.raw_data = None
        self.clean_data = None
        self.feature_data = None
        self.returns_data = None
        
    def run_data_collection(self):
        """Step 1: Collect and prepare data"""
        logger.info("="*60)
        logger.info("STEP 1: DATA COLLECTION AND PREPARATION")
        logger.info("="*60)
        
        # Collect data
        logger.info("Collecting financial data...")
        self.raw_data = self.collector.get_combined_data()
        
        if self.raw_data.empty:
            logger.error("No data collected. Exiting.")
            return False
        
        logger.info(f"Data collected successfully. Shape: {self.raw_data.shape}")
        logger.info(f"Date range: {self.raw_data['Date'].min()} to {self.raw_data['Date'].max()}")
        logger.info(f"Assets: {self.raw_data['Symbol'].unique()}")
        
        # Clean and preprocess data
        logger.info("Cleaning and preprocessing data...")
        self.clean_data = self.preprocessor.clean_data(self.raw_data)
        self.feature_data = self.preprocessor.engineer_features(self.clean_data)
        self.returns_data = self.preprocessor.prepare_returns_data(self.clean_data)
        
        logger.info(f"Data preprocessing completed:")
        logger.info(f"  - Clean data: {self.clean_data.shape}")
        logger.info(f"  - Feature data: {self.feature_data.shape}")
        logger.info(f"  - Returns data: {self.returns_data.shape}")
        
        return True
    
    def run_exploratory_analysis(self):
        """Step 2: Exploratory data analysis"""
        logger.info("="*60)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
        logger.info("="*60)
        
        # Basic statistics
        logger.info("Calculating basic statistics...")
        stats_by_asset = self.clean_data.groupby('Symbol')['Close'].describe()
        logger.info("\nAsset Statistics:")
        logger.info(stats_by_asset)
        
        # Risk metrics
        logger.info("\nCalculating risk metrics...")
        risk_metrics = {}
        for symbol in ['TSLA', 'BND', 'SPY']:
            asset_data = self.feature_data[self.feature_data['Symbol'] == symbol]
            returns = asset_data['Daily_Return'].dropna()
            
            risk_metrics[symbol] = {
                'Mean_Return': returns.mean() * 252,
                'Volatility': returns.std() * np.sqrt(252),
                'Sharpe_Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'Max_Drawdown': (asset_data['Close'] / asset_data['Close'].expanding().max() - 1).min()
            }
        
        risk_df = pd.DataFrame(risk_metrics).T
        logger.info("\nRisk Metrics:")
        logger.info(risk_df)
        
        # Stationarity tests
        logger.info("\nPerforming stationarity tests...")
        for symbol in ['TSLA', 'BND', 'SPY']:
            asset_data = self.feature_data[self.feature_data['Symbol'] == symbol]
            price_test = self.preprocessor.test_stationarity(asset_data['Close'], f"{symbol} Price")
            returns_test = self.preprocessor.test_stationarity(asset_data['Daily_Return'], f"{symbol} Returns")
            
            logger.info(f"{symbol}: Price stationary: {price_test['is_stationary']}, "
                       f"Returns stationary: {returns_test['is_stationary']}")
        
        # Correlation analysis
        logger.info("\nCorrelation analysis:")
        correlation_matrix = self.returns_data.corr()
        logger.info(correlation_matrix)
        
        return True
    
    def run_forecasting_models(self):
        """Step 3: Time series forecasting"""
        logger.info("="*60)
        logger.info("STEP 3: TIME SERIES FORECASTING")
        logger.info("="*60)
        
        # Focus on TSLA for forecasting
        tsla_data = self.clean_data[self.clean_data['Symbol'] == 'TSLA'].set_index('Date')['Close']
        
        # Split data for training/testing
        train_size = int(len(tsla_data) * 0.8)
        train_data = tsla_data[:train_size]
        test_data = tsla_data[train_size:]
        
        logger.info(f"Training data: {len(train_data)} observations")
        logger.info(f"Test data: {len(test_data)} observations")
        
        # ARIMA Model
        logger.info("\nTraining ARIMA model...")
        try:
            arima_model = ARIMAModel()
            arima_model.fit(train_data, auto_order=True)
            
            # Evaluate ARIMA
            arima_metrics = arima_model.evaluate(test_data)
            logger.info("ARIMA Model Performance:")
            for metric, value in arima_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Make future predictions
            arima_predictions, arima_conf_int = arima_model.predict(steps=30)
            logger.info(f"ARIMA future predictions (next 30 days):")
            logger.info(f"  Mean: {arima_predictions.mean():.2f}")
            logger.info(f"  Range: {arima_predictions.min():.2f} - {arima_predictions.max():.2f}")
            
        except Exception as e:
            logger.error(f"ARIMA model failed: {str(e)}")
        
        # LSTM Model
        logger.info("\nTraining LSTM model...")
        try:
            lstm_model = LSTMModel(lookback=60, units=50, dropout=0.2)
            lstm_model.fit(train_data, epochs=50, batch_size=32)
            
            # Evaluate LSTM
            lstm_metrics = lstm_model.evaluate(test_data)
            logger.info("LSTM Model Performance:")
            for metric, value in lstm_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Make future predictions
            lstm_predictions, _ = lstm_model.predict(tsla_data, steps=30)
            logger.info(f"LSTM future predictions (next 30 days):")
            logger.info(f"  Mean: {lstm_predictions.mean():.2f}")
            logger.info(f"  Range: {lstm_predictions.min():.2f} - {lstm_predictions.max():.2f}")
            
        except Exception as e:
            logger.error(f"LSTM model failed: {str(e)}")
        
        return True
    
    def run_portfolio_optimization(self):
        """Step 4: Portfolio optimization"""
        logger.info("="*60)
        logger.info("STEP 4: PORTFOLIO OPTIMIZATION")
        logger.info("="*60)
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        
        # Prepare data
        optimizer.prepare_data(self.returns_data)
        
        # Calculate expected returns and covariance
        expected_returns = optimizer.calculate_expected_returns(method='mean')
        covariance_matrix = optimizer.calculate_covariance_matrix(method='sample')
        
        logger.info("Expected Returns (Annualized):")
        logger.info(expected_returns)
        
        logger.info("\nCovariance Matrix (Annualized):")
        logger.info(covariance_matrix)
        
        # Optimize portfolios
        logger.info("\nOptimizing portfolios...")
        
        # Maximum Sharpe Ratio portfolio
        sharpe_weights = optimizer.optimize_portfolio(objective='sharpe')
        sharpe_performance = optimizer.get_portfolio_performance(sharpe_weights)
        
        logger.info("Maximum Sharpe Ratio Portfolio:")
        for asset, weight in sharpe_weights.items():
            logger.info(f"  {asset}: {weight:.4f}")
        logger.info("Performance:")
        for metric, value in sharpe_performance.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Minimum volatility portfolio
        min_vol_weights = optimizer.optimize_portfolio(objective='min_volatility')
        min_vol_performance = optimizer.get_portfolio_performance(min_vol_weights)
        
        logger.info("\nMinimum Volatility Portfolio:")
        for asset, weight in min_vol_weights.items():
            logger.info(f"  {asset}: {weight:.4f}")
        logger.info("Performance:")
        for metric, value in min_vol_performance.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Generate efficient frontier
        logger.info("\nGenerating efficient frontier...")
        frontier_df = optimizer.generate_efficient_frontier_points(num_portfolios=100)
        logger.info(f"Efficient frontier generated with {len(frontier_df)} points")
        
        return {
            'sharpe_weights': sharpe_weights,
            'min_vol_weights': min_vol_weights,
            'sharpe_performance': sharpe_performance,
            'min_vol_performance': min_vol_performance,
            'frontier_df': frontier_df
        }
    
    def run_backtesting(self, optimal_weights):
        """Step 5: Strategy backtesting"""
        logger.info("="*60)
        logger.info("STEP 5: STRATEGY BACKTESTING")
        logger.info("="*60)
        
        # Initialize backtester
        backtester = PortfolioBacktester(initial_capital=100000, transaction_cost=0.001)
        backtester.set_data(self.returns_data)
        
        # Create strategies
        from portfolio.backtester import create_equal_weight_strategy, create_buy_and_hold_strategy
        
        strategies = {
            'Equal_Weight': create_equal_weight_strategy(),
            'Max_Sharpe': lambda data: optimal_weights['sharpe_weights'],
            'Min_Volatility': lambda data: optimal_weights['min_vol_weights'],
            'Buy_and_Hold_TSLA': create_buy_and_hold_strategy({'TSLA': 1.0, 'BND': 0.0, 'SPY': 0.0})
        }
        
        # Run backtests
        logger.info("Running backtests...")
        results = backtester.compare_strategies(strategies, start_date='2020-01-01')
        
        logger.info("Backtest Results:")
        logger.info(results)
        
        return results
    
    def generate_report(self, optimization_results, backtest_results):
        """Generate final analysis report"""
        logger.info("="*60)
        logger.info("FINAL ANALYSIS REPORT")
        logger.info("="*60)
        
        # Create summary
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_summary': {
                'total_observations': len(self.clean_data),
                'date_range': f"{self.clean_data['Date'].min()} to {self.clean_data['Date'].max()}",
                'assets': list(self.clean_data['Symbol'].unique())
            },
            'optimization_results': optimization_results,
            'backtest_results': backtest_results
        }
        
        logger.info("Analysis completed successfully!")
        logger.info(f"Report generated on: {report['analysis_date']}")
        
        return report
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        logger.info("Starting Portfolio Management Optimization Analysis")
        logger.info("="*80)
        
        try:
            # Step 1: Data collection
            if not self.run_data_collection():
                return None
            
            # Step 2: Exploratory analysis
            self.run_exploratory_analysis()
            
            # Step 3: Forecasting
            self.run_forecasting_models()
            
            # Step 4: Portfolio optimization
            optimization_results = self.run_portfolio_optimization()
            
            # Step 5: Backtesting
            backtest_results = self.run_backtesting(optimization_results)
            
            # Generate final report
            report = self.generate_report(optimization_results, backtest_results)
            
            logger.info("="*80)
            logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return None

def main():
    """Main function"""
    # Create and run pipeline
    pipeline = PortfolioAnalysisPipeline()
    report = pipeline.run_complete_analysis()
    
    if report:
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        print(f"Analysis completed on: {report['analysis_date']}")
        print(f"Data analyzed: {report['data_summary']['total_observations']} observations")
        print(f"Assets: {', '.join(report['data_summary']['assets'])}")
        print("="*80)
    else:
        print("Analysis failed. Check logs for details.")

if __name__ == "__main__":
    main() 