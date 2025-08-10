"""
ARIMA Model for Time Series Forecasting
Implements ARIMA and SARIMA models for financial time series prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ARIMAModel:
    """ARIMA model for time series forecasting"""
    
    def __init__(self, order: Tuple[int, int, int] = None, seasonal_order: Tuple[int, int, int, int] = None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        
    def find_optimal_order(self, series: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5,
                          seasonal: bool = True, m: int = 12) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]]]:
        """
        Find optimal ARIMA order using auto_arima
        
        Args:
            series: Time series data
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            seasonal: Whether to include seasonal components
            m: Seasonal period
            
        Returns:
            Tuple of (order, seasonal_order)
        """
        logger.info("Finding optimal ARIMA order...")
        
        try:
            # Use auto_arima to find optimal parameters
            auto_model = auto_arima(
                series,
                start_p=0, start_q=0,
                max_p=max_p, max_d=max_d, max_q=max_q,
                seasonal=seasonal,
                m=m,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )
            
            order = auto_model.order
            seasonal_order = auto_model.seasonal_order if seasonal else None
            
            logger.info(f"Optimal order: {order}")
            if seasonal_order:
                logger.info(f"Optimal seasonal order: {seasonal_order}")
            
            return order, seasonal_order
            
        except Exception as e:
            logger.error(f"Error finding optimal order: {str(e)}")
            # Return default order
            return (1, 1, 1), None
    
    def check_stationarity(self, series: pd.Series) -> Dict:
        """
        Check if series is stationary using Augmented Dickey-Fuller test
        
        Args:
            series: Time series to test
            
        Returns:
            Dictionary with test results
        """
        result = adfuller(series.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def make_stationary(self, series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """
        Make series stationary by differencing
        
        Args:
            series: Time series to make stationary
            max_diff: Maximum number of differences to apply
            
        Returns:
            Tuple of (stationary_series, number_of_differences)
        """
        diff_count = 0
        current_series = series.copy()
        
        for i in range(max_diff + 1):
            # Check stationarity
            stationarity_result = self.check_stationarity(current_series)
            
            if stationarity_result['is_stationary']:
                logger.info(f"Series is stationary after {diff_count} differences")
                break
            
            if i < max_diff:
                # Apply differencing
                current_series = current_series.diff().dropna()
                diff_count += 1
                logger.info(f"Applied differencing {diff_count}")
        
        return current_series, diff_count
    
    def fit(self, series: pd.Series, auto_order: bool = True) -> 'ARIMAModel':
        """
        Fit ARIMA model to the data
        
        Args:
            series: Time series data
            auto_order: Whether to automatically find optimal order
            
        Returns:
            Fitted model
        """
        logger.info("Fitting ARIMA model...")
        
        # Remove NaN values
        series_clean = series.dropna()
        
        if auto_order:
            # Find optimal order
            self.order, self.seasonal_order = self.find_optimal_order(series_clean)
        
        try:
            # Create and fit model
            if self.seasonal_order:
                self.model = ARIMA(series_clean, order=self.order, seasonal_order=self.seasonal_order)
            else:
                self.model = ARIMA(series_clean, order=self.order)
            
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            logger.info("ARIMA model fitted successfully")
            logger.info(f"Model summary:\n{self.fitted_model.summary()}")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise
        
        return self
    
    def predict(self, steps: int = 30, return_conf_int: bool = True) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Make predictions using fitted model
        
        Args:
            steps: Number of steps to forecast
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info(f"Making predictions for {steps} steps...")
        
        try:
            # Make forecast
            forecast = self.fitted_model.forecast(steps=steps)
            
            # Get confidence intervals if requested
            conf_int = None
            if return_conf_int:
                conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()
            
            logger.info("Predictions completed successfully")
            
            return forecast, conf_int
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, test_series: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_series: Test data for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        logger.info("Evaluating model performance...")
        
        # Make predictions for test period
        steps = len(test_series)
        predictions, _ = self.predict(steps=steps, return_conf_int=False)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - test_series))
        mse = np.mean((predictions - test_series) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_series - predictions) / test_series)) * 100
        
        # Calculate directional accuracy
        actual_direction = np.diff(test_series) > 0
        pred_direction = np.diff(predictions) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }
        
        logger.info("Model evaluation completed:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def get_residuals(self) -> pd.Series:
        """
        Get model residuals
        
        Returns:
            Residuals series
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting residuals")
        
        return self.fitted_model.resid
    
    def check_residuals(self) -> Dict:
        """
        Perform residual analysis
        
        Returns:
            Dictionary with residual analysis results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before checking residuals")
        
        residuals = self.get_residuals()
        
        # Ljung-Box test for autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        # Basic statistics
        residual_stats = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis(),
            'ljung_box_statistic': lb_test['lb_stat'].iloc[-1],
            'ljung_box_pvalue': lb_test['lb_pvalue'].iloc[-1]
        }
        
        logger.info("Residual analysis completed")
        return residual_stats
    
    def plot_diagnostics(self):
        """
        Plot model diagnostics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting diagnostics")
        
        try:
            import matplotlib.pyplot as plt
            
            # Create diagnostic plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Residuals plot
            residuals = self.get_residuals()
            axes[0, 0].plot(residuals)
            axes[0, 0].set_title('Residuals')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Residuals')
            
            # Residuals histogram
            axes[0, 1].hist(residuals, bins=30, alpha=0.7)
            axes[0, 1].set_title('Residuals Distribution')
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')
            
            # ACF of residuals
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(residuals, ax=axes[1, 1], lags=40)
            axes[1, 1].set_title('ACF of Residuals')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting diagnostics")

def main():
    """Main function to demonstrate ARIMA modeling"""
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
    
    # Get TSLA data for modeling
    tsla_data = clean_data[clean_data['Symbol'] == 'TSLA'].set_index('Date')['Close']
    
    # Split data into train and test
    train_size = int(len(tsla_data) * 0.8)
    train_data = tsla_data[:train_size]
    test_data = tsla_data[train_size:]
    
    print(f"Training data: {len(train_data)} observations")
    print(f"Test data: {len(test_data)} observations")
    
    # Initialize and fit ARIMA model
    arima_model = ARIMAModel()
    arima_model.fit(train_data, auto_order=True)
    
    # Evaluate model
    metrics = arima_model.evaluate(test_data)
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Make future predictions
    future_predictions, conf_int = arima_model.predict(steps=30)
    print(f"\nFuture predictions (next 30 days):")
    print(f"Mean prediction: {future_predictions.mean():.2f}")
    print(f"Prediction range: {future_predictions.min():.2f} - {future_predictions.max():.2f}")

if __name__ == "__main__":
    main() 