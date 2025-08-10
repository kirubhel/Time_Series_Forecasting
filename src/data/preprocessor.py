"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and preparation for modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FinancialDataPreprocessor:
    """Preprocesses financial data for time series analysis and portfolio optimization"""
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = []
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the financial data
        
        Args:
            df: Raw financial data
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning process...")
        
        # Make a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Convert Date column to datetime
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        
        # Sort by Date and Symbol
        df_clean = df_clean.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['Symbol', 'Date']).reset_index(drop=True)
        
        # Remove rows with zero or negative prices
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_columns:
            if col in df_clean.columns:
                df_clean = df_clean[df_clean[col] > 0]
        
        logger.info(f"Data cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with handled missing values
        """
        # Forward fill for price data within each symbol
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_columns:
            if col in df.columns:
                df[col] = df.groupby('Symbol')[col].fillna(method='ffill')
        
        # For volume, fill with 0 if missing
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
        
        # Remove any remaining rows with missing values
        df = df.dropna()
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features
        
        Args:
            df: Clean financial data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        df_features = df.copy()
        
        # Calculate daily returns
        df_features['Daily_Return'] = df_features.groupby('Symbol')['Close'].pct_change()
        
        # Calculate log returns
        df_features['Log_Return'] = np.log(df_features['Close'] / df_features.groupby('Symbol')['Close'].shift(1))
        
        # Calculate volatility (rolling standard deviation of returns)
        df_features['Volatility_20d'] = df_features.groupby('Symbol')['Daily_Return'].rolling(window=20).std().reset_index(0, drop=True)
        df_features['Volatility_60d'] = df_features.groupby('Symbol')['Daily_Return'].rolling(window=60).std().reset_index(0, drop=True)
        
        # Moving averages
        df_features['MA_20'] = df_features.groupby('Symbol')['Close'].rolling(window=20).mean().reset_index(0, drop=True)
        df_features['MA_60'] = df_features.groupby('Symbol')['Close'].rolling(window=60).mean().reset_index(0, drop=True)
        df_features['MA_200'] = df_features.groupby('Symbol')['Close'].rolling(window=200).mean().reset_index(0, drop=True)
        
        # Price momentum indicators
        df_features['Price_Momentum_5d'] = df_features.groupby('Symbol')['Close'].pct_change(periods=5)
        df_features['Price_Momentum_20d'] = df_features.groupby('Symbol')['Close'].pct_change(periods=20)
        
        # RSI (Relative Strength Index)
        df_features['RSI'] = self._calculate_rsi(df_features)
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df_features)
        df_features = pd.concat([df_features, bb_data], axis=1)
        
        # Volume indicators
        if 'Volume' in df_features.columns:
            df_features['Volume_MA_20'] = df_features.groupby('Symbol')['Volume'].rolling(window=20).mean().reset_index(0, drop=True)
            df_features['Volume_Ratio'] = df_features['Volume'] / df_features['Volume_MA_20']
        
        # Remove rows with NaN values (from rolling calculations)
        df_features = df_features.dropna()
        
        logger.info(f"Feature engineering completed. Shape: {df_features.shape}")
        return df_features
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)
        
        Args:
            df: DataFrame with price data
            period: RSI period
            
        Returns:
            RSI values
        """
        rsi_values = []
        
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol]['Close']
            
            # Calculate price changes
            delta = symbol_data.diff()
            
            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            rsi_values.extend(rsi.values)
        
        return pd.Series(rsi_values, index=df.index)
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            df: DataFrame with price data
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Bollinger Bands
        """
        bb_data = pd.DataFrame(index=df.index)
        
        for symbol in df['Symbol'].unique():
            symbol_mask = df['Symbol'] == symbol
            symbol_data = df[symbol_mask]['Close']
            
            # Calculate moving average
            ma = symbol_data.rolling(window=period).mean()
            
            # Calculate standard deviation
            std = symbol_data.rolling(window=period).std()
            
            # Calculate bands
            upper_band = ma + (std * std_dev)
            lower_band = ma - (std * std_dev)
            
            # Add to result
            bb_data.loc[symbol_mask, 'BB_Upper'] = upper_band
            bb_data.loc[symbol_mask, 'BB_Lower'] = lower_band
            bb_data.loc[symbol_mask, 'BB_Middle'] = ma
        
        return bb_data
    
    def prepare_returns_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare returns data for portfolio optimization
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with returns data
        """
        logger.info("Preparing returns data for portfolio optimization...")
        
        # Pivot data to have assets as columns and dates as index
        returns_df = df.pivot(index='Date', columns='Symbol', values='Daily_Return')
        
        # Remove any remaining NaN values
        returns_df = returns_df.dropna()
        
        logger.info(f"Returns data prepared. Shape: {returns_df.shape}")
        return returns_df
    
    def test_stationarity(self, series: pd.Series, title: str = "Time Series") -> Dict:
        """
        Perform Augmented Dickey-Fuller test for stationarity
        
        Args:
            series: Time series to test
            title: Title for the test
            
        Returns:
            Dictionary with test results
        """
        from statsmodels.tsa.stattools import adfuller
        
        logger.info(f"Testing stationarity for {title}")
        
        # Perform ADF test
        result = adfuller(series.dropna())
        
        # Extract results
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        # Determine if series is stationary
        is_stationary = p_value < 0.05
        
        results = {
            'adf_statistic': adf_statistic,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'title': title
        }
        
        logger.info(f"ADF Test for {title}:")
        logger.info(f"  ADF Statistic: {adf_statistic:.4f}")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Is Stationary: {is_stationary}")
        
        return results
    
    def scale_features(self, df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        """
        Scale features for machine learning models
        
        Args:
            df: DataFrame with features
            columns: Columns to scale
            method: Scaling method ('standard' or 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features using {method} method...")
        
        df_scaled = df.copy()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        # Scale the specified columns
        df_scaled[columns] = self.scaler.fit_transform(df_scaled[columns])
        
        self.feature_columns = columns
        
        logger.info("Feature scaling completed")
        return df_scaled
    
    def create_sequences(self, data: np.ndarray, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model
        
        Args:
            data: Input data
            lookback: Number of time steps to look back
            
        Returns:
            Tuple of (X, y) arrays for LSTM
        """
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def get_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for the dataset
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Summary statistics DataFrame
        """
        # Basic statistics
        summary = df.describe()
        
        # Add additional statistics
        if 'Daily_Return' in df.columns:
            summary.loc['skewness'] = df.groupby('Symbol')['Daily_Return'].skew()
            summary.loc['kurtosis'] = df.groupby('Symbol')['Daily_Return'].kurtosis()
        
        return summary

def main():
    """Main function to demonstrate preprocessing"""
    from collector import FinancialDataCollector
    
    # Collect data
    collector = FinancialDataCollector()
    raw_data = collector.get_combined_data()
    
    if raw_data.empty:
        print("No data available for preprocessing")
        return
    
    # Initialize preprocessor
    preprocessor = FinancialDataPreprocessor()
    
    # Clean data
    clean_data = preprocessor.clean_data(raw_data)
    
    # Engineer features
    feature_data = preprocessor.engineer_features(clean_data)
    
    # Prepare returns data
    returns_data = preprocessor.prepare_returns_data(clean_data)
    
    # Test stationarity for TSLA returns
    tsla_returns = clean_data[clean_data['Symbol'] == 'TSLA']['Daily_Return']
    stationarity_result = preprocessor.test_stationarity(tsla_returns, "TSLA Daily Returns")
    
    print("Preprocessing completed successfully!")
    print(f"Original data shape: {raw_data.shape}")
    print(f"Clean data shape: {clean_data.shape}")
    print(f"Feature data shape: {feature_data.shape}")
    print(f"Returns data shape: {returns_data.shape}")

if __name__ == "__main__":
    main() 