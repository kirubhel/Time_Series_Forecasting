"""
Data Collection Module for Financial Data
Fetches historical data for TSLA, BND, and SPY from YFinance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataCollector:
    """Collects financial data from YFinance for portfolio analysis"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.assets = {
            'TSLA': 'Tesla Inc.',
            'BND': 'Vanguard Total Bond Market ETF',
            'SPY': 'SPDR S&P 500 ETF Trust'
        }
        self.start_date = '2015-07-01'
        self.end_date = '2025-07-31'
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_asset_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical data for a single asset
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical data
        """
        try:
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Download data from YFinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Add symbol column
            data['Symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all assets
        
        Returns:
            Dictionary with asset symbols as keys and DataFrames as values
        """
        all_data = {}
        
        for symbol in self.assets.keys():
            data = self.fetch_asset_data(symbol, self.start_date, self.end_date)
            if not data.empty:
                all_data[symbol] = data
                
                # Save individual asset data
                filename = f"{self.data_dir}/{symbol}_data.csv"
                data.to_csv(filename, index=False)
                logger.info(f"Saved {symbol} data to {filename}")
        
        return all_data
    
    def load_cached_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from cached CSV files
        
        Returns:
            Dictionary with asset symbols as keys and DataFrames as values
        """
        cached_data = {}
        
        for symbol in self.assets.keys():
            filename = f"{self.data_dir}/{symbol}_data.csv"
            if os.path.exists(filename):
                data = pd.read_csv(filename)
                data['Date'] = pd.to_datetime(data['Date'])
                cached_data[symbol] = data
                logger.info(f"Loaded cached data for {symbol}")
        
        return cached_data
    
    def get_combined_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get combined data for all assets
        
        Args:
            force_refresh: If True, fetch fresh data from YFinance
            
        Returns:
            Combined DataFrame with all assets
        """
        if force_refresh:
            all_data = self.fetch_all_data()
        else:
            all_data = self.load_cached_data()
            
            # If no cached data, fetch fresh data
            if not all_data:
                all_data = self.fetch_all_data()
        
        if not all_data:
            logger.error("No data available")
            return pd.DataFrame()
        
        # Combine all data
        combined_data = []
        for symbol, data in all_data.items():
            combined_data.append(data)
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Save combined data
        combined_filename = f"{self.data_dir}/combined_data.csv"
        combined_df.to_csv(combined_filename, index=False)
        logger.info(f"Saved combined data to {combined_filename}")
        
        return combined_df
    
    def get_asset_info(self) -> Dict[str, Dict]:
        """
        Get basic information about each asset
        
        Returns:
            Dictionary with asset information
        """
        asset_info = {}
        
        for symbol in self.assets.keys():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                asset_info[symbol] = {
                    'name': self.assets[symbol],
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 'N/A'),
                    'pe_ratio': info.get('trailingPE', 'N/A'),
                    'dividend_yield': info.get('dividendYield', 'N/A')
                }
                
            except Exception as e:
                logger.error(f"Error fetching info for {symbol}: {str(e)}")
                asset_info[symbol] = {
                    'name': self.assets[symbol],
                    'sector': 'N/A',
                    'industry': 'N/A',
                    'market_cap': 'N/A',
                    'pe_ratio': 'N/A',
                    'dividend_yield': 'N/A'
                }
        
        return asset_info

def main():
    """Main function to demonstrate data collection"""
    collector = FinancialDataCollector()
    
    # Fetch data
    print("Fetching financial data...")
    combined_data = collector.get_combined_data()
    
    if not combined_data.empty:
        print(f"Successfully collected data for {len(combined_data)} records")
        print(f"Date range: {combined_data['Date'].min()} to {combined_data['Date'].max()}")
        print(f"Assets: {combined_data['Symbol'].unique()}")
        
        # Display basic statistics
        print("\nBasic Statistics:")
        print(combined_data.groupby('Symbol')['Close'].describe())
    else:
        print("Failed to collect data")

if __name__ == "__main__":
    main() 