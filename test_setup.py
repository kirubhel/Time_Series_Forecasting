#!/usr/bin/env python3
"""
Test script to verify project setup
"""

import sys
import os

# Add src to path
sys.path.append('src')

def test_imports():
    """Test that all modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test data modules
        from data.collector import FinancialDataCollector
        print("✓ FinancialDataCollector imported successfully")
        
        from data.preprocessor import FinancialDataPreprocessor
        print("✓ FinancialDataPreprocessor imported successfully")
        
        # Test model modules
        from models.arima_model import ARIMAModel
        print("✓ ARIMAModel imported successfully")
        
        from models.lstm_model import LSTMModel
        print("✓ LSTMModel imported successfully")
        
        # Test portfolio modules
        from portfolio.optimizer import PortfolioOptimizer
        print("✓ PortfolioOptimizer imported successfully")
        
        from portfolio.backtester import PortfolioBacktester
        print("✓ PortfolioBacktester imported successfully")
        
        # Test utility modules
        from utils.visualization import plot_price_evolution
        print("✓ Visualization utilities imported successfully")
        
        print("\nAll imports successful! Project setup is correct.")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_data_collector():
    """Test data collector functionality"""
    try:
        print("\nTesting data collector...")
        
        from data.collector import FinancialDataCollector
        
        collector = FinancialDataCollector()
        print("✓ FinancialDataCollector initialized successfully")
        
        # Test asset info
        asset_info = collector.get_asset_info()
        print(f"✓ Asset info retrieved: {list(asset_info.keys())}")
        
        print("Data collector test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Data collector test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("PORTFOLIO ANALYSIS PROJECT SETUP TEST")
    print("="*50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test data collector
    collector_ok = test_data_collector()
    
    print("\n" + "="*50)
    if imports_ok and collector_ok:
        print("✓ ALL TESTS PASSED!")
        print("Project is ready for analysis.")
        return True
    else:
        print("✗ SOME TESTS FAILED!")
        print("Please check the errors above and fix them.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 