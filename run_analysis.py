#!/usr/bin/env python3
"""
Simple script to run the portfolio analysis
"""

import sys
import os
import logging

# Add src to path
sys.path.append('src')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the analysis"""
    try:
        # Import and run the main analysis
        from main_analysis import PortfolioAnalysisPipeline
        
        logger.info("Starting Portfolio Analysis...")
        
        # Create and run pipeline
        pipeline = PortfolioAnalysisPipeline()
        report = pipeline.run_complete_analysis()
        
        if report:
            logger.info("Analysis completed successfully!")
            logger.info(f"Report generated on: {report['analysis_date']}")
            
            # Print summary
            print("\n" + "="*80)
            print("ANALYSIS SUMMARY")
            print("="*80)
            print(f"Analysis completed on: {report['analysis_date']}")
            print(f"Data analyzed: {report['data_summary']['total_observations']} observations")
            print(f"Assets: {', '.join(report['data_summary']['assets'])}")
            print("="*80)
            
            return True
        else:
            logger.error("Analysis failed!")
            return False
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install required dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 