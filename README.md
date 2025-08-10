# Time Series Forecasting for Portfolio Management Optimization

## Project Overview
This project implements advanced time series forecasting models to optimize portfolio management strategies for GMF Investments. The analysis focuses on three key assets: Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY).

## Business Objective
Guide Me in Finance (GMF) Investments aims to leverage cutting-edge technology and data-driven insights to provide clients with tailored investment strategies. By integrating advanced time series forecasting models, GMF predicts market trends, optimizes asset allocation, and enhances portfolio performance.

## Key Features
- **Data Analysis**: Historical financial data extraction and preprocessing using YFinance
- **Time Series Forecasting**: Implementation of ARIMA/SARIMA and LSTM models
- **Portfolio Optimization**: Modern Portfolio Theory (MPT) implementation with Efficient Frontier analysis
- **Risk Management**: Value at Risk (VaR) and Sharpe Ratio calculations
- **Backtesting**: Strategy validation through historical simulation

## Project Structure
```
week11/
├── data/                   # Data storage and caching
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   ├── models/           # Forecasting models
│   ├── portfolio/        # Portfolio optimization
│   └── utils/            # Utility functions
├── reports/              # Generated reports
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Run data collection: `python src/data/collector.py`
2. Execute analysis notebooks in `notebooks/`
3. Generate reports: `python src/reports/generate_reports.py`

## Key Dates
- **Discussion**: Wednesday, 06 Aug 2025
- **Interim Submission**: Sunday, 10 Aug 2025, 20:00 UTC
- **Final Submission**: Tuesday, 12 Aug 2025, 20:00 UTC

## Team
- **Tutors**: Mahlet, Rediet, Kerod, Rehmet 