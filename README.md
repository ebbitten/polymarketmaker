# Binary Market Maker Bot

A comprehensive market making system for binary prediction markets, featuring market analysis, risk management, and backtesting capabilities.

## Table of Contents

* [Overview](#overview)
* [System Components](#system-components)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Detailed Configuration](#detailed-configuration)
* [Backtesting Guide](#backtesting-guide)
* [Production Deployment](#production-deployment)
* [Risk Management](#risk-management)

## Overview

This system provides a complete framework for market making in binary prediction markets, with the following key features:

* Market analysis and selection
* Dynamic position sizing and risk management
* Comprehensive backtesting framework
* Production-ready market making bot

## System Components

### 1. Market Maker Bot (`market_maker.py`)
* Core market making logic
* Order management
* Position tracking
* API integration stubs

### 2. Market Analyzer (`market_analyzer.py`)
* Market quality assessment
* Liquidity analysis
* Volatility measurement
* Market inefficiency detection

### 3. Risk Manager (`risk_manager.py`)
* Position sizing
* Risk scoring
* Capital allocation
* Time-to-expiry management

### 4. Backtester (`backtester.py`)
* Historical data processing
* Strategy testing
* Performance analysis
* Risk metrics calculation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/binary-market-maker.git
cd binary-market-maker

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
```

## Quick Start

### 1. Configure API Settings

Create a `config.json` file:
```json
{
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "base_url": "https://your-exchange-api.com",
    "initial_capital": 100000.0,
    "max_position_size": 10000.0,
    "min_trade_size": 100.0
}
```

### 2. Basic Usage

```python
from market_maker import MarketMaker
from market_analyzer import MarketAnalyzer
from risk_manager import RiskManager

# Initialize components
config = load_config('config.json')
market_maker = MarketMaker(config)
analyzer = MarketAnalyzer(config)
risk_manager = RiskManager(config)

# Start market making
market_maker.run()
```

### 3. Run Backtest

```python
from backtester import MarketMakerBacktester

# Load historical data
data_path = "path/to/historical_data.csv"
backtest_config = create_backtest_config()
backtester = MarketMakerBacktester(backtest_config)

# Define strategy
def simple_strategy(timestamp, market_states, positions, cash):
    signals = []
    for market_id, state in market_states.items():
        # Your strategy logic here
        pass
    return signals

# Run backtest
results = backtester.run_backtest(simple_strategy, historical_data)
print(f"Total PnL: {results.total_pnl}")
```

## Detailed Configuration

### Market Analyzer Configuration

```python
analyzer_config = {
    'liquidity_weight': 0.4,
    'volatility_weight': 0.3,
    'inefficiency_weight': 0.3,
    'target_daily_volume': 10000,
    'target_daily_trades': 100,
    'target_trade_size': 100,
    'target_volatility': 0.02,
    'max_trend': 0.1,
    'max_price_move': 0.05
}
```

### Risk Manager Configuration

```python
risk_config = {
    'emergency_exit_hours': 1,
    'high_risk_hours': 24,
    'time_decay_factor': 0.2,
    'max_single_market_allocation': 0.2,
    'max_capital_use': 0.8,
    'optimal_position_factor': 0.7,
    'max_daily_volume_percent': 0.1
}
```

### Backtester Configuration

```python
backtest_config = {
    'initial_capital': 100000.0,
    'base_spread': 0.002,
    'reference_volume': 100000.0,
    'impact_factor': 0.1,
    'maker_threshold': 0.0001,
    'maker_fee': 0.001,
    'taker_fee': 0.002
}
```

## Backtesting Guide

### Data Format

Historical data CSV format:
```csv
timestamp,market_id,price,volume,yes_price,no_price,expiry
2024-01-01 00:00:00,market_1,0.65,1000,0.65,0.35,1704067200
```

### Running Backtests

#### 1. Simple Backtest
```python
results = backtester.run_backtest(simple_strategy, historical_data)
```

#### 2. Parameter Optimization
```python
from itertools import product

# Define parameter ranges
spreads = [0.001, 0.002, 0.003]
sizes = [1000, 2000, 3000]

best_result = None
best_params = None

for spread, size in product(spreads, sizes):
    config['base_spread'] = spread
    config['max_position_size'] = size
    
    result = backtester.run_backtest(simple_strategy, historical_data)
    
    if not best_result or result.total_pnl > best_result.total_pnl:
        best_result = result
        best_params = (spread, size)
```

#### 3. Analysis of Results
```python
# Plot daily PnL
results.daily_metrics.plot(y='pnl')

# Print risk metrics
print("Risk Metrics:")
for metric, value in results.risk_metrics.items():
    print(f"{metric}: {value}")
```

## Production Deployment

### System Requirements
* Python 3.8+
* 2GB RAM minimum
* Stable internet connection
* API access to your chosen prediction market

### Deployment Steps

#### 1. Set up monitoring
```python
import logging

logging.basicConfig(
    filename='market_maker.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
```

#### 2. Configure error handling
```python
try:
    market_maker.run()
except Exception as e:
    logging.error(f"Critical error: {e}")
    # Implement notification system
    notify_admin(f"Market maker stopped: {e}")
```

#### 3. Position management
```python
# Regular position checks
def check_positions():
    for market_id, position in positions.items():
        risk_metrics = risk_manager.assess_market_risk(market_id)
        if risk_metrics.time_adjusted_risk > config['max_risk']:
            reduce_position(market_id, position)
```

## Risk Management

### Position Sizing Rules

#### 1. Time-based sizing:
```python
max_position = base_position * time_factor
```

#### 2. Liquidity-based sizing:
```python
max_position = min(
    max_position,
    daily_volume * config['max_daily_volume_percent']
)
```

#### 3. Capital allocation:
```python
max_allocation = total_capital * config['max_single_market_allocation']
```

### Emergency Procedures

#### 1. Market Shutdown
```python
def emergency_shutdown():
    # Close all positions
    for market_id, position in positions.items():
        close_position(market_id)
    
    # Cancel all orders
    cancel_all_orders()
    
    # Notify administrators
    notify_admin("Emergency shutdown initiated")
```

#### 2. Risk Limits
```python
def check_risk_limits():
    portfolio_var = calculate_portfolio_var()
    if portfolio_var > config['max_portfolio_var']:
        reduce_overall_exposure()
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
