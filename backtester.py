from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from collections import defaultdict

@dataclass
class BacktestTrade:
    timestamp: datetime
    market_id: str
    side: str  # 'yes' or 'no'
    size: float
    price: float
    trade_type: str  # 'maker' or 'taker'
    fees: float

@dataclass
class BacktestResult:
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    trade_count: int
    maker_ratio: float
    avg_position_size: float
    avg_daily_volume: float
    win_rate: float
    positions_at_expiry: Dict[str, float]
    daily_metrics: pd.DataFrame
    risk_metrics: Dict[str, float]

class MarketMakerBacktester:
    def __init__(self, config: Dict):
        self.config = config
        self.trades: List[BacktestTrade] = []
        self.positions: Dict[str, float] = defaultdict(float)
        self.cash: float = config['initial_capital']
        self.market_states: Dict[str, Dict] = {}
        
    def load_historical_data(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load and process historical market data
        Expected format: JSON or CSV with timestamps, prices, volumes
        Returns: Dictionary of DataFrames by market_id
        """
        # Example data structure
        data = pd.read_csv(file_path)  # or json.load()
        markets = {}
        
        for market_id in data['market_id'].unique():
            market_data = data[data['market_id'] == market_id].copy()
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
            market_data.set_index('timestamp', inplace=True)
            market_data.sort_index(inplace=True)
            
            # Calculate derived features
            market_data['returns'] = market_data['price'].pct_change()
            market_data['volatility'] = market_data['returns'].rolling(window=24).std()
            market_data['volume_ma'] = market_data['volume'].rolling(window=24).mean()
            
            markets[market_id] = market_data
            
        return markets

    def simulate_order_book(self, 
                          current_price: float, 
                          volume: float,
                          timestamp: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate order book state based on price and volume
        Returns: (bids DataFrame, asks DataFrame)
        """
        # Simulate bid-ask spread based on volume and volatility
        base_spread = self.config['base_spread']
        volume_factor = np.log1p(volume) / np.log1p(self.config['reference_volume'])
        spread = base_spread / volume_factor
        
        # Generate synthetic order book levels
        levels = 10
        bid_prices = [current_price * (1 - spread * (i + 1)/levels) for i in range(levels)]
        ask_prices = [current_price * (1 + spread * (i + 1)/levels) for i in range(levels)]
        
        # Simulate sizes using exponential decay
        base_size = volume / levels
        sizes = [base_size * np.exp(-i/3) for i in range(levels)]
        
        bids = pd.DataFrame({
            'price': bid_prices,
            'size': sizes
        })
        
        asks = pd.DataFrame({
            'price': ask_prices,
            'size': sizes
        })
        
        return bids, asks

    def estimate_market_impact(self, 
                             size: float, 
                             current_price: float, 
                             volume: float) -> float:
        """
        Estimate price impact of a trade
        Returns: Estimated executed price after impact
        """
        impact_factor = self.config['impact_factor']
        volume_ratio = size / max(volume, 1e-6)
        impact = impact_factor * volume_ratio * current_price
        return impact

    def simulate_execution(self, 
                         market_id: str,
                         side: str,
                         size: float,
                         limit_price: float,
                         timestamp: datetime,
                         market_data: pd.DataFrame) -> Optional[BacktestTrade]:
        """
        Simulate trade execution with market impact
        Returns: BacktestTrade if executed, None if failed
        """
        current_price = market_data.loc[timestamp, 'price']
        current_volume = market_data.loc[timestamp, 'volume']
        
        # Simulate order book
        bids, asks = self.simulate_order_book(current_price, current_volume, timestamp)
        
        # Calculate market impact
        impact = self.estimate_market_impact(size, current_price, current_volume)
        
        # Check if trade is possible given limit price
        executed_price = None
        if side == 'yes':
            min_ask = asks['price'].min()
            impacted_price = min_ask + impact
            if impacted_price <= limit_price:
                executed_price = impacted_price
        else:  # 'no'
            max_bid = bids['price'].max()
            impacted_price = max_bid - impact
            if impacted_price >= limit_price:
                executed_price = impacted_price
                
        if executed_price is None:
            return None
            
        # Determine if trade was maker or taker
        trade_type = 'maker' if abs(executed_price - current_price) < self.config['maker_threshold'] else 'taker'
        
        # Calculate fees
        fees = size * executed_price * (
            self.config['maker_fee'] if trade_type == 'maker' 
            else self.config['taker_fee']
        )
        
        return BacktestTrade(
            timestamp=timestamp,
            market_id=market_id,
            side=side,
            size=size,
            price=executed_price,
            trade_type=trade_type,
            fees=fees
        )

    def calculate_pnl(self, 
                     market_id: str, 
                     timestamp: datetime,
                     market_data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate realized and unrealized PnL
        Returns: (realized_pnl, unrealized_pnl)
        """
        position = self.positions[market_id]
        if position == 0:
            return 0.0, 0.0
            
        current_price = market_data.loc[timestamp, 'price']
        
        # Calculate average entry price from trades
        market_trades = [t for t in self.trades if t.market_id == market_id]
        if not market_trades:
            return 0.0, 0.0
            
        avg_entry_price = sum(t.price * t.size for t in market_trades) / sum(t.size for t in market_trades)
        
        unrealized_pnl = position * (current_price - avg_entry_price)
        realized_pnl = sum(
            t.size * (t.price - avg_entry_price) - t.fees
            for t in market_trades
            if t.timestamp <= timestamp
        )
        
        return realized_pnl, unrealized_pnl

    def run_backtest(self, 
                    strategy: callable, 
                    historical_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Run backtest of market making strategy
        Args:
            strategy: Function that generates trading signals
            historical_data: Dictionary of market data by market_id
        Returns: BacktestResult with performance metrics
        """
        daily_metrics = []
        
        # Iterate through time steps
        timestamps = sorted(set.union(*[set(df.index) for df in historical_data.values()]))
        
        for timestamp in timestamps:
            # Update market states
            for market_id, market_data in historical_data.items():
                if timestamp in market_data.index:
                    self.market_states[market_id] = {
                        'price': market_data.loc[timestamp, 'price'],
                        'volume': market_data.loc[timestamp, 'volume'],
                        'volatility': market_data.loc[timestamp, 'volatility']
                    }
            
            # Get strategy signals
            signals = strategy(
                timestamp,
                self.market_states,
                self.positions,
                self.cash
            )
            
            # Execute trades
            for signal in signals:
                trade = self.simulate_execution(
                    signal['market_id'],
                    signal['side'],
                    signal['size'],
                    signal['price'],
                    timestamp,
                    historical_data[signal['market_id']]
                )
                
                if trade:
                    self.trades.append(trade)
                    self.positions[trade.market_id] += (
                        trade.size if trade.side == 'yes' else -trade.size
                    )
                    self.cash -= trade.size * trade.price + trade.fees
            
            # Calculate daily metrics
            daily_pnl = 0
            daily_volume = 0
            
            for market_id in historical_data.keys():
                if market_id in self.positions and self.positions[market_id] != 0:
                    realized, unrealized = self.calculate_pnl(
                        market_id, timestamp, historical_data[market_id]
                    )
                    daily_pnl += realized + unrealized
                    
                daily_volume += sum(
                    t.size * t.price
                    for t in self.trades
                    if t.market_id == market_id and t.timestamp.date() == timestamp.date()
                )
            
            daily_metrics.append({
                'timestamp': timestamp,
                'pnl': daily_pnl,
                'volume': daily_volume,
                'cash': self.cash,
                'positions': self.positions.copy()
            })
        
        # Calculate final metrics
        daily_df = pd.DataFrame(daily_metrics)
        
        result = BacktestResult(
            total_pnl=daily_df['pnl'].sum(),
            realized_pnl=sum(t.size * t.price - t.fees for t in self.trades),
            unrealized_pnl=daily_df['pnl'].iloc[-1] - daily_df['pnl'].sum(),
            sharpe_ratio=self.calculate_sharpe_ratio(daily_df['pnl']),
            max_drawdown=self.calculate_max_drawdown(daily_df['pnl']),
            trade_count=len(self.trades),
            maker_ratio=sum(1 for t in self.trades if t.trade_type == 'maker') / len(self.trades),
            avg_position_size=np.mean([abs(pos) for pos in self.positions.values()]),
            avg_daily_volume=daily_df['volume'].mean(),
            win_rate=self.calculate_win_rate(),
            positions_at_expiry=self.positions.copy(),
            daily_metrics=daily_df,
            risk_metrics=self.calculate_risk_metrics(daily_df)
        )
        
        return result

    def calculate_sharpe_ratio(self, pnl_series: pd.Series) -> float:
        """Calculate Sharpe ratio of PnL series"""
        returns = pnl_series.pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()

    def calculate_max_drawdown(self, pnl_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_pnl = pnl_series.cumsum()
        running_max = cum_pnl.expanding().max()
        drawdowns = cum_pnl - running_max
        return abs(drawdowns.min())

    def calculate_win_rate(self) -> float:
        """Calculate ratio of profitable trades"""
        profitable_trades = sum(
            1 for t in self.trades 
            if (t.side == 'yes' and t.price > self.market_states[t.market_id]['price']) or
               (t.side == 'no' and t.price < self.market_states[t.market_id]['price'])
        )
        return profitable_trades / len(self.trades) if self.trades else 0.0

    def calculate_risk_metrics(self, daily_metrics: pd.DataFrame) -> Dict[str, float]:
        """Calculate various risk metrics"""
        return {
            'var_95': np.percentile(daily_metrics['pnl'], 5),
            'var_99': np.percentile(daily_metrics['pnl'], 1),
            'avg_leverage': daily_metrics['volume'].mean() / self.config['initial_capital'],
            'max_leverage': daily_metrics['volume'].max() / self.config['initial_capital'],
            'avg_turnover': daily_metrics['volume'].mean() / daily_metrics['cash'].mean(),
            'position_concentration': max(abs(pos) for pos in self.positions.values()) / self.config['initial_capital']
        }

def create_backtest_config() -> Dict:
    return {
        'initial_capital': 100000.0,
        'base_spread': 0.002,
        'reference_volume': 100000.0,
        'impact_factor': 0.1,
        'maker_threshold': 0.0001,
        'maker_fee': 0.001,
        'taker_fee': 0.002,
        'min_trade_size': 100.0,
        'max_position_size': 10000.0
    }
