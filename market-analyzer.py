from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime, timedelta

@dataclass
class MarketMetrics:
    id: str
    name: str
    liquidity_score: float
    volatility_score: float
    inefficiency_score: float
    overall_score: float
    volume_24h: float
    trade_count_24h: int
    avg_spread: float
    price_volatility: float
    
class MarketAnalyzer:
    def __init__(self, api_client, config: Dict):
        self.api = api_client
        self.config = config
        
    def get_historical_data(self, market_id: str, lookback_hours: int = 24) -> List[Dict]:
        """
        Stub for getting historical trade/price data
        Returns: List of trades/prices with timestamps
        """
        # In real implementation, would return:
        # - List of trades with prices, sizes, timestamps
        # - Regular price samples (e.g., minute bars)
        return [
            {
                'timestamp': datetime.now() - timedelta(minutes=i),
                'price': 0.5 + 0.1 * np.sin(i/10),  # Example price pattern
                'volume': 100 * np.random.random(),
                'trade_count': int(5 * np.random.random())
            }
            for i in range(lookback_hours * 60)
        ]

    def calculate_liquidity_score(self, market_id: str, data: List[Dict]) -> float:
        """
        Score market's liquidity based on:
        - 24h volume
        - Trade frequency
        - Average trade size
        - Typical bid-ask spread
        """
        volume_24h = sum(trade['volume'] for trade in data)
        trade_count = sum(trade['trade_count'] for trade in data)
        avg_trade_size = volume_24h / max(trade_count, 1)
        
        # Normalize scores relative to configuration thresholds
        volume_score = min(volume_24h / self.config['target_daily_volume'], 1.0)
        trade_freq_score = min(trade_count / self.config['target_daily_trades'], 1.0)
        size_score = min(avg_trade_size / self.config['target_trade_size'], 1.0)
        
        return (volume_score * 0.4 + 
                trade_freq_score * 0.4 + 
                size_score * 0.2)

    def calculate_volatility_score(self, data: List[Dict]) -> float:
        """
        Score market's volatility based on:
        - Price volatility (standard deviation of returns)
        - Price trend
        - Sharp price moves
        """
        prices = [d['price'] for d in data]
        returns = np.diff(prices) / prices[:-1]
        
        volatility = np.std(returns)
        trend = abs(prices[-1] - prices[0]) / prices[0]
        max_move = max(abs(r) for r in returns)
        
        # Higher score = more favorable volatility characteristics
        volatility_score = min(volatility / self.config['target_volatility'], 1.0)
        trend_score = min(trend / self.config['max_trend'], 1.0)
        move_score = min(max_move / self.config['max_price_move'], 1.0)
        
        return (volatility_score * 0.5 + 
                (1 - trend_score) * 0.3 + 
                (1 - move_score) * 0.2)

    def calculate_inefficiency_score(self, market: Market, data: List[Dict]) -> float:
        """
        Score market's pricing inefficiencies:
        - Deviation from efficient market price (e.g., yes + no ≠ 1)
        - Mean reversion characteristics
        - Predictable patterns
        """
        # Check if yes + no prices sum to 1
        price_inefficiency = abs(1 - (market.yes_price + market.no_price))
        
        # Simple mean reversion test
        prices = np.array([d['price'] for d in data])
        mean_price = np.mean(prices)
        mean_reversion = np.mean([
            (p - mean_price) * (next_p - p)
            for p, next_p in zip(prices[:-1], prices[1:])
        ])
        
        # Higher score = more inefficiencies to profit from
        return (min(price_inefficiency / self.config['max_price_inefficiency'], 1.0) * 0.6 +
                min(abs(mean_reversion) / self.config['target_mean_reversion'], 1.0) * 0.4)

    def analyze_market(self, market: Market) -> MarketMetrics:
        """Analyze a single market and return metrics"""
        historical_data = self.get_historical_data(market.id)
        
        liquidity_score = self.calculate_liquidity_score(market.id, historical_data)
        volatility_score = self.calculate_volatility_score(historical_data)
        inefficiency_score = self.calculate_inefficiency_score(market, historical_data)
        
        # Combine scores based on strategy preferences
        overall_score = (liquidity_score * self.config['liquidity_weight'] +
                        volatility_score * self.config['volatility_weight'] +
                        inefficiency_score * self.config['inefficiency_weight'])
        
        return MarketMetrics(
            id=market.id,
            name=market.name,
            liquidity_score=liquidity_score,
            volatility_score=volatility_score,
            inefficiency_score=inefficiency_score,
            overall_score=overall_score,
            volume_24h=sum(d['volume'] for d in historical_data),
            trade_count_24h=sum(d['trade_count'] for d in historical_data),
            avg_spread=0.02,  # Would calculate from order book in real implementation
            price_volatility=np.std([d['price'] for d in historical_data])
        )

    def find_best_markets(self, min_score: float = 0.6) -> List[MarketMetrics]:
        """Find and rank the best markets for market making"""
        markets = self.api.get_markets()
        
        # Analyze all markets
        market_metrics = [self.analyze_market(market) for market in markets]
        
        # Filter and sort by overall score
        qualified_markets = [
            metrics for metrics in market_metrics 
            if metrics.overall_score >= min_score
        ]
        qualified_markets.sort(key=lambda x: x.overall_score, reverse=True)
        
        return qualified_markets

def create_default_config() -> Dict:
    return {
        # Scoring weights
        'liquidity_weight': 0.4,
        'volatility_weight': 0.3,
        'inefficiency_weight': 0.3,
        
        # Liquidity thresholds
        'target_daily_volume': 10000,
        'target_daily_trades': 100,
        'target_trade_size': 100,
        
        # Volatility thresholds
        'target_volatility': 0.02,
        'max_trend': 0.1,
        'max_price_move': 0.05,
        
        # Inefficiency thresholds
        'max_price_inefficiency': 0.02,
        'target_mean_reversion': 0.01
    }
