from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np

@dataclass
class RiskMetrics:
    market_id: str
    max_position: float
    optimal_position: float
    max_trade_size: float
    current_risk_score: float
    time_adjusted_risk: float
    capital_allocation: float
    suggested_spread: float

class RiskManager:
    def __init__(self, api_client, config: Dict):
        self.api = api_client
        self.config = config
        
    def get_portfolio_state(self) -> Dict:
        """
        Stub for getting current portfolio state
        Returns: Dictionary with current positions and capital
        """
        return {
            'total_capital': 100000.0,
            'free_capital': 50000.0,
            'positions': self.api.get_positions(),
            'unrealized_pnl': 5000.0,
            'realized_pnl': 10000.0
        }

    def calculate_time_risk_factor(self, market: Market) -> float:
        """
        Calculate risk factor based on time to expiry
        Returns lower values (more risk) as expiry approaches
        """
        time_to_expiry = market.expiry - time.time()
        hours_to_expiry = time_to_expiry / 3600
        
        if hours_to_expiry < self.config['emergency_exit_hours']:
            return 0.0  # Signal to exit all positions
        
        if hours_to_expiry < self.config['high_risk_hours']:
            # Exponentially increase risk factor as expiry approaches
            return np.exp(-self.config['time_decay_factor'] * 
                         (self.config['high_risk_hours'] - hours_to_expiry))
        
        return 1.0  # Normal risk for markets far from expiry

    def calculate_capital_allocation(self, 
                                   market: Market, 
                                   metrics: MarketMetrics,
                                   portfolio_state: Dict) -> float:
        """
        Determine how much capital to allocate to this market
        Based on:
        - Market quality metrics
        - Time to expiry
        - Current portfolio allocation
        - Available capital
        """
        time_factor = self.calculate_time_risk_factor(market)
        if time_factor == 0.0:
            return 0.0  # No new capital for near-expiry markets
        
        # Base allocation on market quality
        base_allocation = (portfolio_state['total_capital'] * 
                         self.config['max_single_market_allocation'] *
                         metrics.overall_score)
        
        # Adjust for time risk
        time_adjusted_allocation = base_allocation * time_factor
        
        # Check portfolio concentration limits
        current_allocation = sum(
            pos.size * pos.entry_price 
            for pos in portfolio_state['positions']
            if pos.market_id == market.id
        )
        
        max_new_allocation = min(
            time_adjusted_allocation - current_allocation,
            portfolio_state['free_capital'] * self.config['max_capital_use']
        )
        
        return max(0, max_new_allocation)

    def calculate_position_limits(self,
                                market: Market,
                                metrics: MarketMetrics,
                                allocation: float) -> Dict[str, float]:
        """
        Calculate position limits based on:
        - Capital allocation
        - Market liquidity
        - Time to expiry
        - Current volatility
        """
        time_factor = self.calculate_time_risk_factor(market)
        
        # Base position size on allocation and current price
        max_position_value = allocation / max(market.yes_price, market.no_price)
        
        # Adjust for liquidity
        daily_volume = metrics.volume_24h
        max_volume_based = daily_volume * self.config['max_daily_volume_percent']
        
        # Take the more conservative limit
        max_position = min(max_position_value, max_volume_based) * time_factor
        
        # Calculate optimal position size (usually smaller than max)
        optimal_position = max_position * self.config['optimal_position_factor']
        
        # Maximum single trade size based on recent volume
        max_trade = min(
            optimal_position * self.config['max_trade_size_factor'],
            daily_volume * self.config['max_trade_volume_percent']
        )
        
        return {
            'max_position': max_position,
            'optimal_position': optimal_position,
            'max_trade_size': max_trade
        }

    def calculate_spread_adjustment(self,
                                  market: Market,
                                  metrics: MarketMetrics,
                                  time_factor: float) -> float:
        """
        Calculate required spread based on:
        - Time to expiry
        - Market volatility
        - Current liquidity
        """
        # Base spread on market volatility
        base_spread = metrics.price_volatility * self.config['volatility_spread_factor']
        
        # Adjust for time to expiry (wider spreads near expiry)
        time_spread = base_spread / time_factor if time_factor > 0 else float('inf')
        
        # Adjust for liquidity
        liquidity_spread = self.config['min_spread'] / metrics.liquidity_score
        
        return max(
            self.config['min_spread'],
            min(
                self.config['max_spread'],
                max(base_spread, time_spread, liquidity_spread)
            )
        )

    def assess_market_risk(self, market: Market, metrics: MarketMetrics) -> RiskMetrics:
        """Main risk assessment function for a market"""
        portfolio_state = self.get_portfolio_state()
        time_factor = self.calculate_time_risk_factor(market)
        
        # If too close to expiry, signal to exit
        if time_factor == 0.0:
            return RiskMetrics(
                market_id=market.id,
                max_position=0.0,
                optimal_position=0.0,
                max_trade_size=0.0,
                current_risk_score=1.0,
                time_adjusted_risk=1.0,
                capital_allocation=0.0,
                suggested_spread=float('inf')
            )
        
        # Calculate capital allocation
        allocation = self.calculate_capital_allocation(
            market, metrics, portfolio_state
        )
        
        # Get position limits
        position_limits = self.calculate_position_limits(
            market, metrics, allocation
        )
        
        # Calculate spread requirements
        suggested_spread = self.calculate_spread_adjustment(
            market, metrics, time_factor
        )
        
        # Current risk score (0-1, higher = more risky)
        current_risk = (
            (1 - time_factor) * 0.4 +
            (1 - metrics.liquidity_score) * 0.3 +
            metrics.volatility_score * 0.3
        )
        
        return RiskMetrics(
            market_id=market.id,
            max_position=position_limits['max_position'],
            optimal_position=position_limits['optimal_position'],
            max_trade_size=position_limits['max_trade_size'],
            current_risk_score=current_risk,
            time_adjusted_risk=current_risk / time_factor if time_factor > 0 else 1.0,
            capital_allocation=allocation,
            suggested_spread=suggested_spread
        )

def create_risk_config() -> Dict:
    return {
        # Time-based risk parameters
        'emergency_exit_hours': 1,  # Exit all positions
        'high_risk_hours': 24,     # Begin reducing exposure
        'time_decay_factor': 0.2,  # How quickly to reduce exposure
        
        # Capital allocation parameters
        'max_single_market_allocation': 0.2,  # Max 20% of capital per market
        'max_capital_use': 0.8,              # Keep 20% capital in reserve
        'optimal_position_factor': 0.7,      # Target 70% of max position
        
        # Trading size limits
        'max_daily_volume_percent': 0.1,     # Max 10% of daily volume
        'max_trade_size_factor': 0.1,        # Max trade 10% of target position
        'max_trade_volume_percent': 0.01,    # Max 1% of daily volume per trade
        
        # Spread parameters
        'min_spread': 0.001,
        'max_spread': 0.05,
        'volatility_spread_factor': 2.0,     # Spread multiplier based on volatility
        
        # Risk limits
        'max_portfolio_var': 0.1,            # Maximum portfolio VaR
        'max_correlation': 0.7               # Maximum position correlation
    }
