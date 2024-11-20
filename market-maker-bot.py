from dataclasses import dataclass
from typing import List, Dict, Optional
import time
import random
import logging

@dataclass
class Market:
    id: str
    name: str
    yes_price: float
    no_price: float
    volume: float
    expiry: int  # Unix timestamp

@dataclass
class Position:
    market_id: str
    size: float  # Positive for yes, negative for no
    entry_price: float

class APIClient:
    """Stub API client for prediction market platform"""
    
    def get_markets(self) -> List[Market]:
        """
        Get list of active markets
        Returns: List of Market objects with current prices and metadata
        """
        # Stub implementation
        return [
            Market(
                id="market_1",
                name="Will X happen?",
                yes_price=0.65,
                no_price=0.35,
                volume=1000.0,
                expiry=int(time.time()) + 86400
            )
        ]
    
    def get_positions(self) -> List[Position]:
        """
        Get current positions
        Returns: List of Position objects with sizes and entry prices
        """
        # Stub implementation
        return [
            Position(
                market_id="market_1",
                size=100.0,
                entry_price=0.60
            )
        ]
    
    def place_order(self, market_id: str, side: str, size: float, price: float) -> bool:
        """
        Place a new order
        Args:
            market_id: Market identifier
            side: 'yes' or 'no'
            size: Order size
            price: Limit price
        Returns: True if order was accepted
        """
        # Stub implementation
        return True

class MarketMaker:
    def __init__(self, api_client: APIClient, config: Dict):
        self.api = api_client
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def get_target_position(self, market: Market) -> Optional[float]:
        """Calculate target position based on current market prices"""
        # Simple strategy: Be neutral when prices near 0.5
        # Long when prices < 0.4, short when prices > 0.6
        mid_price = market.yes_price
        if mid_price < 0.4:
            return self.config['max_position']
        elif mid_price > 0.6:
            return -self.config['max_position']
        return 0.0
        
    def run_once(self):
        """Single iteration of market making loop"""
        try:
            markets = self.api.get_markets()
            positions = {p.market_id: p for p in self.api.get_positions()}
            
            for market in markets:
                if market.expiry - time.time() < self.config['min_time_to_expiry']:
                    continue
                    
                target = self.get_target_position(market)
                if target is None:
                    continue
                    
                current = positions.get(market.id, Position(market.id, 0.0, 0.0))
                position_delta = target - current.size
                
                if abs(position_delta) < self.config['min_trade_size']:
                    continue
                    
                # Add random spread around mid price
                spread = random.uniform(
                    self.config['min_spread'],
                    self.config['max_spread']
                )
                
                if position_delta > 0:
                    self.api.place_order(
                        market.id,
                        'yes',
                        position_delta,
                        market.yes_price * (1 - spread/2)
                    )
                else:
                    self.api.place_order(
                        market.id,
                        'no',
                        -position_delta,
                        market.no_price * (1 + spread/2)
                    )
                    
        except Exception as e:
            self.logger.error(f"Error in market maker loop: {e}")

def main():
    config = {
        'max_position': 1000.0,
        'min_trade_size': 10.0,
        'min_spread': 0.01,
        'max_spread': 0.02,
        'min_time_to_expiry': 3600,
        'loop_interval': 5
    }
    
    api = APIClient()
    maker = MarketMaker(api, config)
    
    while True:
        maker.run_once()
        time.sleep(config['loop_interval'])

if __name__ == '__main__':
    main()
