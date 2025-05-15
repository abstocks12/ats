"""
Market data models for the Automated Trading System.
Defines the structure for market data collections.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import json

class MarketData:
    """Market data model for storing price and volume data"""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str, timestamp: datetime,
                 open_price: float, high_price: float, low_price: float, close_price: float,
                 volume: int, indicators: Dict[str, float] = None):
        """
        Initialize market data model
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Timeframe (e.g. '1min', '5min', 'day')
            timestamp (datetime): Bar timestamp
            open_price (float): Opening price
            high_price (float): Highest price
            low_price (float): Lowest price
            close_price (float): Closing price
            volume (int): Volume
            indicators (dict, optional): Technical indicators
        """
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.timestamp = timestamp
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price
        self.volume = volume
        self.indicators = indicators or {}
        self.created_at = datetime.now()
        self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert market data to dictionary"""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "indicators": self.indicators,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create market data from dictionary"""
        return cls(
            symbol=data["symbol"],
            exchange=data["exchange"],
            timeframe=data["timeframe"],
            timestamp=data["timestamp"],
            open_price=data["open"],
            high_price=data["high"],
            low_price=data["low"],
            close_price=data["close"],
            volume=data["volume"],
            indicators=data.get("indicators", {})
        )
    
    @classmethod
    def from_zerodha(cls, symbol: str, exchange: str, timeframe: str, historical_data: Dict[str, Any]) -> List['MarketData']:
        """Create market data from Zerodha historical data"""
        market_data_list = []
        
        for record in historical_data.get('data', []):
            try:
                # Zerodha provides data as list: [timestamp, open, high, low, close, volume]
                if len(record) < 6:
                    continue
                
                timestamp = datetime.fromtimestamp(record[0] / 1000)  # Convert milliseconds to seconds
                
                market_data = cls(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    open_price=record[1],
                    high_price=record[2],
                    low_price=record[3],
                    close_price=record[4],
                    volume=record[5]
                )
                
                market_data_list.append(market_data)
                
            except (IndexError, ValueError, TypeError) as e:
                continue
        
        return market_data_list
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.symbol}:{self.exchange} at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - O:{self.open} H:{self.high} L:{self.low} C:{self.close}"