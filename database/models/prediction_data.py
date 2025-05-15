"""
Prediction data models for the Automated Trading System.
Defines the structure for prediction data collections.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union

class PredictionData:
    """Prediction data model for storing price movement predictions"""
    
    def __init__(self, symbol: str, exchange: str, date: datetime, prediction: str,
                 confidence: float, timeframe: str, supporting_factors: Optional[List[Dict[str, Any]]] = None,
                 target_price: Optional[float] = None, stop_loss: Optional[float] = None,
                 model_id: Optional[str] = None):
        """
        Initialize prediction data model
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            date (datetime): Prediction date
            prediction (str): Prediction ('up', 'down', 'sideways')
            confidence (float): Confidence score (0.0 to 1.0)
            timeframe (str): Prediction timeframe ('intraday', 'short_term', 'medium_term', 'long_term')
            supporting_factors (list, optional): List of supporting factors with weights
            target_price (float, optional): Target price
            stop_loss (float, optional): Stop loss price
            model_id (str, optional): Model ID used for prediction
        """
        self.symbol = symbol
        self.exchange = exchange
        self.date = date
        self.prediction = prediction
        self.confidence = confidence
        self.timeframe = timeframe
        self.supporting_factors = supporting_factors or []
        self.target_price = target_price
        self.stop_loss = stop_loss
        self.model_id = model_id
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction data to dictionary"""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "date": self.date,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "timeframe": self.timeframe,
            "supporting_factors": self.supporting_factors,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "model_id": self.model_id,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionData':
        """Create prediction data from dictionary"""
        return cls(
            symbol=data["symbol"],
            exchange=data["exchange"],
            date=data["date"],
            prediction=data["prediction"],
            confidence=data["confidence"],
            timeframe=data["timeframe"],
            supporting_factors=data.get("supporting_factors", []),
            target_price=data.get("target_price"),
            stop_loss=data.get("stop_loss"),
            model_id=data.get("model_id")
        )
    
    def __str__(self) -> str:
        """String representation"""
        confidence_pct = f"{self.confidence * 100:.1f}%"
        return f"{self.symbol}:{self.exchange} - {self.prediction.upper()} ({confidence_pct}) - {self.date.strftime('%Y-%m-%d')}"