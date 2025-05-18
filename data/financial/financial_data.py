# Create file: data/financial/financial_data.py
"""
Financial Data model for the Automated Trading System.
Provides structure for financial data.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional

class FinancialData:
    """Financial data model"""
    
    def __init__(self, symbol: str, exchange: str, report_type: str, period: str, 
                data: Dict[str, Any], ratios: Dict[str, float] = None, metadata: Dict[str, Any] = None):
        """
        Initialize financial data
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            report_type (str): Report type (e.g., 'quarterly', 'annual')
            period (str): Time period (e.g., '2025-Q1', '2024-FY')
            data (dict): Financial data
            ratios (dict, optional): Financial ratios
            metadata (dict, optional): Additional metadata
        """
        self.symbol = symbol
        self.exchange = exchange
        self.report_type = report_type
        self.period = period
        self.data = data
        self.ratios = ratios or {}
        self.metadata = metadata or {
            "source": "financial_data",
            "collected_at": datetime.now()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "report_type": self.report_type,
            "period": self.period,
            "data": self.data,
            "ratios": self.ratios,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FinancialData':
        """Create from dictionary"""
        return cls(
            symbol=data.get("symbol"),
            exchange=data.get("exchange"),
            report_type=data.get("report_type"),
            period=data.get("period"),
            data=data.get("data", {}),
            ratios=data.get("ratios", {}),
            metadata=data.get("metadata", {})
        )