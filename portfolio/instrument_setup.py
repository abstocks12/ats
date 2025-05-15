"""
Instrument Setup for the Automated Trading System.
Configures and sets up instruments for trading.
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings, portfolio_config
from database.connection_manager import get_db
from utils.logging_utils import setup_logger, log_error
from utils.helper_functions import normalize_symbol, get_instrument_type

class InstrumentSetup:
    """
    Instrument Setup for configuring trading instruments
    """
    
    def __init__(self, db=None):
        """
        Initialize Instrument Setup
        
        Args:
            db: Database connector (optional, will use global connection if not provided)
        """
        self.logger = setup_logger(__name__)
        self.db = db or get_db()
    
    def setup_instrument(self, symbol: str, exchange: str, instrument_type: Optional[str] = None,
                        sector: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Set up an instrument for trading
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            instrument_type (str, optional): Instrument type (equity, futures, options)
            sector (str, optional): Sector of the instrument
            **kwargs: Additional configuration parameters
            
        Returns:
            dict: Instrument configuration
        """
        try:
            # Normalize symbol and exchange
            symbol = normalize_symbol(symbol)
            exchange = exchange.upper()
            
            # Guess instrument type if not provided
            if not instrument_type:
                instrument_type = get_instrument_type(symbol)
            else:
                instrument_type = instrument_type.lower()
            
            # Get base trading parameters
            trading_params = portfolio_config.get_instrument_params(instrument_type, sector)
            
            # Create configuration
            config = {
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": instrument_type,
                "timeframes": kwargs.get("timeframes", self._get_default_timeframes(instrument_type)),
                "strategies": kwargs.get("strategies", trading_params.get("strategies", ["technical"])),
                "position_size_percent": kwargs.get("position_size_percent", trading_params.get("position_size_percent", 5.0)),
                "max_risk_percent": kwargs.get("max_risk_percent", trading_params.get("max_risk_percent", 1.0)),
                "stop_loss_percent": kwargs.get("stop_loss_percent", trading_params.get("stop_loss_percent", 2.0)),
                "target_percent": kwargs.get("target_percent", trading_params.get("target_percent", 6.0)),
                "trading_timeframe": kwargs.get("timeframe", trading_params.get("default_timeframe", "intraday")),
                "trailing_stop": kwargs.get("trailing_stop", trading_params.get("trailing_stop", True)),
                "partial_booking": kwargs.get("partial_booking", trading_params.get("partial_booking", False)),
                "data_sources": self._get_data_sources(instrument_type, sector),
                "indicators": self._get_default_indicators(instrument_type, trading_params.get("strategies", ["technical"])),
                "sector": sector.lower() if sector else None
            }
            
            # Add strategy-specific parameters
            strategy_params = {}
            for strategy in config["strategies"]:
                try:
                    strategy_config = portfolio_config.get_strategy_params(strategy)
                    strategy_params[strategy] = strategy_config
                except ValueError:
                    # Strategy not found, skip
                    pass
            
            config["strategy_params"] = strategy_params
            
            # Add technical analysis parameters
            config["technical_params"] = {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "ema_periods": [9, 21, 50, 200],
                "sma_periods": [20, 50, 200],
                "bollinger_period": 20,
                "bollinger_std": 2.0,
                "atr_period": 14
            }
            
            # Log the configuration
            self.logger.info(f"Set up instrument {symbol}:{exchange} with configuration")
            
            return config
                
        except Exception as e:
            log_error(e, context={"action": "setup_instrument", "symbol": symbol, "exchange": exchange})
            
            # Return minimal configuration
            return {
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": instrument_type or "equity",
                "timeframes": ["day", "60min", "5min"],
                "strategies": ["technical"],
                "data_sources": ["historical", "news"]
            }
    
    def _get_default_timeframes(self, instrument_type: str) -> List[str]:
        """
        Get default timeframes for an instrument type
        
        Args:
            instrument_type (str): Instrument type
            
        Returns:
            list: List of timeframes
        """
        if instrument_type == "equity":
            return ["day", "60min", "15min", "5min", "1min"]
        elif instrument_type == "futures":
            return ["day", "60min", "15min", "5min"]
        elif instrument_type == "options":
            return ["day", "60min", "15min"]
        else:
            return ["day", "60min", "5min"]
    
    def _get_data_sources(self, instrument_type: str, sector: Optional[str] = None) -> List[str]:
        """
        Get data sources for an instrument type
        
        Args:
            instrument_type (str): Instrument type
            sector (str, optional): Sector of the instrument
            
        Returns:
            list: List of data sources
        """
        sources = ["historical", "news"]
        
        if instrument_type == "equity":
            sources.append("financial")
        
        if sector:
            sources.append("global")
        
        return sources
    
    def _get_default_indicators(self, instrument_type: str, strategies: List[str]) -> Dict[str, List[str]]:
        """
        Get default technical indicators for an instrument type and strategies
        
        Args:
            instrument_type (str): Instrument type
            strategies (list): List of strategies
            
        Returns:
            dict: Dictionary of indicators by timeframe
        """
        # Base indicators for all timeframes
        base_indicators = ["sma_20", "sma_50", "sma_200", "ema_9", "ema_21", "rsi_14", "macd"]
        
        # Add strategy-specific indicators
        if "trend_following" in strategies:
            base_indicators.extend(["adx_14", "supertrend"])
        
        if "mean_reversion" in strategies:
            base_indicators.extend(["bollinger_bands", "stochastic", "cci_20"])
        
        if "volatility" in strategies:
            base_indicators.extend(["atr_14", "volatility_bands"])
        
        # Timeframe-specific indicators
        indicators = {
            "day": base_indicators + ["volume_sma_20", "weekly_pivot"],
            "60min": base_indicators + ["volume_sma_20", "daily_pivot"],
            "15min": base_indicators,
            "5min": ["sma_20", "ema_9", "rsi_14", "macd"],
            "1min": ["ema_9", "rsi_7"]
        }
        
        return indicators
    
    def get_strategy_parameters(self, strategy: str) -> Dict[str, Any]:
        """
        Get parameters for a specific strategy
        
        Args:
            strategy (str): Strategy name
            
        Returns:
            dict: Strategy parameters
        """
        try:
            return portfolio_config.get_strategy_params(strategy)
        except ValueError:
            self.logger.warning(f"Strategy {strategy} not found in configuration")
            return {}
    
    def configure_from_database(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Configure instrument from database
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            dict: Instrument configuration
        """
        try:
            # Normalize symbol and exchange
            symbol = normalize_symbol(symbol)
            exchange = exchange.upper()
            
            # Get instrument from database
            from portfolio.portfolio_manager import PortfolioManager
            portfolio_manager = PortfolioManager(self.db)
            instrument = portfolio_manager.get_instrument(symbol, exchange)
            
            if not instrument:
                self.logger.warning(f"Instrument {symbol}:{exchange} not found in portfolio")
                return self.setup_instrument(symbol, exchange)
            
            # Extract configuration
            instrument_type = instrument.get("instrument_type", "equity")
            sector = instrument.get("sector")
            trading_config = instrument.get("trading_config", {})
            
            # Set up instrument with database config
            return self.setup_instrument(
                symbol=symbol,
                exchange=exchange,
                instrument_type=instrument_type,
                sector=sector,
                **trading_config
            )
                
        except Exception as e:
            log_error(e, context={"action": "configure_from_database", "symbol": symbol, "exchange": exchange})
            return self.setup_instrument(symbol, exchange)