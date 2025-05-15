"""
Portfolio Manager for the Automated Trading System.
Manages the collection of instruments in the portfolio and their configurations.
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

class PortfolioManager:
    """
    Portfolio Manager for managing trading instruments
    """
    
    def __init__(self, db=None):
        """
        Initialize Portfolio Manager
        
        Args:
            db: Database connector (optional, will use global connection if not provided)
        """
        self.logger = setup_logger(__name__)
        self.db = db or get_db()
    
    def add_instrument(self, symbol: str, exchange: str, instrument_type: Optional[str] = None, 
                      sector: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Add an instrument to the portfolio
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            instrument_type (str, optional): Instrument type (equity, futures, options)
            sector (str, optional): Sector of the instrument
            **kwargs: Additional parameters
            
        Returns:
            str: Instrument ID if successful, None otherwise
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
            
            # Check if instrument already exists
            existing = self.db.find_one(
                "portfolio",
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "active"
                }
            )
            
            if existing:
                self.logger.warning(f"Instrument {symbol}:{exchange} already exists in portfolio")
                return existing.get("_id")
            
            # Get trading parameters based on instrument type and sector
            trading_params = portfolio_config.get_instrument_params(instrument_type, sector)
            
            # Create instrument document
            instrument = {
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": instrument_type,
                "status": "active",
                "added_date": datetime.now(),
                "data_collection_status": {
                    "historical": False,
                    "financial": False,
                    "news": False,
                    "global": False
                },
                "trading_config": {
                    "enabled": False,  # Start disabled until data is collected
                    "strategies": kwargs.get("strategies", trading_params.get("strategies", ["technical"])),
                    "position_size_percent": kwargs.get("position_size_percent", trading_params.get("position_size_percent", 5.0)),
                    "max_risk_percent": kwargs.get("max_risk_percent", trading_params.get("max_risk_percent", 1.0)),
                    "stop_loss_percent": kwargs.get("stop_loss_percent", trading_params.get("stop_loss_percent", 2.0)),
                    "target_percent": kwargs.get("target_percent", trading_params.get("target_percent", 6.0)),
                    "timeframe": kwargs.get("timeframe", trading_params.get("default_timeframe", "intraday")),
                    "trailing_stop": kwargs.get("trailing_stop", trading_params.get("trailing_stop", True)),
                    "partial_booking": kwargs.get("partial_booking", trading_params.get("partial_booking", False))
                }
            }
            
            # Add sector if provided
            if sector:
                instrument["sector"] = sector.lower()
                
                # Check if industry is provided in kwargs
                if "industry" in kwargs:
                    instrument["industry"] = kwargs["industry"].lower()
            
            # Add any additional metadata
            if "metadata" in kwargs:
                instrument["metadata"] = kwargs["metadata"]
            
            # Add to database
            instrument_id = self.db.insert_one("portfolio", instrument)
            
            if instrument_id:
                self.logger.info(f"Added instrument {symbol}:{exchange} to portfolio")
                
                # Trigger data collection pipeline
                from portfolio.data_pipeline_trigger import DataPipelineTrigger
                pipeline = DataPipelineTrigger(self.db)
                pipeline.trigger_data_collection(symbol, exchange, instrument_type)
                
                return str(instrument_id)
            else:
                self.logger.error(f"Failed to add instrument {symbol}:{exchange} to portfolio")
                return None
                
        except Exception as e:
            log_error(e, context={"action": "add_instrument", "symbol": symbol, "exchange": exchange})
            return None
    
    def remove_instrument(self, symbol: str, exchange: str, force: bool = False) -> bool:
        """
        Remove an instrument from the portfolio
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            force (bool): Force removal even with open positions
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Normalize symbol and exchange
            symbol = normalize_symbol(symbol)
            exchange = exchange.upper()
            
            # Get instrument
            instrument = self.db.find_one(
                "portfolio",
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "active"
                }
            )
            
            if not instrument:
                self.logger.warning(f"Instrument {symbol}:{exchange} not found in portfolio")
                return False
            
            # Check for open positions if not forced
            if not force:
                # Query open positions for this instrument
                open_positions = self.db.find(
                    "trades",
                    {
                        "symbol": symbol,
                        "exchange": exchange,
                        "status": "open"
                    }
                )
                
                if open_positions:
                    self.logger.warning(f"Instrument {symbol}:{exchange} has open positions - use force=True to remove anyway")
                    return False
            
            # Update instrument status to inactive
            result = self.db.update_one(
                "portfolio",
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "active"
                },
                {
                    "$set": {
                        "status": "inactive",
                        "removed_date": datetime.now()
                    }
                }
            )
            
            if result:
                self.logger.info(f"Removed instrument {symbol}:{exchange} from portfolio")
                return True
            else:
                self.logger.error(f"Failed to remove instrument {symbol}:{exchange} from portfolio")
                return False
                
        except Exception as e:
            log_error(e, context={"action": "remove_instrument", "symbol": symbol, "exchange": exchange})
            return False
    
    def get_instrument(self, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        """
        Get an instrument from the portfolio
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            dict: Instrument document if found, None otherwise
        """
        try:
            # Normalize symbol and exchange
            symbol = normalize_symbol(symbol)
            exchange = exchange.upper()
            
            return self.db.find_one(
                "portfolio",
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "active"
                }
            )
        except Exception as e:
            log_error(e, context={"action": "get_instrument", "symbol": symbol, "exchange": exchange})
            return None
    
    def get_active_instruments(self, instrument_type: Optional[str] = None, 
                              sector: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all active instruments in the portfolio
        
        Args:
            instrument_type (str, optional): Filter by instrument type
            sector (str, optional): Filter by sector
            
        Returns:
            list: List of instrument documents
        """
        try:
            # Build query
            query = {"status": "active"}
            
            if instrument_type:
                query["instrument_type"] = instrument_type.lower()
            
            if sector:
                query["sector"] = sector.lower()
            
            return self.db.find("portfolio", query)
        except Exception as e:
            log_error(e, context={"action": "get_active_instruments"})
            return []
    
    def update_instrument_config(self, symbol: str, exchange: str, 
                                config_updates: Dict[str, Any]) -> bool:
        """
        Update an instrument's trading configuration
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            config_updates (dict): Configuration updates
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Normalize symbol and exchange
            symbol = normalize_symbol(symbol)
            exchange = exchange.upper()
            
            # Check if instrument exists
            instrument = self.get_instrument(symbol, exchange)
            
            if not instrument:
                self.logger.warning(f"Instrument {symbol}:{exchange} not found in portfolio")
                return False
            
            # Build update
            updates = {}
            
            for key, value in config_updates.items():
                updates[f"trading_config.{key}"] = value
            
            # Update instrument
            result = self.db.update_one(
                "portfolio",
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "active"
                },
                {"$set": updates}
            )
            
            if result:
                self.logger.info(f"Updated trading configuration for {symbol}:{exchange}")
                return True
            else:
                self.logger.error(f"Failed to update trading configuration for {symbol}:{exchange}")
                return False
                
        except Exception as e:
            log_error(e, context={"action": "update_instrument_config", "symbol": symbol, "exchange": exchange})
            return False
    
    def enable_trading(self, symbol: str, exchange: str) -> bool:
        """
        Enable trading for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.update_instrument_config(symbol, exchange, {"enabled": True})
    
    def disable_trading(self, symbol: str, exchange: str) -> bool:
        """
        Disable trading for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.update_instrument_config(symbol, exchange, {"enabled": False})
    
    def update_data_collection_status(self, symbol: str, exchange: str, 
                                     data_type: str, status: bool) -> bool:
        """
        Update data collection status for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            data_type (str): Data type (historical, financial, news, global)
            status (bool): Collection status
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Normalize symbol and exchange
            symbol = normalize_symbol(symbol)
            exchange = exchange.upper()
            
            # Update status
            result = self.db.update_one(
                "portfolio",
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "active"
                },
                {"$set": {f"data_collection_status.{data_type}": status}}
            )
            
            if result:
                self.logger.info(f"Updated {data_type} data collection status for {symbol}:{exchange} to {status}")
                
                # If all data collection is complete, enable trading
                if status and data_type != "all":
                    instrument = self.get_instrument(symbol, exchange)
                    
                    if instrument and all(instrument.get("data_collection_status", {}).values()):
                        self.enable_trading(symbol, exchange)
                        self.logger.info(f"All data collection complete for {symbol}:{exchange}, trading enabled")
                
                return True
            else:
                self.logger.error(f"Failed to update {data_type} data collection status for {symbol}:{exchange}")
                return False
                
        except Exception as e:
            log_error(e, context={"action": "update_data_collection_status", "symbol": symbol, "exchange": exchange})
            return False
    
    def get_trading_parameters(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get trading parameters for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            dict: Trading parameters
        """
        try:
            # Get instrument
            instrument = self.get_instrument(symbol, exchange)
            
            if not instrument:
                self.logger.warning(f"Instrument {symbol}:{exchange} not found in portfolio")
                
                # Return default parameters based on instrument type
                instrument_type = get_instrument_type(symbol)
                return portfolio_config.get_instrument_params(instrument_type)
            
            return instrument.get("trading_config", {})
        except Exception as e:
            log_error(e, context={"action": "get_trading_parameters", "symbol": symbol, "exchange": exchange})
            
            # Return default parameters
            return portfolio_config.get_instrument_params("equity")
    
    def check_position_limit(self, symbol: str, exchange: str) -> bool:
        """
        Check if an instrument has reached position limit
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            bool: True if position limit not reached, False otherwise
        """
        try:
            # Get current open positions for this instrument
            open_positions = self.db.find(
                "trades",
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "open"
                }
            )
            
            # Get instrument config
            instrument = self.get_instrument(symbol, exchange)
            
            if not instrument:
                return False
            
            # Get maximum positions from config
            max_positions = instrument.get("trading_config", {}).get("max_positions", 1)
            
            # Check if limit reached
            return len(open_positions) < max_positions
            
        except Exception as e:
            log_error(e, context={"action": "check_position_limit", "symbol": symbol, "exchange": exchange})
            return False
    
    def get_portfolio_exposure(self) -> Dict[str, Any]:
        """
        Get current portfolio exposure metrics
        
        Returns:
            dict: Portfolio exposure metrics
        """
        try:
            # Get all open positions
            open_positions = self.db.find(
                "trades",
                {"status": "open"}
            )
            
            # Initialize metrics
            total_exposure = 0.0
            sector_exposure = {}
            instrument_exposure = {}
            
            for position in open_positions:
                symbol = position.get("symbol", "")
                exchange = position.get("exchange", "")
                quantity = position.get("quantity", 0)
                entry_price = position.get("entry_price", 0.0)
                current_price = position.get("current_price", entry_price)
                
                # Get instrument details
                instrument = self.get_instrument(symbol, exchange)
                sector = instrument.get("sector", "unknown") if instrument else "unknown"
                
                # Calculate position value
                position_value = quantity * current_price
                total_exposure += position_value
                
                # Add to sector exposure
                if sector in sector_exposure:
                    sector_exposure[sector] += position_value
                else:
                    sector_exposure[sector] = position_value
                
                # Add to instrument exposure
                instrument_key = f"{symbol}:{exchange}"
                if instrument_key in instrument_exposure:
                    instrument_exposure[instrument_key] += position_value
                else:
                    instrument_exposure[instrument_key] = position_value
            
            # Get latest portfolio value from performance
            latest_performance = self.db.find_one(
                "performance",
                {},
                sort=[("date", -1)]
            )
            
            portfolio_value = latest_performance.get("portfolio_value", 1000000) if latest_performance else 1000000
            
            # Calculate exposure percentages
            exposure_percent = (total_exposure / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            sector_exposure_percent = {}
            for sector, exposure in sector_exposure.items():
                sector_exposure_percent[sector] = (exposure / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            instrument_exposure_percent = {}
            for instrument, exposure in instrument_exposure.items():
                instrument_exposure_percent[instrument] = (exposure / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            return {
                "total_exposure": total_exposure,
                "exposure_percent": exposure_percent,
                "sector_exposure": sector_exposure,
                "sector_exposure_percent": sector_exposure_percent,
                "instrument_exposure": instrument_exposure,
                "instrument_exposure_percent": instrument_exposure_percent,
                "portfolio_value": portfolio_value,
                "position_count": len(open_positions)
            }
            
        except Exception as e:
            log_error(e, context={"action": "get_portfolio_exposure"})
            return {
                "total_exposure": 0.0,
                "exposure_percent": 0.0,
                "sector_exposure": {},
                "sector_exposure_percent": {},
                "instrument_exposure": {},
                "instrument_exposure_percent": {},
                "portfolio_value": 1000000,
                "position_count": 0
            }