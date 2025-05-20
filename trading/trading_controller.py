"""
Trading Controller Module - Manages the trading system's start/stop functionality
"""

import time
import threading
import os
import signal
import logging
from datetime import datetime
from trading.zerodha_manager import get_zerodha_connector, ensure_zerodha_connection
from utils.logging_utils import setup_logger

class TradingController:
    """
    Controls the trading system's operation, starting and stopping the trading process.
    Manages market hours, position tracking, and trading thread lifecycle.
    """
    def __init__(self, db_connector, mode="paper"):
        """
        Initialize the trading controller
        
        Args:
            db_connector (MongoDBConnector): Database connection
            mode (str): Trading mode, either 'paper' or 'live'
        """
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        self.trading_mode = mode
        self.trading_active = False
        self.trading_thread = None
        self._pid_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '.trading_pid'
        )
        
        self.logger = setup_logger(__name__)
        self.mode = mode
        
        # Initialize state
        self.is_active = False
        self.current_positions = {}
        self.orders = {}
        
        # Initialize trading engine
        self.engine = None
        # Initialize market data connector
        if mode == "live":
            # Ensure Zerodha connection
            if ensure_zerodha_connection():
                self.zerodha_connector = get_zerodha_connector()
            else:
                self.zerodha_connector = None
                self.logger.warning("Running without Zerodha connection, features will be limited")
        else:
            # In paper trading mode, we can use simulated connector
            self.zerodha_connector = get_zerodha_connector()
        # Initialize market hours manager
        from trading.market_hours import MarketHours
        self.market_hours = MarketHours()
        
        # Initialize position manager
        from trading.position_manager import PositionManager
        self.position_manager = PositionManager(self.db)
        
        # Check if trading was already running (recover from crash)
        self._check_previous_instance()
    
    def _check_previous_instance(self):
        """Check if a previous instance was running and handle accordingly"""
        if os.path.exists(self._pid_file):
            try:
                with open(self._pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process is still running
                try:
                    os.kill(pid, 0)  # This will raise an exception if process is not running
                    self.logger.warning(f"Previous trading instance (PID: {pid}) still running")
                except OSError:
                    # Process not running, clean up pid file
                    os.remove(self._pid_file)
                    self.logger.warning("Found stale PID file from crashed instance, removed it")
            except (ValueError, IOError) as e:
                # Invalid PID file
                os.remove(self._pid_file)
                self.logger.warning(f"Invalid PID file: {e}, removed it")
    
    def _write_pid_file(self):
        """Write current PID to file for crash recovery"""
        with open(self._pid_file, 'w') as f:
            f.write(str(os.getpid()))
    
    def _remove_pid_file(self):
        """Remove PID file when trading stops"""
        if os.path.exists(self._pid_file):
            os.remove(self._pid_file)
    
    def is_trading_active(self):
        """Check if trading is currently active"""
        return self.trading_active
        
    def start_trading(self, instruments=None):
        """
        Start automated trading for all or specific instruments
        
        Args:
            instruments (list): List of instrument documents to trade (default: None, for all enabled)
        
        Returns:
            bool: True if trading started successfully, False otherwise
        """
        if self.trading_active:
            self.logger.warning("Trading already active")
            return False
        
        self.trading_active = True
        
        # Write PID file for crash recovery
        self._write_pid_file()
        
        # Get active instruments if not specified
        if not instruments:
            try:
                self.logger.info("Enabling trading for all active instruments")
                result = self.db.update_many(
                    "portfolio",
                    {"status": "active"},
                    {"$set": {"trading_config.enabled": True}}
                )
                self.logger.info(f"Enabled trading for {result} instruments")
            except Exception as e:
                self.logger.warning(f"Failed to update trading_config.enabled: {e}")
            # Use the portfolio manager to get active instruments
            from portfolio.portfolio_manager import PortfolioManager
            portfolio_manager = PortfolioManager(self.db)
            instruments = portfolio_manager.get_active_instruments()
            self.logger.info(instruments)
            
            # Filter for trading_config.enabled = True
            instruments = [i for i in instruments if i.get("trading_config", {}).get("enabled", False)]
        
        if not instruments:
            self.logger.warning("No active instruments found for trading")
            self.trading_active = False
            self._remove_pid_file()
            return False
        
        self.logger.info(f"Starting trading in {self.trading_mode} mode for {len(instruments)} instruments")
        
        # Log trading system status
        self._log_system_status(instruments)
        
        # Start trading in a background thread
        self.trading_thread = threading.Thread(
            target=self._trading_loop,
            args=(instruments,)
        )
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        return True
    
    def stop_trading(self, close_positions=False):
        """
        Stop automated trading
        
        Args:
            close_positions (bool): Whether to close all open positions
        
        Returns:
            bool: True if trading stopped successfully, False otherwise
        """
        if not self.trading_active:
            self.logger.warning("Trading not active")
            return False
        
        self.logger.info("Stopping trading")
        self.trading_active = False
        
        # Close positions if requested
        if close_positions:
            try:
                closed_count = self.position_manager.close_all_positions()
                self.logger.info(f"Closed {closed_count} positions")
            except Exception as e:
                self.logger.error(f"Error closing positions: {e}")
                # Continue with shutdown even if closing positions fails
        
        # Wait for trading thread to terminate
        if self.trading_thread:
            self.trading_thread.join(timeout=30)
            self.trading_thread = None
        
        # Remove PID file
        self._remove_pid_file()
        
        return True
    
    def _log_system_status(self, instruments):
        """Log trading system status"""
        # List of instruments
        symbols = [f"{i['symbol']}@{i['exchange']}" for i in instruments]
        self.logger.info(f"Trading instruments: {', '.join(symbols)}")
        
        # Mode
        self.logger.info(f"Trading mode: {self.trading_mode}")
        
        # Market hours
        market_open = self.market_hours.is_market_open()
        self.logger.info(f"Market is currently {'open' if market_open else 'closed'}")
        
        if not market_open:
            next_open = self.market_hours.get_next_market_open()
            self.logger.info(f"Next market open: {next_open.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _trading_loop(self, instruments):
        """
        Main trading loop that runs continuously while trading is active
        
        Args:
            instruments (list): List of instrument documents to trade
        """
        self.logger.info("Trading loop started")
        
        while self.trading_active:
            try:
                # Check if market is open
                if not self.market_hours.is_market_open():
                    self.logger.info("Market closed, waiting...")
                    
                    # Log when market will open next
                    next_open = self.market_hours.get_next_market_open()
                    self.logger.info(f"Next market open: {next_open.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Sleep until next check (every minute)
                    time.sleep(60)
                    continue
                
                # Execute one trading cycle
                self._execute_trading_cycle(instruments)
                
                # Small delay between cycles
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}", exc_info=True)
                time.sleep(30)  # Longer delay after error
    
    def _execute_trading_cycle(self, instruments):
        """
        Execute one complete trading cycle
        
        Args:
            instruments (list): List of instrument documents to trade
        """
        # Update real-time data for all instruments
        self._update_realtime_data(instruments)
        
        # Get latest predictions for instruments
        predictions = self._get_latest_predictions(instruments)
        
        # Process each prediction
        for prediction in predictions:
            try:
                # Check confidence threshold
                if prediction.get("confidence", 0) < 0.6:
                    continue
                
                # Check if we already have a position
                existing_position = self.position_manager.get_position(
                    prediction["symbol"],  
                    prediction["exchange"]
                )
                
                if existing_position:
                    # Update existing position
                    self._update_position(existing_position, prediction)
                else:
                    # Consider new position
                    self._consider_new_position(prediction)
            except Exception as e:
                self.logger.error(f"Error processing prediction for {prediction['symbol']}: {e}", exc_info=True)
        
        # Update all existing positions (e.g., trailing stops)
        self.position_manager.update_all_positions()
    
    def _update_realtime_data(self, instruments):
        """
        Update real-time market data for all instruments
        
        Args:
            instruments (list): List of instrument documents to trade
        """
        try:
            from data.market.real_time import RealTimeDataCollector
            collector = RealTimeDataCollector(self.db)
            
            # Batch collect real-time data for all instruments
            symbols = [instrument["symbol"] for instrument in instruments]
            exchanges = [instrument["exchange"] for instrument in instruments]
            
            collector.collect_batch(symbols, exchanges)
            
        except Exception as e:
            self.logger.error(f"Error updating real-time data: {e}", exc_info=True)
    
    def _get_latest_predictions(self, instruments):
        """
        Get latest predictions for all active instruments
        
        Args:
            instruments (list): List of instrument documents to trade
            
        Returns:
            list: List of prediction documents
        """
        predictions = []
        
        for instrument in instruments:
            try:
                # Get latest prediction from database
                latest = self.db.predictions_collection.find_one(
                    {
                        "symbol": instrument["symbol"],
                        "exchange": instrument["exchange"]
                    },
                    sort=[("date", -1)]
                )
                
                if latest:
                    predictions.append(latest)
                else:
                    # Generate a new prediction if none exists
                    self._generate_prediction(instrument)
                
            except Exception as e:
                self.logger.error(f"Error getting prediction for {instrument['symbol']}: {e}")
        
        return predictions
    
    def _generate_prediction(self, instrument):
        """
        Generate a new prediction for an instrument
        
        Args:
            instrument (dict): Instrument document
        """
        try:
            from ml.prediction.daily_predictor import DailyPredictor
            predictor = DailyPredictor(self.db)
            
            # Generate prediction
            prediction = predictor.generate_prediction(
                symbol=instrument["symbol"],
                exchange=instrument["exchange"]
            )
            # trading/trading_controller.py
            # Around line 359
            if prediction:
                self.logger.info(f"Generated new prediction for {instrument['symbol']}: {prediction['prediction']} (confidence: {prediction['confidence']:.2f})")
            else:
                self.logger.warning(f"Failed to generate prediction for {instrument['symbol']}")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error generating prediction for {instrument['symbol']}: {e}", exc_info=True)
            return None
    
    def _consider_new_position(self, prediction):
        """
        Evaluate whether to open a new position based on prediction
        
        Args:
            prediction (dict): Prediction document
        """
        try:
            # Get instrument configuration
            instrument = self.db.portfolio_collection.find_one({
                "symbol": prediction["symbol"],
                "exchange": prediction["exchange"]
            })
            
            if not instrument:
                return
            
            # Calculate position size
            from core.risk.position_sizing import calculate_position_size
            position_size = calculate_position_size(
                prediction,
                instrument["trading_config"],
                self.db  # For portfolio context
            )
            
            if position_size <= 0:
                return  # Skip if position size is too small
            
            # Get entry parameters
            from core.strategies.technical import get_entry_parameters
            entry_params = get_entry_parameters(prediction, self.db)
            
            # Execute the trade
            if self.trading_mode == "live":
                from trading.order_executor import execute_order
                order_result = execute_order(
                    symbol=prediction["symbol"],
                    exchange=prediction["exchange"],
                    order_type="buy" if prediction["prediction"] == "up" else "sell",
                    quantity=position_size,
                    price=entry_params.get("limit_price"),
                    stop_loss=entry_params.get("stop_loss"),
                    target=entry_params.get("target"),
                    db=self.db
                )
                
                self.logger.info(f"Opened new position for {prediction['symbol']}: {order_result}")
            else:
                # Paper trading - simulate order
                self.logger.info(f"[PAPER] Would open position for {prediction['symbol']}: {position_size} shares, direction: {prediction['prediction']}")
                
                # Add to paper trading positions
                self.position_manager.add_paper_position(
                    prediction["symbol"],
                    prediction["exchange"],
                    "buy" if prediction["prediction"] == "up" else "sell",
                    position_size,
                    entry_params
                )
            
        except Exception as e:
            self.logger.error(f"Error considering new position for {prediction['symbol']}: {e}", exc_info=True)
    
    def _update_position(self, position, prediction):
        """
        Update an existing position based on new prediction
        
        Args:
            position (dict): Position document
            prediction (dict): Prediction document
        """
        try:
            # Check if we should exit based on prediction
            if (prediction["prediction"] == "down" and position["position_type"] == "long") or \
               (prediction["prediction"] == "up" and position["position_type"] == "short"):
                
                # Exit the position
                if self.trading_mode == "live":
                    from trading.order_executor import close_position
                    result = close_position(position, db=self.db)
                    self.logger.info(f"Closed position for {position['symbol']}: {result}")
                else:
                    # Paper trading - simulate closing
                    self.logger.info(f"[PAPER] Would close position for {position['symbol']} based on prediction reversal")
                    self.position_manager.close_paper_position(
                        position["symbol"],
                        position["exchange"]
                    )
                
                return
            
            # Update stop loss and targets
            from core.risk.stop_management import update_stop_loss
            new_stop = update_stop_loss(position, prediction, self.db)
            
            if new_stop and new_stop != position.get("stop_loss"):
                # Update the stop loss
                if self.trading_mode == "live":
                    from trading.order_executor import update_order
                    result = update_order(
                        position_id=position["_id"],
                        updates={"stop_loss": new_stop},
                        db=self.db
                    )
                    self.logger.info(f"Updated stop loss for {position['symbol']}: {new_stop}")
                else:
                    # Paper trading - simulate update
                    self.logger.info(f"[PAPER] Would update stop loss for {position['symbol']}: {new_stop}")
                    self.position_manager.update_paper_position(
                        position["symbol"],
                        position["exchange"],
                        {"stop_loss": new_stop}
                    )
            
        except Exception as e:
            self.logger.error(f"Error updating position for {position['symbol']}: {e}", exc_info=True)


def force_stop_trading():
    """
    Force stop the trading system by killing the process
    This is a last resort when normal shutdown fails
    """
    pid_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        '.trading_pid'
    )
    
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Try to terminate the process
            os.kill(pid, signal.SIGTERM)
            
            # Wait a bit and check if it's still running
            time.sleep(3)
            try:
                os.kill(pid, 0)  # This will raise an exception if process is not running
                
                # If we got here, process is still running, use SIGKILL
                os.kill(pid, signal.SIGKILL)
                logging.warning(f"Force killed trading process (PID: {pid})")
            except OSError:
                # Process already terminated
                logging.info(f"Trading process (PID: {pid}) terminated successfully")
            
            # Remove PID file
            os.remove(pid_file)
            
        except (ValueError, IOError, OSError) as e:
            logging.error(f"Error force stopping trading: {e}")
            
            # Try to remove PID file anyway
            if os.path.exists(pid_file):
                os.remove(pid_file)