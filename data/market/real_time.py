"""
Real-Time Data Module - Collects and processes real-time market data
"""

import logging
import time
from datetime import datetime
import threading
import queue
import pandas as pd
import numpy as np

from data.market.zerodha_connector import ZerodhaConnector

class RealTimeDataCollector:
    """
    Collects and processes real-time market data using Zerodha API.
    Supports websocket connection for streaming data and batch data collection.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the real-time data collector
        
        Args:
            db_connector (MongoDBConnector): Database connection
        """
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        self.zerodha = ZerodhaConnector()
        
        # Initialize real-time data cache
        self.real_time_data = {}
        self.last_update_time = {}
        
        # Initialize data queue for async processing
        self.data_queue = queue.Queue()
        self.is_processing = False
        self.processor_thread = None
        
        # Initialize websocket connection status
        self.is_connected = False
        self.subscribed_instruments = set()
        
        # Initialize callbacks
        self.tick_callbacks = []
    
    def start(self):
        """
        Start real-time data collection
        
        Returns:
            bool: True if started successfully
        """
        if self.is_processing:
            self.logger.warning("Real-time data collector already running")
            return True
        
        # Connect to websocket
        success = self.zerodha.connect_ticker()
        
        if not success:
            self.logger.error("Failed to connect to Zerodha ticker")
            return False
        
        # Add callbacks
        self.zerodha.add_tick_callback(self._on_ticks)
        self.zerodha.add_connection_callback(self._on_connection_change)
        
        # Start processor thread
        self.is_processing = True
        self.processor_thread = threading.Thread(target=self._process_data_queue)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
        self.logger.info("Real-time data collector started")
        return True
    
    def stop(self):
        """
        Stop real-time data collection
        """
        if not self.is_processing:
            self.logger.warning("Real-time data collector not running")
            return
        
        # Stop processing
        self.is_processing = False
        
        # Wait for processor thread to terminate
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
            self.processor_thread = None
        
        # Clear data
        self.real_time_data = {}
        self.last_update_time = {}
        self.subscribed_instruments = set()
        
        self.logger.info("Real-time data collector stopped")
    
    def subscribe(self, symbols, exchanges=None):
        """
        Subscribe to real-time data for instruments
        
        Args:
            symbols (list): List of instrument symbols
            exchanges (list, optional): List of exchanges corresponding to symbols
            
        Returns:
            bool: True if subscribed successfully
        """
        if not symbols:
            return False
        
        # Create a list of (symbol, exchange) tuples
        instruments = []
        if exchanges:
            instruments = list(zip(symbols, exchanges))
        else:
            # Default to NSE
            instruments = [(symbol, "NSE") for symbol in symbols]
        
        # Subscribe to websocket
        success = self.zerodha.subscribe_symbols([i[0] for i in instruments], [i[1] for i in instruments])
        
        if success:
            # Add to subscribed instruments
            for symbol, exchange in instruments:
                key = f"{symbol}@{exchange}"
                self.subscribed_instruments.add(key)
            
            self.logger.info(f"Subscribed to {len(instruments)} instruments")
        
        return success
    
    def collect_batch(self, symbols, exchanges=None):
        """
        Collect real-time data for multiple instruments in a batch
        
        Args:
            symbols (list): List of instrument symbols
            exchanges (list, optional): List of exchanges corresponding to symbols
            
        Returns:
            dict: Dictionary of real-time data
        """
        if not symbols:
            return {}
        
        # Create a list of (symbol, exchange) tuples
        instruments = []
        if exchanges:
            instruments = list(zip(symbols, exchanges))
        else:
            # Default to NSE
            instruments = [(symbol, "NSE") for symbol in symbols]
        
        # Collect data
        data = {}
        
        # Get quotes from Zerodha
        symbol_list = [i[0] for i in instruments]
        exchange_list = [i[1] for i in instruments]
        quotes = self.zerodha.get_quote(symbol_list, exchange_list)
        
        # Process quotes
        for symbol, exchange in instruments:
            key = f"{exchange}:{symbol}"
            if key in quotes:
                # Extract relevant data
                quote = quotes[key]
                
                # Process and store the data
                processed_data = self._process_quote(symbol, exchange, quote)
                
                if processed_data:
                    # Update cache
                    cache_key = f"{symbol}@{exchange}"
                    self.real_time_data[cache_key] = processed_data
                    self.last_update_time[cache_key] = datetime.now()
                    
                    # Add to result
                    data[cache_key] = processed_data
                    
                    # Store in database
                    self._store_real_time_data(processed_data)
        
        return data
    
    def get_latest_data(self, symbol, exchange, max_age_seconds=60):
        """
        Get latest real-time data for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            max_age_seconds (int): Maximum age of data in seconds
            
        Returns:
            dict: Real-time data or None if not available or too old
        """
        key = f"{symbol}@{exchange}"
        
        # Check if data exists in cache
        if key in self.real_time_data and key in self.last_update_time:
            # Check age
            age = (datetime.now() - self.last_update_time[key]).total_seconds()
            
            if age <= max_age_seconds:
                return self.real_time_data[key]
        
        # Data not in cache or too old, fetch from database
        data = self.db.real_time_data_collection.find_one(
            {
                "symbol": symbol,
                "exchange": exchange
            },
            sort=[("timestamp", -1)]
        )
        
        if data:
            # Check age
            age = (datetime.now() - data["timestamp"]).total_seconds()
            
            if age <= max_age_seconds:
                return data
        
        # If we get here, need to collect new data
        self.collect_batch([symbol], [exchange])
        
        # Try cache again
        if key in self.real_time_data:
            return self.real_time_data[key]
        
        return None
    
    def add_tick_callback(self, callback):
        """
        Add callback for tick data
        
        Args:
            callback (function): Callback function(symbol, exchange, data)
        """
        self.tick_callbacks.append(callback)
    
    def _on_ticks(self, ticks):
        """
        Callback for tick data from websocket
        
        Args:
            ticks (list): List of tick data
        """
        for tick in ticks:
            # Get instrument info
            token = tick.get("instrument_token")
            instrument = self.zerodha.token_to_instrument.get(token, {})
            
            if not instrument:
                continue
            
            symbol = instrument.get("symbol")
            exchange = instrument.get("exchange")
            
            if not symbol or not exchange:
                continue
            
            # Add to queue for processing
            self.data_queue.put((symbol, exchange, tick))
    
    def _on_connection_change(self, is_connected):
        """
        Callback for websocket connection changes
        
        Args:
            is_connected (bool): Connection status
        """
        self.is_connected = is_connected
        
        if is_connected:
            self.logger.info("Connected to Zerodha websocket")
            
            # Resubscribe to instruments
            if self.subscribed_instruments:
                symbols = []
                exchanges = []
                
                for key in self.subscribed_instruments:
                    symbol, exchange = key.split("@")
                    symbols.append(symbol)
                    exchanges.append(exchange)
                
                self.zerodha.subscribe_symbols(symbols, exchanges)
        else:
            self.logger.warning("Disconnected from Zerodha websocket")
    
    def _process_data_queue(self):
        """
        Process data from the queue
        """
        while self.is_processing:
            try:
                # Get data from queue with timeout
                try:
                    symbol, exchange, tick = self.data_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process tick
                processed_data = self._process_tick(symbol, exchange, tick)
                
                if not processed_data:
                    continue
                
                # Update cache
                key = f"{symbol}@{exchange}"
                self.real_time_data[key] = processed_data
                self.last_update_time[key] = datetime.now()
                
                # Store in database
                self._store_real_time_data(processed_data)
                
                # Notify callbacks
                for callback in self.tick_callbacks:
                    try:
                        callback(symbol, exchange, processed_data)
                    except Exception as e:
                        self.logger.error(f"Error in tick callback: {e}")
                
                # Mark task as done
                self.data_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing tick data: {e}")
    
    def _process_tick(self, symbol, exchange, tick):
        """
        Process tick data
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            tick (dict): Tick data
            
        Returns:
            dict: Processed data
        """
        try:
            # Extract relevant data
            timestamp = datetime.now()
            if "timestamp" in tick:
                if isinstance(tick["timestamp"], str):
                    timestamp = datetime.fromisoformat(tick["timestamp"].replace("Z", "+00:00"))
                else:
                    timestamp = tick["timestamp"]
            
            # Get price data
            last_price = tick.get("last_price")
            
            if not last_price:
                return None
            
            # Create data record
            data = {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": timestamp,
                "last_price": last_price,
                "last_quantity": tick.get("last_quantity"),
                "average_price": tick.get("average_price"),
                "volume": tick.get("volume"),
                "buy_quantity": tick.get("buy_quantity"),
                "sell_quantity": tick.get("sell_quantity"),
                "open": tick.get("ohlc", {}).get("open"),
                "high": tick.get("ohlc", {}).get("high"),
                "low": tick.get("ohlc", {}).get("low"),
                "close": tick.get("ohlc", {}).get("close"),
                "change": tick.get("change"),
                "tick_timestamp": tick.get("timestamp")
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing tick for {symbol}@{exchange}: {e}")
            return None
    
    def _process_quote(self, symbol, exchange, quote):
        """
        Process quote data
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            quote (dict): Quote data
            
        Returns:
            dict: Processed data
        """
        try:
            # Extract timestamp
            timestamp = datetime.now()
            if "timestamp" in quote:
                if isinstance(quote["timestamp"], str):
                    timestamp = datetime.fromisoformat(quote["timestamp"].replace("Z", "+00:00"))
                else:
                    timestamp = quote["timestamp"]
            
            # Get price data
            last_price = quote.get("last_price")
            
            if not last_price:
                return None
            
            # Create data record
            data = {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": timestamp,
                "last_price": last_price,
                "last_quantity": quote.get("last_quantity"),
                "average_price": quote.get("average_price"),
                "volume": quote.get("volume"),
                "buy_quantity": quote.get("buy_quantity"),
                "sell_quantity": quote.get("sell_quantity"),
                "open": quote.get("ohlc", {}).get("open"),
                "high": quote.get("ohlc", {}).get("high"),
                "low": quote.get("ohlc", {}).get("low"),
                "close": quote.get("ohlc", {}).get("close"),
                "change": quote.get("change"),
                "quote_timestamp": quote.get("timestamp")
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing quote for {symbol}@{exchange}: {e}")
            return None
    
    def _store_real_time_data(self, data):
        """
        Store real-time data in the database
        
        Args:
            data (dict): Real-time data
        """
        try:
            # Create a copy to avoid modifying the original
            db_data = data.copy()
            
            # Store in real-time collection
            self.db.real_time_data_collection.update_one(
                {
                    "symbol": data["symbol"],
                    "exchange": data["exchange"]
                },
                {"$set": db_data},
                upsert=True
            )
            
            # Check if we need to create a 1-minute candle
            self._update_minute_candle(data)
            
        except Exception as e:
            self.logger.error(f"Error storing real-time data: {e}")
    
    def _update_minute_candle(self, data):
        """
        Update or create a 1-minute candle from real-time data
        
        Args:
            data (dict): Real-time data
        """
        try:
            symbol = data["symbol"]
            exchange = data["exchange"]
            timestamp = data["timestamp"]
            
            # Round down to the nearest minute
            candle_timestamp = timestamp.replace(second=0, microsecond=0)
            
            # Check if candle exists
            existing_candle = self.db.market_data_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "1min",
                "timestamp": candle_timestamp
            })
            
            if existing_candle:
                # Update existing candle
                updates = {}
                
                # Update high if new price is higher
                if data["last_price"] > existing_candle["high"]:
                    updates["high"] = data["last_price"]
                
                # Update low if new price is lower
                if data["last_price"] < existing_candle["low"]:
                    updates["low"] = data["last_price"]
                
                # Update close and volume
                updates["close"] = data["last_price"]
                updates["volume"] = data["volume"]  # Total volume for the day
                
                # Calculate volume for the minute (approximate)
                if "last_volume" in existing_candle:
                    minute_volume = data["volume"] - existing_candle["last_volume"]
                    updates["minute_volume"] = max(0, minute_volume)  # Ensure non-negative
                
                # Store last volume for next update
                updates["last_volume"] = data["volume"]
                
                # Update the candle
                if updates:
                    self.db.market_data_collection.update_one(
                        {"_id": existing_candle["_id"]},
                        {"$set": updates}
                    )
            else:
                # Create new candle
                new_candle = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": "1min",
                    "timestamp": candle_timestamp,
                    "open": data["last_price"],
                    "high": data["last_price"],
                    "low": data["last_price"],
                    "close": data["last_price"],
                    "volume": data["volume"],
                    "last_volume": data["volume"],
                    "minute_volume": 0  # Will be updated in next tick
                }
                
                # Insert the candle
                self.db.market_data_collection.insert_one(new_candle)
            
        except Exception as e:
            self.logger.error(f"Error updating minute candle: {e}")
    
    def update_ohlc_from_realtime(self, symbol, exchange, timeframe="day"):
        """
        Update end-of-day or last candle data from real-time data
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Candle timeframe (e.g., 'day', '60min')
            
        Returns:
            bool: True if updated successfully
        """
        try:
            # Get real-time data
            real_time = self.get_latest_data(symbol, exchange)
            
            if not real_time:
                self.logger.warning(f"No real-time data available for {symbol}@{exchange}")
                return False
            
            # Get latest candle
            latest_candle = self.db.market_data_collection.find_one(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe
                },
                sort=[("timestamp", -1)]
            )
            
            if not latest_candle:
                self.logger.warning(f"No {timeframe} candle found for {symbol}@{exchange}")
                return False
            
            # Check if candle is for today (or current period for intraday timeframes)
            # This logic would need to be more sophisticated for intraday timeframes
            is_current_period = self._is_current_period(latest_candle["timestamp"], timeframe)
            
            if not is_current_period:
                self.logger.info(f"Latest {timeframe} candle for {symbol}@{exchange} is not for the current period")
                return False
            
            # Update candle
            updates = {}
            
            # Update high if needed
            if real_time["last_price"] > latest_candle["high"]:
                updates["high"] = real_time["last_price"]
            
            # Update low if needed
            if real_time["last_price"] < latest_candle["low"]:
                updates["low"] = real_time["last_price"]
            
            # Update close
            updates["close"] = real_time["last_price"]
            
            # Update volume if available
            if "volume" in real_time and real_time["volume"] > latest_candle.get("volume", 0):
                updates["volume"] = real_time["volume"]
            
            # Update the candle
            if updates:
                self.db.market_data_collection.update_one(
                    {"_id": latest_candle["_id"]},
                    {"$set": updates}
                )
                
                self.logger.debug(f"Updated {timeframe} candle for {symbol}@{exchange} with real-time data")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating OHLC from real-time: {e}")
            return False
    
    def _is_current_period(self, timestamp, timeframe):
        """
        Check if timestamp is in the current period based on timeframe
        
        Args:
            timestamp (datetime): Timestamp to check
            timeframe (str): Candle timeframe
            
        Returns:
            bool: True if timestamp is in current period
        """
        now = datetime.now()
        
        if timeframe == "day":
            # Check if same day
            return timestamp.date() == now.date()
        elif timeframe == "60min":
            # Check if within last hour
            return (now - timestamp).total_seconds() < 3600
        elif timeframe == "30min":
            # Check if within last 30 minutes
            return (now - timestamp).total_seconds() < 1800
        elif timeframe == "15min":
            # Check if within last 15 minutes
            return (now - timestamp).total_seconds() < 900
        elif timeframe == "5min":
            # Check if within last 5 minutes
            return (now - timestamp).total_seconds() < 300
        elif timeframe == "1min":
            # Check if within last minute
            return (now - timestamp).total_seconds() < 60
        else:
            # Default to day
            return timestamp.date() == now.date()