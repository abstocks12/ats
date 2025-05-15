"""
Zerodha Connector Module - Provides connection to Zerodha APIs
"""

import os
import time
import logging
import json
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from kiteconnect import KiteTicker

class ZerodhaConnector:
    """
    Provides connection to Zerodha Kite APIs for market data and trading.
    Handles authentication, rate limiting, and API call management.
    """
    
    def __init__(self, api_key=None, api_secret=None, access_token=None, auto_reconnect=True):
        """
        Initialize the Zerodha connector
        
        Args:
            api_key (str, optional): Zerodha API key (default: from environment)
            api_secret (str, optional): Zerodha API secret (default: from environment)
            access_token (str, optional): Access token (default: from saved file or None)
            auto_reconnect (bool, optional): Whether to auto reconnect websocket (default: True)
        """
        self.logger = logging.getLogger(__name__)
        
        # Get API credentials
        self.api_key = api_key or os.environ.get('ZERODHA_API_KEY')
        self.api_secret = api_secret or os.environ.get('ZERODHA_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            self.logger.warning("Zerodha API credentials not provided, using simulated mode")
            self.simulated_mode = True
        else:
            self.simulated_mode = False
        
        # Create a directory for storing tokens
        self.token_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'auth_tokens'
        )
        os.makedirs(self.token_dir, exist_ok=True)
        
        self.token_file = os.path.join(self.token_dir, 'zerodha_token.json')
        
        # Initialize KiteConnect instance
        self.kite = KiteConnect(api_key=self.api_key)
        
        # Initialize access token
        self.access_token = access_token
        if not self.access_token:
            self._load_access_token()
        
        # Set the access token
        if self.access_token:
            self.kite.set_access_token(self.access_token)
        
        # Initialize ticker (websocket connection)
        self.ticker = None
        self.ticker_connected = False
        self.auto_reconnect = auto_reconnect
        
        # Rate limiting
        self.last_api_call_time = 0
        self.min_time_between_calls = 0.5  # seconds
        
        # Initialize instrument mapping
        self.instrument_tokens = {}
        self.token_to_instrument = {}
        
        # Initialize callbacks
        self.on_tick_callbacks = []
        self.on_connection_callbacks = []
        
        # Load instruments if not in simulated mode
        if not self.simulated_mode and self.access_token:
            self._load_instruments()
    
    def _load_access_token(self):
        """Load access token from file"""
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r') as f:
                    token_data = json.load(f)
                
                # Check if token is expired (valid for one day)
                expiry_time = datetime.fromisoformat(token_data.get('expiry', '2000-01-01'))
                if expiry_time > datetime.now():
                    self.access_token = token_data.get('access_token')
                    self.logger.info("Loaded valid access token from file")
                else:
                    self.logger.warning("Access token expired, needs re-authentication")
        except Exception as e:
            self.logger.error(f"Error loading access token: {e}")
    
    def _save_access_token(self):
        """Save access token to file"""
        try:
            token_data = {
                'access_token': self.access_token,
                'api_key': self.api_key,
                'expiry': (datetime.now() + timedelta(days=1)).isoformat()
            }
            
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f)
            
            self.logger.info("Saved access token to file")
        except Exception as e:
            self.logger.error(f"Error saving access token: {e}")
    
    def _load_instruments(self):
        """Load instrument details from Zerodha"""
        try:
            instruments = self.kite.instruments()
            
            # Map instruments to tokens
            for instrument in instruments:
                key = f"{instrument['tradingsymbol']}:{instrument['exchange']}"
                self.instrument_tokens[key] = instrument['instrument_token']
                self.token_to_instrument[instrument['instrument_token']] = {
                    'symbol': instrument['tradingsymbol'],
                    'exchange': instrument['exchange'],
                    'lot_size': instrument.get('lot_size', 1),
                    'tick_size': instrument.get('tick_size', 0.05),
                    'instrument_type': instrument.get('instrument_type', 'EQ')
                }
            
            self.logger.info(f"Loaded {len(instruments)} instruments from Zerodha")
        except Exception as e:
            self.logger.error(f"Error loading instruments: {e}")
    
    def _rate_limit(self):
        """Apply rate limiting for API calls"""
        current_time = time.time()
        elapsed = current_time - self.last_api_call_time
        
        if elapsed < self.min_time_between_calls:
            sleep_time = self.min_time_between_calls - elapsed
            time.sleep(sleep_time)
        
        self.last_api_call_time = time.time()
    
    def generate_login_url(self):
        """
        Generate login URL for Zerodha authentication
        
        Returns:
            str: Login URL
        """
        if self.simulated_mode:
            return "SIMULATED_MODE_NO_LOGIN_URL"
        
        return self.kite.login_url()
    
    def generate_session(self, request_token):
        """
        Generate session from request token
        
        Args:
            request_token (str): Request token from Zerodha callback
            
        Returns:
            bool: True if successful
        """
        if self.simulated_mode:
            self.logger.info("Simulated mode: Skipping session generation")
            return True
        
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            
            # Save the token for future use
            self._save_access_token()
            
            # Load instruments
            self._load_instruments()
            
            self.logger.info("Successfully generated session and set access token")
            return True
        except Exception as e:
            self.logger.error(f"Error generating session: {e}")
            return False
    
    def get_quote(self, symbols, exchanges=None):
        """
        Get quotes for instruments
        
        Args:
            symbols (list): List of instrument symbols
            exchanges (list, optional): List of exchanges corresponding to symbols
            
        Returns:
            dict: Quote data
        """
        if self.simulated_mode:
            return self._get_simulated_quotes(symbols, exchanges)
        
        try:
            self._rate_limit()
            
            # Format the instruments in exchange:symbol format
            instruments = []
            if exchanges:
                for symbol, exchange in zip(symbols, exchanges):
                    instruments.append(f"{exchange}:{symbol}")
            else:
                # Default to NSE
                for symbol in symbols:
                    instruments.append(f"NSE:{symbol}")
            
            quotes = self.kite.quote(instruments)
            return quotes
        except Exception as e:
            self.logger.error(f"Error getting quotes: {e}")
            # Fallback to simulated quotes in case of error
            return self._get_simulated_quotes(symbols, exchanges)
    
    def get_historical_data(self, symbol, exchange, interval, from_date, to_date):
        """
        Get historical data for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            interval (str): Candle interval (minute, day, etc.)
            from_date (datetime): Start date
            to_date (datetime): End date
            
        Returns:
            list: Historical data
        """
        if self.simulated_mode:
            return self._get_simulated_historical(symbol, exchange, interval, from_date, to_date)
        
        try:
            self._rate_limit()
            
            # Get instrument token
            key = f"{symbol}:{exchange}"
            instrument_token = self.instrument_tokens.get(key)
            
            if not instrument_token:
                self.logger.error(f"Instrument not found: {key}")
                return []
            
            # Convert interval to Zerodha format
            zerodha_interval = self._convert_interval(interval)
            
            # Get historical data
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=zerodha_interval
            )
            
            return data
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            # Fallback to simulated data in case of error
            return self._get_simulated_historical(symbol, exchange, interval, from_date, to_date)
    
    def get_instrument_token(self, symbol, exchange):
        """
        Get instrument token for a symbol
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            int: Instrument token
        """
        key = f"{symbol}:{exchange}"
        return self.instrument_tokens.get(key)
    
    def connect_ticker(self, instrument_tokens=None):
        """
        Connect to Zerodha websocket for real-time data
        
        Args:
            instrument_tokens (list, optional): List of instrument tokens to subscribe
            
        Returns:
            bool: True if connected successfully
        """
        if self.simulated_mode:
            self.logger.info("Simulated mode: Simulating ticker connection")
            self.ticker_connected = True
            return True
        
        if not self.access_token:
            self.logger.error("Cannot connect ticker: No access token")
            return False
        
        try:
            # Initialize ticker
            self.ticker = KiteTicker(self.api_key, self.access_token)
            
            # Set callbacks
            self.ticker.on_ticks = self._on_ticks
            self.ticker.on_connect = self._on_connect
            self.ticker.on_close = self._on_close
            self.ticker.on_error = self._on_error
            self.ticker.on_reconnect = self._on_reconnect
            self.ticker.on_noreconnect = self._on_noreconnect
            
            # Connect
            self.ticker.connect(threaded=True)
            
            # Subscribe to tokens if provided
            if instrument_tokens:
                self.ticker.subscribe(instrument_tokens)
            
            self.logger.info("Ticker connection initiated")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting ticker: {e}")
            return False
    
    def subscribe_symbols(self, symbols, exchanges=None):
        """
        Subscribe to symbols for real-time data
        
        Args:
            symbols (list): List of instrument symbols
            exchanges (list, optional): List of exchanges corresponding to symbols
            
        Returns:
            bool: True if subscribed successfully
        """
        if self.simulated_mode:
            self.logger.info(f"Simulated mode: Simulating subscription to {symbols}")
            return True
        
        if not self.ticker_connected:
            return self.connect_ticker()
        
        try:
            # Get instrument tokens
            tokens = []
            if exchanges:
                for symbol, exchange in zip(symbols, exchanges):
                    key = f"{symbol}:{exchange}"
                    token = self.instrument_tokens.get(key)
                    if token:
                        tokens.append(token)
            else:
                # Default to NSE
                for symbol in symbols:
                    key = f"{symbol}:NSE"
                    token = self.instrument_tokens.get(key)
                    if token:
                        tokens.append(token)
            
            if not tokens:
                self.logger.error("No valid instrument tokens found for subscription")
                return False
            
            # Subscribe
            self.ticker.subscribe(tokens)
            self.logger.info(f"Subscribed to {len(tokens)} instruments")
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing: {e}")
            return False
    
    def add_tick_callback(self, callback):
        """
        Add callback for tick data
        
        Args:
            callback (function): Callback function
        """
        self.on_tick_callbacks.append(callback)
    
    def add_connection_callback(self, callback):
        """
        Add callback for connection events
        
        Args:
            callback (function): Callback function
        """
        self.on_connection_callbacks.append(callback)
    
    def place_order(self, **params):
        """
        Place an order on Zerodha
        
        Args:
            **params: Order parameters
            
        Returns:
            str: Order ID or None if failed
        """
        if self.simulated_mode:
            # Generate a fake order ID
            import uuid
            return f"SIMULATED_{uuid.uuid4()}"
        
        try:
            self._rate_limit()
            
            order_id = self.kite.place_order(
                variety=params.get("variety", "regular"),
                exchange=params.get("exchange", "NSE"),
                tradingsymbol=params.get("symbol"),
                transaction_type=params.get("transaction_type", "BUY"),
                quantity=params.get("quantity", 1),
                product=params.get("product", "MIS"),
                order_type=params.get("order_type", "MARKET"),
                price=params.get("price"),
                trigger_price=params.get("trigger_price"),
                tag=params.get("tag", "AutoTrader")
            )
            
            self.logger.info(f"Order placed successfully: {order_id}")
            return order_id
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def get_order_status(self, order_id):
        """
        Get status of an order
        
        Args:
            order_id (str): Order ID
            
        Returns:
            dict: Order details
        """
        if self.simulated_mode:
            return {
                "status": "COMPLETE",
                "filled_quantity": 1,
                "average_price": 100.0
            }
        
        try:
            self._rate_limit()
            orders = self.kite.orders()
            
            for order in orders:
                if order["order_id"] == order_id:
                    return order
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return None
    
    def get_positions(self):
        """
        Get current positions
        
        Returns:
            dict: Positions data
        """
        if self.simulated_mode:
            return {"net": [], "day": []}
        
        try:
            self._rate_limit()
            positions = self.kite.positions()
            return positions
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {"net": [], "day": []}
    
    def get_holdings(self):
        """
        Get holdings
        
        Returns:
            list: Holdings data
        """
        if self.simulated_mode:
            return []
        
        try:
            self._rate_limit()
            holdings = self.kite.holdings()
            return holdings
        except Exception as e:
            self.logger.error(f"Error getting holdings: {e}")
            return []
    
    def _on_ticks(self, ws, ticks):
        """Ticker callback for tick data"""
        for callback in self.on_tick_callbacks:
            try:
                callback(ticks)
            except Exception as e:
                self.logger.error(f"Error in tick callback: {e}")
    
    def _on_connect(self, ws, response):
        """Ticker callback for connection"""
        self.ticker_connected = True
        self.logger.info("Ticker connected")
        
        for callback in self.on_connection_callbacks:
            try:
                callback(True)
            except Exception as e:
                self.logger.error(f"Error in connection callback: {e}")
    
    def _on_close(self, ws, code, reason):
        """Ticker callback for connection close"""
        self.ticker_connected = False
        self.logger.warning(f"Ticker connection closed: {code} - {reason}")
        
        for callback in self.on_connection_callbacks:
            try:
                callback(False)
            except Exception as e:
                self.logger.error(f"Error in connection callback: {e}")
    
    def _on_error(self, ws, code, reason):
        """Ticker callback for errors"""
        self.logger.error(f"Ticker error: {code} - {reason}")
    
    def _on_reconnect(self, ws, attempt_count):
        """Ticker callback for reconnection"""
        self.logger.info(f"Ticker reconnecting: attempt {attempt_count}")
    
    def _on_noreconnect(self, ws):
        """Ticker callback for failed reconnection"""
        self.ticker_connected = False
        self.logger.error("Ticker failed to reconnect")
        
        # Notify callbacks
        for callback in self.on_connection_callbacks:
            try:
                callback(False)
            except Exception as e:
                self.logger.error(f"Error in connection callback: {e}")
    
    def _convert_interval(self, interval):
        """
        Convert interval to Zerodha format
        
        Args:
            interval (str): Internal interval format
            
        Returns:
            str: Zerodha interval format
        """
        mapping = {
            "1min": "minute",
            "5min": "5minute",
            "15min": "15minute",
            "30min": "30minute",
            "60min": "60minute",
            "day": "day",
            "week": "week"
        }
        
        return mapping.get(interval, "day")
    
    def _get_simulated_quotes(self, symbols, exchanges=None):
        """
        Generate simulated quotes for testing
        
        Args:
            symbols (list): List of instrument symbols
            exchanges (list, optional): List of exchanges
            
        Returns:
            dict: Simulated quote data
        """
        quotes = {}
        
        for i, symbol in enumerate(symbols):
            exchange = exchanges[i] if exchanges and i < len(exchanges) else "NSE"
            
            # Generate a stable but semi-random price based on symbol name
            base_price = sum(ord(c) for c in symbol) % 1000 + 100
            
            # Add some variation based on current time
            import random
            random.seed(int(time.time()) // 300)  # Change every 5 minutes
            variation = random.uniform(-5, 5)
            price = base_price + variation
            
            key = f"{exchange}:{symbol}"
            quotes[key] = {
                "instrument_token": hash(key) % 100000,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_price": price,
                "last_quantity": random.randint(1, 100) * 10,
                "buy_quantity": random.randint(1, 100) * 100,
                "sell_quantity": random.randint(1, 100) * 100,
                "volume": random.randint(10, 1000) * 100,
                "average_price": price * random.uniform(0.998, 1.002),
                "ohlc": {
                    "open": price * random.uniform(0.99, 1.01),
                    "high": price * random.uniform(1.01, 1.03),
                    "low": price * random.uniform(0.97, 0.99),
                    "close": price * random.uniform(0.99, 1.01)
                }
            }
        
        return quotes
    
    def _get_simulated_historical(self, symbol, exchange, interval, from_date, to_date):
        """
        Generate simulated historical data for testing
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            interval (str): Candle interval
            from_date (datetime): Start date
            to_date (datetime): End date
            
        Returns:
            list: Simulated historical data
        """
        # Generate a stable base price based on symbol name
        base_price = sum(ord(c) for c in symbol) % 1000 + 100
        
        # Determine time delta based on interval
        if interval == "1min":
            delta = timedelta(minutes=1)
        elif interval == "5min":
            delta = timedelta(minutes=5)
        elif interval == "15min":
            delta = timedelta(minutes=15)
        elif interval == "30min":
            delta = timedelta(minutes=30)
        elif interval == "60min":
            delta = timedelta(hours=1)
        elif interval == "day":
            delta = timedelta(days=1)
        else:
            delta = timedelta(days=1)
        
        # Generate data points
        data = []
        current_date = from_date
        import random
        price = base_price
        
        while current_date <= to_date:
            # Skip weekends for daily data
            if interval == "day" and current_date.weekday() >= 5:
                current_date += delta
                continue
            
            # Generate OHLC data with some randomness
            random.seed(int(current_date.timestamp()))
            
            # Price movement
            change_percent = random.uniform(-1, 1)
            price = price * (1 + change_percent/100)
            
            # OHLC
            open_price = price
            high_price = price * random.uniform(1, 1.02)
            low_price = price * random.uniform(0.98, 1)
            close_price = price * random.uniform(0.99, 1.01)
            
            # Volume - higher for shorter intervals
            volume_multiplier = 1 if interval == "day" else 24 // (delta.seconds // 3600)
            volume = random.randint(1000, 10000) * volume_multiplier
            
            data_point = {
                "date": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume
            }
            
            data.append(data_point)
            current_date += delta
        
        return data