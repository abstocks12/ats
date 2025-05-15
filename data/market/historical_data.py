"""
Historical Data Collector Module - Collects and manages historical market data
"""

import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from data.market.zerodha_connector import ZerodhaConnector

class HistoricalDataCollector:
    """
    Collects and manages historical market data.
    Supports different timeframes and handles data gaps.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the historical data collector
        
        Args:
            db_connector (MongoDBConnector): Database connection
        """
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        self.zerodha = ZerodhaConnector()
    
    def collect_data(self, symbol, exchange, timeframe="day", days=365, end_date=None, max_retries=3):
        """
        Collect historical data for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Candle timeframe (default: day)
            days (int): Number of days to collect (default: 365)
            end_date (datetime, optional): End date (default: today)
            max_retries (int): Maximum number of retries for API errors
            
        Returns:
            bool: True if successful
        """
        self.logger.info(f"Collecting {timeframe} historical data for {symbol}@{exchange} for the last {days} days")
        
        # Determine date range
        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # For intraday data, we need to make multiple requests due to API limitations
        if timeframe in ["1min", "5min", "15min", "30min", "60min"]:
            return self._collect_intraday_data(symbol, exchange, timeframe, start_date, end_date, max_retries)
        else:
            # For daily/weekly data, we can make a single request
            return self._collect_daily_data(symbol, exchange, timeframe, start_date, end_date, max_retries)
    
    def _collect_intraday_data(self, symbol, exchange, timeframe, start_date, end_date, max_retries):
        """
        Collect intraday historical data in chunks
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Candle timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            max_retries (int): Maximum number of retries for API errors
            
        Returns:
            bool: True if successful
        """
        # Zerodha's API has a limit of 60 days for intraday data
        max_days_per_request = 30  # Using 30 to be safe
        
        # Break down the date range into chunks
        current_start = start_date
        success_count = 0
        total_records = 0
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=max_days_per_request), end_date)
            
            # Market hours only (9:15 AM to 3:30 PM, Indian market)
            adjusted_start = datetime.combine(current_start.date(), datetime.min.time()).replace(hour=9, minute=15)
            adjusted_end = datetime.combine(current_end.date(), datetime.min.time()).replace(hour=15, minute=30)
            
            # Get data
            retry_count = 0
            while retry_count < max_retries:
                try:
                    data = self.zerodha.get_historical_data(
                        symbol, exchange, timeframe, adjusted_start, adjusted_end
                    )
                    
                    if data:
                        # Process and store the data
                        processed_count = self._process_and_store_data(symbol, exchange, timeframe, data)
                        total_records += processed_count
                        success_count += 1
                    
                    # Break out of retry loop on success
                    break
                except Exception as e:
                    retry_count += 1
                    self.logger.error(f"Error collecting data (attempt {retry_count}): {e}")
                    if retry_count < max_retries:
                        time.sleep(2 ** retry_count)  # Exponential backoff
            
            # Move to next chunk
            current_start = current_end + timedelta(days=1)
        
        self.logger.info(
            f"Collected {total_records} {timeframe} data points for {symbol}@{exchange} "
            f"({success_count} successful requests)"
        )
        
        # Calculate technical indicators for the newly collected data
        self._calculate_indicators(symbol, exchange, timeframe)
        
        return total_records > 0
    
    def _collect_daily_data(self, symbol, exchange, timeframe, start_date, end_date, max_retries):
        """
        Collect daily historical data
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Candle timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            max_retries (int): Maximum number of retries for API errors
            
        Returns:
            bool: True if successful
        """
        retry_count = 0
        total_records = 0
        
        while retry_count < max_retries:
            try:
                data = self.zerodha.get_historical_data(
                    symbol, exchange, timeframe, start_date, end_date
                )
                
                if data:
                    # Process and store the data
                    processed_count = self._process_and_store_data(symbol, exchange, timeframe, data)
                    total_records = processed_count
                
                # Break out of retry loop on success
                break
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Error collecting data (attempt {retry_count}): {e}")
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)  # Exponential backoff
        
        self.logger.info(f"Collected {total_records} {timeframe} data points for {symbol}@{exchange}")
        
        # Calculate technical indicators for the newly collected data
        self._calculate_indicators(symbol, exchange, timeframe)
        
        return total_records > 0
    
    def _process_and_store_data(self, symbol, exchange, timeframe, data):
        """
        Process and store historical data
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Candle timeframe
            data (list): Historical data from Zerodha
            
        Returns:
            int: Number of records processed
        """
        if not data:
            return 0
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(data)
        
        # Handle date/timestamp conversion
        if 'date' in df.columns:
            if isinstance(df['date'].iloc[0], str):
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                df['timestamp'] = df['date']
        
        # Prepare data for MongoDB
        records = []
        for _, row in df.iterrows():
            record = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": row['timestamp'],
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume'])
            }
            records.append(record)
        
        # Check for existing records to avoid duplicates
        existing_count = 0
        new_records = []
        
        for record in records:
            exists = self.db.market_data_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": record["timestamp"]
            })
            
            if not exists:
                new_records.append(record)
            else:
                existing_count += 1
        
        # Insert new records
        if new_records:
            result = self.db.market_data_collection.insert_many(new_records)
            insert_count = len(result.inserted_ids)
            self.logger.debug(f"Inserted {insert_count} new records (skipped {existing_count} existing records)")
        else:
            insert_count = 0
            self.logger.debug(f"No new records to insert (skipped {existing_count} existing records)")
        
        return insert_count + existing_count
    
    def _calculate_indicators(self, symbol, exchange, timeframe):
        """
        Calculate technical indicators and update the data
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Candle timeframe
        """
        try:
            # Fetch data for calculation
            data = list(self.db.market_data_collection.find(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe
                },
                sort=[("timestamp", 1)]
            ))
            
            if not data:
                self.logger.warning(f"No data found for {symbol}@{exchange} ({timeframe}) to calculate indicators")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Calculate indicators
            indicators = self._compute_indicators(df)
            
            # Update records with indicators
            for i, record in enumerate(data):
                if i < len(indicators):
                    self.db.market_data_collection.update_one(
                        {"_id": record["_id"]},
                        {"$set": {"indicators": indicators[i]}}
                    )
            
            self.logger.info(f"Updated indicators for {len(data)} {timeframe} records for {symbol}@{exchange}")
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
    
    def _compute_indicators(self, df):
        """
        Compute technical indicators for the data
        
        Args:
            df (DataFrame): Price data
            
        Returns:
            list: List of indicator dictionaries for each row
        """
        # Extract price and volume data
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        
        # Initialize indicators list
        indicators = []
        
        # Calculate SMA
        sma_periods = [9, 20, 50, 200]
        sma_values = {}
        
        for period in sma_periods:
            sma = self._calculate_sma(closes, period)
            sma_values[f'sma_{period}'] = sma
        
        # Calculate EMA
        ema_periods = [9, 12, 26]
        ema_values = {}
        
        for period in ema_periods:
            ema = self._calculate_ema(closes, period)
            ema_values[f'ema_{period}'] = ema
        
        # Calculate RSI
        rsi_14 = self._calculate_rsi(closes, 14)
        
        # Calculate MACD
        macd, macd_signal, macd_hist = self._calculate_macd(closes)
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes, 20, 2)
        
        # Calculate ATR
        atr_14 = self._calculate_atr(highs, lows, closes, 14)
        
        # Assemble indicators for each row
        for i in range(len(df)):
            indicator_dict = {}
            
            # Add SMAs
            for period in sma_periods:
                key = f'sma_{period}'
                indicator_dict[key] = float(sma_values[key][i]) if i < len(sma_values[key]) and not np.isnan(sma_values[key][i]) else None
            
            # Add EMAs
            for period in ema_periods:
                key = f'ema_{period}'
                indicator_dict[key] = float(ema_values[key][i]) if i < len(ema_values[key]) and not np.isnan(ema_values[key][i]) else None
            
            # Add RSI
            indicator_dict['rsi_14'] = float(rsi_14[i]) if i < len(rsi_14) and not np.isnan(rsi_14[i]) else None
            
            # Add MACD
            indicator_dict['macd'] = float(macd[i]) if i < len(macd) and not np.isnan(macd[i]) else None
            indicator_dict['macd_signal'] = float(macd_signal[i]) if i < len(macd_signal) and not np.isnan(macd_signal[i]) else None
            indicator_dict['macd_hist'] = float(macd_hist[i]) if i < len(macd_hist) and not np.isnan(macd_hist[i]) else None
            
            # Add Bollinger Bands
            indicator_dict['bb_upper'] = float(bb_upper[i]) if i < len(bb_upper) and not np.isnan(bb_upper[i]) else None
            indicator_dict['bb_middle'] = float(bb_middle[i]) if i < len(bb_middle) and not np.isnan(bb_middle[i]) else None
            indicator_dict['bb_lower'] = float(bb_lower[i]) if i < len(bb_lower) and not np.isnan(bb_lower[i]) else None
            
            # Add ATR
            indicator_dict['atr_14'] = float(atr_14[i]) if i < len(atr_14) and not np.isnan(atr_14[i]) else None
            
            indicators.append(indicator_dict)
        
        return indicators
    
    def _calculate_sma(self, prices, period):
        """
        Calculate Simple Moving Average
        
        Args:
            prices (ndarray): Array of price data
            period (int): SMA period
            
        Returns:
            ndarray: SMA values
        """
        sma = np.zeros_like(prices)
        sma[:] = np.nan
        
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        
        return sma
    
    def _calculate_ema(self, prices, period):
        """
        Calculate Exponential Moving Average
        
        Args:
            prices (ndarray): Array of price data
            period (int): EMA period
            
        Returns:
            ndarray: EMA values
        """
        ema = np.zeros_like(prices)
        ema[:] = np.nan
        
        if len(prices) < period:
            return ema
        
        # Start with SMA for the first value
        ema[period - 1] = np.mean(prices[:period])
        
        # Calculate multiplier
        multiplier = 2 / (period + 1)
        
        # Calculate EMA for subsequent values
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index
        
        Args:
            prices (ndarray): Array of price data
            period (int): RSI period
            
        Returns:
            ndarray: RSI values
        """
        rsi = np.zeros_like(prices)
        rsi[:] = np.nan
        
        if len(prices) < period + 1:
            return rsi
        
        # Calculate price changes
        deltas = np.diff(prices)
        deltas = np.append(0, deltas)
        
        # Get gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Initialize averages with simple averages
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        avg_gain[:] = np.nan
        avg_loss[:] = np.nan
        
        avg_gain[period] = np.mean(gains[1:period+1])
        avg_loss[period] = np.mean(losses[1:period+1])
        
        # Calculate averages with smoothing
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
        
        # Calculate RS and RSI
        rs = np.zeros_like(prices)
        rs[:] = np.nan
        
        for i in range(period, len(prices)):
            if avg_loss[i] == 0:
                rs[i] = 100  # To avoid division by zero
            else:
                rs[i] = avg_gain[i] / avg_loss[i]
            
            rsi[i] = 100 - (100 / (1 + rs[i]))
        
        return rsi
    
    def _calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices (ndarray): Array of price data
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal EMA period
            
        Returns:
            tuple: (MACD, Signal, Histogram)
        """
        macd = np.zeros_like(prices)
        signal = np.zeros_like(prices)
        histogram = np.zeros_like(prices)
        
        macd[:] = np.nan
        signal[:] = np.nan
        histogram[:] = np.nan
        
        if len(prices) < slow_period + signal_period:
            return macd, signal, histogram
        
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, fast_period)
        ema_slow = self._calculate_ema(prices, slow_period)
        
        # Calculate MACD line
        for i in range(slow_period - 1, len(prices)):
            macd[i] = ema_fast[i] - ema_slow[i]
        
        # Calculate signal line (EMA of MACD)
        signal_start = slow_period + signal_period - 2
        signal[signal_start] = np.mean(macd[slow_period-1:signal_start+1])
        
        multiplier = 2 / (signal_period + 1)
        for i in range(signal_start + 1, len(prices)):
            signal[i] = (macd[i] - signal[i-1]) * multiplier + signal[i-1]
        
        # Calculate histogram
        for i in range(signal_start, len(prices)):
            histogram[i] = macd[i] - signal[i]
        
        return macd, signal, histogram
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """
        Calculate Bollinger Bands
        
        Args:
            prices (ndarray): Array of price data
            period (int): SMA period
            std_dev (float): Standard deviation multiplier
            
        Returns:
            tuple: (Upper Band, Middle Band, Lower Band)
        """
        upper = np.zeros_like(prices)
        middle = np.zeros_like(prices)
        lower = np.zeros_like(prices)
        
        upper[:] = np.nan
        middle[:] = np.nan
        lower[:] = np.nan
        
        if len(prices) < period:
            return upper, middle, lower
        
        # Calculate the middle band (SMA)
        middle = self._calculate_sma(prices, period)
        
        # Calculate standard deviation
        std = np.zeros_like(prices)
        std[:] = np.nan
        
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1])
        
        # Calculate the upper and lower bands
        for i in range(period - 1, len(prices)):
            upper[i] = middle[i] + (std_dev * std[i])
            lower[i] = middle[i] - (std_dev * std[i])
        
        return upper, middle, lower
    
    def _calculate_atr(self, highs, lows, closes, period=14):
        """
        Calculate Average True Range
        
        Args:
            highs (ndarray): Array of high prices
            lows (ndarray): Array of low prices
            closes (ndarray): Array of close prices
            period (int): ATR period
            
        Returns:
            ndarray: ATR values
        """
        atr = np.zeros_like(closes)
        atr[:] = np.nan
        
        if len(closes) < period + 1:
            return atr
        
        # Calculate true ranges
        tr = np.zeros_like(closes)
        tr[:] = np.nan
        
        tr[0] = highs[0] - lows[0]  # First TR is high - low
        
        for i in range(1, len(closes)):
            # True range is the maximum of:
            # 1. Current high - current low
            # 2. Abs(current high - previous close)
            # 3. Abs(current low - previous close)
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        
        # Calculate first ATR as simple average of TR
        atr[period] = np.mean(tr[1:period+1])
        
        # Calculate subsequent ATRs with smoothing
        for i in range(period + 1, len(closes)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr
    
    def get_data(self, symbol, exchange, timeframe="day", limit=100, end_date=None):
        """
        Get historical data from the database
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Candle timeframe
            limit (int): Number of records to retrieve
            end_date (datetime, optional): End date (default: latest available)
            
        Returns:
            list: Historical data records
        """
        query = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe
        }
        
        if end_date:
            query["timestamp"] = {"$lte": end_date}
        
        data = list(self.db.market_data_collection.find(
            query,
            sort=[("timestamp", -1)],
            limit=limit
        ))
        
        # Reverse to get chronological order
        data.reverse()
        
        return data
    
    def get_latest_data(self, symbol, exchange, timeframe="day"):
        """
        Get latest data record for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Candle timeframe
            
        Returns:
            dict: Latest data record or None if not available
        """
        data = self.db.market_data_collection.find_one(
            {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            },
            sort=[("timestamp", -1)]
        )
        
        return data
    
    def backfill_missing_data(self, symbol, exchange, timeframe="day", days=365):
        """
        Check for and fill missing data points
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Candle timeframe
            days (int): Number of days to check
            
        Returns:
            int: Number of points added
        """
        self.logger.info(f"Checking for missing {timeframe} data for {symbol}@{exchange}")
        
        # Get existing data
        existing_data = self.get_data(symbol, exchange, timeframe, limit=10000)
        
        if not existing_data:
            self.logger.warning(f"No existing data found for {symbol}@{exchange}. Cannot check for gaps.")
            return 0
        
        # Convert to DataFrame for gap analysis
        df = pd.DataFrame(existing_data)
        
        # Check for time gaps based on timeframe
        time_delta = self._get_timeframe_delta(timeframe)
        expected_timestamps = self._generate_expected_timestamps(
            df['timestamp'].min(),
            df['timestamp'].max(),
            time_delta
        )
        
        # Find missing timestamps
        existing_timestamps = set(df['timestamp'])
        missing_timestamps = expected_timestamps - existing_timestamps
        
        if not missing_timestamps:
            self.logger.info(f"No missing data points found for {symbol}@{exchange}")
            return 0
        
        self.logger.info(f"Found {len(missing_timestamps)} missing data points for {symbol}@{exchange}")
        
        # Collect missing data
        missing_count = 0
        for timestamp in sorted(missing_timestamps):
            # For each missing timestamp, get data from the API
            start_date = timestamp - time_delta
            end_date = timestamp + time_delta
            
            try:
                data = self.zerodha.get_historical_data(
                    symbol, exchange, timeframe, start_date, end_date
                )
                
                if data:
                    # Process and store the data
                    processed_count = self._process_and_store_data(symbol, exchange, timeframe, data)
                    missing_count += processed_count
            except Exception as e:
                self.logger.error(f"Error collecting missing data for {timestamp}: {e}")
        
        self.logger.info(f"Added {missing_count} missing data points for {symbol}@{exchange}")
        
        # Recalculate indicators if data was added
        if missing_count > 0:
            self._calculate_indicators(symbol, exchange, timeframe)
        
        return missing_count
    
    def _get_timeframe_delta(self, timeframe):
        """
        Get timedelta for a timeframe
        
        Args:
            timeframe (str): Candle timeframe
            
        Returns:
            timedelta: Time delta for the timeframe
        """
        if timeframe == "1min":
            return timedelta(minutes=1)
        elif timeframe == "5min":
            return timedelta(minutes=5)
        elif timeframe == "15min":
            return timedelta(minutes=15)
        elif timeframe == "30min":
            return timedelta(minutes=30)
        elif timeframe == "60min":
            return timedelta(hours=1)
        elif timeframe == "day":
            return timedelta(days=1)
        elif timeframe == "week":
            return timedelta(weeks=1)
        else:
            return timedelta(days=1)
    
    def _generate_expected_timestamps(self, start_date, end_date, time_delta):
        """
        Generate a set of expected timestamps between start and end dates
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            time_delta (timedelta): Time delta between points
            
        Returns:
            set: Set of expected timestamps
        """
        # For daily and weekly data, filter for trading days only
        from trading.market_hours import MarketHours
        market_hours = MarketHours()
        
        expected_timestamps = set()
        current_date = start_date
        
        while current_date <= end_date:
            # For daily/weekly data, check if it's a trading day
            if time_delta >= timedelta(days=1):
                if market_hours.is_valid_trading_day(current_date.date()):
                    expected_timestamps.add(current_date)
            else:
                # For intraday data, check if it's a trading day and within market hours
                if market_hours.is_valid_trading_day(current_date.date()):
                    time_component = current_date.time()
                    if time_component >= market_hours.regular_open and time_component < market_hours.regular_close:
                        expected_timestamps.add(current_date)
            
            current_date += time_delta
        
        return expected_timestamps