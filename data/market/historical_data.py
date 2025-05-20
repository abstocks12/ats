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
        self.partitioner = db_connector.get_partitioner()
    
    def _collect_simulated_data(self, symbol, exchange, timeframe, start_date, end_date):
        """
        Collect simulated historical data when Zerodha authentication is not available
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Candle timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            bool: True if successful
        """
        self.logger.info(f"Generating simulated {timeframe} data for {symbol}@{exchange}")
        
        try:
            # Get simulated data directly
            data = self.zerodha.get_historical_data(
                symbol, exchange, timeframe, start_date, end_date
            )
            
            if not data:
                self.logger.warning(f"No simulated data generated for {symbol}@{exchange}")
                return False
            
            self.logger.info(f"Generated {len(data)} simulated {timeframe} data points")
            
            # Process and store the data, but skip indicator calculation for simulation
            processed_count = 0
            if data:
                # Convert to DataFrame for easier processing
                df = pd.DataFrame(data)
                
                # Handle date/timestamp conversion
                if 'date' in df.columns:
                    if isinstance(df['date'].iloc[0], str):
                        df['timestamp'] = pd.to_datetime(df['date'])
                    else:
                        df['timestamp'] = df['date']
                
                # Prepare data for MongoDB - simplified version without indicators
                records = []
                for _, row in df.iterrows():
                    record = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "timeframe": timeframe,
                        "timestamp": row['timestamp'] if 'timestamp' in row else pd.to_datetime(row['date']),
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "close": float(row['close']),
                        "volume": int(row['volume']),
                        "simulated": True  # Mark as simulated data
                    }
                    records.append(record)
                
                # Insert new records
                if records:
                    try:
                        if hasattr(self.db, 'insert_many'):
                            result = self.db.insert_many("market_data", records)
                            processed_count = len(result) if result else 0
                        elif hasattr(self.db, 'market_data_collection'):
                            result = self.db.market_data_collection.insert_many(records)
                            processed_count = len(result.inserted_ids) if result else 0
                        else:
                            result = self.db["market_data"].insert_many(records)
                            processed_count = len(result.inserted_ids) if result else 0
                    except Exception as e:
                        self.logger.error(f"Error inserting simulated market data: {e}")
                        # Try one-by-one insertion as last resort
                        single_insert_count = 0
                        for record in records:
                            try:
                                if hasattr(self.db, 'insert_one'):
                                    self.db.insert_one("market_data", record)
                                elif hasattr(self.db, 'market_data_collection'):
                                    self.db.market_data_collection.insert_one(record)
                                else:
                                    self.db["market_data"].insert_one(record)
                                single_insert_count += 1
                            except Exception as e2:
                                self.logger.error(f"Error in single record insertion: {e2}")
                        
                        processed_count = single_insert_count
                
                self.logger.info(f"Processed {processed_count} simulated {timeframe} records for {symbol}@{exchange}")
            
            return processed_count > 0
        
        except Exception as e:
            self.logger.error(f"Error in simulation mode: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def collect_data(self, symbol, exchange, timeframes=None, days=365, end_date=None, max_retries=3):
        """
        Collect historical data for an instrument at multiple timeframes
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframes (list): List of candle timeframes (default: ["day", "60min", "15min", "5min", "1min"])
            days (int): Number of days to collect (default: 365)
            end_date (datetime, optional): End date (default: today)
            max_retries (int): Maximum number of retries for API errors
            
        Returns:
            dict: Results by timeframe
        """
        # Default timeframes if not specified
        if timeframes is None:
            timeframes = ["day", "60min", "15min", "5min", "1min"]
        
        self.logger.info(f"Collecting historical data for {symbol}@{exchange} at timeframes: {timeframes}")
        
        # Determine date range
        end_date = end_date or datetime.now()
        
        # Days to collect for each timeframe (to avoid excessive data)
        timeframe_days = {
            "day": days,
            "60min": min(days, 90),  # 3 months for hourly
            "30min": min(days, 60),  # 2 months for 30min
            "15min": min(days, 30),  # 1 month for 15min
            "5min": min(days, 15),   # 15 days for 5min
            "1min": min(days, 7)     # 7 days for 1min
        }
        
        # Check if Zerodha is in simulation mode
        is_simulation = hasattr(self.zerodha, 'is_authenticated') and not self.zerodha.is_authenticated()
        if is_simulation:
            self.logger.warning("Zerodha is in simulation mode. Will use simulated data.")
        
        # Collect data for each timeframe
        results = {}
        
        # IMPORTANT: Do NOT iterate over the characters in the timeframe string!
        for timeframe in timeframes:
            try:
                days_to_collect = timeframe_days.get(timeframe, 30)  # Default to 30 days
                start_date = end_date - timedelta(days=days_to_collect)
                
                self.logger.info(f"Collecting {timeframe} data for {symbol}@{exchange} - {days_to_collect} days")
                
                # For intraday data (less than daily)
                if timeframe in ["1min", "5min", "15min", "30min", "60min"]:
                    result = self._collect_intraday_data(symbol, exchange, timeframe, start_date, end_date, max_retries)
                else:
                    # For daily/weekly data
                    result = self._collect_daily_data(symbol, exchange, timeframe, start_date, end_date, max_retries)
                
                results[timeframe] = result
                
            except Exception as e:
                self.logger.error(f"Error collecting {timeframe} data for {symbol}@{exchange}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                results[timeframe] = False
        
        # Save collection results to database for tracking
        try:
            self._save_collection_results(symbol, exchange, results)
        except Exception as e:
            self.logger.error(f"Error in _save_collection_results: {e}")
        
        return results

    def _save_collection_results(self, symbol, exchange, results):
        """
        Save data collection results to database for tracking and monitoring
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            results (dict): Collection results by timeframe
        """
        try:
            # Create a results record
            collection_record = {
                "symbol": symbol,
                "exchange": exchange,
                "collection_time": datetime.now(),
                "results": {k: bool(v) for k, v in results.items()},  # Convert to bool values for JSON serialization
                "success_count": sum(1 for result in results.values() if result),
                "failure_count": sum(1 for result in results.values() if not result),
                "timeframes_collected": list(results.keys())
            }
            
            # Insert into data_collection_logs collection
            self.db.data_collection_logs_collection.insert_one(collection_record)
            self.logger.info(f"Saved collection results for {symbol}@{exchange}")
                    
        except Exception as e:
            self.logger.error(f"Error preparing collection results: {e}")
            
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
                    # IMPORTANT: Pass the whole timeframe string directly
                    # DO NOT loop over the characters in the timeframe
                    data = self.zerodha.get_historical_data(
                        symbol=symbol, 
                        exchange=exchange, 
                        interval_input=timeframe,  # Pass the entire string
                        from_date=adjusted_start, 
                        to_date=adjusted_end
                    )
                    
                    # Fix: Check if data is not None and is not empty properly for DataFrame
                    if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
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

    def collect_all_timeframes(self, symbol, exchange, days=30):
        """
        Collect data for all standard timeframes for a symbol
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            days (int): Number of days to collect
            
        Returns:
            dict: Results by timeframe
        """
        timeframes = ["day", "60min", "15min", "5min", "1min"]
        return self.collect_data(symbol, exchange, timeframes, days)

    def _collect_daily_data(self, symbol, exchange, timeframe, start_date, end_date, max_retries, is_simulation=False):
        """
        Collect daily historical data
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Candle timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            max_retries (int): Maximum number of retries for API errors
            is_simulation (bool): Whether running in simulation mode
            
        Returns:
            bool: True if successful
        """
        retry_count = 0
        total_records = 0
        
        while retry_count < max_retries:
            try:
                # IMPORTANT: Pass the whole timeframe string directly
                # DO NOT loop over the characters in the timeframe
                data = self.zerodha.get_historical_data(
                    symbol=symbol, 
                    exchange=exchange, 
                    interval_input=timeframe,  # Pass the entire string as one parameter
                    from_date=start_date, 
                    to_date=end_date
                )
                
                # Fix: Check if data is not None and is not empty properly for DataFrame
                if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                    # Process and store the data
                    processed_count = self._process_and_store_data(symbol, exchange, timeframe, data, is_simulation)
                    total_records = processed_count
                
                # Break out of retry loop on success
                break
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Error collecting data (attempt {retry_count}): {e}")
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)  # Exponential backoff
        
        self.logger.info(f"Collected {total_records} {timeframe} data points for {symbol}@{exchange}")
        
        # Calculate technical indicators only if not in simulation mode
        if total_records > 0 and not is_simulation:
            try:
                self._calculate_indicators(symbol, exchange, timeframe)
            except Exception as e:
                self.logger.error(f"Error calculating indicators: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        return total_records > 0

    def save_market_data(self, symbol, timeframe, data):
        """Save market data using time-based partitioning."""
        for record in data:
            # Get the appropriate partition for this timestamp
            timestamp = record.get("timestamp")
            if not timestamp:
                continue
                
            partition_name = self.partitioner.get_partition_for_date(
                "market_data_collection", 
                timestamp
            )
            
            # Insert into the correct partition
            try:
                self.db[partition_name].insert_one(record)
            except Exception as e:
                self.logger.error(f"Error saving data to {partition_name}: {e}")
                
    def _process_and_store_data(self, symbol, exchange, timeframe, data, is_simulation=False):
        """
        Process and store historical data
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframe (str): Candle timeframe
            data (DataFrame or list): Historical data
            is_simulation (bool): Whether running in simulation mode
            
        Returns:
            int: Number of records processed
        """
        # Fix: Check if data is not None and is not empty properly for DataFrame
        if data is None:
            self.logger.warning(f"No data received for {symbol}@{exchange} ({timeframe})")
            return 0
            
        # Convert to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Check if DataFrame is empty
        if df.empty:
            self.logger.warning(f"Empty DataFrame for {symbol}@{exchange} ({timeframe})")
            return 0
        
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
                "timestamp": row['timestamp'] if 'timestamp' in row else pd.to_datetime(row['date']),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume']),
                "simulated": is_simulation  # Mark as simulated data
            }
            records.append(record)
        
        # Check for existing records to avoid duplicates
        existing_count = 0
        new_records = []
        
        for record in records:
            # Use consistent method to check for existing records
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
        
        # Insert new records - using consistent method
        insert_count = 0
        if new_records:
            try:
                # Use consistent database access method
                result = self.db.market_data_collection.insert_many(new_records)
                insert_count = len(result.inserted_ids) if result else 0
                
                self.logger.info(f"Successfully inserted {insert_count} new {timeframe} records for {symbol}@{exchange}")
            except Exception as e:
                self.logger.error(f"Error inserting market data: {e}")
                
                # Try one-by-one insertion as fallback
                single_insert_count = 0
                for record in new_records:
                    try:
                        self.db.market_data_collection.insert_one(record)
                        single_insert_count += 1
                    except Exception as e2:
                        self.logger.error(f"Error in single record insertion: {e2}")
                
                insert_count = single_insert_count
                self.logger.info(f"Inserted {single_insert_count} records individually after bulk insert failed")
        
        total_count = insert_count + existing_count
        self.logger.debug(f"Processed {total_count} {timeframe} records for {symbol}@{exchange} (new: {insert_count}, existing: {existing_count})")
        
        return total_count

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
        
        # Handle edge case with all zeros
        if np.all(gains[1:period+1] == 0):
            avg_gain[period] = 0
        else:
            avg_gain[period] = np.mean(gains[1:period+1])
            
        if np.all(losses[1:period+1] == 0):
            avg_loss[period] = 0
        else:
            avg_loss[period] = np.mean(losses[1:period+1])
        
        # Calculate averages with smoothing
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
        
        # Calculate RS and RSI
        for i in range(period, len(prices)):
            if avg_loss[i] == 0:
                # If no losses, RSI is 100
                rsi[i] = 100
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
        
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
                # IMPORTANT: Pass the whole timeframe string as one parameter
                data = self.zerodha.get_historical_data(
                    symbol=symbol, 
                    exchange=exchange, 
                    interval_input=timeframe,  # Pass the entire string
                    from_date=start_date, 
                    to_date=end_date
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
    
# Simple self-test when run directly
# Main script patch for demonstration
if __name__ == "__main__":
    import sys
    from database.connection_manager import get_db
    
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python historical_data.py SYMBOL EXCHANGE [DAYS]")
        sys.exit(1)
    
    symbol = sys.argv[1]
    exchange = sys.argv[2]
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get database connection
    db = get_db()
    
    # Create collector and collect data
    collector = HistoricalDataCollector(db)
    
    # IMPORTANT: Define timeframes as individual strings, NOT to be iterated over
    timeframes = ["day", "60min", "15min", "5min", "1min"]
    results = collector.collect_data(symbol, exchange, timeframes, days)
    
    # Print results
    print(f"\nCollection Results for {symbol}@{exchange}:")
    for timeframe, count in results.items():
        print(f"  {timeframe}: {'Success' if count else 'Failed'}")
    
    # Check database to confirm storage
    try:
        if hasattr(db, 'count_documents'):
            count = db.count_documents("market_data", {"symbol": symbol, "exchange": exchange})
        elif hasattr(db, 'market_data_collection'):
            count = db.market_data_collection.count_documents({"symbol": symbol, "exchange": exchange})
        else:
            count = db["market_data"].count_documents({"symbol": symbol, "exchange": exchange})
        
        print(f"\nFound {count} total records in database for {symbol}@{exchange}")
    except Exception as e:
        print(f"Error checking database: {e}")