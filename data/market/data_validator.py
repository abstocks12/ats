"""
Market Data Validator and Processor

This module handles validation, cleaning, and preprocessing of market data.
It ensures data quality and integrity before analysis and trading.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from pymongo import UpdateOne, UpdateMany

class MarketDataValidator:
    """
    Validates and processes market data to ensure quality and integrity.
    Provides methods for cleaning, error detection, and data transformation.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the market data validator with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Define validation thresholds
        self.thresholds = {
            # Price movement thresholds (%)
            "price_jump_threshold": 15.0,  # Suspicious price jump (%)
            "price_gap_threshold": 10.0,   # Suspicious price gap (%)
            
            # Volume thresholds
            "volume_spike_factor": 5.0,    # Volume spike factor vs average
            "zero_volume_allowed": False,  # Whether zero volume is acceptable
            
            # Missing data thresholds
            "max_missing_days": 5,         # Maximum allowed consecutive missing days
            "max_missing_pct": 10.0,       # Maximum allowed missing data percentage
            
            # OHLC validity checks
            "high_low_min_diff": 0.01,     # Minimum difference between high and low (%)
            
            # Time-based checks
            "max_staleness_hours": 24,     # Maximum allowed data staleness (hours)
            
            # Duplicate check windows (days)
            "duplicate_check_window": 30   # Window for checking duplicates (days)
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for this module."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def validate_market_data(self, symbol: str, timeframe: str = "day", 
                           days: int = 30) -> Dict[str, Any]:
        """
        Run validation checks on market data.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe ('1min', '5min', '15min', 'hour', 'day', etc.)
            days: Number of days to validate
            
        Returns:
            Dictionary containing validation results
        """
        self.logger.info(f"Validating {timeframe} data for {symbol} over {days} days")
        
        # Retrieve data from database
        data = self._get_market_data(symbol, timeframe, days)
        
        if not data:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": "error",
                "error": "No data found",
                "valid": False
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data)
        
        # Run validation checks
        validation_results = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now(),
            "data_points": len(df),
            "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            "checks": {},
            "valid": True,
            "issues_found": 0,
            "warnings": 0
        }
        
        # Store data copy for reference
        orig_df = df.copy()
        
        # 1. Check for duplicates
        duplicate_check = self._check_duplicates(df)
        validation_results["checks"]["duplicates"] = duplicate_check
        if not duplicate_check["valid"]:
            validation_results["valid"] = False
            validation_results["issues_found"] += duplicate_check["duplicate_count"]
        
        # 2. Check for missing data
        missing_check = self._check_missing_data(df, timeframe)
        validation_results["checks"]["missing_data"] = missing_check
        if not missing_check["valid"]:
            validation_results["valid"] = False
            validation_results["issues_found"] += missing_check["missing_points"]
        
        # 3. Check for price anomalies
        price_check = self._check_price_anomalies(df)
        validation_results["checks"]["price_anomalies"] = price_check
        if not price_check["valid"]:
            validation_results["valid"] = False
            validation_results["issues_found"] += len(price_check["anomalies"])
        
        # 4. Check for volume anomalies
        volume_check = self._check_volume_anomalies(df)
        validation_results["checks"]["volume_anomalies"] = volume_check
        if not volume_check["valid"]:
            validation_results["valid"] = False
            validation_results["issues_found"] += len(volume_check["anomalies"])
        
        # 5. Check OHLC validity
        ohlc_check = self._check_ohlc_validity(df)
        validation_results["checks"]["ohlc_validity"] = ohlc_check
        if not ohlc_check["valid"]:
            validation_results["valid"] = False
            validation_results["issues_found"] += len(ohlc_check["invalid_points"])
        
        # 6. Check for data staleness
        staleness_check = self._check_data_staleness(df, timeframe)
        validation_results["checks"]["data_staleness"] = staleness_check
        if not staleness_check["valid"]:
            validation_results["valid"] = False
            validation_results["issues_found"] += 1
        
        # If issues found but we still have enough data, mark as warning
        if not validation_results["valid"] and len(df) > 10:
            validation_results["warnings"] = validation_results["issues_found"]
            validation_results["status"] = "warning"
        elif not validation_results["valid"]:
            validation_results["status"] = "error"
        else:
            validation_results["status"] = "valid"
        
        # Save validation results
        self._save_validation_results(validation_results)
        
        return validation_results
    
    def _get_market_data(self, symbol: str, timeframe: str, days: int) -> List[Dict[str, Any]]:
        """
        Get market data from database.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            days: Number of days
            
        Returns:
            List of market data documents
        """
        try:
            # Calculate start date based on timeframe and days
            if timeframe == "day":
                # For daily data, use calendar days
                start_date = datetime.now() - timedelta(days=days)
            else:
                # For intraday data, use a buffer to account for non-trading days
                buffer_factor = 2 if days <= 30 else 1.5  # Larger buffer for shorter periods
                start_date = datetime.now() - timedelta(days=int(days * buffer_factor))
            
            # Query the database
            cursor = self.db.market_data_collection.find(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": {"$gte": start_date}
                }
            ).sort("timestamp", 1)
            
            data = list(cursor)
            
            if not data:
                self.logger.warning(f"No {timeframe} data found for {symbol} since {start_date}")
            else:
                self.logger.info(f"Retrieved {len(data)} {timeframe} data points for {symbol}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving market data: {e}")
            return []
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for duplicate data points.
        
        Args:
            df: DataFrame containing market data
            
        Returns:
            Dictionary with duplicate check results
        """
        # Check for exact duplicates
        duplicates = df[df.duplicated(subset=["timestamp"], keep="first")]
        duplicate_count = len(duplicates)
        
        # Remove duplicates from df for further processing
        if duplicate_count > 0:
            df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
        
        return {
            "valid": duplicate_count == 0,
            "duplicate_count": duplicate_count,
            "duplicate_dates": duplicates["timestamp"].tolist() if not duplicates.empty else []
        }
    
    def _check_missing_data(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        Check for missing data points.
        
        Args:
            df: DataFrame containing market data
            timeframe: Data timeframe
            
        Returns:
            Dictionary with missing data check results
        """
        # Sort dataframe by timestamp
        df = df.sort_values("timestamp")
        
        # Determine expected frequency based on timeframe
        if timeframe == "day":
            # Business days only (skip weekends)
            freq = "B"
        elif timeframe == "hour":
            # Hourly during trading hours
            freq = "H"
        elif timeframe == "1min":
            freq = "1min"
        elif timeframe == "5min":
            freq = "5min"
        elif timeframe == "15min":
            freq = "15min"
        else:
            # Default to daily
            freq = "B"
        
        # Create a date range at the expected frequency
        if timeframe == "day":
            # For daily data, use business days
            expected_dates = pd.date_range(
                start=df["timestamp"].min(),
                end=df["timestamp"].max(),
                freq=freq
            )
            
            # Filter expected dates to trading days only
            # This is a simplification - ideally we would check against a calendar of trading days
            expected_dates = expected_dates[expected_dates.dayofweek < 5]  # 0-4 are Monday-Friday
        else:
            # For intraday data, use trading hours (9:15 AM to 3:30 PM for Indian markets)
            # This is a simplification - would need a more sophisticated approach for production
            expected_dates = []
            current_date = df["timestamp"].min().date()
            end_date = df["timestamp"].max().date()
            
            while current_date <= end_date:
                # Skip weekends
                if current_date.weekday() < 5:  # 0-4 are Monday-Friday
                    if timeframe == "hour":
                        # Trading hours: 9:15 AM to 3:30 PM (round to hours)
                        for hour in range(9, 16):  # 9 AM to 3 PM
                            expected_dates.append(datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour))
                    elif timeframe in ["1min", "5min", "15min"]:
                        # Trading minutes: 9:15 AM to 3:30 PM
                        start_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=9, minutes=15)
                        end_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=15, minutes=30)
                        
                        # Generate expected times at the appropriate frequency
                        if timeframe == "1min":
                            delta = timedelta(minutes=1)
                        elif timeframe == "5min":
                            delta = timedelta(minutes=5)
                        else:  # 15min
                            delta = timedelta(minutes=15)
                        
                        current_time = start_time
                        while current_time <= end_time:
                            expected_dates.append(current_time)
                            current_time += delta
                
                current_date += timedelta(days=1)
            
            expected_dates = pd.DatetimeIndex(expected_dates)
        
        # Find missing dates
        actual_dates = pd.DatetimeIndex(df["timestamp"])
        missing_dates = expected_dates.difference(actual_dates)
        
        # Calculate longest gap
        if len(missing_dates) > 0:
            # Create a series of consecutive missing dates
            missing_series = pd.Series(missing_dates)
            
            # Calculate differences between consecutive dates
            date_diffs = missing_series.sort_values().diff()
            
            # Find longest gap in trading days
            if timeframe == "day":
                # For daily data, measure gaps in calendar days
                longest_gap = date_diffs.max().days if not date_diffs.empty else 0
            else:
                # For intraday data, measure gaps according to the expected frequency
                freq_multiplier = {
                    "hour": 60,
                    "1min": 1,
                    "5min": 5,
                    "15min": 15
                }
                
                # Convert timedelta to minutes and divide by the frequency
                longest_gap = date_diffs.max().total_seconds() / 60 / freq_multiplier.get(timeframe, 1) if not date_diffs.empty else 0
        else:
            longest_gap = 0
        
        # Calculate missing percentage
        missing_pct = (len(missing_dates) / len(expected_dates)) * 100 if len(expected_dates) > 0 else 0
        
        # Determine if the missing data is critical
        if timeframe == "day":
            # For daily data, check against max_missing_days
            valid = longest_gap <= self.thresholds["max_missing_days"]
        else:
            # For intraday data, primarily check percentage
            valid = missing_pct <= self.thresholds["max_missing_pct"]
        
        return {
            "valid": valid,
            "missing_points": len(missing_dates),
            "missing_percentage": missing_pct,
            "longest_gap": longest_gap,
            "threshold": self.thresholds["max_missing_days"] if timeframe == "day" else self.thresholds["max_missing_pct"],
            "missing_dates": missing_dates.tolist() if len(missing_dates) > 0 else []
        }
    
    def _check_price_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for price anomalies like extreme jumps or gaps.
        
        Args:
            df: DataFrame containing market data
            
        Returns:
            Dictionary with price anomaly check results
        """
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        # Calculate price changes
        df['price_change_pct'] = (df['close'].pct_change() * 100).fillna(0)
        
        # Calculate gap between previous close and current open
        df['gap_pct'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100).fillna(0)
        
        # Find price jumps
        price_jumps = df[abs(df['price_change_pct']) > self.thresholds["price_jump_threshold"]]
        
        # Find gaps
        gaps = df[abs(df['gap_pct']) > self.thresholds["price_gap_threshold"]]
        
        # Combine anomalies
        anomalies = []
        
        for _, row in price_jumps.iterrows():
            anomalies.append({
                "timestamp": row["timestamp"],
                "type": "price_jump",
                "value": row["price_change_pct"],
                "threshold": self.thresholds["price_jump_threshold"]
            })
        
        for _, row in gaps.iterrows():
            anomalies.append({
                "timestamp": row["timestamp"],
                "type": "price_gap",
                "value": row["gap_pct"],
                "threshold": self.thresholds["price_gap_threshold"]
            })
        
        return {
            "valid": len(anomalies) == 0,
            "anomalies": anomalies
        }
    
    def _check_volume_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for volume anomalies like spikes or zero values.
        
        Args:
            df: DataFrame containing market data
            
        Returns:
            Dictionary with volume anomaly check results
        """
        # Check for zero or missing volumes
        zero_volumes = df[df['volume'] == 0] if 'volume' in df.columns else pd.DataFrame()
        
        # Calculate average volume
        avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
        
        # Check for volume spikes
        volume_spikes = df[df['volume'] > avg_volume * self.thresholds["volume_spike_factor"]] if 'volume' in df.columns else pd.DataFrame()
        
        # Combine anomalies
        anomalies = []
        
        if not self.thresholds["zero_volume_allowed"]:
            for _, row in zero_volumes.iterrows():
                anomalies.append({
                    "timestamp": row["timestamp"],
                    "type": "zero_volume",
                    "value": 0,
                    "threshold": "non-zero"
                })
        
        for _, row in volume_spikes.iterrows():
            anomalies.append({
                "timestamp": row["timestamp"],
                "type": "volume_spike",
                "value": row["volume"],
                "avg_volume": avg_volume,
                "spike_factor": row["volume"] / avg_volume if avg_volume > 0 else float('inf'),
                "threshold": self.thresholds["volume_spike_factor"]
            })
        
        return {
            "valid": len(anomalies) == 0,
            "anomalies": anomalies
        }
    
    def _check_ohlc_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check the validity of OHLC data.
        
        Args:
            df: DataFrame containing market data
            
        Returns:
            Dictionary with OHLC validity check results
        """
        invalid_points = []
        
        # Check if high >= low
        high_low_invalid = df[df['high'] < df['low']]
        
        for _, row in high_low_invalid.iterrows():
            invalid_points.append({
                "timestamp": row["timestamp"],
                "type": "high_lower_than_low",
                "high": row["high"],
                "low": row["low"]
            })
        
        # Check if high < open or high < close
        high_invalid = df[(df['high'] < df['open']) | (df['high'] < df['close'])]
        
        for _, row in high_invalid.iterrows():
            invalid_points.append({
                "timestamp": row["timestamp"],
                "type": "high_lower_than_open_or_close",
                "high": row["high"],
                "open": row["open"],
                "close": row["close"]
            })
        
        # Check if low > open or low > close
        low_invalid = df[(df['low'] > df['open']) | (df['low'] > df['close'])]
        
        for _, row in low_invalid.iterrows():
            invalid_points.append({
                "timestamp": row["timestamp"],
                "type": "low_higher_than_open_or_close",
                "low": row["low"],
                "open": row["open"],
                "close": row["close"]
            })
        
        # Check if high-low range is too small
        min_diff_pct = self.thresholds["high_low_min_diff"]
        min_diff_check = df[((df['high'] - df['low']) / df['low'] * 100) < min_diff_pct]
        
        for _, row in min_diff_check.iterrows():
            invalid_points.append({
                "timestamp": row["timestamp"],
                "type": "high_low_too_close",
                "high": row["high"],
                "low": row["low"],
                "diff_pct": ((row["high"] - row["low"]) / row["low"] * 100),
                "threshold": min_diff_pct
            })
        
        return {
            "valid": len(invalid_points) == 0,
            "invalid_points": invalid_points
        }
    
    def _check_data_staleness(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        Check if data is stale (too old).
        
        Args:
            df: DataFrame containing market data
            timeframe: Data timeframe
            
        Returns:
            Dictionary with data staleness check results
        """
        if df.empty:
            return {
                "valid": False,
                "latest_data": None,
                "hours_since_update": float('inf'),
                "threshold": self.thresholds["max_staleness_hours"]
            }
        
        # Get latest timestamp
        latest_timestamp = df["timestamp"].max()
        
        # Calculate hours since latest update
        hours_since_update = (datetime.now() - latest_timestamp).total_seconds() / 3600
        
        # Check if data is stale based on timeframe
        if timeframe == "day":
            # For daily data, we only expect updates once per day
            # Check if we're missing today's data (during market hours) or yesterday's (after market hours)
            
            # Current time
            now = datetime.now()
            
            # Check if market is currently open (9:15 AM to 3:30 PM, Monday-Friday)
            market_open = (
                now.hour > 9 or (now.hour == 9 and now.minute >= 15)
            ) and (
                now.hour < 15 or (now.hour == 15 and now.minute <= 30)
            ) and now.weekday() < 5
            
            if market_open:
                # During market hours, we should have today's data
                valid = latest_timestamp.date() == now.date()
            else:
                # After market hours, we should have yesterday's data or today's data
                if now.hour < 9:
                    # Before market opens, yesterday's data is fine
                    latest_allowed = (now - timedelta(days=1)).date()
                    # If yesterday was a weekend, we need Friday's data
                    if latest_allowed.weekday() >= 5:  # Weekend
                        latest_allowed = latest_allowed - timedelta(days=latest_allowed.weekday() - 4)
                    valid = latest_timestamp.date() >= latest_allowed
                else:
                    # After market closes, we should have today's data
                    valid = latest_timestamp.date() == now.date()
        else:
            # For intraday data, check hours_since_update
            valid = hours_since_update <= self.thresholds["max_staleness_hours"]
        
        return {
            "valid": valid,
            "latest_data": latest_timestamp,
            "hours_since_update": hours_since_update,
            "threshold": self.thresholds["max_staleness_hours"]
        }
    
    def _save_validation_results(self, results: Dict[str, Any]) -> bool:
        """
        Save validation results to database.
        
        Args:
            results: Validation results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a document for the validation results
            document = {
                "symbol": results["symbol"],
                "timeframe": results["timeframe"],
                "timestamp": results["timestamp"],
                "status": results["status"],
                "valid": results["valid"],
                "issues_found": results["issues_found"],
                "warnings": results["warnings"],
                "checks": results["checks"]
            }
            
            # Insert into database
            self.db.market_data_validation_collection.insert_one(document)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving validation results: {e}")
            return False