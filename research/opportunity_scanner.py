"""
Opportunity Scanner

This module provides functionality to scan for trading opportunities across various instruments.
It leverages volatility and correlation analysis to identify high-potential trading setups.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import math
from scipy import stats
import talib
from concurrent.futures import ThreadPoolExecutor

class OpportunityScanner:
    """
    Scans for trading opportunities across different instruments.
    Identifies high-probability trading setups based on technical, volatility,
    and correlation characteristics.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the opportunity scanner with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Get volatility and correlation analyzers if available
        self.volatility_analyzer = None
        self.correlation_analyzer = None
        
        try:
            from research.volatility_analyzer import VolatilityAnalyzer
            self.volatility_analyzer = VolatilityAnalyzer(db_connector)
        except ImportError:
            self.logger.warning("VolatilityAnalyzer not found. Some scanning features will be limited.")
        
        try:
            from research.correlation_analyzer import CorrelationAnalyzer
            self.correlation_analyzer = CorrelationAnalyzer(db_connector)
        except ImportError:
            self.logger.warning("CorrelationAnalyzer not found. Some scanning features will be limited.")
        
        # Define scan parameters
        self.scan_params = {
            # Volatility-based parameters
            "high_volatility_percentile": 80,        # Percentile for high volatility
            "low_volatility_percentile": 20,         # Percentile for low volatility
            "volatility_expansion_threshold": 1.5,   # Threshold for volatility expansion
            "volatility_contraction_threshold": 0.5, # Threshold for volatility contraction
            
            # Trend parameters
            "strong_trend_adx": 25,                  # ADX threshold for strong trend
            "trend_lookback_periods": 5,             # Periods to confirm trend
            
            # Pattern detection
            "consolidation_range_percent": 3.0,      # Maximum range for consolidation
            "consolidation_min_periods": 5,          # Minimum periods for consolidation
            "breakout_threshold_percent": 2.0,       # Minimum move for breakout
            
            # Support/Resistance
            "support_resistance_lookback": 30,       # Periods to find support/resistance
            "support_resistance_threshold": 0.5,     # Proximity threshold percentage
            
            # Moving Averages
            "fast_ma_period": 20,                    # Fast moving average period
            "slow_ma_period": 50,                    # Slow moving average period
            "ma_crossover_confirmation": 2,          # Confirmation periods for MA crossover
            
            # Mean Reversion
            "overbought_rsi": 70,                    # RSI threshold for overbought
            "oversold_rsi": 30,                      # RSI threshold for oversold
            "mean_reversion_zscore": 2.0,            # Z-score threshold for mean reversion
            
            # Pair Trading
            "pair_zscore_threshold": 2.5,            # Z-score threshold for pair trading
            "pair_correlation_threshold": 0.7,       # Minimum correlation for pair trading
            
            # Technical Strength
            "technical_signal_threshold": 2,         # Number of agreeing signals needed
            
            # Scanner operation
            "max_opportunities": 10,                 # Maximum opportunities to return
            "max_threads": 8                         # Maximum threads for parallel scanning
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
    
    def scan_all_opportunities(self, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Scan all active instruments for trading opportunities.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            Dictionary with all identified opportunities
        """
        try:
            self.logger.info(f"Scanning all instruments for trading opportunities")
            
            # Get all active instruments
            instruments = self._get_active_instruments(exchange)
            
            if not instruments:
                return {
                    "status": "error",
                    "error": "No active instruments found"
                }
            
            self.logger.info(f"Found {len(instruments)} active instruments to scan")
            
            # Create opportunity categories
            opportunity_categories = {
                "breakout_opportunities": [],
                "trend_following_opportunities": [],
                "mean_reversion_opportunities": [],
                "volatility_based_opportunities": [],
                "pair_trading_opportunities": [],
                "support_resistance_opportunities": []
            }
            
            # Scan each instrument for opportunities
            with ThreadPoolExecutor(max_workers=min(self.scan_params["max_threads"], len(instruments))) as executor:
                # Create a list of futures
                future_to_instrument = {
                    executor.submit(self._scan_single_instrument, instrument): instrument
                    for instrument in instruments
                }
                
                # Process completed futures
                for future in future_to_instrument:
                    instrument = future_to_instrument[future]
                    try:
                        opportunities = future.result()
                        
                        # Add opportunities to their respective categories
                        for opp in opportunities:
                            category = opp.get("category", "other")
                            if category in opportunity_categories:
                                opportunity_categories[category].append(opp)
                    except Exception as e:
                        self.logger.error(f"Error scanning {instrument['symbol']}: {e}")
            
            # Scan for pair trading opportunities
            self.logger.info("Scanning for pair trading opportunities")
            pair_opportunities = self._scan_pair_opportunities(instruments, exchange)
            opportunity_categories["pair_trading_opportunities"].extend(pair_opportunities)
            
            # Filter and sort opportunities
            for category in opportunity_categories:
                # Sort by score (descending)
                opportunity_categories[category].sort(key=lambda x: x.get("score", 0), reverse=True)
                
                # Limit to max opportunities per category
                opportunity_categories[category] = opportunity_categories[category][:self.scan_params["max_opportunities"]]
            
            # Create a combined list of top opportunities across all categories
            all_opportunities = []
            for category, opportunities in opportunity_categories.items():
                all_opportunities.extend(opportunities)
            
            # Sort by score
            all_opportunities.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Limit to max opportunities overall
            all_opportunities = all_opportunities[:self.scan_params["max_opportunities"]]
            
            # Generate summary of opportunities
            summary = self._generate_opportunity_summary(opportunity_categories, all_opportunities)
            
            # Save opportunities to database
            self._save_opportunities(opportunity_categories, all_opportunities)
            
            return {
                "status": "success",
                "timestamp": datetime.now(),
                "total_instruments_scanned": len(instruments),
                "opportunity_categories": opportunity_categories,
                "top_opportunities": all_opportunities[:self.scan_params["max_opportunities"]],
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error scanning opportunities: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _scan_single_instrument(self, instrument: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Scan a single instrument for trading opportunities.
        
        Args:
            instrument: Instrument data
            
        Returns:
            List of opportunities
        """
        opportunities = []
        
        symbol = instrument["symbol"]
        exchange = instrument.get("exchange", "NSE")
        
        try:
            # Get market data
            data = self._get_market_data(symbol, exchange)
            
            if not data or len(data) < 30:
                return []
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Calculate technical indicators
            df = self._calculate_indicators(df)
            
            # Scan for different types of opportunities
            
            # 1. Breakout opportunities
            breakout = self._scan_breakout_opportunity(df, symbol, exchange)
            if breakout:
                opportunities.append(breakout)
            
            # 2. Trend following opportunities
            trend = self._scan_trend_following_opportunity(df, symbol, exchange)
            if trend:
                opportunities.append(trend)
            
            # 3. Mean reversion opportunities
            reversion = self._scan_mean_reversion_opportunity(df, symbol, exchange)
            if reversion:
                opportunities.append(reversion)
            
            # 4. Volatility-based opportunities
            volatility = self._scan_volatility_opportunity(df, symbol, exchange)
            if volatility:
                opportunities.append(volatility)
            
            # 5. Support/resistance opportunities
            support_resistance = self._scan_support_resistance_opportunity(df, symbol, exchange)
            if support_resistance:
                opportunities.append(support_resistance)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error scanning {symbol}: {e}")
            return []
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for opportunity scanning.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with added indicators
        """
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Moving Averages
            df['ma_fast'] = talib.SMA(df['close'].values, timeperiod=self.scan_params["fast_ma_period"])
            df['ma_slow'] = talib.SMA(df['close'].values, timeperiod=self.scan_params["slow_ma_period"])
            
            # Calculate MACD
            macd, macd_signal, _ = talib.MACD(df['close'].values)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd - macd_signal
            
            # Calculate RSI
            df['rsi'] = talib.RSI(df['close'].values)
            
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'].values)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle
            
            # Calculate ADX for trend strength
            df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values)
            
            # Calculate ATR for volatility
            df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values)
            
            # Calculate Stochastic
            df['slowk'], df['slowd'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
            
            # Calculate CCI
            df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values)
            
            # Calculate momentum
            df['roc'] = talib.ROC(df['close'].values)
            
            # Calculate price relative to moving averages
            df['price_to_ma_fast'] = df['close'] / df['ma_fast'] - 1
            df['price_to_ma_slow'] = df['close'] / df['ma_slow'] - 1
            
            # Calculate volatility
            df['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
            
            # Z-score for mean reversion (using 20-day mean)
            df['zscore'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return df
    
    def _scan_breakout_opportunity(self, df: pd.DataFrame, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        """
        Scan for breakout opportunities.
        
        Args:
            df: DataFrame with market data and indicators
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Opportunity dictionary or None
        """
        try:
            # Check if we have enough data
            if len(df) < 30:
                return None
            
            recent_data = df.iloc[-20:]
            
            # 1. Look for consolidation followed by breakout
            consolidation = True
            
            # Check range of recent candles (excluding the latest few)
            consolidation_data = recent_data.iloc[:-3]
            
            if len(consolidation_data) < 5:
                return None
            
            consolidation_high = consolidation_data['high'].max()
            consolidation_low = consolidation_data['low'].min()
            consolidation_range_pct = (consolidation_high - consolidation_low) / consolidation_low * 100
            
            # Check if the range is within the consolidation threshold
            if consolidation_range_pct > self.scan_params["consolidation_range_percent"]:
                consolidation = False
            
            # Check volatility contraction
            recent_volatility = recent_data['volatility'].iloc[-1]
            prev_volatility = df['volatility'].iloc[-30:-20].mean()
            
            volatility_contracting = recent_volatility < prev_volatility * self.scan_params["volatility_contraction_threshold"]
            
            # Check for breakout
            latest_close = df['close'].iloc[-1]
            latest_high = df['high'].iloc[-1]
            latest_volume = df.get('volume', pd.Series([0] * len(df))).iloc[-1]
            
            # Calculate average volume
            avg_volume = df.get('volume', pd.Series([0] * len(df))).iloc[-20:-1].mean()
            volume_surge = latest_volume > avg_volume * 1.5 if avg_volume > 0 else False
            
            # Check for breakout above consolidation
            breakout_up = latest_close > consolidation_high * (1 + self.scan_params["breakout_threshold_percent"] / 100)
            
            # Check for breakdown below consolidation
            breakout_down = latest_close < consolidation_low * (1 - self.scan_params["breakout_threshold_percent"] / 100)
            
            # Score the opportunity
            score = 0
            direction = "neutral"
            
            if consolidation and volatility_contracting:
                if breakout_up:
                    score += 3
                    direction = "long"
                    if volume_surge:
                        score += 1
                elif breakout_down:
                    score += 3
                    direction = "short"
                    if volume_surge:
                        score += 1
            
            # Add other confirming factors
            if direction == "long":
                # Confirming factors for upward breakout
                if df['macd_histogram'].iloc[-1] > 0 and df['macd_histogram'].iloc[-2] < 0:
                    score += 1  # MACD crossover
                
                if df['close'].iloc[-1] > df['ma_fast'].iloc[-1] > df['ma_slow'].iloc[-1]:
                    score += 1  # Price above both MAs, fast MA above slow MA
                
                if df['adx'].iloc[-1] > self.scan_params["strong_trend_adx"]:
                    score += 1  # Strong trend
                
            elif direction == "short":
                # Confirming factors for downward breakout
                if df['macd_histogram'].iloc[-1] < 0 and df['macd_histogram'].iloc[-2] > 0:
                    score += 1  # MACD crossover
                
                if df['close'].iloc[-1] < df['ma_fast'].iloc[-1] < df['ma_slow'].iloc[-1]:
                    score += 1  # Price below both MAs, fast MA below slow MA
                
                if df['adx'].iloc[-1] > self.scan_params["strong_trend_adx"]:
                    score += 1  # Strong trend
            
            # Create opportunity if score is high enough
            if score >= 3 and direction != "neutral":
                # Calculate distance to potential profit target and stop loss
                if direction == "long":
                    # Target based on previous range projection
                    range_size = consolidation_high - consolidation_low
                    target = latest_close + range_size
                    stop_loss = consolidation_high * 0.98  # Slightly below breakout level
                else:
                    # Target based on previous range projection
                    range_size = consolidation_high - consolidation_low
                    target = latest_close - range_size
                    stop_loss = consolidation_low * 1.02  # Slightly above breakdown level
                
                risk_reward = abs(target - latest_close) / abs(latest_close - stop_loss)
                
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "opportunity_type": "Breakout",
                    "category": "breakout_opportunities",
                    "direction": direction,
                    "score": score,
                    "current_price": latest_close,
                    "entry_price": latest_close,
                    "target_price": target,
                    "stop_loss": stop_loss,
                    "risk_reward_ratio": risk_reward,
                    "timestamp": datetime.now(),
                    "signals": {
                        "consolidation": consolidation,
                        "volatility_contracting": volatility_contracting,
                        "breakout_up": breakout_up,
                        "breakout_down": breakout_down,
                        "volume_surge": volume_surge
                    },
                    "description": f"{'Bullish' if direction == 'long' else 'Bearish'} breakout from consolidation"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scanning breakout for {symbol}: {e}")
            return None
    
    def _scan_trend_following_opportunity(self, df: pd.DataFrame, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        """
        Scan for trend following opportunities.
        
        Args:
            df: DataFrame with market data and indicators
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Opportunity dictionary or None
        """
        try:
            # Check if we have enough data
            if len(df) < 50:
                return None
            
            latest_close = df['close'].iloc[-1]
            
            # Identify trend direction and strength
            trend_direction = "neutral"
            score = 0
            
            # Check moving average alignment
            ma_fast_trend = df['ma_fast'].iloc[-1] > df['ma_fast'].iloc[-2]
            ma_slow_trend = df['ma_slow'].iloc[-1] > df['ma_slow'].iloc[-2]
            
            if ma_fast_trend and ma_slow_trend and df['ma_fast'].iloc[-1] > df['ma_slow'].iloc[-1]:
                trend_direction = "up"
                score += 1
            elif not ma_fast_trend and not ma_slow_trend and df['ma_fast'].iloc[-1] < df['ma_slow'].iloc[-1]:
                trend_direction = "down"
                score += 1
            
            # Check ADX for trend strength
            strong_trend = df['adx'].iloc[-1] > self.scan_params["strong_trend_adx"]
            if strong_trend:
                score += 1
            
            # Check consecutive candles in the same direction
            consecutive_up = True
            consecutive_down = True
            
            for i in range(1, self.scan_params["trend_lookback_periods"] + 1):
                if df['close'].iloc[-i] <= df['close'].iloc[-i-1]:
                    consecutive_up = False
                if df['close'].iloc[-i] >= df['close'].iloc[-i-1]:
                    consecutive_down = False
            
            if consecutive_up and trend_direction == "up":
                score += 1
            elif consecutive_down and trend_direction == "down":
                score += 1
            
            # Check MACD
            macd_positive = df['macd'].iloc[-1] > 0
            macd_above_signal = df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]
            
            if macd_positive and macd_above_signal and trend_direction == "up":
                score += 1
            elif not macd_positive and not macd_above_signal and trend_direction == "down":
                score += 1
            
            # Check RSI
            if trend_direction == "up" and df['rsi'].iloc[-1] > 50 and df['rsi'].iloc[-1] < 70:
                score += 1  # Strong but not overbought
            elif trend_direction == "down" and df['rsi'].iloc[-1] < 50 and df['rsi'].iloc[-1] > 30:
                score += 1  # Strong but not oversold
            
            # Check for pullback to moving average (entry opportunity in an existing trend)
            pullback_to_ma = False
            
            if trend_direction == "up":
                if df['low'].iloc[-1] <= df['ma_fast'].iloc[-1] <= df['close'].iloc[-1]:
                    pullback_to_ma = True
                    score += 1
            elif trend_direction == "down":
                if df['high'].iloc[-1] >= df['ma_fast'].iloc[-1] >= df['close'].iloc[-1]:
                    pullback_to_ma = True
                    score += 1
            
            # Create opportunity if score is high enough
            if score >= 4 and trend_direction != "neutral":
                direction = "long" if trend_direction == "up" else "short"
                
                # Calculate target and stop loss
                atr = df['atr'].iloc[-1]
                
                if direction == "long":
                    target = latest_close + (3 * atr)
                    stop_loss = latest_close - (1.5 * atr)
                else:
                    target = latest_close - (3 * atr)
                    stop_loss = latest_close + (1.5 * atr)
                
                risk_reward = abs(target - latest_close) / abs(latest_close - stop_loss)
                
                trend_type = "pullback" if pullback_to_ma else "continuation"
                
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "opportunity_type": f"Trend Following - {trend_type.capitalize()}",
                    "category": "trend_following_opportunities",
                    "direction": direction,
                    "score": score,
                    "current_price": latest_close,
                    "entry_price": latest_close,
                    "target_price": target,
                    "stop_loss": stop_loss,
                    "risk_reward_ratio": risk_reward,
                    "timestamp": datetime.now(),
                    "signals": {
                        "strong_trend": strong_trend,
                        "ma_alignment": ma_fast_trend == ma_slow_trend,
                        "consecutive_candles": consecutive_up if direction == "long" else consecutive_down,
                        "macd_confirmation": macd_positive if direction == "long" else not macd_positive,
                        "pullback_to_ma": pullback_to_ma
                    },
                    "description": f"{'Bullish' if direction == 'long' else 'Bearish'} trend following setup with {trend_type}"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scanning trend following for {symbol}: {e}")
            return None
    
    def _scan_mean_reversion_opportunity(self, df: pd.DataFrame, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        """
        Scan for mean reversion opportunities.
        
        Args:
            df: DataFrame with market data and indicators
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Opportunity dictionary or None
        """
        try:
            # Check if we have enough data
            if len(df) < 30:
                return None
            
            latest_close = df['close'].iloc[-1]
            
            # 1. Check RSI for overbought/oversold conditions
            rsi = df['rsi'].iloc[-1]
            rsi_prev = df['rsi'].iloc[-2]
            
            oversold = rsi < self.scan_params["oversold_rsi"]
            overbought = rsi > self.scan_params["overbought_rsi"]
            
            # Check for RSI divergence
            rsi_divergence = False
            direction = "neutral"
            
            if oversold and rsi > rsi_prev and df['close'].iloc[-1] < df['close'].iloc[-2]:
                # Positive divergence (price making lower lows, RSI making higher lows)
                rsi_divergence = True
                direction = "long"
            elif overbought and rsi < rsi_prev and df['close'].iloc[-1] > df['close'].iloc[-2]:
                # Negative divergence (price making higher highs, RSI making lower highs)
                rsi_divergence = True
                direction = "short"
            
            # 2. Check Bollinger Bands
            bb_squeeze = df['bb_width'].iloc[-1] < df['bb_width'].iloc[-20:].mean() * 0.8
            
            bb_signal = "neutral"
            if latest_close < df['bb_lower'].iloc[-1]:
                bb_signal = "long"  # Price below lower band
            elif latest_close > df['bb_upper'].iloc[-1]:
                bb_signal = "short"  # Price above upper band
            
            # 3. Check Z-score for extreme deviations
            zscore = df['zscore'].iloc[-1]
            zscore_signal = "neutral"
            
            if zscore < -self.scan_params["mean_reversion_zscore"]:
                zscore_signal = "long"  # Price significantly below mean
            elif zscore > self.scan_params["mean_reversion_zscore"]:
                zscore_signal = "short"  # Price significantly above mean
            
            # 4. Check distance from moving average
            ma_distance = df['price_to_ma_slow'].iloc[-1] * 100  # Convert to percentage
            ma_signal = "neutral"
            
            if ma_distance < -10:  # Price more than 10% below slow MA
                ma_signal = "long"
            elif ma_distance > 10:  # Price more than 10% above slow MA
                ma_signal = "short"
            
            # Score the opportunity
            score = 0
            
            # Determine consistent direction from signals
            signals = [
                direction if rsi_divergence else "neutral",
                bb_signal,
                zscore_signal,
                ma_signal
            ]
            
            # Count signals in each direction
            long_signals = signals.count("long")
            short_signals = signals.count("short")
            
            final_direction = "neutral"
            if long_signals > short_signals and long_signals >= 2:
                final_direction = "long"
                score += long_signals
            elif short_signals > long_signals and short_signals >= 2:
                final_direction = "short"
                score += short_signals
            
            # Add for RSI divergence which is a strong signal
            if rsi_divergence and direction == final_direction:
                score += 1
            
            # Add for Bollinger Band squeeze (volatility contraction)
            if bb_squeeze:
                score += 1
            
            # Create opportunity if score is high enough
            if score >= 3 and final_direction != "neutral":
                # Calculate target and stop loss
                if final_direction == "long":
                    # Target is typically the mean or upper band
                    target = df['bb_middle'].iloc[-1]
                    # Stop is typically a new low
                    stop_loss = min(df['low'].iloc[-5:]) * 0.99
                else:
                    # Target is typically the mean or lower band
                    target = df['bb_middle'].iloc[-1]
                    # Stop is typically a new high
                    stop_loss = max(df['high'].iloc[-5:]) * 1.01
                
                risk_reward = abs(target - latest_close) / abs(latest_close - stop_loss)
                
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "opportunity_type": "Mean Reversion",
                    "category": "mean_reversion_opportunities",
                    "direction": final_direction,
                    "score": score,
                    "current_price": latest_close,
                    "entry_price": latest_close,
                    "target_price": target,
                    "stop_loss": stop_loss,
                    "risk_reward_ratio": risk_reward,
                    "timestamp": datetime.now(),
                    "signals": {
                        "rsi": rsi,
                        "rsi_divergence": rsi_divergence,
                        "bb_squeeze": bb_squeeze,
                        "bb_signal": bb_signal,
                        "zscore": zscore,
                        "zscore_signal": zscore_signal,
                        "ma_distance": ma_distance,
                        "ma_signal": ma_signal
                    },
                    "description": f"{'Bullish' if final_direction == 'long' else 'Bearish'} mean reversion setup"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scanning mean reversion for {symbol}: {e}")
            return None
    
    def _scan_volatility_opportunity(self, df: pd.DataFrame, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        """
        Scan for volatility-based opportunities.
        
        Args:
            df: DataFrame with market data and indicators
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Opportunity dictionary or None
        """
        try:
            # Check if we have enough data
            if len(df) < 30:
                return None
            
            latest_close = df['close'].iloc[-1]
            
            # Get current and historical volatility
            current_volatility = df['volatility'].iloc[-1]
            historical_volatility = df['volatility'].iloc[-30:-1].mean()
            
            # Compare current to historical
            volatility_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1
            
            # Check for volatility expansion/contraction
            # Check for volatility expansion/contraction
            volatility_expanding = volatility_ratio > self.scan_params["volatility_expansion_threshold"]
            volatility_contracting = volatility_ratio < self.scan_params["volatility_contraction_threshold"]
            
            # Check Bollinger Band width
            bb_width = df['bb_width'].iloc[-1]
            bb_width_history = df['bb_width'].iloc[-30:-1].mean()
            bb_width_ratio = bb_width / bb_width_history if bb_width_history > 0 else 1
            
            # Check ATR
            atr = df['atr'].iloc[-1]
            atr_history = df['atr'].iloc[-30:-1].mean()
            atr_ratio = atr / atr_history if atr_history > 0 else 1
            
            # Look for volatility expansion setups
            expansion_opportunity = None
            contraction_opportunity = None
            
            # Volatility Expansion Opportunity
            if volatility_expanding and bb_width_ratio > 1.3 and atr_ratio > 1.3:
                # Determine direction
                direction = "neutral"
                score = 3  # Start with base score for volatility expansion
                
                # Check recent price action
                if df['close'].iloc[-1] > df['close'].iloc[-5]:
                    up_candles = sum(1 for i in range(1, 6) if df['close'].iloc[-i] > df['open'].iloc[-i])
                    if up_candles >= 3:
                        direction = "long"
                elif df['close'].iloc[-1] < df['close'].iloc[-5]:
                    down_candles = sum(1 for i in range(1, 6) if df['close'].iloc[-i] < df['open'].iloc[-i])
                    if down_candles >= 3:
                        direction = "short"
                
                # Check for breakout with the expansion
                if direction == "long" and df['close'].iloc[-1] > max(df['high'].iloc[-20:-1]):
                    score += 2  # Strong breakout
                elif direction == "short" and df['close'].iloc[-1] < min(df['low'].iloc[-20:-1]):
                    score += 2  # Strong breakdown
                
                # Check volume confirmation
                if 'volume' in df.columns:
                    latest_volume = df['volume'].iloc[-1]
                    avg_volume = df['volume'].iloc[-20:-1].mean()
                    if latest_volume > avg_volume * 1.5:
                        score += 1
                
                if direction != "neutral" and score >= 4:
                    # Calculate target and stop loss
                    if direction == "long":
                        target = latest_close + (atr * 3)
                        stop_loss = latest_close - (atr * 1.5)
                    else:
                        target = latest_close - (atr * 3)
                        stop_loss = latest_close + (atr * 1.5)
                    
                    risk_reward = abs(target - latest_close) / abs(latest_close - stop_loss)
                    
                    expansion_opportunity = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "opportunity_type": "Volatility Expansion",
                        "category": "volatility_based_opportunities",
                        "direction": direction,
                        "score": score,
                        "current_price": latest_close,
                        "entry_price": latest_close,
                        "target_price": target,
                        "stop_loss": stop_loss,
                        "risk_reward_ratio": risk_reward,
                        "timestamp": datetime.now(),
                        "signals": {
                            "volatility_ratio": volatility_ratio,
                            "bb_width_ratio": bb_width_ratio,
                            "atr_ratio": atr_ratio,
                            "volume_surge": latest_volume > avg_volume * 1.5 if 'volume' in df.columns else False
                        },
                        "description": f"{'Bullish' if direction == 'long' else 'Bearish'} volatility expansion setup"
                    }
            
            # Volatility Contraction Opportunity (often precedes expansion)
            if volatility_contracting and bb_width_ratio < 0.7 and atr_ratio < 0.7:
                # Volatility contraction often precedes a strong move, but direction is uncertain
                score = 3  # Base score for volatility contraction
                
                # Look for directional clues
                direction = "neutral"
                
                # Check if price is near support/resistance
                price_near_top = latest_close > df['bb_upper'].iloc[-1] * 0.95
                price_near_bottom = latest_close < df['bb_lower'].iloc[-1] * 1.05
                
                if price_near_top:
                    direction = "short"  # Potential breakdown from resistance
                    score += 1
                elif price_near_bottom:
                    direction = "long"   # Potential breakout from support
                    score += 1
                
                # Check for consolidation pattern
                if max(df['high'].iloc[-10:]) - min(df['low'].iloc[-10:]) < atr * 2:
                    score += 1  # Tight consolidation
                
                if direction != "neutral" and score >= 4:
                    # For contraction setups, we often wait for the breakout
                    # so the entry would be a conditional order
                    if direction == "long":
                        entry_price = max(df['high'].iloc[-5:]) * 1.01  # Breakout level
                        target = entry_price + (atr * 3)
                        stop_loss = min(df['low'].iloc[-5:]) * 0.99
                    else:
                        entry_price = min(df['low'].iloc[-5:]) * 0.99  # Breakdown level
                        target = entry_price - (atr * 3)
                        stop_loss = max(df['high'].iloc[-5:]) * 1.01
                    
                    risk_reward = abs(target - entry_price) / abs(entry_price - stop_loss)
                    
                    contraction_opportunity = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "opportunity_type": "Volatility Contraction",
                        "category": "volatility_based_opportunities",
                        "direction": direction,
                        "score": score,
                        "current_price": latest_close,
                        "entry_price": entry_price,  # Conditional entry
                        "target_price": target,
                        "stop_loss": stop_loss,
                        "risk_reward_ratio": risk_reward,
                        "timestamp": datetime.now(),
                        "signals": {
                            "volatility_ratio": volatility_ratio,
                            "bb_width_ratio": bb_width_ratio,
                            "atr_ratio": atr_ratio,
                            "price_location": "near_top" if price_near_top else "near_bottom" if price_near_bottom else "middle"
                        },
                        "description": f"{'Bullish' if direction == 'long' else 'Bearish'} volatility contraction setup"
                    }
            
            # Return the higher-scored opportunity
            if expansion_opportunity and contraction_opportunity:
                return expansion_opportunity if expansion_opportunity["score"] >= contraction_opportunity["score"] else contraction_opportunity
            elif expansion_opportunity:
                return expansion_opportunity
            elif contraction_opportunity:
                return contraction_opportunity
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scanning volatility for {symbol}: {e}")
            return None
    
    def _scan_support_resistance_opportunity(self, df: pd.DataFrame, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        """
        Scan for support/resistance trading opportunities.
        
        Args:
            df: DataFrame with market data and indicators
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Opportunity dictionary or None
        """
        try:
            # Check if we have enough data
            if len(df) < 50:
                return None
            
            latest_close = df['close'].iloc[-1]
            
            # Find potential support and resistance levels
            levels = self._find_support_resistance_levels(df)
            
            if not levels or len(levels) < 2:
                return None
            
            # Find the nearest level to current price
            nearest_level = None
            nearest_distance = float('inf')
            level_type = None
            
            for level in levels:
                distance = abs(latest_close - level["price"])
                distance_pct = distance / latest_close * 100
                
                if distance_pct < nearest_distance:
                    nearest_distance = distance_pct
                    nearest_level = level
                    level_type = level["type"]
            
            # Check if price is near a level
            if nearest_distance > self.scan_params["support_resistance_threshold"] * 100:
                return None  # Not close enough to any level
            
            # Determine direction based on level type and price position
            direction = "neutral"
            if level_type == "support" and latest_close > nearest_level["price"] * 1.01:
                direction = "long"  # Bounced off support
            elif level_type == "resistance" and latest_close < nearest_level["price"] * 0.99:
                direction = "short"  # Rejected at resistance
            
            if direction == "neutral":
                return None  # No clear direction
            
            # Score the opportunity
            score = 3  # Base score for being near a significant level
            
            # Check for confirmation signals
            
            # 1. Check for candlestick patterns
            if direction == "long":
                # Look for bullish candles
                if df['close'].iloc[-1] > df['open'].iloc[-1] and (df['close'].iloc[-1] - df['open'].iloc[-1]) > (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.6:
                    score += 1  # Strong bullish candle
                
                # Check for hammer pattern
                lower_wick = df['open'].iloc[-1] - df['low'].iloc[-1]
                if lower_wick > (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.6:
                    score += 1  # Hammer pattern
            
            elif direction == "short":
                # Look for bearish candles
                if df['close'].iloc[-1] < df['open'].iloc[-1] and (df['open'].iloc[-1] - df['close'].iloc[-1]) > (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.6:
                    score += 1  # Strong bearish candle
                
                # Check for shooting star pattern
                upper_wick = df['high'].iloc[-1] - df['open'].iloc[-1]
                if upper_wick > (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.6:
                    score += 1  # Shooting star pattern
            
            # 2. Check for RSI confirmation
            if direction == "long" and df['rsi'].iloc[-1] > df['rsi'].iloc[-2] and df['rsi'].iloc[-1] > 40:
                score += 1  # Rising RSI
            elif direction == "short" and df['rsi'].iloc[-1] < df['rsi'].iloc[-2] and df['rsi'].iloc[-1] < 60:
                score += 1  # Falling RSI
            
            # 3. Check volume for confirmation
            if 'volume' in df.columns:
                latest_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].iloc[-20:-1].mean()
                if latest_volume > avg_volume * 1.2:
                    score += 1  # Higher than average volume
            
            # Create opportunity if score is high enough
            if score >= 4:
                # Calculate target and stop loss
                atr = df['atr'].iloc[-1]
                
                if direction == "long":
                    # Target is the next resistance level or a multiple of ATR
                    higher_levels = [l["price"] for l in levels if l["price"] > latest_close]
                    if higher_levels:
                        target = min(higher_levels)  # Nearest resistance
                    else:
                        target = latest_close + (atr * 2)
                    
                    # Stop is below the support level
                    stop_loss = nearest_level["price"] * 0.99
                else:
                    # Target is the next support level or a multiple of ATR
                    lower_levels = [l["price"] for l in levels if l["price"] < latest_close]
                    if lower_levels:
                        target = max(lower_levels)  # Nearest support
                    else:
                        target = latest_close - (atr * 2)
                    
                    # Stop is above the resistance level
                    stop_loss = nearest_level["price"] * 1.01
                
                risk_reward = abs(target - latest_close) / abs(latest_close - stop_loss)
                
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "opportunity_type": f"{level_type.capitalize()} Trade",
                    "category": "support_resistance_opportunities",
                    "direction": direction,
                    "score": score,
                    "current_price": latest_close,
                    "entry_price": latest_close,
                    "target_price": target,
                    "stop_loss": stop_loss,
                    "risk_reward_ratio": risk_reward,
                    "timestamp": datetime.now(),
                    "signals": {
                        "level_type": level_type,
                        "level_price": nearest_level["price"],
                        "distance_percent": nearest_distance,
                        "level_strength": nearest_level["strength"]
                    },
                    "description": f"{'Bullish' if direction == 'long' else 'Bearish'} {level_type} trade"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scanning support/resistance for {symbol}: {e}")
            return None
    
    def _find_support_resistance_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find support and resistance levels.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            List of support/resistance levels
        """
        levels = []
        
        # Use the last N periods for analysis
        lookback = min(len(df) - 1, self.scan_params["support_resistance_lookback"])
        
        if lookback < 20:
            return []
        
        # Use high and low values to identify levels
        highs = df['high'].iloc[-lookback:-1].values  # Exclude the most recent candle
        lows = df['low'].iloc[-lookback:-1].values
        
        # Function to find local maxima/minima
        def find_local_extrema(data, min_strength=2):
            extrema = []
            for i in range(1, len(data) - 1):
                # Check if this point is a local maximum
                if data[i] > data[i-1] and data[i] > data[i+1]:
                    # Check strength (how many nearby points are lower)
                    strength = 0
                    for j in range(max(0, i-5), min(len(data), i+6)):
                        if i != j and data[i] > data[j]:
                            strength += 1
                    
                    if strength >= min_strength:
                        extrema.append({"price": data[i], "index": i, "strength": strength})
            
            return extrema
        
        # Find local maxima (potential resistance)
        resistance_levels = find_local_extrema(highs)
        
        # Find local minima (potential support)
        support_levels = find_local_extrema(-lows)
        for level in support_levels:
            level["price"] = -level["price"]
        
        # Tag levels as support or resistance
        for level in resistance_levels:
            level["type"] = "resistance"
            levels.append(level)
        
        for level in support_levels:
            level["type"] = "support"
            levels.append(level)
        
        # Cluster nearby levels
        if levels:
            clustered_levels = []
            levels.sort(key=lambda x: x["price"])
            
            current_cluster = [levels[0]]
            
            for i in range(1, len(levels)):
                current_price = levels[i]["price"]
                prev_price = current_cluster[-1]["price"]
                
                # If this level is close to the previous one, add to cluster
                if abs(current_price - prev_price) / prev_price < 0.01:  # 1% threshold
                    current_cluster.append(levels[i])
                else:
                    # Process the current cluster
                    if current_cluster:
                        avg_price = sum(l["price"] for l in current_cluster) / len(current_cluster)
                        avg_strength = sum(l["strength"] for l in current_cluster) / len(current_cluster)
                        level_type = max(set(l["type"] for l in current_cluster), key=[l["type"] for l in current_cluster].count)
                        
                        clustered_levels.append({
                            "price": avg_price,
                            "strength": avg_strength,
                            "type": level_type
                        })
                    
                    # Start a new cluster
                    current_cluster = [levels[i]]
            
            # Add the last cluster
            if current_cluster:
                avg_price = sum(l["price"] for l in current_cluster) / len(current_cluster)
                avg_strength = sum(l["strength"] for l in current_cluster) / len(current_cluster)
                level_type = max(set(l["type"] for l in current_cluster), key=[l["type"] for l in current_cluster].count)
                
                clustered_levels.append({
                    "price": avg_price,
                    "strength": avg_strength,
                    "type": level_type
                })
            
            # Sort by strength
            clustered_levels.sort(key=lambda x: x["strength"], reverse=True)
            
            return clustered_levels
        
        return []
    
    def _scan_pair_opportunities(self, instruments: List[Dict[str, Any]], exchange: str) -> List[Dict[str, Any]]:
        """
        Scan for pair trading opportunities.
        
        Args:
            instruments: List of instruments
            exchange: Stock exchange
            
        Returns:
            List of pair trading opportunities
        """
        try:
            self.logger.info("Scanning for pair trading opportunities")
            
            if not self.correlation_analyzer:
                self.logger.warning("Correlation analyzer not available, skipping pair scan")
                return []
            
            opportunities = []
            
            # Get symbols
            symbols = [instrument["symbol"] for instrument in instruments]
            
            # If we have too many symbols, focus on the most liquid ones
            if len(symbols) > 30:
                # Sort by trading volume or other liquidity metric
                instruments.sort(key=lambda x: x.get("liquidity", 0), reverse=True)
                symbols = [instrument["symbol"] for instrument in instruments[:30]]
            
            if len(symbols) < 5:
                return []  # Need at least a few symbols for meaningful analysis
            
            # 1. Check for cointegrated pairs
            try:
                # Get price data for all symbols
                price_data = {}
                for symbol in symbols:
                    data = self._get_market_data(symbol, exchange)
                    if not data or len(data) < 60:
                        continue
                    
                    df = pd.DataFrame(data)
                    if "close" not in df.columns or "timestamp" not in df.columns:
                        continue
                    
                    # Extract closing prices with timestamps
                    price_data[symbol] = df[["timestamp", "close"]].sort_values("timestamp")
                
                valid_symbols = list(price_data.keys())
                
                if len(valid_symbols) < 5:
                    return []
                
                # Combine prices into a single DataFrame
                combined_df = None
                for symbol, data in price_data.items():
                    if combined_df is None:
                        combined_df = data.rename(columns={"close": symbol}).set_index("timestamp")
                    else:
                        combined_df[symbol] = data.set_index("timestamp")["close"]
                
                # Fill missing values
                combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
                
                # Calculate correlation matrix
                correlation_matrix = combined_df.corr()
                
                # Find highly correlated pairs
                correlated_pairs = []
                
                for i in range(len(valid_symbols)):
                    for j in range(i+1, len(valid_symbols)):
                        symbol1 = valid_symbols[i]
                        symbol2 = valid_symbols[j]
                        correlation = correlation_matrix.loc[symbol1, symbol2]
                        
                        if correlation >= self.scan_params["pair_correlation_threshold"]:
                            correlated_pairs.append({
                                "symbol1": symbol1,
                                "symbol2": symbol2,
                                "correlation": correlation
                            })
                
                # Sort by correlation
                correlated_pairs.sort(key=lambda x: x["correlation"], reverse=True)
                
                # Test for cointegration
                from statsmodels.tsa.stattools import coint
                
                cointegrated_pairs = []
                
                for pair in correlated_pairs[:20]:  # Test top 20 correlated pairs
                    symbol1 = pair["symbol1"]
                    symbol2 = pair["symbol2"]
                    
                    prices1 = combined_df[symbol1].values
                    prices2 = combined_df[symbol2].values
                    
                    # Run cointegration test
                    score, pvalue, _ = coint(prices1, prices2)
                    
                    # Check if cointegrated at 5% significance level
                    if pvalue < 0.05:
                        # Calculate hedge ratio using linear regression
                        import statsmodels.api as sm
                        model = sm.OLS(prices1, prices2).fit()
                        hedge_ratio = model.params[0]
                        
                        cointegrated_pairs.append({
                            "symbol1": symbol1,
                            "symbol2": symbol2,
                            "correlation": pair["correlation"],
                            "p_value": pvalue,
                            "hedge_ratio": hedge_ratio
                        })
                
                # For each cointegrated pair, calculate the spread and z-score
                for pair in cointegrated_pairs:
                    symbol1 = pair["symbol1"]
                    symbol2 = pair["symbol2"]
                    hedge_ratio = pair["hedge_ratio"]
                    
                    # Calculate spread
                    spread = combined_df[symbol1] - hedge_ratio * combined_df[symbol2]
                    
                    # Calculate z-score
                    mean = spread.mean()
                    std = spread.std()
                    z_score = (spread.iloc[-1] - mean) / std
                    
                    pair["spread_mean"] = mean
                    pair["spread_std"] = std
                    pair["z_score"] = z_score
                    
                    # Latest prices
                    price1 = combined_df[symbol1].iloc[-1]
                    price2 = combined_df[symbol2].iloc[-1]
                    
                    pair["price1"] = price1
                    pair["price2"] = price2
                    
                    # Check if z-score is extreme enough for trading
                    if abs(z_score) >= self.scan_params["pair_zscore_threshold"]:
                        # Determine direction
                        direction = "neutral"
                        if z_score > 0:
                            # Spread is positive (above mean) - short spread
                            # Short symbol1, long symbol2
                            direction = "short_spread"
                        else:
                            # Spread is negative (below mean) - long spread
                            # Long symbol1, short symbol2
                            direction = "long_spread"
                        
                        # Calculate entry amounts
                        amount1 = 100000  # Notional amount for symbol1
                        shares1 = amount1 / price1
                        shares2 = shares1 * hedge_ratio
                        amount2 = shares2 * price2
                        
                        # Target is mean reversion
                        target_z = 0  # Target mean z-score
                        
                        # Stop is extended move away from mean
                        if z_score > 0:
                            stop_z = z_score + 1.0  # Stop if spread widens further
                        else:
                            stop_z = z_score - 1.0  # Stop if spread widens further
                        
                        # Calculate target and stop prices
                        target_spread = mean
                        stop_spread = mean + stop_z * std
                        
                        # Risk-reward
                        risk = abs(stop_spread - spread.iloc[-1])
                        reward = abs(target_spread - spread.iloc[-1])
                        risk_reward = reward / risk if risk > 0 else 0
                        
                        # Create opportunity
                        if risk_reward >= 1.5:
                            opportunity = {
                                "symbol": f"{symbol1}/{symbol2}",
                                "exchange": exchange,
                                "opportunity_type": "Pair Trading",
                                "category": "pair_trading_opportunities",
                                "direction": direction,
                                "score": 4 + min(abs(z_score) - self.scan_params["pair_zscore_threshold"], 2),
                                "timestamp": datetime.now(),
                                "pair_details": {
                                    "symbol1": symbol1,
                                    "symbol2": symbol2,
                                    "price1": price1,
                                    "price2": price2,
                                    "correlation": pair["correlation"],
                                    "hedge_ratio": hedge_ratio,
                                    "z_score": z_score,
                                    "shares1": shares1 if direction == "long_spread" else -shares1,
                                    "shares2": -shares2 if direction == "long_spread" else shares2,
                                    "amount1": amount1 if direction == "long_spread" else -amount1,
                                    "amount2": -amount2 if direction == "long_spread" else amount2
                                },
                                "risk_reward_ratio": risk_reward,
                                "signals": {
                                    "cointegration": pair["p_value"],
                                    "correlation": pair["correlation"],
                                    "z_score": z_score
                                },
                                "description": f"{'Long' if direction == 'long_spread' else 'Short'} the spread between {symbol1} and {symbol2} (z-score: {z_score:.2f})"
                            }
                            
                            opportunities.append(opportunity)
                
                return opportunities[:self.scan_params["max_opportunities"]]
                
            except Exception as e:
                self.logger.error(f"Error scanning for cointegrated pairs: {e}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error scanning pair opportunities: {e}")
            return []
    
    def _get_active_instruments(self, exchange: str) -> List[Dict[str, Any]]:
        """
        Get all active instruments for scanning.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            List of active instruments
        """
        try:
            # Query the portfolio collection for active instruments
            cursor = self.db.portfolio_collection.find({
                "exchange": exchange,
                "status": "active",
                "trading_config.enabled": True
            })
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting active instruments: {e}")
            return []
    
    def _get_market_data(self, symbol: str, exchange: str) -> List[Dict[str, Any]]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            List of market data documents
        """
        try:
            # Default to 100 days of daily data
            days = 100
            timeframe = "day"
            
            # Calculate the start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Create query
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            # Get data from database
            cursor = self.db.market_data_collection.find(query).sort("timestamp", 1)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return []
    
    def _save_opportunities(self, categories: Dict[str, List[Dict[str, Any]]], 
                         all_opportunities: List[Dict[str, Any]]) -> None:
        """
        Save opportunities to database.
        
        Args:
            categories: Opportunity categories
            all_opportunities: Combined list of opportunities
        """
        try:
            # Create document
            document = {
                "type": "opportunity_scan",
                "timestamp": datetime.now(),
                "categories": categories,
                "all_opportunities": all_opportunities
            }
            
            # Insert into database
            self.db.opportunity_scan_collection.insert_one(document)
            
        except Exception as e:
            self.logger.error(f"Error saving opportunities: {e}")
    
    def _generate_opportunity_summary(self, categories: Dict[str, List[Dict[str, Any]]],
                                   all_opportunities: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of identified opportunities.
        
        Args:
            categories: Opportunity categories
            all_opportunities: Combined list of opportunities
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Overall summary
        total_opportunities = sum(len(opps) for opps in categories.values())
        
        if total_opportunities == 0:
            return "No trading opportunities identified in the current scan."
        
        summary_parts.append(f"Identified {total_opportunities} trading opportunities across {len(categories)} categories.")
        
        # Category breakdown
        category_counts = {category: len(opportunities) for category, opportunities in categories.items() if opportunities}
        
        if category_counts:
            summary_parts.append("Breakdown by category:")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                category_name = category.replace("_opportunities", "").replace("_", " ").title()
                summary_parts.append(f"- {category_name}: {count}")
        
        # Top opportunities
        if all_opportunities:
            top_three = all_opportunities[:3]
            
            summary_parts.append("\nTop opportunities:")
            for i, opp in enumerate(top_three):
                description = opp.get("description", "Unknown opportunity")
                score = opp.get("score", 0)
                risk_reward = opp.get("risk_reward_ratio", 0)
                
                summary_parts.append(f"{i+1}. {description} (Score: {score}, RR: {risk_reward:.2f})")
        
        # Market context
        # Market context
            if all_opportunities:
                # Count direction bias
                long_count = sum(1 for opp in all_opportunities if opp.get("direction") == "long")
                short_count = sum(1 for opp in all_opportunities if opp.get("direction") == "short")
                
                if long_count > short_count * 2:
                    summary_parts.append("\nStrong bullish bias observed with significantly more long opportunities than short.")
                elif short_count > long_count * 2:
                    summary_parts.append("\nStrong bearish bias observed with significantly more short opportunities than long.")
                elif long_count > short_count:
                    summary_parts.append("\nSlightly bullish bias observed with more long opportunities than short.")
                elif short_count > long_count:
                    summary_parts.append("\nSlightly bearish bias observed with more short opportunities than long.")
                else:
                    summary_parts.append("\nBalanced market with equal long and short opportunities.")
                
                # Check opportunity types
                breakout_count = len(categories.get("breakout_opportunities", []))
                trend_count = len(categories.get("trend_following_opportunities", []))
                reversion_count = len(categories.get("mean_reversion_opportunities", []))
                
                if breakout_count > trend_count and breakout_count > reversion_count:
                    summary_parts.append("Market favors breakout strategies in the current environment.")
                elif trend_count > breakout_count and trend_count > reversion_count:
                    summary_parts.append("Market favors trend following strategies in the current environment.")
                elif reversion_count > breakout_count and reversion_count > trend_count:
                    summary_parts.append("Market favors mean reversion strategies in the current environment.")
        
        return "\n".join(summary_parts)
    
    def get_opportunity_details(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Get detailed opportunity analysis for a specific symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with opportunity details
        """
        try:
            self.logger.info(f"Getting opportunity details for {symbol}")
            
            # Get market data
            data = self._get_market_data(symbol, exchange)
            
            if not data or len(data) < 30:
                return {
                    "status": "error",
                    "error": "Insufficient data"
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            # Get opportunity signals
            
            # 1. Technical signals
            technical_signals = self._get_technical_signals(df)
            
            # 2. Price levels
            price_levels = self._find_support_resistance_levels(df)
            
            # 3. Volatility analysis
            volatility_analysis = self._analyze_volatility(df, symbol, exchange)
            
            # 4. Recent opportunities
            recent_opportunities = self._get_recent_opportunities(symbol, exchange)
            
            # 5. Related opportunities (sector, correlations)
            related_opportunities = self._get_related_opportunities(symbol, exchange)
            
            # Generate opportunity summary
            opportunity_summary = self._generate_symbol_opportunity_summary(
                symbol, technical_signals, price_levels, 
                volatility_analysis, recent_opportunities
            )
            
            # Current price and recent trend
            latest_close = df['close'].iloc[-1]
            week_change = (latest_close / df['close'].iloc[-6] - 1) * 100 if len(df) >= 6 else None
            month_change = (latest_close / df['close'].iloc[-22] - 1) * 100 if len(df) >= 22 else None
            
            return {
                "status": "success",
                "symbol": symbol,
                "exchange": exchange,
                "current_price": latest_close,
                "week_change_percent": week_change,
                "month_change_percent": month_change,
                "timestamp": datetime.now(),
                "technical_signals": technical_signals,
                "price_levels": price_levels,
                "volatility_analysis": volatility_analysis,
                "recent_opportunities": recent_opportunities,
                "related_opportunities": related_opportunities,
                "opportunity_summary": opportunity_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error getting opportunity details for {symbol}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_technical_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get technical signals from indicator data.
        
        Args:
            df: DataFrame with market data and indicators
            
        Returns:
            Dictionary with technical signals
        """
        signals = {
            "bullish": [],
            "bearish": [],
            "neutral": []
        }
        
        # Price vs Moving Averages
        if df['close'].iloc[-1] > df['ma_fast'].iloc[-1]:
            signals["bullish"].append("Price above fast MA")
        else:
            signals["bearish"].append("Price below fast MA")
        
        if df['close'].iloc[-1] > df['ma_slow'].iloc[-1]:
            signals["bullish"].append("Price above slow MA")
        else:
            signals["bearish"].append("Price below slow MA")
        
        if df['ma_fast'].iloc[-1] > df['ma_slow'].iloc[-1]:
            signals["bullish"].append("Fast MA above slow MA")
        else:
            signals["bearish"].append("Fast MA below slow MA")
        
        # MA Crossover
        if df['ma_fast'].iloc[-1] > df['ma_slow'].iloc[-1] and df['ma_fast'].iloc[-2] <= df['ma_slow'].iloc[-2]:
            signals["bullish"].append("Bullish MA crossover")
        elif df['ma_fast'].iloc[-1] < df['ma_slow'].iloc[-1] and df['ma_fast'].iloc[-2] >= df['ma_slow'].iloc[-2]:
            signals["bearish"].append("Bearish MA crossover")
        
        # MACD
        if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
            signals["bullish"].append("MACD above signal line")
        else:
            signals["bearish"].append("MACD below signal line")
        
        if df['macd'].iloc[-1] > 0:
            signals["bullish"].append("MACD positive")
        else:
            signals["bearish"].append("MACD negative")
        
        if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
            signals["bullish"].append("Bullish MACD crossover")
        elif df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
            signals["bearish"].append("Bearish MACD crossover")
        
        # RSI
        if df['rsi'].iloc[-1] < 30:
            signals["bullish"].append("RSI oversold")
        elif df['rsi'].iloc[-1] > 70:
            signals["bearish"].append("RSI overbought")
        else:
            signals["neutral"].append(f"RSI neutral ({df['rsi'].iloc[-1]:.1f})")
        
        # Bollinger Bands
        if df['close'].iloc[-1] < df['bb_lower'].iloc[-1]:
            signals["bullish"].append("Price below lower Bollinger Band")
        elif df['close'].iloc[-1] > df['bb_upper'].iloc[-1]:
            signals["bearish"].append("Price above upper Bollinger Band")
        else:
            signals["neutral"].append("Price within Bollinger Bands")
        
        # Bollinger Band Width
        bb_width = df['bb_width'].iloc[-1]
        bb_width_prev = df['bb_width'].iloc[-5:].mean()
        
        if bb_width < bb_width_prev * 0.8:
            signals["neutral"].append("Bollinger Band squeeze (contracting volatility)")
        elif bb_width > bb_width_prev * 1.2:
            signals["neutral"].append("Bollinger Band expansion (expanding volatility)")
        
        # ADX (Trend Strength)
        if df['adx'].iloc[-1] > 25:
            signals["neutral"].append(f"Strong trend (ADX: {df['adx'].iloc[-1]:.1f})")
        else:
            signals["neutral"].append(f"Weak trend (ADX: {df['adx'].iloc[-1]:.1f})")
        
        # Stochastic
        if df['slowk'].iloc[-1] < 20 and df['slowd'].iloc[-1] < 20:
            signals["bullish"].append("Stochastic oversold")
        elif df['slowk'].iloc[-1] > 80 and df['slowd'].iloc[-1] > 80:
            signals["bearish"].append("Stochastic overbought")
        
        if df['slowk'].iloc[-1] > df['slowd'].iloc[-1] and df['slowk'].iloc[-2] <= df['slowd'].iloc[-2]:
            signals["bullish"].append("Bullish Stochastic crossover")
        elif df['slowk'].iloc[-1] < df['slowd'].iloc[-1] and df['slowk'].iloc[-2] >= df['slowd'].iloc[-2]:
            signals["bearish"].append("Bearish Stochastic crossover")
        
        # CCI
        if df['cci'].iloc[-1] < -100:
            signals["bullish"].append("CCI oversold")
        elif df['cci'].iloc[-1] > 100:
            signals["bearish"].append("CCI overbought")
        
        # ROC (Momentum)
        if df['roc'].iloc[-1] > 0:
            signals["bullish"].append("Positive momentum")
        else:
            signals["bearish"].append("Negative momentum")
        
        # Count signals
        bullish_count = len(signals["bullish"])
        bearish_count = len(signals["bearish"])
        neutral_count = len(signals["neutral"])
        
        # Determine overall bias
        if bullish_count > bearish_count:
            overall_bias = "bullish"
        elif bearish_count > bullish_count:
            overall_bias = "bearish"
        else:
            overall_bias = "neutral"
        
        # Calculate signal strength (0-100)
        total_signals = bullish_count + bearish_count
        if total_signals > 0:
            signal_strength = int(max(bullish_count, bearish_count) / total_signals * 100)
        else:
            signal_strength = 50
        
        return {
            "bullish_signals": signals["bullish"],
            "bearish_signals": signals["bearish"],
            "neutral_signals": signals["neutral"],
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "overall_bias": overall_bias,
            "signal_strength": signal_strength
        }
    
    def _analyze_volatility(self, df: pd.DataFrame, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get volatility analysis for a symbol.
        
        Args:
            df: DataFrame with market data
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with volatility analysis
        """
        # If volatility analyzer is available, use it
        if self.volatility_analyzer:
            vol_analysis = self.volatility_analyzer.get_recent_volatility_analysis(symbol, exchange)
            if vol_analysis:
                return vol_analysis
        
        # Otherwise do a simplified analysis
        current_volatility = df['volatility'].iloc[-1]
        historical_volatility = df['volatility'].iloc[-30:-1].mean()
        
        # Compare current to historical
        volatility_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1
        
        # Check for volatility expansion/contraction
        if volatility_ratio > 1.5:
            volatility_state = "expanding"
        elif volatility_ratio < 0.7:
            volatility_state = "contracting"
        else:
            volatility_state = "stable"
        
        # Determine volatility regime
        if current_volatility > df['volatility'].quantile(0.8):
            volatility_regime = "high"
        elif current_volatility < df['volatility'].quantile(0.2):
            volatility_regime = "low"
        else:
            volatility_regime = "normal"
        
        # Analyze ATR
        atr = df['atr'].iloc[-1]
        atr_pct = atr / df['close'].iloc[-1] * 100
        
        # Generate volatility summary
        if volatility_regime == "high" and volatility_state == "expanding":
            summary = f"Volatility is high and expanding. Consider wider stops and reduced position sizes. Daily ATR: {atr_pct:.2f}% of price."
        elif volatility_regime == "high" and volatility_state == "stable":
            summary = f"Volatility is high but stable. Maintain wider stops. Daily ATR: {atr_pct:.2f}% of price."
        elif volatility_regime == "high" and volatility_state == "contracting":
            summary = f"Volatility is high but contracting. Be alert for potential regime change. Daily ATR: {atr_pct:.2f}% of price."
        elif volatility_regime == "normal" and volatility_state == "expanding":
            summary = f"Volatility is normal but expanding. Consider adjusting position sizing. Daily ATR: {atr_pct:.2f}% of price."
        elif volatility_regime == "normal" and volatility_state == "stable":
            summary = f"Volatility is normal and stable. Standard trading approach appropriate. Daily ATR: {atr_pct:.2f}% of price."
        elif volatility_regime == "normal" and volatility_state == "contracting":
            summary = f"Volatility is normal and contracting. Be alert for potential breakout setups. Daily ATR: {atr_pct:.2f}% of price."
        elif volatility_regime == "low" and volatility_state == "expanding":
            summary = f"Volatility is low but expanding. Be alert for new trends developing. Daily ATR: {atr_pct:.2f}% of price."
        elif volatility_regime == "low" and volatility_state == "stable":
            summary = f"Volatility is low and stable. Consider strategies that benefit from quiet markets. Daily ATR: {atr_pct:.2f}% of price."
        else:  # low and contracting
            summary = f"Volatility is low and contracting further. Be alert for extremely tight ranges and potential explosive moves. Daily ATR: {atr_pct:.2f}% of price."
        
        return {
            "current_volatility": current_volatility,
            "historical_volatility": historical_volatility,
            "volatility_ratio": volatility_ratio,
            "volatility_state": volatility_state,
            "volatility_regime": volatility_regime,
            "atr": atr,
            "atr_percent": atr_pct,
            "summary": summary
        }
    
    def _get_recent_opportunities(self, symbol: str, exchange: str) -> List[Dict[str, Any]]:
        """
        Get recent trading opportunities for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            List of recent opportunities
        """
        try:
            # Query for recent opportunities
            cursor = self.db.opportunity_scan_collection.find({
                "timestamp": {"$gte": datetime.now() - timedelta(days=7)}
            }).sort("timestamp", -1).limit(5)
            
            scans = list(cursor)
            
            if not scans:
                return []
            
            # Extract opportunities for the specific symbol
            opportunities = []
            
            for scan in scans:
                if "all_opportunities" not in scan:
                    continue
                
                for opp in scan["all_opportunities"]:
                    if opp.get("symbol") == symbol and opp.get("exchange") == exchange:
                        # Add scan timestamp
                        opp["scan_timestamp"] = scan["timestamp"]
                        opportunities.append(opp)
            
            # Sort by timestamp
            opportunities.sort(key=lambda x: x.get("scan_timestamp", datetime.now()), reverse=True)
            
            return opportunities[:5]  # Return most recent 5
            
        except Exception as e:
            self.logger.error(f"Error getting recent opportunities for {symbol}: {e}")
            return []
    
    def _get_related_opportunities(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get related opportunities (same sector, correlated).
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with related opportunities
        """
        try:
            # Try to find the sector for this symbol
            instrument = self.db.portfolio_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            })
            
            if not instrument:
                return {"sector_opportunities": [], "correlated_opportunities": []}
            
            sector = instrument.get("sector", "unknown")
            
            # Get recent scan
            recent_scan = self.db.opportunity_scan_collection.find_one(
                {}, sort=[("timestamp", -1)]
            )
            
            if not recent_scan or "all_opportunities" not in recent_scan:
                return {"sector_opportunities": [], "correlated_opportunities": []}
            
            # Find opportunities in the same sector
            sector_opportunities = []
            
            # Get all instruments in this sector
            sector_instruments = list(self.db.portfolio_collection.find({
                "sector": sector,
                "exchange": exchange
            }))
            sector_symbols = [instr["symbol"] for instr in sector_instruments]
            
            for opp in recent_scan["all_opportunities"]:
                opp_symbol = opp.get("symbol", "")
                if opp_symbol in sector_symbols and opp_symbol != symbol:
                    sector_opportunities.append(opp)
            
            # Find correlated opportunities using correlation analyzer
            correlated_opportunities = []
            
            if self.correlation_analyzer:
                # Find correlated symbols
                correlated_symbols = []
                
                try:
                    analysis = self.correlation_analyzer.get_recent_correlation_analysis([symbol], exchange)
                    
                    if analysis and "high_correlation_pairs" in analysis:
                        for pair in analysis["high_correlation_pairs"]:
                            if pair["symbol1"] == symbol and pair["symbol2"] not in correlated_symbols:
                                correlated_symbols.append(pair["symbol2"])
                            elif pair["symbol2"] == symbol and pair["symbol1"] not in correlated_symbols:
                                correlated_symbols.append(pair["symbol1"])
                except:
                    pass
                
                # Find opportunities for correlated symbols
                for opp in recent_scan["all_opportunities"]:
                    opp_symbol = opp.get("symbol", "")
                    if opp_symbol in correlated_symbols:
                        correlated_opportunities.append(opp)
            
            return {
                "sector_opportunities": sector_opportunities[:5],  # Top 5
                "correlated_opportunities": correlated_opportunities[:5]  # Top 5
            }
            
        except Exception as e:
            self.logger.error(f"Error getting related opportunities for {symbol}: {e}")
            return {"sector_opportunities": [], "correlated_opportunities": []}
    
    def _generate_symbol_opportunity_summary(self, symbol: str, 
                                         technical_signals: Dict[str, Any],
                                         price_levels: List[Dict[str, Any]],
                                         volatility_analysis: Dict[str, Any],
                                         recent_opportunities: List[Dict[str, Any]]) -> str:
        """
        Generate an opportunity summary for a symbol.
        
        Args:
            symbol: Stock symbol
            technical_signals: Technical signals dictionary
            price_levels: Price levels list
            volatility_analysis: Volatility analysis dictionary
            recent_opportunities: Recent opportunities list
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Overall bias
        overall_bias = technical_signals.get("overall_bias", "neutral")
        signal_strength = technical_signals.get("signal_strength", 0)
        
        if overall_bias == "bullish":
            summary_parts.append(f"{symbol} shows a bullish bias with {signal_strength}% signal strength.")
        elif overall_bias == "bearish":
            summary_parts.append(f"{symbol} shows a bearish bias with {signal_strength}% signal strength.")
        else:
            summary_parts.append(f"{symbol} shows a neutral bias with mixed signals.")
        
        # Key technical signals
        bullish_signals = technical_signals.get("bullish_signals", [])
        bearish_signals = technical_signals.get("bearish_signals", [])
        
        if bullish_signals:
            summary_parts.append(f"Key bullish signals: {', '.join(bullish_signals[:3])}.")
        
        if bearish_signals:
            summary_parts.append(f"Key bearish signals: {', '.join(bearish_signals[:3])}.")
        
        # Volatility state
        if volatility_analysis:
            volatility_summary = volatility_analysis.get("summary", "")
            if volatility_summary:
                summary_parts.append(volatility_summary)
        
        # Support and resistance levels
        if price_levels:
            # Sort by distance from last price
            last_price = price_levels[0].get("last_price", 0)
            if last_price > 0:
                for level in price_levels:
                    level["distance"] = abs(level["price"] - last_price) / last_price
                
                price_levels.sort(key=lambda x: x.get("distance", 0))
            
            # Mention nearest levels
            nearest_support = None
            nearest_resistance = None
            
            for level in price_levels:
                if level["type"] == "support" and nearest_support is None:
                    nearest_support = level
                elif level["type"] == "resistance" and nearest_resistance is None:
                    nearest_resistance = level
                
                if nearest_support and nearest_resistance:
                    break
            
            if nearest_support:
                summary_parts.append(f"Nearest support at {nearest_support['price']:.2f}.")
            
            if nearest_resistance:
                summary_parts.append(f"Nearest resistance at {nearest_resistance['price']:.2f}.")
        
        # Recent opportunities
        if recent_opportunities:
            latest_opp = recent_opportunities[0]
            opp_type = latest_opp.get("opportunity_type", "")
            direction = latest_opp.get("direction", "")
            timestamp = latest_opp.get("scan_timestamp", datetime.now())
            days_ago = (datetime.now() - timestamp).days
            
            if days_ago == 0:
                time_str = "today"
            elif days_ago == 1:
                time_str = "yesterday"
            else:
                time_str = f"{days_ago} days ago"
            
            if direction == "long":
                summary_parts.append(f"A bullish {opp_type} opportunity was identified {time_str}.")
            elif direction == "short":
                summary_parts.append(f"A bearish {opp_type} opportunity was identified {time_str}.")
        
        # Timeframe recommendations based on volatility
        if volatility_analysis:
            vol_regime = volatility_analysis.get("volatility_regime", "normal")
            vol_state = volatility_analysis.get("volatility_state", "stable")
            
            if vol_regime == "high":
                if overall_bias == "bullish":
                    summary_parts.append("Consider short-term bullish trades with tight stops due to high volatility.")
                elif overall_bias == "bearish":
                    summary_parts.append("Consider short-term bearish trades with tight stops due to high volatility.")
            elif vol_regime == "low" and vol_state == "contracting":
                summary_parts.append("Consider breakout strategies as low and contracting volatility often precedes significant moves.")
            elif overall_bias == "bullish" and vol_regime == "normal":
                summary_parts.append("Conditions favor medium-term bullish trend-following strategies.")
            elif overall_bias == "bearish" and vol_regime == "normal":
                summary_parts.append("Conditions favor medium-term bearish trend-following strategies.")
        
        return " ".join(summary_parts)

# Helper functions for opportunity scanner
def is_hammer_candle(row: pd.Series) -> bool:
    """Check if a candle is a hammer pattern."""
    body_size = abs(row['close'] - row['open'])
    total_range = row['high'] - row['low']
    
    if total_range == 0:
        return False
    
    body_percent = body_size / total_range
    
    if body_percent > 0.5:
        return False
    
    if row['close'] >= row['open']:  # Bullish
        lower_wick = row['open'] - row['low']
    else:  # Bearish
        lower_wick = row['close'] - row['low']
    
    lower_percent = lower_wick / total_range
    
    return lower_percent > 0.6

def is_shooting_star_candle(row: pd.Series) -> bool:
    """Check if a candle is a shooting star pattern."""
    body_size = abs(row['close'] - row['open'])
    total_range = row['high'] - row['low']
    
    if total_range == 0:
        return False
    
    body_percent = body_size / total_range
    
    if body_percent > 0.5:
        return False
    
    if row['close'] >= row['open']:  # Bullish
        upper_wick = row['high'] - row['close']
    else:  # Bearish
        upper_wick = row['high'] - row['open']
    
    upper_percent = upper_wick / total_range
    
    return upper_percent > 0.6