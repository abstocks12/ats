"""
Technical Trading Strategy Module

This module implements technical analysis-based trading strategies including:
- Trend following
- Momentum
- Breakout
- Support/resistance
- Pattern recognition
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import math
import talib
from talib import abstract

class TechnicalStrategy:
    """
    Implements technical analysis-based trading strategies.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the strategy with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Strategy parameters (configurable)
        self.params = {
            # General parameters
            "lookback_period": 100,  # Days for technical analysis
            "min_data_points": 30,  # Minimum data points needed
            
            # Trend following parameters
            "ma_short": 20,  # Short moving average period
            "ma_medium": 50,  # Medium moving average period
            "ma_long": 200,  # Long moving average period
            
            # Momentum parameters
            "rsi_period": 14,  # RSI period
            "rsi_overbought": 70,  # RSI overbought threshold
            "rsi_oversold": 30,  # RSI oversold threshold
            
            # Breakout parameters
            "breakout_period": 20,  # Lookback period for breakout
            "breakout_threshold": 0.03,  # Minimum breakout size (percent)
            
            # Support/Resistance parameters
            "sr_lookback": 60,  # Lookback period for support/resistance
            "sr_threshold": 0.02,  # Support/resistance threshold (percent)
            "sr_touch_count": 2,  # Minimum touches to confirm level
            
            # Pattern recognition parameters
            "pattern_confirmation": 3,  # Candles for pattern confirmation
            
            # Position sizing
            "risk_per_trade": 0.01,  # 1% risk per trade
            
            # Filters
            "min_volume_ratio": 1.2,  # Minimum volume ratio to average
            "min_atr_percent": 0.01,  # Minimum ATR as percent of price
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
    
    def analyze_trend(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze trend for a given symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            # Get market data
            data = self._get_market_data(symbol, exchange, days=self.params["lookback_period"])
            
            if not data or len(data) < self.params["min_data_points"]:
                self.logger.warning(f"Insufficient data for trend analysis of {symbol}")
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(data).sort_values("timestamp")
            
            # Calculate various moving averages
            df['ma_short'] = df['close'].rolling(window=self.params["ma_short"]).mean()
            df['ma_medium'] = df['close'].rolling(window=self.params["ma_medium"]).mean()
            df['ma_long'] = df['close'].rolling(window=self.params["ma_long"]).mean()
            
            # Get current values
            current_price = df['close'].iloc[-1]
            current_ma_short = df['ma_short'].iloc[-1]
            current_ma_medium = df['ma_medium'].iloc[-1]
            current_ma_long = df['ma_long'].iloc[-1] if len(df) >= self.params["ma_long"] else None
            
            # Identify trend based on moving averages
            short_term_trend = "bullish" if current_price > current_ma_short else "bearish"
            medium_term_trend = "bullish" if current_price > current_ma_medium else "bearish"
            long_term_trend = "neutral"
            
            if current_ma_long is not None:
                long_term_trend = "bullish" if current_price > current_ma_long else "bearish"
            
            # Check for moving average crossovers
            crossovers = []
            
            # Short-Medium MA crossover
            if len(df) > 2:
                prev_short = df['ma_short'].iloc[-2]
                prev_medium = df['ma_medium'].iloc[-2]
                
                if (current_ma_short > current_ma_medium and prev_short <= prev_medium):
                    crossovers.append({
                        "type": "golden_cross_short_medium",
                        "description": f"{self.params['ma_short']}-day MA crossed above {self.params['ma_medium']}-day MA",
                        "signal": "bullish"
                    })
                elif (current_ma_short < current_ma_medium and prev_short >= prev_medium):
                    crossovers.append({
                        "type": "death_cross_short_medium",
                        "description": f"{self.params['ma_short']}-day MA crossed below {self.params['ma_medium']}-day MA",
                        "signal": "bearish"
                    })
            
            # Medium-Long MA crossover
            if current_ma_long is not None and len(df) > 2:
                prev_medium = df['ma_medium'].iloc[-2]
                prev_long = df['ma_long'].iloc[-2]
                
                if (current_ma_medium > current_ma_long and prev_medium <= prev_long):
                    crossovers.append({
                        "type": "golden_cross_medium_long",
                        "description": f"{self.params['ma_medium']}-day MA crossed above {self.params['ma_long']}-day MA",
                        "signal": "bullish"
                    })
                elif (current_ma_medium < current_ma_long and prev_medium >= prev_long):
                    crossovers.append({
                        "type": "death_cross_medium_long",
                        "description": f"{self.params['ma_medium']}-day MA crossed below {self.params['ma_long']}-day MA",
                        "signal": "bearish"
                    })
            
            # Determine trend strength
            trend_alignment_score = 0
            
            if short_term_trend == medium_term_trend:
                trend_alignment_score += 1
            
            if medium_term_trend == long_term_trend:
                trend_alignment_score += 1
            
            if short_term_trend == long_term_trend:
                trend_alignment_score += 1
            
            trend_strength = "weak"
            if trend_alignment_score == 3:
                trend_strength = "strong"
            elif trend_alignment_score == 2:
                trend_strength = "moderate"
            
            # Calculate ADX for trend strength if available
            adx = None
            try:
                if len(df) >= 14:
                    # TA-Lib names need high, low, close as uppercase
                    df_talib = df.rename(columns={'high': 'High', 'low': 'Low', 'close': 'Close'})
                    adx = talib.ADX(df_talib['High'].values, df_talib['Low'].values, df_talib['Close'].values, timeperiod=14)
                    adx = adx[-1]
                    
                    # ADX > 25 indicates strong trend
                    if adx > 25:
                        trend_strength = "strong"
                    elif adx < 20:
                        trend_strength = "weak"
            except Exception as e:
                self.logger.warning(f"Error calculating ADX: {e}")
            
            # Determine overall trend
            overall_trend = "neutral"
            
            if short_term_trend == medium_term_trend:
                overall_trend = short_term_trend
            
            if medium_term_trend == long_term_trend and trend_strength != "weak":
                overall_trend = medium_term_trend
            
            # Check if price is near a moving average (potential support/resistance)
            ma_proximity = []
            price_tolerance = 0.02  # 2% tolerance
            
            if abs(current_price - current_ma_short) / current_price < price_tolerance:
                ma_proximity.append({
                    "ma": f"{self.params['ma_short']}-day",
                    "value": current_ma_short,
                    "type": "support" if current_price > current_ma_short else "resistance"
                })
            
            if abs(current_price - current_ma_medium) / current_price < price_tolerance:
                ma_proximity.append({
                    "ma": f"{self.params['ma_medium']}-day",
                    "value": current_ma_medium,
                    "type": "support" if current_price > current_ma_medium else "resistance"
                })
            
            if current_ma_long is not None and abs(current_price - current_ma_long) / current_price < price_tolerance:
                ma_proximity.append({
                    "ma": f"{self.params['ma_long']}-day",
                    "value": current_ma_long,
                    "type": "support" if current_price > current_ma_long else "resistance"
                })
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "current_price": current_price,
                "moving_averages": {
                    "short": current_ma_short,
                    "medium": current_ma_medium,
                    "long": current_ma_long
                },
                "trends": {
                    "short_term": short_term_trend,
                    "medium_term": medium_term_trend,
                    "long_term": long_term_trend,
                    "overall": overall_trend
                },
                "trend_strength": trend_strength,
                "adx": adx,
                "crossovers": crossovers,
                "ma_proximity": ma_proximity,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend for {symbol}: {e}")
            return {}
    
    def analyze_momentum(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze momentum indicators for a given symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with momentum analysis results
        """
        try:
            # Get market data
            data = self._get_market_data(symbol, exchange, days=self.params["lookback_period"])
            
            if not data or len(data) < self.params["min_data_points"]:
                self.logger.warning(f"Insufficient data for momentum analysis of {symbol}")
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(data).sort_values("timestamp")
            
            # Prepare data for TA-Lib (which expects uppercase column names)
            df_talib = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            
            # Calculate RSI
            rsi = None
            try:
                rsi = talib.RSI(df_talib['Close'].values, timeperiod=self.params["rsi_period"])
                current_rsi = rsi[-1]
            except Exception as e:
                self.logger.warning(f"Error calculating RSI: {e}")
                current_rsi = None
            
            # Calculate MACD
            macd = None
            macd_signal = None
            macd_hist = None
            try:
                macd, macd_signal, macd_hist = talib.MACD(
                    df_talib['Close'].values, 
                    fastperiod=12, 
                    slowperiod=26, 
                    signalperiod=9
                )
                current_macd = macd[-1]
                current_macd_signal = macd_signal[-1]
                current_macd_hist = macd_hist[-1]
            except Exception as e:
                self.logger.warning(f"Error calculating MACD: {e}")
                current_macd = None
                current_macd_signal = None
                current_macd_hist = None
            
            # Calculate Stochastic Oscillator
            slowk = None
            slowd = None
            try:
                slowk, slowd = talib.STOCH(
                    df_talib['High'].values,
                    df_talib['Low'].values,
                    df_talib['Close'].values,
                    fastk_period=14,
                    slowk_period=3,
                    slowk_matype=0,
                    slowd_period=3,
                    slowd_matype=0
                )
                current_slowk = slowk[-1]
                current_slowd = slowd[-1]
            except Exception as e:
                self.logger.warning(f"Error calculating Stochastic: {e}")
                current_slowk = None
                current_slowd = None
            
            # Calculate Rate of Change (ROC)
            roc = None
            try:
                roc = talib.ROC(df_talib['Close'].values, timeperiod=10)
                current_roc = roc[-1]
            except Exception as e:
                self.logger.warning(f"Error calculating ROC: {e}")
                current_roc = None
            
            # Calculate Money Flow Index (MFI)
            mfi = None
            try:
                if 'Volume' in df_talib.columns:
                    mfi = talib.MFI(
                        df_talib['High'].values,
                        df_talib['Low'].values,
                        df_talib['Close'].values,
                        df_talib['Volume'].values,
                        timeperiod=14
                    )
                    current_mfi = mfi[-1]
                else:
                    current_mfi = None
            except Exception as e:
                self.logger.warning(f"Error calculating MFI: {e}")
                current_mfi = None
            
            # Determine momentum signals
            momentum_signals = []
            
            # RSI signals
            if current_rsi is not None:
                if current_rsi < self.params["rsi_oversold"]:
                    momentum_signals.append({
                        "indicator": "RSI",
                        "value": current_rsi,
                        "signal": "oversold",
                        "interpretation": "bullish",
                        "description": f"RSI ({current_rsi:.2f}) below oversold threshold ({self.params['rsi_oversold']})"
                    })
                elif current_rsi > self.params["rsi_overbought"]:
                    momentum_signals.append({
                        "indicator": "RSI",
                        "value": current_rsi,
                        "signal": "overbought",
                        "interpretation": "bearish",
                        "description": f"RSI ({current_rsi:.2f}) above overbought threshold ({self.params['rsi_overbought']})"
                    })
                else:
                    # Check for RSI divergence
                    if len(df) > 10 and rsi is not None:
                        # Look for bullish divergence (price makes lower low but RSI makes higher low)
                        if (df['close'].iloc[-1] < df['close'].iloc[-5]) and (rsi[-1] > rsi[-5]):
                            momentum_signals.append({
                                "indicator": "RSI",
                                "value": current_rsi,
                                "signal": "bullish_divergence",
                                "interpretation": "bullish",
                                "description": "RSI showing bullish divergence (price making lower lows, RSI making higher lows)"
                            })
                        # Look for bearish divergence (price makes higher high but RSI makes lower high)
                        elif (df['close'].iloc[-1] > df['close'].iloc[-5]) and (rsi[-1] < rsi[-5]):
                            momentum_signals.append({
                                "indicator": "RSI",
                                "value": current_rsi,
                                "signal": "bearish_divergence",
                                "interpretation": "bearish",
                                "description": "RSI showing bearish divergence (price making higher highs, RSI making lower highs)"
                            })
            
            # MACD signals
            if current_macd is not None and current_macd_signal is not None:
                # MACD line crosses above signal line (bullish)
                if len(df) > 1 and macd is not None and macd_signal is not None:
                    if macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]:
                        momentum_signals.append({
                            "indicator": "MACD",
                            "value": current_macd,
                            "signal": "bullish_crossover",
                            "interpretation": "bullish",
                            "description": "MACD line crossed above signal line"
                        })
                    # MACD line crosses below signal line (bearish)
                    elif macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]:
                        momentum_signals.append({
                            "indicator": "MACD",
                            "value": current_macd,
                            "signal": "bearish_crossover",
                            "interpretation": "bearish",
                            "description": "MACD line crossed below signal line"
                        })
                
                # MACD histogram turning positive or negative
                if current_macd_hist is not None and len(df) > 1 and macd_hist is not None:
                    if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                        momentum_signals.append({
                            "indicator": "MACD_Histogram",
                            "value": current_macd_hist,
                            "signal": "turned_positive",
                            "interpretation": "bullish",
                            "description": "MACD histogram turned positive"
                        })
                    elif macd_hist[-1] < 0 and macd_hist[-2] >= 0:
                        momentum_signals.append({
                            "indicator": "MACD_Histogram",
                            "value": current_macd_hist,
                            "signal": "turned_negative",
                            "interpretation": "bearish",
                            "description": "MACD histogram turned negative"
                        })
            
            # Stochastic signals
            if current_slowk is not None and current_slowd is not None:
                # Oversold conditions
                if current_slowk < 20 and current_slowd < 20:
                    momentum_signals.append({
                        "indicator": "Stochastic",
                        "value": current_slowk,
                        "signal": "oversold",
                        "interpretation": "bullish",
                        "description": "Stochastic oscillator in oversold territory"
                    })
                # Overbought conditions
                elif current_slowk > 80 and current_slowd > 80:
                    momentum_signals.append({
                        "indicator": "Stochastic",
                        "value": current_slowk,
                        "signal": "overbought",
                        "interpretation": "bearish",
                        "description": "Stochastic oscillator in overbought territory"
                    })
                
                # Stochastic crossovers
                if len(df) > 1 and slowk is not None and slowd is not None:
                    if slowk[-1] > slowd[-1] and slowk[-2] <= slowd[-2]:
                        momentum_signals.append({
                            "indicator": "Stochastic",
                            "value": current_slowk,
                            "signal": "bullish_crossover",
                            "interpretation": "bullish",
                            "description": "Stochastic %K crossed above %D"
                        })
                    elif slowk[-1] < slowd[-1] and slowk[-2] >= slowd[-2]:
                        momentum_signals.append({
                            "indicator": "Stochastic",
                            "value": current_slowk,
                            "signal": "bearish_crossover",
                            "interpretation": "bearish",
                            "description": "Stochastic %K crossed below %D"
                        })
            
            # MFI signals
            if current_mfi is not None:
                if current_mfi < 20:
                    momentum_signals.append({
                        "indicator": "MFI",
                        "value": current_mfi,
                        "signal": "oversold",
                        "interpretation": "bullish",
                        "description": "Money Flow Index below 20, indicating oversold conditions"
                    })
                elif current_mfi > 80:
                    momentum_signals.append({
                        "indicator": "MFI",
                        "value": current_mfi,
                        "signal": "overbought",
                        "interpretation": "bearish",
                        "description": "Money Flow Index above 80, indicating overbought conditions"
                    })
            
            # Determine overall momentum
            bullish_signals = sum(1 for signal in momentum_signals if signal["interpretation"] == "bullish")
            bearish_signals = sum(1 for signal in momentum_signals if signal["interpretation"] == "bearish")
            
            overall_momentum = "neutral"
            if bullish_signals > bearish_signals:
                overall_momentum = "bullish"
            elif bearish_signals > bullish_signals:
                overall_momentum = "bearish"
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "current_price": df['close'].iloc[-1],
                "indicators": {
                    "rsi": current_rsi,
                    "macd": current_macd,
                    "macd_signal": current_macd_signal,
                    "macd_histogram": current_macd_hist,
                    "stochastic_k": current_slowk,
                    "stochastic_d": current_slowd,
                    "roc": current_roc,
                    "mfi": current_mfi
                },
                "momentum_signals": momentum_signals,
                "overall_momentum": overall_momentum,
                "bullish_signals": bullish_signals,
                "bearish_signals": bearish_signals,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum for {symbol}: {e}")
            return {}
    
    def identify_support_resistance(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Identify support and resistance levels.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            # Get market data
            data = self._get_market_data(symbol, exchange, days=self.params["sr_lookback"])
            
            if not data or len(data) < self.params["min_data_points"]:
                self.logger.warning(f"Insufficient data for support/resistance analysis of {symbol}")
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(data).sort_values("timestamp")
            
            # Calculate Pivot Points (using the last day's data)
            last_high = df['high'].iloc[-1]
            last_low = df['low'].iloc[-1]
            last_close = df['close'].iloc[-1]
            
            pivot = (last_high + last_low + last_close) / 3
            
            s1 = 2 * pivot - last_high
            s2 = pivot - (last_high - last_low)
            s3 = s2 - (last_high - last_low)
            
            r1 = 2 * pivot - last_low
            r2 = pivot + (last_high - last_low)
            r3 = r2 + (last_high - last_low)
            
            pivot_points = {
                "pivot": pivot,
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "r1": r1,
                "r2": r2,
                "r3": r3
            }
            
            # Identify historical support and resistance levels
            sr_levels = self._find_historical_support_resistance(df)
            
            # Identify Fibonacci retracement levels
            fib_levels = self._calculate_fibonacci_levels(df)
            
            # Identify price patterns
            patterns = self._identify_price_patterns(df)
            
            # Current price
            current_price = df['close'].iloc[-1]
            
            # Find nearest levels
            nearest_support = None
            nearest_resistance = None
            support_distance = float('inf')
            resistance_distance = float('inf')
            
            # Check historical levels
            for level in sr_levels:
                price = level["price"]
                if price < current_price and (current_price - price) < support_distance:
                    nearest_support = level
                    support_distance = current_price - price
                elif price > current_price and (price - current_price) < resistance_distance:
                    nearest_resistance = level
                    resistance_distance = price - current_price
            
            # Check pivot levels
            for name, value in pivot_points.items():
                if name != "pivot":
                    if value < current_price and (current_price - value) < support_distance:
                        nearest_support = {"price": value, "type": f"pivot_{name}"}
                        support_distance = current_price - value
                    elif value > current_price and (value - current_price) < resistance_distance:
                        nearest_resistance = {"price": value, "type": f"pivot_{name}"}
                        resistance_distance = value - current_price
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "current_price": current_price,
                "pivot_points": pivot_points,
                "support_resistance_levels": sr_levels,
                "fibonacci_levels": fib_levels,
                "patterns": patterns,
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying support/resistance for {symbol}: {e}")
            return {}
    
    def analyze_breakout(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze for breakout opportunities.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with breakout analysis results
        """
        try:
            # Get market data
            data = self._get_market_data(symbol, exchange, days=self.params["lookback_period"])
            
            if not data or len(data) < self.params["min_data_points"]:
                self.logger.warning(f"Insufficient data for breakout analysis of {symbol}")
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(data).sort_values("timestamp")
            
            # Current price
            current_price = df['close'].iloc[-1]
            
            # Identify support and resistance levels
            sr_analysis = self.identify_support_resistance(symbol, exchange)
            sr_levels = sr_analysis.get("support_resistance_levels", [])
            
            # Get recent high and low
            lookback = min(len(df), self.params["breakout_period"])
            recent_high = df['high'].iloc[-lookback:].max()
            recent_low = df['low'].iloc[-lookback:].min()
            
            # Check if price is near a breakout level
            breakout_signals = []
            
            # Check for breakout above recent high
            if abs(current_price - recent_high) / current_price < 0.01:
                # Price is near the recent high
                volumes = df['volume'].iloc[-5:].values if 'volume' in df.columns else None
                avg_volume = df['volume'].iloc[-20:].mean() if 'volume' in df.columns else None
                
                # Check for increasing volume on approach to breakout
                volume_increasing = False
                if volumes is not None and avg_volume is not None and volumes[-1] > avg_volume * self.params["min_volume_ratio"]:
                    volume_increasing = True
                
                breakout_signals.append({
                    "type": "potential_breakout_high",
                    "price_level": recent_high,
                    "volume_confirmation": volume_increasing,
                    "description": f"Price approaching breakout above recent high ({recent_high:.2f})"
                })
            # Check if we've already broken out above recent high
            elif current_price > recent_high and (current_price - recent_high) / recent_high < self.params["breakout_threshold"]:
                volumes = df['volume'].iloc[-5:].values if 'volume' in df.columns else None
                avg_volume = df['volume'].iloc[-20:].mean() if 'volume' in df.columns else None
                
                # Check for increasing volume on breakout
                volume_confirmation = False
                if volumes is not None and avg_volume is not None and volumes[-1] > avg_volume * self.params["min_volume_ratio"]:
                    volume_confirmation = True
                
                breakout_signals.append({
                    "type": "confirmed_breakout_high",
                    "price_level": recent_high,
                    "breakout_size": (current_price - recent_high) / recent_high * 100,
                    "volume_confirmation": volume_confirmation,
                    "description": f"Price broke out above recent high ({recent_high:.2f})"
                })
            
            # Check for breakdown below recent low
            if abs(current_price - recent_low) / current_price < 0.01:
                # Price is near the recent low
                volumes = df['volume'].iloc[-5:].values if 'volume' in df.columns else None
                avg_volume = df['volume'].iloc[-20:].mean() if 'volume' in df.columns else None
                
                # Check for increasing volume on approach to breakdown
                volume_increasing = False
                if volumes is not None and avg_volume is not None and volumes[-1] > avg_volume * self.params["min_volume_ratio"]:
                    volume_increasing = True
                
                breakout_signals.append({
                    "type": "potential_breakdown_low",
                    "price_level": recent_low,
                    "volume_confirmation": volume_increasing,
                    "description": f"Price approaching breakdown below recent low ({recent_low:.2f})"
                })
            # Check if we've already broken down below recent low
            elif current_price < recent_low and (recent_low - current_price) / recent_low < self.params["breakout_threshold"]:
                volumes = df['volume'].iloc[-5:].values if 'volume' in df.columns else None
                avg_volume = df['volume'].iloc[-20:].mean() if 'volume' in df.columns else None
                
                # Check for increasing volume on breakdown
                volume_confirmation = False
                if volumes is not None and avg_volume is not None and volumes[-1] > avg_volume * self.params["min_volume_ratio"]:
                    volume_confirmation = True
                
                breakout_signals.append({
                    "type": "confirmed_breakdown_low",
                    "price_level": recent_low,
                    "breakout_size": (recent_low - current_price) / recent_low * 100,
                    "volume_confirmation": volume_confirmation,
                    "description": f"Price broke down below recent low ({recent_low:.2f})"
                })
            
            # Check for breakouts from established support/resistance levels
            for level in sr_levels:
                level_price = level["price"]
                
                # Breakout above resistance
                if current_price > level_price and level["type"] == "resistance" and (current_price - level_price) / level_price < self.params["breakout_threshold"]:
                    volumes = df['volume'].iloc[-5:].values if 'volume' in df.columns else None
                    avg_volume = df['volume'].iloc[-20:].mean() if 'volume' in df.columns else None
                    
                    # Check for increasing volume on breakout
                    volume_confirmation = False
                    if volumes is not None and avg_volume is not None and volumes[-1] > avg_volume * self.params["min_volume_ratio"]:
                        volume_confirmation = True
                    
                    breakout_signals.append({
                        "type": "breakout_resistance",
                        "price_level": level_price,
                        "breakout_size": (current_price - level_price) / level_price * 100,
                        "level_strength": level.get("strength", 1),
                        "volume_confirmation": volume_confirmation,
                        "description": f"Price broke out above resistance level ({level_price:.2f})"
                    })
                
                # Breakdown below support
                if current_price < level_price and level["type"] == "support" and (level_price - current_price) / level_price < self.params["breakout_threshold"]:
                    volumes = df['volume'].iloc[-5:].values if 'volume' in df.columns else None
                    avg_volume = df['volume'].iloc[-20:].mean() if 'volume' in df.columns else None
                    
                    # Check for increasing volume on breakdown
                    volume_confirmation = False
                    if volumes is not None and avg_volume is not None and volumes[-1] > avg_volume * self.params["min_volume_ratio"]:
                        volume_confirmation = True
                    
                    breakout_signals.append({
                        "type": "breakdown_support",
                        "price_level": level_price,
                        "breakout_size": (level_price - current_price) / level_price * 100,
                        "level_strength": level.get("strength", 1),
                        "volume_confirmation": volume_confirmation,
                        "description": f"Price broke down below support level ({level_price:.2f})"
                    })
            
            # Determine if there are any valid breakout signals
            valid_breakouts = []
            for signal in breakout_signals:
                # Check volume confirmation
                if signal.get("volume_confirmation", False):
                    valid_breakouts.append(signal)
                # Check if the breakout size is significant
                elif "breakout_size" in signal and signal["breakout_size"] > 1.0:
                    valid_breakouts.append(signal)
                # Check if the level is strong
                elif "level_strength" in signal and signal["level_strength"] >= 2:
                    valid_breakouts.append(signal)
            
            # Determine overall breakout status
            overall_signal = "no_breakout"
            if valid_breakouts:
                # Check for bullish breakout signals
                bullish_breakouts = [s for s in valid_breakouts if s["type"] in ["confirmed_breakout_high", "breakout_resistance"]]
                bearish_breakouts = [s for s in valid_breakouts if s["type"] in ["confirmed_breakdown_low", "breakdown_support"]]
                
                if bullish_breakouts:
                    overall_signal = "bullish_breakout"
                elif bearish_breakouts:
                    overall_signal = "bearish_breakout"
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "current_price": current_price,
                "recent_high": recent_high,
                "recent_low": recent_low,
                "breakout_signals": breakout_signals,
                "valid_breakouts": valid_breakouts,
                "overall_signal": overall_signal,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing breakout for {symbol}: {e}")
            return {}
    
    def generate_technical_signals(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Generate comprehensive technical trading signals.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with trading signals
        """
        try:
            # Collect all technical analyses
            trend_analysis = self.analyze_trend(symbol, exchange)
            momentum_analysis = self.analyze_momentum(symbol, exchange)
            support_resistance = self.identify_support_resistance(symbol, exchange)
            breakout_analysis = self.analyze_breakout(symbol, exchange)
            
            # Get ATR
            # Get ATR for volatility assessment and risk management
            atr = self._calculate_atr(symbol, exchange)
            
            # Calculate current price
            current_price = None
            if trend_analysis:
                current_price = trend_analysis.get("current_price")
            elif momentum_analysis:
                current_price = momentum_analysis.get("current_price")
            elif breakout_analysis:
                current_price = breakout_analysis.get("current_price")
            
            if not current_price:
                return {"status": "error", "message": "Could not determine current price"}
            
            # Aggregate signals
            signals = []
            
            # 1. Trend signals
            if trend_analysis:
                trends = trend_analysis.get("trends", {})
                overall_trend = trends.get("overall")
                
                if overall_trend == "bullish":
                    signals.append({
                        "type": "trend",
                        "signal": "bullish",
                        "strength": trend_analysis.get("trend_strength", "moderate"),
                        "description": "Overall trend analysis indicates bullish conditions"
                    })
                elif overall_trend == "bearish":
                    signals.append({
                        "type": "trend",
                        "signal": "bearish",
                        "strength": trend_analysis.get("trend_strength", "moderate"),
                        "description": "Overall trend analysis indicates bearish conditions"
                    })
                
                # Add crossover signals
                for crossover in trend_analysis.get("crossovers", []):
                    signals.append({
                        "type": "crossover",
                        "signal": crossover.get("signal"),
                        "strength": "strong" if "long" in crossover.get("type", "") else "moderate",
                        "description": crossover.get("description")
                    })
            
            # 2. Momentum signals
            if momentum_analysis:
                momentum_signals = momentum_analysis.get("momentum_signals", [])
                for signal in momentum_signals:
                    signals.append({
                        "type": "momentum",
                        "signal": signal.get("interpretation"),
                        "strength": "strong" if signal.get("indicator") == "MACD" else "moderate",
                        "description": signal.get("description")
                    })
            
            # 3. Support/Resistance signals
            if support_resistance:
                nearest_support = support_resistance.get("nearest_support")
                nearest_resistance = support_resistance.get("nearest_resistance")
                
                if nearest_support:
                    support_price = nearest_support.get("price")
                    support_distance = (current_price - support_price) / current_price * 100
                    
                    if support_distance < 1.0:
                        signals.append({
                            "type": "support",
                            "signal": "bullish",
                            "strength": "moderate",
                            "description": f"Price near support at {support_price:.2f} ({support_distance:.2f}% below current price)"
                        })
                
                if nearest_resistance:
                    resistance_price = nearest_resistance.get("price")
                    resistance_distance = (resistance_price - current_price) / current_price * 100
                    
                    if resistance_distance < 1.0:
                        signals.append({
                            "type": "resistance",
                            "signal": "bearish",
                            "strength": "moderate",
                            "description": f"Price near resistance at {resistance_price:.2f} ({resistance_distance:.2f}% above current price)"
                        })
            
            # 4. Breakout signals
            if breakout_analysis:
                valid_breakouts = breakout_analysis.get("valid_breakouts", [])
                
                for breakout in valid_breakouts:
                    if breakout["type"] in ["confirmed_breakout_high", "breakout_resistance"]:
                        signals.append({
                            "type": "breakout",
                            "signal": "bullish",
                            "strength": "strong" if breakout.get("volume_confirmation", False) else "moderate",
                            "description": breakout.get("description")
                        })
                    elif breakout["type"] in ["confirmed_breakdown_low", "breakdown_support"]:
                        signals.append({
                            "type": "breakout",
                            "signal": "bearish",
                            "strength": "strong" if breakout.get("volume_confirmation", False) else "moderate",
                            "description": breakout.get("description")
                        })
            
            # Determine overall signal
            bullish_signals = [s for s in signals if s["signal"] == "bullish"]
            bearish_signals = [s for s in signals if s["signal"] == "bearish"]
            
            # Weight signals by strength
            bullish_score = sum(2 if s["strength"] == "strong" else 1 for s in bullish_signals)
            bearish_score = sum(2 if s["strength"] == "strong" else 1 for s in bearish_signals)
            
            overall_signal = "neutral"
            if bullish_score > bearish_score + 1:
                overall_signal = "bullish"
            elif bearish_score > bullish_score + 1:
                overall_signal = "bearish"
            
            # Calculate risk management parameters
            stop_loss = None
            target_price = None
            
            if overall_signal == "bullish":
                # Set stop loss based on support or ATR
                if support_resistance and support_resistance.get("nearest_support"):
                    stop_loss = support_resistance["nearest_support"]["price"] * 0.99  # Just below support
                elif atr:
                    stop_loss = current_price - 2 * atr  # 2 ATR stop
                
                # Set target based on resistance or risk-reward ratio
                if support_resistance and support_resistance.get("nearest_resistance"):
                    target_price = support_resistance["nearest_resistance"]["price"]  # Resistance as target
                elif stop_loss:
                    risk = current_price - stop_loss
                    target_price = current_price + 2 * risk  # 1:2 risk-reward
            
            elif overall_signal == "bearish":
                # Set stop loss based on resistance or ATR
                if support_resistance and support_resistance.get("nearest_resistance"):
                    stop_loss = support_resistance["nearest_resistance"]["price"] * 1.01  # Just above resistance
                elif atr:
                    stop_loss = current_price + 2 * atr  # 2 ATR stop
                
                # Set target based on support or risk-reward ratio
                if support_resistance and support_resistance.get("nearest_support"):
                    target_price = support_resistance["nearest_support"]["price"]  # Support as target
                elif stop_loss:
                    risk = stop_loss - current_price
                    target_price = current_price - 2 * risk  # 1:2 risk-reward
            
            # Calculate risk-reward ratio
            risk_reward_ratio = None
            if stop_loss and target_price:
                risk = abs(current_price - stop_loss)
                reward = abs(target_price - current_price)
                
                if risk > 0:
                    risk_reward_ratio = reward / risk
            
            # Generate final signal
            trading_signal = {
                "strategy": "technical",
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": datetime.now(),
                "current_price": current_price,
                "atr": atr,
                "signal": overall_signal,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "risk_reward_ratio": risk_reward_ratio,
                "signals": signals,
                "bullish_score": bullish_score,
                "bearish_score": bearish_score
            }
            
            return trading_signal
            
        except Exception as e:
            self.logger.error(f"Error generating technical signals for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def scan_for_technical_setups(self, symbols: List[str], exchange: str = "NSE") -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan for technical trading setups across multiple symbols.
        
        Args:
            symbols: List of stock symbols to analyze
            exchange: Stock exchange
            
        Returns:
            Dictionary with opportunities for each setup type
        """
        results = {
            "trend_following": [],
            "momentum": [],
            "breakout": [],
            "support_resistance": []
        }
        
        for symbol in symbols:
            try:
                # Get comprehensive technical signals
                signals = self.generate_technical_signals(symbol, exchange)
                
                if not signals or "status" in signals and signals["status"] == "error":
                    continue
                
                # Sort into appropriate categories based on signal type
                signal_types = [s["type"] for s in signals.get("signals", [])]
                
                # Only add signals with good risk-reward
                if signals.get("risk_reward_ratio", 0) >= 1.5:
                    # Categorize based on predominant signal type
                    if "trend" in signal_types and signals.get("signal") != "neutral":
                        results["trend_following"].append(signals)
                    
                    if "momentum" in signal_types and signals.get("signal") != "neutral":
                        results["momentum"].append(signals)
                    
                    if "breakout" in signal_types:
                        results["breakout"].append(signals)
                    
                    if "support" in signal_types or "resistance" in signal_types:
                        results["support_resistance"].append(signals)
            
            except Exception as e:
                self.logger.error(f"Error scanning technical setups for {symbol}: {e}")
        
        return results
    
    def _find_historical_support_resistance(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find historical support and resistance levels.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            List of identified support and resistance levels
        """
        levels = []
        
        # Get high and low values
        highs = df['high'].values
        lows = df['low'].values
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Function to find local maxima/minima
        def is_local_extrema(data, idx, is_high=True, window=5):
            if idx < window or idx >= len(data) - window:
                return False
            
            if is_high:
                return all(data[idx] > data[j] for j in range(idx-window, idx)) and \
                       all(data[idx] > data[j] for j in range(idx+1, idx+window+1))
            else:
                return all(data[idx] < data[j] for j in range(idx-window, idx)) and \
                       all(data[idx] < data[j] for j in range(idx+1, idx+window+1))
        
        # Find swing highs (potential resistance)
        for i in range(len(highs)):
            if is_local_extrema(highs, i, is_high=True):
                price = highs[i]
                
                # Check if there are multiple touches near this level
                touches = 0
                for j in range(len(highs)):
                    if i != j and abs(highs[j] - price) / price < self.params["sr_threshold"]:
                        touches += 1
                
                if touches >= self.params["sr_touch_count"] - 1:  # -1 because we already counted one touch
                    # Check if the level is relevant (not too far from current price)
                    if abs(price - current_price) / current_price < 0.15:  # Within 15%
                        levels.append({
                            "price": price,
                            "type": "resistance",
                            "strength": touches + 1,
                            "date": df['timestamp'].iloc[i] if 'timestamp' in df.columns else None
                        })
        
        # Find swing lows (potential support)
        for i in range(len(lows)):
            if is_local_extrema(lows, i, is_high=False):
                price = lows[i]
                
                # Check if there are multiple touches near this level
                touches = 0
                for j in range(len(lows)):
                    if i != j and abs(lows[j] - price) / price < self.params["sr_threshold"]:
                        touches += 1
                
                if touches >= self.params["sr_touch_count"] - 1:
                    # Check if the level is relevant (not too far from current price)
                    if abs(price - current_price) / current_price < 0.15:  # Within 15%
                        levels.append({
                            "price": price,
                            "type": "support",
                            "strength": touches + 1,
                            "date": df['timestamp'].iloc[i] if 'timestamp' in df.columns else None
                        })
        
        # Cluster similar levels
        clustered_levels = []
        
        for level in sorted(levels, key=lambda x: x["price"]):
            if not clustered_levels:
                clustered_levels.append(level)
                continue
            
            # Check if this level is close to any existing clustered level
            merged = False
            for i, existing_level in enumerate(clustered_levels):
                if abs(level["price"] - existing_level["price"]) / existing_level["price"] < self.params["sr_threshold"]:
                    # Merge by taking the average price and summing strength
                    new_price = (level["price"] + existing_level["price"]) / 2
                    new_strength = level["strength"] + existing_level["strength"]
                    
                    # Keep the type (support/resistance) of the stronger level
                    new_type = level["type"] if level["strength"] > existing_level["strength"] else existing_level["type"]
                    
                    clustered_levels[i] = {
                        "price": new_price,
                        "type": new_type,
                        "strength": new_strength,
                        "date": existing_level["date"]  # Keep the earlier date
                    }
                    merged = True
                    break
            
            if not merged:
                clustered_levels.append(level)
        
        # Sort by price
        clustered_levels.sort(key=lambda x: x["price"])
        
        return clustered_levels
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with Fibonacci levels
        """
        # Find recent high and low
        lookback = min(len(df), 60)  # 60 days for Fibonacci calculation
        high = df['high'].iloc[-lookback:].max()
        low = df['low'].iloc[-lookback:].min()
        current = df['close'].iloc[-1]
        
        # Determine trend direction
        if current > (high + low) / 2:
            # Uptrend - calculate retracements from low to high
            diff = high - low
            
            return {
                "trend": "uptrend",
                "swing_low": low,
                "swing_high": high,
                "0.0": low,
                "0.236": low + 0.236 * diff,
                "0.382": low + 0.382 * diff,
                "0.5": low + 0.5 * diff,
                "0.618": low + 0.618 * diff,
                "0.786": low + 0.786 * diff,
                "1.0": high
            }
        else:
            # Downtrend - calculate retracements from high to low
            diff = high - low
            
            return {
                "trend": "downtrend",
                "swing_high": high,
                "swing_low": low,
                "0.0": high,
                "0.236": high - 0.236 * diff,
                "0.382": high - 0.382 * diff,
                "0.5": high - 0.5 * diff,
                "0.618": high - 0.618 * diff,
                "0.786": high - 0.786 * diff,
                "1.0": low
            }
    
    def _identify_price_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify common price patterns.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        try:
            # TA-Lib requires specific column names
            df_talib = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            
            # Function to recognize pattern with TA-Lib
            def recognize_pattern(pattern_func, pattern_name, pattern_type):
                try:
                    result = pattern_func(df_talib['Open'].values, df_talib['High'].values, df_talib['Low'].values, df_talib['Close'].values)
                    if result[-1] != 0:
                        signal = "bullish" if result[-1] > 0 else "bearish"
                        
                        # Check for confirmation
                        confirmed = False
                        if len(df) > self.params["pattern_confirmation"]:
                            # Check if price has moved in expected direction since pattern formation
                            if signal == "bullish" and df['close'].iloc[-1] > df['close'].iloc[-self.params["pattern_confirmation"]]:
                                confirmed = True
                            elif signal == "bearish" and df['close'].iloc[-1] < df['close'].iloc[-self.params["pattern_confirmation"]]:
                                confirmed = True
                        
                        return {
                            "name": pattern_name,
                            "type": pattern_type,
                            "signal": signal,
                            "confirmed": confirmed,
                            "date": df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None
                        }
                    return None
                except Exception as e:
                    self.logger.warning(f"Error recognizing pattern {pattern_name}: {e}")
                    return None
            
            # Check for candlestick patterns
            pattern_funcs = [
                (talib.CDLDOJI, "Doji", "reversal"),
                (talib.CDLHAMMER, "Hammer", "bullish_reversal"),
                (talib.CDLINVERTEDHAMMER, "Inverted Hammer", "bullish_reversal"),
                (talib.CDLENGULFING, "Engulfing", "reversal"),
                (talib.CDLMORNINGSTAR, "Morning Star", "bullish_reversal"),
                (talib.CDLEVENINGSTAR, "Evening Star", "bearish_reversal"),
                (talib.CDLHARAMI, "Harami", "reversal"),
                (talib.CDLMARUBOZU, "Marubozu", "continuation"),
                (talib.CDLSHOOTINGSTAR, "Shooting Star", "bearish_reversal"),
                (talib.CDLHANGINGMAN, "Hanging Man", "bearish_reversal")
            ]
            
            for func, name, pattern_type in pattern_funcs:
                pattern = recognize_pattern(func, name, pattern_type)
                if pattern:
                    patterns.append(pattern)
            
            # Check for chart patterns
            # Head and Shoulders
            # TODO: Implement chart pattern recognition
            
        except Exception as e:
            self.logger.error(f"Error identifying price patterns: {e}")
        
        return patterns
    
    def _calculate_atr(self, symbol: str, exchange: str, period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range (ATR).
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            period: ATR period
            
        Returns:
            ATR value or None if not available
        """
        try:
            # Get market data
            data = self._get_market_data(symbol, exchange, days=period * 2)  # Get enough data for calculation
            
            if not data or len(data) < period:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data).sort_values("timestamp")
            
            # TA-Lib requires specific column names
            df_talib = df.rename(columns={'high': 'High', 'low': 'Low', 'close': 'Close'})
            
            # Calculate ATR
            atr = talib.ATR(df_talib['High'].values, df_talib['Low'].values, df_talib['Close'].values, timeperiod=period)
            
            # Return the most recent ATR value
            return atr[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR for {symbol}: {e}")
            return None
    
    def _get_market_data(self, symbol: str, exchange: str, days: int = 100) -> List[Dict[str, Any]]:
        """
        Get market data from database.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            days: Number of days to retrieve
            
        Returns:
            List of market data documents
        """
        try:
            # Calculate the start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Create query
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "day",
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