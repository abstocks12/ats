"""
Statistical Arbitrage Strategy Module

This module implements statistical arbitrage strategies including:
- Pair trading
- Mean reversion
- Statistical factor models
- Cointegration-based approaches
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats

class StatisticalArbitrageStrategy:
    """
    Implements statistical arbitrage strategies for automated trading.
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
            "lookback_period": 60,  # Days for calculating relationships
            "zscore_entry_threshold": 2.0,  # Z-score threshold for entry
            "zscore_exit_threshold": 0.5,  # Z-score threshold for exit
            "max_position_days": 20,  # Maximum holding period
            "confidence_threshold": 0.95,  # Confidence threshold for pair selection
            "min_correlation": 0.7,  # Minimum correlation for pair selection
            "max_pairs": 5,  # Maximum number of pairs to trade simultaneously
            "mean_reversion_lookback": 20,  # Lookback period for mean reversion
            "bollinger_band_std": 2.0,  # Standard deviation for Bollinger Bands
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
    
    def find_cointegrated_pairs(self, symbols: List[str], exchange: str = "NSE") -> List[Dict[str, Any]]:
        """
        Find cointegrated pairs among the provided symbols.
        
        Args:
            symbols: List of stock symbols to analyze
            exchange: Stock exchange
            
        Returns:
            List of cointegrated pairs with statistics
        """
        try:
            self.logger.info(f"Finding cointegrated pairs among {len(symbols)} symbols")
            
            # Get price data for all symbols
            price_data = {}
            
            for symbol in symbols:
                data = self._get_market_data(symbol, exchange, days=self.params["lookback_period"])
                
                if not data or len(data) < 30:
                    continue
                
                df = pd.DataFrame(data)
                
                # Extract closing prices with timestamps
                symbol_data = df[["timestamp", "close"]].sort_values("timestamp")
                price_data[symbol] = symbol_data
            
            if len(price_data) < 2:
                self.logger.warning("Insufficient price data for pair analysis")
                return []
            
            # Create a combined dataframe with all closing prices
            combined_df = None
            
            for symbol, data in price_data.items():
                if combined_df is None:
                    combined_df = data.rename(columns={"close": symbol}).set_index("timestamp")
                else:
                    combined_df[symbol] = data.set_index("timestamp")["close"]
            
            # Fill missing values
            combined_df = combined_df.fillna(method="ffill").fillna(method="bfill")
            
            # Find cointegrated pairs
            n = len(combined_df.columns)
            pvalue_matrix = np.ones((n, n))
            keys = combined_df.columns
            pairs = []
            
            # Calculate correlation matrix
            correlation_matrix = combined_df.corr()
            
            # For each pair of stocks, check for cointegration
            for i in range(n):
                for j in range(i+1, n):
                    stock1 = keys[i]
                    stock2 = keys[j]
                    
                    # Skip if correlation is too low
                    correlation = correlation_matrix.loc[stock1, stock2]
                    if abs(correlation) < self.params["min_correlation"]:
                        continue
                    
                    # Check for cointegration
                    try:
                        # Use price series
                        stock1_prices = combined_df[stock1].values
                        stock2_prices = combined_df[stock2].values
                        
                        # Perform cointegration test
                        result = coint(stock1_prices, stock2_prices)
                        pvalue = result[1]
                        
                        pvalue_matrix[i, j] = pvalue
                        
                        # If p-value is less than threshold, consider pair cointegrated
                        if pvalue < (1.0 - self.params["confidence_threshold"]):
                            # Calculate hedge ratio using OLS
                            model = sm.OLS(stock1_prices, stock2_prices).fit()
                            hedge_ratio = model.params[0]
                            
                            # Calculate spread series
                            spread = stock1_prices - hedge_ratio * stock2_prices
                            
                            # Calculate z-score
                            spread_mean = np.mean(spread)
                            spread_std = np.std(spread)
                            current_spread = spread[-1]
                            zscore = (current_spread - spread_mean) / spread_std
                            
                            # Calculate half-life of mean reversion
                            spread_lag = np.roll(spread, 1)
                            spread_lag[0] = spread_lag[1]
                            spread_ret = spread - spread_lag
                            spread_lag_1 = sm.add_constant(spread_lag[1:])
                            model = sm.OLS(spread_ret[1:], spread_lag_1).fit()
                            
                            # Calculate half-life
                            half_life = -np.log(2) / model.params[1]
                            
                            # Perform ADF test on spread
                            adf_result = adfuller(spread)
                            adf_pvalue = adf_result[1]
                            
                            pairs.append({
                                "symbol1": stock1,
                                "symbol2": stock2,
                                "hedge_ratio": hedge_ratio,
                                "correlation": correlation,
                                "coint_pvalue": pvalue,
                                "adf_pvalue": adf_pvalue,
                                "zscore": zscore,
                                "half_life": max(0.5, half_life),  # Avoid negative half-life
                                "mean": spread_mean,
                                "std": spread_std,
                                "current_spread": current_spread,
                                "lookback_period": self.params["lookback_period"]
                            })
                    except Exception as e:
                        self.logger.error(f"Error testing cointegration for {stock1}-{stock2}: {e}")
            
            # Sort pairs by lowest p-value (highest confidence)
            pairs.sort(key=lambda x: x["coint_pvalue"])
            
            return pairs[:self.params["max_pairs"]]
            
        except Exception as e:
            self.logger.error(f"Error finding cointegrated pairs: {e}")
            return []
    
    def generate_pair_trading_signals(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate trading signals for cointegrated pairs.
        
        Args:
            pairs: List of cointegrated pairs with statistics
            
        Returns:
            List of trading signals
        """
        signals = []
        
        for pair in pairs:
            symbol1 = pair["symbol1"]
            symbol2 = pair["symbol2"]
            zscore = pair["zscore"]
            
            signal = {
                "strategy": "pair_trading",
                "symbol1": symbol1,
                "symbol2": symbol2,
                "hedge_ratio": pair["hedge_ratio"],
                "timestamp": datetime.now(),
                "zscore": zscore,
                "half_life": pair["half_life"],
                "confidence": 1.0 - pair["coint_pvalue"],
                "signal": "neutral",
                "entry_price1": None,
                "entry_price2": None,
                "target_price1": None,
                "target_price2": None,
                "stop_price1": None,
                "stop_price2": None
            }
            
            # Generate signals based on z-score
            zscore_threshold = self.params["zscore_entry_threshold"]
            
            if zscore > zscore_threshold:
                # Pairs strategy: Short the spread (Short symbol1, Long symbol2)
                signal["signal"] = "short_spread"
                signal["action1"] = "sell"
                signal["action2"] = "buy"
                signal["direction"] = "bearish"
            elif zscore < -zscore_threshold:
                # Pairs strategy: Long the spread (Long symbol1, Short symbol2)
                signal["signal"] = "long_spread"
                signal["action1"] = "buy"
                signal["action2"] = "sell"
                signal["direction"] = "bullish"
            
            # If we have a signal, add price information
            if signal["signal"] != "neutral":
                # Get current prices
                price1 = self._get_current_price(symbol1)
                price2 = self._get_current_price(symbol2)
                
                if price1 and price2:
                    # Set entry prices
                    signal["entry_price1"] = price1
                    signal["entry_price2"] = price2
                    
                    # Set target prices (at mean reversion)
                    if signal["signal"] == "short_spread":
                        # Target is when spread narrows (price1 decreases or price2 increases)
                        mean_spread = pair["mean"]
                        current_spread = pair["current_spread"]
                        spread_change = current_spread - mean_spread
                        
                        signal["target_price1"] = price1 - spread_change / 2
                        signal["target_price2"] = price2 + (spread_change / 2) / pair["hedge_ratio"]
                    else:
                        # Target is when spread widens (price1 increases or price2 decreases)
                        mean_spread = pair["mean"]
                        current_spread = pair["current_spread"]
                        spread_change = mean_spread - current_spread
                        
                        signal["target_price1"] = price1 + spread_change / 2
                        signal["target_price2"] = price2 - (spread_change / 2) / pair["hedge_ratio"]
                    
                    # Set stop prices (at further divergence)
                    if signal["signal"] == "short_spread":
                        # Stop is when spread widens further
                        stop_zscore = zscore + 1.0
                        stop_spread = stop_zscore * pair["std"] + pair["mean"]
                        spread_change = stop_spread - current_spread
                        
                        signal["stop_price1"] = price1 + spread_change / 2
                        signal["stop_price2"] = price2 - (spread_change / 2) / pair["hedge_ratio"]
                    else:
                        # Stop is when spread narrows further
                        stop_zscore = zscore - 1.0
                        stop_spread = stop_zscore * pair["std"] + pair["mean"]
                        spread_change = current_spread - stop_spread
                        
                        signal["stop_price1"] = price1 - spread_change / 2
                        signal["stop_price2"] = price2 + (spread_change / 2) / pair["hedge_ratio"]
                    
                    # Calculate risk-reward ratio
                    if signal["signal"] == "short_spread":
                        risk1 = abs(signal["stop_price1"] - price1)
                        reward1 = abs(signal["target_price1"] - price1)
                        risk2 = abs(signal["stop_price2"] - price2)
                        reward2 = abs(signal["target_price2"] - price2)
                    else:
                        risk1 = abs(signal["stop_price1"] - price1)
                        reward1 = abs(signal["target_price1"] - price1)
                        risk2 = abs(signal["stop_price2"] - price2)
                        reward2 = abs(signal["target_price2"] - price2)
                    
                    # Weight by position size (using hedge ratio)
                    weighted_risk = risk1 + risk2 * pair["hedge_ratio"]
                    weighted_reward = reward1 + reward2 * pair["hedge_ratio"]
                    
                    if weighted_risk > 0:
                        signal["risk_reward_ratio"] = weighted_reward / weighted_risk
                    else:
                        signal["risk_reward_ratio"] = 0
                
                # Add signal to list if risk-reward is favorable
                if signal.get("risk_reward_ratio", 0) >= 1.5:
                    signals.append(signal)
        
        return signals
    
    def analyze_mean_reversion(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze a stock for mean reversion opportunities.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with mean reversion analysis
        """
        try:
            # Get market data
            data = self._get_market_data(symbol, exchange, days=self.params["lookback_period"])
            
            if not data or len(data) < self.params["mean_reversion_lookback"]:
                self.logger.warning(f"Insufficient data for mean reversion analysis of {symbol}")
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(data).sort_values("timestamp")
            
            # Calculate rolling mean and standard deviation
            df['rolling_mean'] = df['close'].rolling(window=self.params["mean_reversion_lookback"]).mean()
            df['rolling_std'] = df['close'].rolling(window=self.params["mean_reversion_lookback"]).std()
            
            # Calculate Bollinger Bands
            std_dev = self.params["bollinger_band_std"]
            df['upper_band'] = df['rolling_mean'] + std_dev * df['rolling_std']
            df['lower_band'] = df['rolling_mean'] - std_dev * df['rolling_std']
            
            # Calculate z-score
            df['zscore'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
            
            # Get current values
            current_price = df['close'].iloc[-1]
            current_mean = df['rolling_mean'].iloc[-1]
            current_upper = df['upper_band'].iloc[-1]
            current_lower = df['lower_band'].iloc[-1]
            current_zscore = df['zscore'].iloc[-1]
            
            # Perform mean reversion tests
            
            # 1. Augmented Dickey-Fuller test
            adf_result = adfuller(df['close'].values)
            adf_pvalue = adf_result[1]
            
            # 2. Hurst exponent calculation
            hurst_exponent = self._calculate_hurst_exponent(df['close'].values)
            
            # 3. Half-life of mean reversion
            prices = df['close'].values
            price_lag = np.roll(prices, 1)
            price_lag[0] = price_lag[1]
            price_ret = prices - price_lag
            price_lag_1 = sm.add_constant(price_lag[1:])
            model = sm.OLS(price_ret[1:], price_lag_1).fit()
            
            # Calculate half-life
            half_life = -np.log(2) / model.params[1] if model.params[1] < 0 else float('inf')
            
            # Determine mean reversion strength
            mean_reverting = False
            mean_reversion_score = 0
            
            # Score based on Hurst exponent (< 0.5 indicates mean reversion)
            if hurst_exponent < 0.5:
                mean_reversion_score += 1
            
            # Score based on ADF test (low p-value rejects unit root, supporting mean reversion)
            if adf_pvalue < 0.05:
                mean_reversion_score += 1
            
            # Score based on half-life (lower is better for mean reversion)
            if 1 <= half_life <= 20:  # Reasonable half-life for trading
                mean_reversion_score += 1
            
            # Consider mean reverting if at least 2 of 3 criteria are met
            if mean_reversion_score >= 2:
                mean_reverting = True
            
            # Determine current position relative to bands
            position = "neutral"
            if current_price <= current_lower:
                position = "below_lower_band"
            elif current_price >= current_upper:
                position = "above_upper_band"
            
            # Generate trading signal
            signal = "neutral"
            if mean_reverting:
                if position == "below_lower_band":
                    signal = "buy"  # Price is below lower band, expected to revert upward
                elif position == "above_upper_band":
                    signal = "sell"  # Price is above upper band, expected to revert downward
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "current_price": current_price,
                "rolling_mean": current_mean,
                "upper_band": current_upper,
                "lower_band": current_lower,
                "zscore": current_zscore,
                "position": position,
                "mean_reverting": mean_reverting,
                "mean_reversion_score": mean_reversion_score,
                "adf_pvalue": adf_pvalue,
                "hurst_exponent": hurst_exponent,
                "half_life": half_life,
                "signal": signal,
                "confidence": mean_reversion_score / 3.0  # Scale from 0 to 1
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing mean reversion for {symbol}: {e}")
            return {}
    
    def generate_mean_reversion_signal(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal based on mean reversion analysis.
        
        Args:
            analysis: Mean reversion analysis result
            
        Returns:
            Trading signal dictionary
        """
        if not analysis or analysis.get("signal") == "neutral":
            return {}
        
        symbol = analysis.get("symbol")
        exchange = analysis.get("exchange")
        signal = analysis.get("signal")
        
        entry_price = analysis.get("current_price")
        
        # Set target and stop prices
        if signal == "buy":
            # Buy signal: below lower band, expect reversion to mean
            target_price = analysis.get("rolling_mean")
            
            # Stop is a further deviation from the lower band
            lower_band = analysis.get("lower_band")
            band_distance = entry_price - lower_band
            stop_price = entry_price - band_distance
            
            # Adjust to ensure stop is below entry
            stop_price = min(stop_price, entry_price * 0.97)
        else:
            # Sell signal: above upper band, expect reversion to mean
            target_price = analysis.get("rolling_mean")
            
            # Stop is a further deviation from the upper band
            upper_band = analysis.get("upper_band")
            band_distance = upper_band - entry_price
            stop_price = entry_price + band_distance
            
            # Adjust to ensure stop is above entry
            stop_price = max(stop_price, entry_price * 1.03)
        
        # Calculate risk-reward ratio
        risk = abs(stop_price - entry_price)
        reward = abs(target_price - entry_price)
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return {
            "strategy": "mean_reversion",
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(),
            "signal": signal,
            "direction": "bullish" if signal == "buy" else "bearish",
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price,
            "risk_reward_ratio": risk_reward_ratio,
            "confidence": analysis.get("confidence", 0),
            "zscore": analysis.get("zscore"),
            "mean_reverting": analysis.get("mean_reverting"),
            "position": analysis.get("position")
        }
    
    def scan_for_statistical_arbitrage(self, symbols: List[str], exchange: str = "NSE") -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan for statistical arbitrage opportunities across multiple strategies.
        
        Args:
            symbols: List of stock symbols to analyze
            exchange: Stock exchange
            
        Returns:
            Dictionary with opportunities for each strategy type
        """
        results = {
            "pair_trading": [],
            "mean_reversion": []
        }
        
        # 1. Pair Trading Analysis
        try:
            pairs = self.find_cointegrated_pairs(symbols, exchange)
            pair_signals = self.generate_pair_trading_signals(pairs)
            results["pair_trading"] = pair_signals
        except Exception as e:
            self.logger.error(f"Error scanning for pair trading opportunities: {e}")
        
        # 2. Mean Reversion Analysis
        try:
            for symbol in symbols:
                analysis = self.analyze_mean_reversion(symbol, exchange)
                if analysis and analysis.get("signal") != "neutral":
                    signal = self.generate_mean_reversion_signal(analysis)
                    if signal and signal.get("risk_reward_ratio", 0) >= 1.5:
                        results["mean_reversion"].append(signal)
        except Exception as e:
            self.logger.error(f"Error scanning for mean reversion opportunities: {e}")
        
        return results
    
    def _get_market_data(self, symbol: str, exchange: str, days: int = 60) -> List[Dict[str, Any]]:
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
    
    def _get_current_price(self, symbol: str, exchange: str = "NSE") -> Optional[float]:
        """
        Get the current price of a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Current price or None if not available
        """
        try:
            # Get the most recent market data
            data = self.db.market_data_collection.find_one(
                {
                    "symbol": symbol,
                    "exchange": exchange
                },
                sort=[("timestamp", -1)]
            )
            
            if data and "close" in data:
                return data["close"]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _calculate_hurst_exponent(self, price_series: np.ndarray, max_lag: int = 20) -> float:
        """
        Calculate the Hurst exponent of a time series.
        
        Args:
            price_series: Array of prices
            max_lag: Maximum lag for calculation
            
        Returns:
            Hurst exponent value
        """
        # Convert to numpy array
        ts = np.array(price_series)
        
        # Calculate logarithmic returns
        returns = np.diff(np.log(ts))
        
        # Calculate range of cumulative sum of returns for different lags
        tau = []
        lagvec = []
        
        # Use lags from 2 to max_lag
        for lag in range(2, max_lag):
            tau.append(lag)
            
            # Reshape returns into lag-sized chunks
            n = len(returns) // lag
            if n == 0:
                break
                
            chunk_size = n * lag
            y = returns[:chunk_size].reshape((n, lag))
            
            # Calculate statistics for each chunk
            mean_y = np.mean(y, axis=1)
            y_demeaned = y - mean_y.reshape((n, 1))
            
            # Calculate range (max - min) of cumulative sum
            y_cumsum = np.cumsum(y_demeaned, axis=1)
            y_range = np.max(y_cumsum, axis=1) - np.min(y_cumsum, axis=1)
            
            # Calculate standard deviation of returns for each chunk
            y_std = np.std(y, axis=1)
            
            # Calculate R/S statistic
            rs = np.mean(y_range / y_std)
            lagvec.append(rs)
        
        # Calculate Hurst exponent using linear regression
        if len(tau) > 1 and len(lagvec) > 1:
            m = np.polyfit(np.log(tau), np.log(lagvec), 1)
            hurst = m[0]
            return hurst
        else:
            return 0.5  # Default to random walk if not enough data