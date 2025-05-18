# technical_features.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class TechnicalFeatureGenerator:
    def __init__(self, db_connector):
        """Initialize the technical feature generator"""
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        
        # Initialize scalers
        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.volume_scaler = StandardScaler()
        self.indicator_scaler = StandardScaler()
    
    def generate_features(self, symbol, exchange, timeframe="day", lookback=200, 
                        include_target=True, target_horizon=5, for_date=None):
        """
        Generate technical features for a symbol
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange (e.g., NSE)
        - timeframe: Data timeframe
        - lookback: Number of historical bars to include
        - include_target: Whether to include target variables
        - target_horizon: Forecast horizon in periods
        - for_date: Optional specific date to generate features for

        Returns:
        - DataFrame with technical features
        """
        try:
            # Get market data
            market_data = self._get_market_data(symbol, exchange, timeframe, lookback, for_date)
            if market_data is None or len(market_data) < 50:  # Need sufficient history
                return None
            
            # Create a copy to avoid modifying original data
            df = market_data.copy()
            
            # Calculate basic price features
            self._add_price_features(df)
            
            # Calculate momentum features
            self._add_momentum_features(df)
            
            # Calculate volatility features
            self._add_volatility_features(df)
            
            # Calculate volume features
            self._add_volume_features(df)
            
            # Calculate pattern features
            self._add_pattern_features(df)
            
            # Calculate support/resistance features
            self._add_support_resistance_features(df)
            
            # Calculate mean-reversion features
            self._add_mean_reversion_features(df)
            
            # Calculate trend features
            self._add_trend_features(df)
            
            # Add cycle features
            self._add_cycle_features(df)
            
            # Add target variables if requested
            if include_target:
                self._add_target_variables(df, target_horizon)
            
            # Drop rows with NaN values (from indicator calculations)
            df = df.dropna()
            
            # Scale features to normalize data
            feature_columns = [col for col in df.columns if col.startswith('feature_')]
            if len(df) > 0 and len(feature_columns) > 0:
                df[feature_columns] = self.indicator_scaler.fit_transform(df[feature_columns])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating technical features: {str(e)}")
            return None
    
    def _get_market_data(self, symbol, exchange, timeframe, lookback, for_date=None):
        """Get historical market data from database"""
        try:
            # Base query for market data
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }
            
            # Add date filter if specified
            if for_date:
                # Find data up to this date
                query["timestamp"] = {"$lte": for_date}
            
            # Get data from database
            data = list(self.db.market_data_collection.find(
                query,
                {"timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
            ).sort("timestamp", -1).limit(lookback))
            
            if not data:
                return None
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return None
    
    def _add_price_features(self, df):
        """Add basic price-based features"""
        try:
            # Calculate log returns
            df['feature_returns_1d'] = np.log(df['close'] / df['close'].shift(1))
            
            # Calculate multi-period returns
            for period in [2, 3, 5, 10, 21]:
                df[f'feature_returns_{period}d'] = np.log(df['close'] / df['close'].shift(period))
            
            # Calculate price ratios
            df['feature_price_to_ma20'] = df['close'] / df['close'].rolling(window=20).mean()
            df['feature_price_to_ma50'] = df['close'] / df['close'].rolling(window=50).mean()
            df['feature_price_to_ma200'] = df['close'] / df['close'].rolling(window=200).mean()
            
            # Calculate price position (0 to 1) within recent range
            for period in [10, 20, 50]:
                df[f'feature_price_position_{period}d'] = (df['close'] - df['low'].rolling(window=period).min()) / \
                                                     (df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min())
            
            # High-Low range compared to previous periods
            df['feature_hl_ratio_1d'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
            
            # Gap features
            df['feature_gap_open'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            
            # Price acceleration (second derivative of price)
            df['feature_price_accel'] = df['feature_returns_1d'] - df['feature_returns_1d'].shift(1)
            
            # Body size relative to range
            df['feature_body_to_range'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            
            # Upper and lower shadows
            df['feature_upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['feature_lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
            
        except Exception as e:
            self.logger.error(f"Error adding price features: {str(e)}")
    
    def _add_momentum_features(self, df):
        """Add momentum-based features"""
        try:
            # Calculate RSI
            df['feature_rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
            df['feature_rsi_5'] = talib.RSI(df['close'].values, timeperiod=5)
            
            # Calculate Stochastic Oscillator
            df['feature_stoch_k'], df['feature_stoch_d'] = talib.STOCH(df['high'].values, 
                                                                   df['low'].values, 
                                                                   df['close'].values)
            
            # Calculate MACD
            df['feature_macd'], df['feature_macd_signal'], df['feature_macd_hist'] = talib.MACD(
                df['close'].values,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            
            # Calculate ROC (Rate of Change)
            for period in [5, 10, 21]:
                df[f'feature_roc_{period}'] = talib.ROC(df['close'].values, timeperiod=period)
            
            # Bollinger Bands %B (position within bands)
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20)
            df['feature_bbands_b'] = (df['close'] - lower) / (upper - lower)
            
            # Calculate CCI (Commodity Channel Index)
            df['feature_cci_20'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=20)
            
            # Williams %R
            df['feature_willr_14'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # Money Flow Index
            df['feature_mfi_14'] = talib.MFI(df['high'].values, df['low'].values, df['close'].values, df['volume'].values, timeperiod=14)
            
            # Average Directional Index
            df['feature_adx_14'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # Ultimate Oscillator
            df['feature_ultosc'] = talib.ULTOSC(df['high'].values, df['low'].values, df['close'].values)
            
            # Changes in momentum indicators
            df['feature_rsi_change'] = df['feature_rsi_14'] - df['feature_rsi_14'].shift(1)
            df['feature_macd_hist_change'] = df['feature_macd_hist'] - df['feature_macd_hist'].shift(1)
            
        except Exception as e:
            self.logger.error(f"Error adding momentum features: {str(e)}")
    
    def _add_volatility_features(self, df):
        """Add volatility-based features"""
        try:
            # Calculate ATR (Average True Range)
            df['feature_atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['feature_atr_ratio'] = df['feature_atr_14'] / df['close']  # ATR as percentage of price
            
            # Historical volatility (std dev of returns)
            for period in [5, 10, 21]:
                df[f'feature_volatility_{period}d'] = df['feature_returns_1d'].rolling(window=period).std()
            
            # Bollinger Band width (indicator of volatility)
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20)
            df['feature_bb_width'] = (upper - lower) / middle
            
            # Volatility ratio (comparing recent to longer-term volatility)
            df['feature_volatility_ratio'] = df['feature_volatility_5d'] / df['feature_volatility_21d']
            
            # Normalized range
            df['feature_norm_range'] = (df['high'] - df['low']) / df['close']
            
            # Parkinson volatility estimator
            df['feature_parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * ((np.log(df['high'] / df['low'])) ** 2))
            
            # Garman-Klass volatility estimator
            df['feature_gk_vol'] = np.sqrt(
                0.5 * (np.log(df['high'] / df['low'])) ** 2 - 
                (2 * np.log(2) - 1) * (np.log(df['close'] / df['open'])) ** 2
            )
            
            # Change in volatility
            df['feature_volatility_change'] = df['feature_volatility_10d'] - df['feature_volatility_10d'].shift(1)
            
            # Volatility of volatility (meta-volatility)
            df['feature_vol_of_vol'] = df['feature_volatility_10d'].rolling(window=10).std()
            
        except Exception as e:
            self.logger.error(f"Error adding volatility features: {str(e)}")
    
    def _add_volume_features(self, df):
        """Add volume-based features"""
        try:
            # Ensure volume column exists
            if 'volume' not in df.columns:
                return
            
            # Volume changes
            df['feature_volume_change'] = df['volume'] / df['volume'].shift(1) - 1
            
            # Volume moving averages
            for period in [5, 10, 20]:
                df[f'feature_volume_ratio_{period}d'] = df['volume'] / df['volume'].rolling(window=period).mean()
            
            # On-Balance Volume (OBV)
            df['feature_obv'] = talib.OBV(df['close'].values, df['volume'].values)
            df['feature_obv_ratio'] = df['feature_obv'] / df['feature_obv'].shift(5)
            
            # Chaikin Money Flow
            df['feature_cmf_20'] = talib.ADOSC(df['high'].values, df['low'].values, 
                                          df['close'].values, df['volume'].values,
                                          fastperiod=3, slowperiod=10)
            
            # Money Flow Volume
            df['mfv'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
            df['feature_mfv_ratio'] = df['mfv'].rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
            
            # Price-Volume Trend
            df['feature_pvt'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']
            df['feature_pvt_ma'] = df['feature_pvt'].rolling(window=20).mean()
            
            # Volume Weighted Average Price (VWAP)
            df['feature_vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            df['feature_price_to_vwap'] = df['close'] / df['feature_vwap']
            
            # Volume Oscillator
            vol_5 = df['volume'].rolling(window=5).mean()
            vol_10 = df['volume'].rolling(window=10).mean()
            df['feature_volume_osc'] = (vol_5 - vol_10) / vol_10 * 100
            
            # Normalized volume
            df['feature_norm_volume'] = (df['volume'] - df['volume'].rolling(window=50).min()) / \
                                   (df['volume'].rolling(window=50).max() - df['volume'].rolling(window=50).min())
            
        except Exception as e:
            self.logger.error(f"Error adding volume features: {str(e)}")
    
    def _add_pattern_features(self, df):
        """Add candlestick pattern features"""
        try:
            # Detect common candlestick patterns
            patterns = {
                'CDL2CROWS': talib.CDL2CROWS,
                'CDL3BLACKCROWS': talib.CDL3BLACKCROWS,
                'CDL3INSIDE': talib.CDL3INSIDE,
                'CDL3LINESTRIKE': talib.CDL3LINESTRIKE,
                'CDL3OUTSIDE': talib.CDL3OUTSIDE,
                'CDL3STARSINSOUTH': talib.CDL3STARSINSOUTH,
                'CDL3WHITESOLDIERS': talib.CDL3WHITESOLDIERS,
                'CDLABANDONEDBABY': talib.CDLABANDONEDBABY,
                'CDLADVANCEBLOCK': talib.CDLADVANCEBLOCK,
                'CDLBELTHOLD': talib.CDLBELTHOLD,
                'CDLBREAKAWAY': talib.CDLBREAKAWAY,
                'CDLCLOSINGMARUBOZU': talib.CDLCLOSINGMARUBOZU,
                'CDLCONCEALBABYSWALL': talib.CDLCONCEALBABYSWALL,
                'CDLCOUNTERATTACK': talib.CDLCOUNTERATTACK,
                'CDLDARKCLOUDCOVER': talib.CDLDARKCLOUDCOVER,
                'CDLDOJI': talib.CDLDOJI,
                'CDLDOJISTAR': talib.CDLDOJISTAR,
                'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI,
                'CDLENGULFING': talib.CDLENGULFING,
                'CDLEVENINGDOJISTAR': talib.CDLEVENINGDOJISTAR,
                'CDLEVENINGSTAR': talib.CDLEVENINGSTAR,
                'CDLGAPSIDESIDEWHITE': talib.CDLGAPSIDESIDEWHITE,
                'CDLGRAVESTONEDOJI': talib.CDLGRAVESTONEDOJI,
                'CDLHAMMER': talib.CDLHAMMER,
                'CDLHANGINGMAN': talib.CDLHANGINGMAN,
                'CDLHARAMI': talib.CDLHARAMI,
                'CDLHARAMICROSS': talib.CDLHARAMICROSS,
                'CDLHIGHWAVE': talib.CDLHIGHWAVE,
                'CDLHIKKAKE': talib.CDLHIKKAKE,
                'CDLHIKKAKEMOD': talib.CDLHIKKAKEMOD,
                'CDLHOMINGPIGEON': talib.CDLHOMINGPIGEON,
                'CDLIDENTICAL3CROWS': talib.CDLIDENTICAL3CROWS,
                'CDLINNECK': talib.CDLINNECK,
                'CDLINVERTEDHAMMER': talib.CDLINVERTEDHAMMER,
                'CDLKICKING': talib.CDLKICKING,
                'CDLKICKINGBYLENGTH': talib.CDLKICKINGBYLENGTH,
                'CDLLADDERBOTTOM': talib.CDLLADDERBOTTOM,
                'CDLLONGLEGGEDDOJI': talib.CDLLONGLEGGEDDOJI,
                'CDLMARUBOZU': talib.CDLMARUBOZU,
                'CDLMATCHINGLOW': talib.CDLMATCHINGLOW,
                'CDLMATHOLD': talib.CDLMATHOLD,
                'CDLMORNINGDOJISTAR': talib.CDLMORNINGDOJISTAR,
                'CDLMORNINGSTAR': talib.CDLMORNINGSTAR,
                'CDLONNECK': talib.CDLONNECK,
                'CDLPIERCING': talib.CDLPIERCING,
                'CDLRICKSHAWMAN': talib.CDLRICKSHAWMAN,
                'CDLRISEFALL3METHODS': talib.CDLRISEFALL3METHODS,
                'CDLSEPARATINGLINES': talib.CDLSEPARATINGLINES,
                'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
                'CDLSHORTLINE': talib.CDLSHORTLINE,
                'CDLSPINNINGTOP': talib.CDLSPINNINGTOP,
                'CDLSTALLEDPATTERN': talib.CDLSTALLEDPATTERN,
                'CDLSTICKSANDWICH': talib.CDLSTICKSANDWICH,
                'CDLTAKURI': talib.CDLTAKURI,
                'CDLTASUKIGAP': talib.CDLTASUKIGAP,
                'CDLTHRUSTING': talib.CDLTHRUSTING,
                'CDLTRISTAR': talib.CDLTRISTAR,
                'CDLUNIQUE3RIVER': talib.CDLUNIQUE3RIVER,
                'CDLUPSIDEGAP2CROWS': talib.CDLUPSIDEGAP2CROWS,
                'CDLXSIDEGAP3METHODS': talib.CDLXSIDEGAP3METHODS
            }
            
            # Group patterns by significance to reduce dimensionality
            bullish_patterns = []
            bearish_patterns = []
            
            # Calculate all patterns
            for name, func in patterns.items():
                pattern_values = func(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
                
                # Add significant patterns to feature list
                if np.any(pattern_values != 0):
                    df[f'feature_{name}'] = pattern_values / 100.0  # Normalize to -1 to 1 range
                    
                    # Track bullish/bearish patterns for aggregation
                    if np.any(pattern_values > 0):
                        bullish_patterns.append(f'feature_{name}')
                    if np.any(pattern_values < 0):
                        bearish_patterns.append(f'feature_{name}')
            
            # Aggregate pattern signals
            if bullish_patterns:
                df['feature_bullish_patterns'] = df[bullish_patterns].clip(lower=0).sum(axis=1) / len(bullish_patterns)
            else:
                df['feature_bullish_patterns'] = 0
                
            if bearish_patterns:
                df['feature_bearish_patterns'] = df[bearish_patterns].clip(upper=0).abs().sum(axis=1) / len(bearish_patterns)
            else:
                df['feature_bearish_patterns'] = 0
                
            # Create a combined sentiment indicator
            df['feature_pattern_sentiment'] = df['feature_bullish_patterns'] - df['feature_bearish_patterns']
            
        except Exception as e:
            self.logger.error(f"Error adding pattern features: {str(e)}")
    
    def _add_support_resistance_features(self, df):
        """Add support and resistance level features"""
        try:
            # Calculate pivot points
            df['feature_pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['feature_r1'] = 2 * df['feature_pivot'] - df['low']
            df['feature_s1'] = 2 * df['feature_pivot'] - df['high']
            
            # Calculate distance to pivot levels
            df['feature_dist_to_pivot'] = (df['close'] - df['feature_pivot']) / df['close']
            df['feature_dist_to_r1'] = (df['close'] - df['feature_r1']) / df['close']
            df['feature_dist_to_s1'] = (df['close'] - df['feature_s1']) / df['close']
            
            # Identify potential support levels (previous lows)
            window_sizes = [5, 10, 20]
            for window in window_sizes:
                # Identify local minima
                df[f'feature_min_{window}'] = df['low'].rolling(window=window, center=True).min()
                
                # Distance to local minimum
                df[f'feature_dist_to_min_{window}'] = (df['close'] - df[f'feature_min_{window}']) / df['close']
                
                # Remove the temporary columns
                df = df.drop(columns=[f'feature_min_{window}'])
                
                # Identify local maxima
                df[f'feature_max_{window}'] = df['high'].rolling(window=window, center=True).max()
                
                # Distance to local maximum
                df[f'feature_dist_to_max_{window}'] = (df['close'] - df[f'feature_max_{window}']) / df['close']
                
                # Remove the temporary columns
                df = df.drop(columns=[f'feature_max_{window}'])
            
            # Detect key price levels using volume profile
            # Simplified implementation - find price levels with high volume
            if 'volume' in df.columns:
                # Create price bins
                price_range = df['high'].max() - df['low'].min()
                bin_width = price_range / 10  # 10 bins across the range
                
                df['price_bin'] = ((df['close'] - df['low'].min()) / bin_width).astype(int)
                
                # Volume by price bin
                volume_profile = df.groupby('price_bin')['volume'].sum()
                
                # Find high volume price levels
                high_volume_bins = volume_profile.nlargest(2).index
                
                # Create features based on distance to these levels
                for i, bin_idx in enumerate(high_volume_bins):
                    price_level = df['low'].min() + (bin_idx + 0.5) * bin_width
                    df[f'feature_dist_to_vol_level_{i+1}'] = (df['close'] - price_level) / df['close']
                
                # Drop temporary column
                df = df.drop(columns=['price_bin'])
            
        except Exception as e:
            self.logger.error(f"Error adding support/resistance features: {str(e)}")
    
    def _add_mean_reversion_features(self, df):
        """Add mean reversion and oscillation features"""
        try:
            # Z-score of price relative to recent mean (20 days)
            df['feature_zscore_20d'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
            
            # Z-score of price relative to longer mean (50 days)
            df['feature_zscore_50d'] = (df['close'] - df['close'].rolling(window=50).mean()) / df['close'].rolling(window=50).std()
            
            # Rate of mean reversion (speed of return to mean)
            df['feature_reversion_speed'] = (df['feature_zscore_20d'] - df['feature_zscore_20d'].shift(1))
            
            # RSI distances from overbought/oversold levels
            df['feature_rsi_ob_distance'] = 70 - df['feature_rsi_14']  # Distance from overbought (70)
            df['feature_rsi_os_distance'] = df['feature_rsi_14'] - 30  # Distance from oversold (30)
            
            # Bollinger Band oscillation
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20)
            df['feature_bb_oscillation'] = (df['close'] - lower) / (upper - lower) - 0.5  # Center around 0
            
            # Hurst Exponent approximation
            for period in [30, 60]:
                if len(df) >= period:
                    # Calculate log returns
                    returns = df['feature_returns_1d'].dropna().values[-period:]
                    if len(returns) == period:
                        # Calculate range over different time windows
                        window_sizes = np.arange(2, 21)
                        ranges = []
                        
                        for w in window_sizes:
                            window_returns = [returns[i:i+w] for i in range(0, period-w+1)]
                            window_ranges = [np.max(x) - np.min(x) for x in window_returns]
                            ranges.append(np.mean(window_ranges))
                        
                        # Calculate Hurst by regression slope
                        if all(r > 0 for r in ranges):
                            log_window = np.log(window_sizes)
                            log_range = np.log(ranges)
                            
                            # Simple linear regression
                            slope = np.cov(log_window, log_range)[0, 1] / np.var(log_window)
                            
                            # Hurst = slope
                            df.loc[df.index[-1], f'feature_hurst_{period}'] = slope
                        else:
                            df.loc[df.index[-1], f'feature_hurst_{period}'] = 0.5  # Default (random walk)
                    else:
                        df.loc[df.index[-1], f'feature_hurst_{period}'] = 0.5
            
            # Forward-fill Hurst values
            for period in [30, 60]:
                df[f'feature_hurst_{period}'] = df[f'feature_hurst_{period}'].fillna(method='ffill')
            
            # Mean-reversion potential based on Hurst
            df['feature_mean_reversion_potential'] = 1 - df['feature_hurst_30']
            
        except Exception as e:
            self.logger.error(f"Error adding mean reversion features: {str(e)}")
    
    def _add_trend_features(self, df):
        """Add trend identification features"""
        try:
            # SMA-based trend indicators
            for period in [10, 20, 50, 200]:
                # Calculate SMA
                sma = df['close'].rolling(window=period).mean()
                
                # Price position relative to SMA
                df[f'feature_price_to_sma_{period}'] = df['close'] / sma
                
                # Trend direction (1 for uptrend, -1 for downtrend)
                df[f'feature_trend_direction_{period}'] = np.where(df['close'] > sma, 1, -1)
                
                # Slope of SMA
                df[f'feature_sma_slope_{period}'] = (sma - sma.shift(5)) / (5 * sma.shift(5))
            
            # Moving Average Convergence/Divergence (MACD)
            # Already calculated in momentum features
            
            # Average Directional Index (ADX)
            # Already calculated in momentum features
            
            # MESA Adaptive Moving Average
            mama, fama = talib.MAMA(df['close'].values)
            df['feature_mama'] = mama
            df['feature_fama'] = fama
            df['feature_mama_crossover'] = np.where(df['feature_mama'] > df['feature_fama'], 1, -1)
            
            # Linear regression slope over different periods
            for period in [5, 10, 20]:
                # Calculate time array for regression
                x = np.arange(period)
                
                # Initialize slope column
                df[f'feature_linreg_slope_{period}'] = np.nan
                
                # Calculate for each point with sufficient history
                for i in range(period, len(df)):
                    y = df['close'].values[i-period:i]
                    slope, _ = np.polyfit(x, y, 1)
                    df.iloc[i, df.columns.get_loc(f'feature_linreg_slope_{period}')] = slope / df['close'].iloc[i]
            
            # Parabolic SAR
            df['feature_psar'] = talib.SAR(df['high'].values, df['low'].values)
            df['feature_psar_position'] = np.where(df['close'] > df['feature_psar'], 1, -1)
            
            # DMI (Directional Movement Index)
            plus_di = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            minus_di = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['feature_plus_di'] = plus_di
            df['feature_minus_di'] = minus_di
            df['feature_di_diff'] = plus_di - minus_di

            # Trend strength based on ADX
            adx = df['feature_adx_14']
            df['feature_trend_strength'] = pd.cut(
                adx, 
                bins=[0, 20, 40, 60, 100], 
                labels=[0, 0.33, 0.67, 1]
            ).astype(float)
            
            # Ichimoku Cloud components
            tenkan_period = 9
            kijun_period = 26
            senkou_b_period = 52
            
            # Tenkan-sen (Conversion Line)
            tenkan_high = df['high'].rolling(window=tenkan_period).max()
            tenkan_low = df['low'].rolling(window=tenkan_period).min()
            df['feature_tenkan_sen'] = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (Base Line)
            kijun_high = df['high'].rolling(window=kijun_period).max()
            kijun_low = df['low'].rolling(window=kijun_period).min()
            df['feature_kijun_sen'] = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Leading Span A)
            df['feature_senkou_span_a'] = ((df['feature_tenkan_sen'] + df['feature_kijun_sen']) / 2).shift(kijun_period)
            
            # Senkou Span B (Leading Span B)
            senkou_high = df['high'].rolling(window=senkou_b_period).max()
            senkou_low = df['low'].rolling(window=senkou_b_period).min()
            df['feature_senkou_span_b'] = ((senkou_high + senkou_low) / 2).shift(kijun_period)
            
            # Ichimoku Cloud status (above/below cloud)
            df['feature_cloud_position'] = np.where(
                df['close'] > df['feature_senkou_span_a'], 
                np.where(df['close'] > df['feature_senkou_span_b'], 1, 0), 
                np.where(df['close'] < df['feature_senkou_span_b'], -1, 0)
            )
            
        except Exception as e:
            self.logger.error(f"Error adding trend features: {str(e)}")
    
    def _add_cycle_features(self, df):
        """Add cyclical and seasonal features"""
        try:
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract time-based features
            df['feature_day_of_week'] = df['timestamp'].dt.dayofweek / 6  # Normalize to 0-1
            df['feature_day_of_month'] = (df['timestamp'].dt.day - 1) / 30  # Normalize to 0-1
            df['feature_week_of_year'] = (df['timestamp'].dt.isocalendar().week - 1) / 52  # Normalize to 0-1
            df['feature_month'] = (df['timestamp'].dt.month - 1) / 11  # Normalize to 0-1
            df['feature_quarter'] = (df['timestamp'].dt.quarter - 1) / 3  # Normalize to 0-1
            
            # Cyclical encoding for day of week (sin/cos transformation)
            df['feature_day_of_week_sin'] = np.sin(2 * np.pi * df['feature_day_of_week'])
            df['feature_day_of_week_cos'] = np.cos(2 * np.pi * df['feature_day_of_week'])
            
            # Cyclical encoding for month
            df['feature_month_sin'] = np.sin(2 * np.pi * df['feature_month'])
            df['feature_month_cos'] = np.cos(2 * np.pi * df['feature_month'])
            
            # Identify month-end effect (last 3 days of month)
            df['feature_month_end'] = np.where(df['timestamp'].dt.day >= 28, 1, 0)
            
            # Monday effect
            df['feature_monday'] = np.where(df['timestamp'].dt.dayofweek == 0, 1, 0)
            
            # Friday effect
            df['feature_friday'] = np.where(df['timestamp'].dt.dayofweek == 4, 1, 0)
            
            # Try to detect cycles using Fourier transform
            if len(df) >= 60:  # Need sufficient data
                # Get returns
                returns = df['feature_returns_1d'].dropna().values
                
                # Perform FFT
                fft_values = np.fft.fft(returns)
                fft_magnitudes = np.abs(fft_values)
                
                # Find dominant frequencies
                n = len(returns)
                freq_indices = np.argsort(fft_magnitudes[1:n//2])[-3:]  # Top 3 frequencies
                
                # Create cycle features based on dominant frequencies
                for i, idx in enumerate(freq_indices):
                    freq = (idx + 1) / n  # Frequency
                    period = n / (idx + 1)  # Period in days
                    
                    # Create sine and cosine features for this cycle
                    t = np.arange(len(df))
                    df[f'feature_cycle_{i+1}_sin'] = np.sin(2 * np.pi * freq * t)
                    df[f'feature_cycle_{i+1}_cos'] = np.cos(2 * np.pi * freq * t)
                    df[f'feature_cycle_{i+1}_period'] = period
            
            # HilbertTransform to detect market cycles
            df['feature_hilbert_sine'], df['feature_hilbert_lead'] = talib.HT_SINE(df['close'].values)
            
            # Detect trend/cycle phase using HT_TRENDMODE
            df['feature_ht_trendmode'] = talib.HT_TRENDMODE(df['close'].values)
            
            # Dominant Cycle Period
            df['feature_ht_dcperiod'] = talib.HT_DCPERIOD(df['close'].values)
            
            # Dominant Cycle Phase
            df['feature_ht_dcphase'] = talib.HT_DCPHASE(df['close'].values)
            
            # Phasing components
            df['feature_ht_phasor_inphase'], df['feature_ht_phasor_quadrature'] = talib.HT_PHASOR(df['close'].values)
            
        except Exception as e:
            self.logger.error(f"Error adding cycle features: {str(e)}")
    
    def _add_target_variables(self, df, horizon=5):
        """Add target variables for prediction"""
        try:
            # Future returns over different horizons
            for h in range(1, horizon + 1):
                # Price return
                df[f'target_return_next_{h}d'] = df['close'].shift(-h) / df['close'] - 1
                
                # Directional target
                df[f'target_direction_next_{h}d'] = np.where(df[f'target_return_next_{h}d'] > 0, 1, 0)
                
                # Significant move target (>1% move)
                threshold = 0.01  # 1%
                df[f'target_significant_move_{h}d'] = np.where(
                    df[f'target_return_next_{h}d'] > threshold, 1,
                    np.where(df[f'target_return_next_{h}d'] < -threshold, -1, 0)
                )
            
            # Volatility target
            df['target_volatility_next_5d'] = df['feature_volatility_5d'].shift(-5)
            
            # High/Low range target
            high_5d = df['high'].shift(-1).rolling(window=5).max().shift(-4)
            low_5d = df['low'].shift(-1).rolling(window=5).min().shift(-4)
            df['target_range_next_5d'] = (high_5d - low_5d) / df['close']
            
        except Exception as e:
            self.logger.error(f"Error adding target variables: {str(e)}")
    
    def get_feature_importance(self, symbol, exchange, target='target_direction_next_1d', 
                             timeframe="day", lookback=200):
        """
        Calculate feature importance for a given target
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange (e.g., NSE)
        - target: Target variable to predict
        - timeframe: Data timeframe
        - lookback: Number of historical bars to include

        Returns:
        - DataFrame with feature importances
        """
        try:
            # Generate features
            df = self.generate_features(symbol, exchange, timeframe, lookback)
            if df is None or len(df) < 50:
                return None
            
            # Get feature and target columns
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            
            # Make sure target exists
            if target not in df.columns:
                self.logger.error(f"Target {target} not found in dataframe")
                return None
            
            # Drop rows with missing target
            df = df.dropna(subset=[target])
            
            if len(df) < 50:
                return None
            
            # Split data for feature importance calculation
            X = df[feature_cols].values
            y = df[target].values
            
            # Calculate feature importance using Random Forest
            try:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
                
                # Choose appropriate model based on target type
                if np.all(np.isin(y, [0, 1])) or np.all(np.isin(y, [-1, 0, 1])):
                    # Classification target
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    mutual_info_func = mutual_info_classif
                else:
                    # Regression target
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    mutual_info_func = mutual_info_regression
                
                # Fit model
                model.fit(X, y)
                
                # Get feature importance from model
                rf_importance = model.feature_importances_
                
                # Calculate mutual information
                mi_importance = mutual_info_func(X, y, random_state=42)
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'random_forest_importance': rf_importance,
                    'mutual_info_importance': mi_importance,
                    'combined_importance': (rf_importance + mi_importance) / 2
                })
                
                # Sort by combined importance
                importance_df = importance_df.sort_values('combined_importance', ascending=False)
                
                return importance_df
                
            except Exception as e:
                self.logger.error(f"Error calculating feature importance: {str(e)}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error in get_feature_importance: {str(e)}")
            return None
    
    def select_features(self, features_df, target, top_n=20, correlation_threshold=0.85):
        """
        Select top features while removing highly correlated ones
        
        Parameters:
        - features_df: DataFrame with features and target
        - target: Target variable
        - top_n: Maximum number of features to select
        - correlation_threshold: Threshold for correlation filtering

        Returns:
        - List of selected feature names
        """
        try:
            # Get feature and target columns
            feature_cols = [col for col in features_df.columns if col.startswith('feature_')]
            
            if len(feature_cols) <= top_n:
                return feature_cols
            
            # Calculate feature importance
            importance_df = self.get_feature_importance(features_df, target)
            
            if importance_df is None:
                # If importance calculation fails, return top_n features
                return feature_cols[:top_n]
            
            # Sort features by importance
            sorted_features = importance_df['feature'].tolist()
            
            # Filter based on correlation
            selected_features = []
            
            for feature in sorted_features:
                if len(selected_features) >= top_n:
                    break
                
                # If it's the first feature, add it
                if not selected_features:
                    selected_features.append(feature)
                    continue
                
                # Check correlation with already selected features
                corr_with_selected = features_df[selected_features + [feature]].corr().abs().loc[feature, selected_features]
                
                # If not highly correlated with any selected feature, add it
                if corr_with_selected.max() < correlation_threshold:
                    selected_features.append(feature)
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error in select_features: {str(e)}")
            return feature_cols[:top_n]  # Fallback to simple selection