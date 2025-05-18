# global_features.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class GlobalFeatureGenerator:
    def __init__(self, db_connector):
        """Initialize the global market feature generator"""
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        
        # Initialize scalers
        self.return_scaler = StandardScaler()
        self.corr_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def generate_features(self, symbol=None, exchange=None, timeframe="day", 
                        lookback=50, for_date=None, include_target=False):
        """
        Generate features from global market data
        
        Parameters:
        - symbol: Symbol for target-specific correlations (optional)
        - exchange: Exchange for the symbol (optional)
        - timeframe: Data timeframe
        - lookback: Number of historical periods to include
        - for_date: Optional specific date to generate features for
        - include_target: Whether to include target variables
        
        Returns:
        - DataFrame with global market features
        """
        try:
            # Get global market data
            market_data = self._get_global_market_data(timeframe, lookback, for_date)
            if market_data is None or len(market_data) < 10:
                return None
            
            # Create a new DataFrame for features
            df_features = pd.DataFrame(index=market_data.index)
            
            # Add global market index features
            self._add_index_features(df_features, market_data)
            
            # Add forex features
            self._add_forex_features(df_features, market_data)
            
            # Add commodity features
            self._add_commodity_features(df_features, market_data)
            
            # Add bond and interest rate features
            self._add_bond_features(df_features, market_data)
            
            # Add volatility index features
            self._add_volatility_features(df_features, market_data)
            
            # Add cross-asset correlation features
            self._add_correlation_features(df_features, market_data)
            
            # Add target-specific correlation features if a symbol is provided
            if symbol and exchange:
                self._add_target_correlation_features(df_features, market_data, symbol, exchange, timeframe)
            
            # Add economic indicator features
            self._add_economic_features(df_features, for_date)
            
            # Add seasonal features
            self._add_seasonal_features(df_features)
            
            # Add market breadth features
            self._add_market_breadth_features(df_features, market_data)
            
            # Add momentum and regime features
            self._add_market_regime_features(df_features, market_data)
            
            # Add target variables if requested
            if include_target and symbol and exchange:
                self._add_target_variables(df_features, symbol, exchange, timeframe, for_date)
            
            # Prefix all features with 'feature_'
            df_features = df_features.rename(columns={col: f'feature_{col}' 
                                                    for col in df_features.columns 
                                                    if not col.startswith(('feature_', 'target_'))})
            
            # Drop rows with NaN values
            df_features = df_features.dropna(how='all')
            
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error generating global features: {str(e)}")
            return None
    
    def _get_global_market_data(self, timeframe, lookback, for_date=None):
        """Get global market data from database"""
        try:
            # Base query for global market data
            query = {
                "data_type": "global_market",
                "timeframe": timeframe
            }
            
            # Add date filter if specified
            if for_date:
                query["timestamp"] = {"$lte": for_date}
            
            # Get data from database
            data = list(self.db.global_market_collection.find(
                query
            ).sort("timestamp", -1).limit(lookback))
            
            if not data:
                return None
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            
            # Set timestamp as index
            df = df.set_index("timestamp")
            
            # Sort by timestamp
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting global market data: {str(e)}")
            return None
    
    def _add_index_features(self, df_features, market_data):
        """Add features based on global market indices"""
        try:
            # Define major indices to track
            indices = [
                # US indices
                "SPX", "DJIA", "NASDAQ", "RUSSELL2000",
                # European indices
                "DAX", "FTSE", "CAC40", "STOXX50",
                # Asian indices
                "NIKKEI", "HSI", "KOSPI", "SHANGHAI",
                # Indian indices
                "NIFTY50", "SENSEX", "NIFTYBANK", "NIFTYMIDCAP"
            ]
            
            # Calculate return features for each index
            for index in indices:
                if index in market_data.columns:
                    # Calculate returns
                    df_features[f'{index}_return_1d'] = market_data[index].pct_change(1)
                    
                    # Calculate multi-period returns
                    for period in [3, 5, 10]:
                        df_features[f'{index}_return_{period}d'] = market_data[index].pct_change(period)
                    
                    # Calculate moving averages and relative strength
                    for ma_period in [10, 20, 50]:
                        # Moving average
                        ma_col = f'{index}_ma_{ma_period}'
                        market_data[ma_col] = market_data[index].rolling(window=ma_period).mean()
                        
                        # Price relative to moving average
                        df_features[f'{index}_rel_ma_{ma_period}'] = market_data[index] / market_data[ma_col] - 1
                    
                    # Calculate momentum
                    df_features[f'{index}_momentum_10d'] = market_data[index].pct_change(10)
                    
                    # Calculate volatility
                    df_features[f'{index}_volatility_10d'] = market_data[index].pct_change().rolling(window=10).std()
                    
                    # Calculate RSI
                    delta = market_data[index].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    
                    rs = gain / loss
                    df_features[f'{index}_rsi_14d'] = 100 - (100 / (1 + rs))
            
            # Add cross-index relationships
            # S&P 500 vs. NIFTY50
            if "SPX" in market_data.columns and "NIFTY50" in market_data.columns:
                df_features['spx_vs_nifty'] = market_data["SPX"].pct_change(5) - market_data["NIFTY50"].pct_change(5)
            
            # DJIA vs. SENSEX
            if "DJIA" in market_data.columns and "SENSEX" in market_data.columns:
                df_features['djia_vs_sensex'] = market_data["DJIA"].pct_change(5) - market_data["SENSEX"].pct_change(5)
            
            # Nikkei vs. Asian indices
            if "NIKKEI" in market_data.columns and "HSI" in market_data.columns:
                df_features['nikkei_vs_hsi'] = market_data["NIKKEI"].pct_change(5) - market_data["HSI"].pct_change(5)
            
            # US vs. Europe
            if "SPX" in market_data.columns and "STOXX50" in market_data.columns:
                df_features['us_vs_europe'] = market_data["SPX"].pct_change(5) - market_data["STOXX50"].pct_change(5)
            
            # Developed vs. Emerging Markets (approximation)
            if "SPX" in market_data.columns and "NIFTY50" in market_data.columns and "SHANGHAI" in market_data.columns:
                developed = market_data["SPX"].pct_change(5)
                emerging = (market_data["NIFTY50"].pct_change(5) + market_data["SHANGHAI"].pct_change(5)) / 2
                df_features['developed_vs_emerging'] = developed - emerging
            
            # Add market leadership features (e.g., NASDAQ vs. S&P 500 for growth vs. value)
            if "NASDAQ" in market_data.columns and "SPX" in market_data.columns:
                df_features['growth_vs_value'] = market_data["NASDAQ"].pct_change(10) - market_data["SPX"].pct_change(10)
            
            # Add small cap vs. large cap features
            if "RUSSELL2000" in market_data.columns and "SPX" in market_data.columns:
                df_features['small_vs_large'] = market_data["RUSSELL2000"].pct_change(10) - market_data["SPX"].pct_change(10)
            
            # Normalize return features
            return_cols = [col for col in df_features.columns if 'return' in col]
            
            if return_cols:
                df_features[return_cols] = self.return_scaler.fit_transform(df_features[return_cols].fillna(0))
            
        except Exception as e:
            self.logger.error(f"Error adding index features: {str(e)}")
    
    def _add_forex_features(self, df_features, market_data):
        """Add features based on forex rates"""
        try:
            # Define key forex pairs to track
            forex_pairs = [
                # USD pairs
                "USDINR", "EURUSD", "GBPUSD", "USDJPY", "USDCNY",
                # Cross rates
                "EURGBP", "EURJPY", "GBPJPY",
                # Commodity currencies
                "AUDUSD", "USDCAD", "NZDUSD"
            ]
            
            # Calculate return features for each pair
            for pair in forex_pairs:
                if pair in market_data.columns:
                    # Calculate returns
                    df_features[f'{pair}_return_1d'] = market_data[pair].pct_change(1)
                    
                    # Calculate multi-period returns
                    for period in [3, 5, 10]:
                        df_features[f'{pair}_return_{period}d'] = market_data[pair].pct_change(period)
                    
                    # Calculate volatility
                    df_features[f'{pair}_volatility_10d'] = market_data[pair].pct_change().rolling(window=10).std()
                    
                    # Calculate Z-score (standard deviations from 20-day mean)
                    mean_20d = market_data[pair].rolling(window=20).mean()
                    std_20d = market_data[pair].rolling(window=20).std()
                    df_features[f'{pair}_zscore_20d'] = (market_data[pair] - mean_20d) / std_20d
            
            # Add USD index features if available
            if "USDX" in market_data.columns:
                df_features['usdx_return_1d'] = market_data["USDX"].pct_change(1)
                df_features['usdx_return_5d'] = market_data["USDX"].pct_change(5)
                df_features['usdx_volatility_10d'] = market_data["USDX"].pct_change().rolling(window=10).std()
            
            # Add INR features (especially important for Indian market)
            if "USDINR" in market_data.columns:
                # Add strength/weakness indicators
                for period in [5, 10, 20]:
                    df_features[f'inr_strength_{period}d'] = -1 * market_data["USDINR"].pct_change(period)  # Negative because INR strengthens when USDINR falls
                
                # Calculate INR volatility
                df_features['inr_volatility_10d'] = market_data["USDINR"].pct_change().rolling(window=10).std()
                
                # Calculate moving averages and relative strength
                for ma_period in [10, 20, 50]:
                    ma_col = f'usdinr_ma_{ma_period}'
                    market_data[ma_col] = market_data["USDINR"].rolling(window=ma_period).mean()
                    df_features[f'usdinr_rel_ma_{ma_period}'] = market_data["USDINR"] / market_data[ma_col] - 1
            
            # Create currency strength indicators
            # USD strength
            usd_pairs = ["USDINR", "EURUSD", "GBPUSD", "USDJPY", "USDCNY", "AUDUSD", "USDCAD", "NZDUSD"]
            usd_pairs_in_data = [pair for pair in usd_pairs if pair in market_data.columns]
            
            if len(usd_pairs_in_data) >= 3:  # Need at least 3 pairs for meaningful index
                # For pairs like EURUSD, GBPUSD where USD is quoted second, take negative return
                # For pairs like USDINR, USDJPY where USD is quoted first, take positive return
                usd_strength_components = []
                
                for pair in usd_pairs_in_data:
                    returns = market_data[pair].pct_change(5)
                    # If USD is the base currency (first in the pair)
                    if pair.startswith("USD"):
                        usd_strength_components.append(returns)
                    # If USD is the quote currency (second in the pair)
                    else:
                        usd_strength_components.append(-returns)
                
                # Average the components
                df_features['usd_strength_index'] = pd.concat(usd_strength_components, axis=1).mean(axis=1)
            
            # EUR strength (similar approach)
            eur_pairs = ["EURUSD", "EURGBP", "EURJPY"]
            eur_pairs_in_data = [pair for pair in eur_pairs if pair in market_data.columns]
            
            if len(eur_pairs_in_data) >= 2:
                eur_strength_components = []
                
                for pair in eur_pairs_in_data:
                    returns = market_data[pair].pct_change(5)
                    # If EUR is the base currency (first in the pair)
                    if pair.startswith("EUR"):
                        eur_strength_components.append(returns)
                    # If EUR is the quote currency (second in the pair)
                    else:
                        eur_strength_components.append(-returns)
                
                df_features['eur_strength_index'] = pd.concat(eur_strength_components, axis=1).mean(axis=1)
            
        except Exception as e:
            self.logger.error(f"Error adding forex features: {str(e)}")
    
    def _add_commodity_features(self, df_features, market_data):
        """Add features based on commodity prices"""
        try:
            # Define key commodities to track
            commodities = [
                # Energy
                "CRUDE_OIL", "BRENT_OIL", "NATURAL_GAS",
                # Metals
                "GOLD", "SILVER", "COPPER", "ALUMINUM",
                # Agriculture
                "WHEAT", "CORN", "SOYBEAN",
                # Indian relevance
                "MCX_GOLD", "MCX_SILVER", "MCX_CRUDEOIL"
            ]
            
            # Calculate return features for each commodity
            for commodity in commodities:
                if commodity in market_data.columns:
                    # Calculate returns
                    df_features[f'{commodity}_return_1d'] = market_data[commodity].pct_change(1)
                    
                    # Calculate multi-period returns
                    for period in [3, 5, 10]:
                        df_features[f'{commodity}_return_{period}d'] = market_data[commodity].pct_change(period)
                    
                    # Calculate volatility
                    df_features[f'{commodity}_volatility_10d'] = market_data[commodity].pct_change().rolling(window=10).std()
                    
                    # Calculate moving averages and relative strength
                    for ma_period in [10, 20, 50]:
                        ma_col = f'{commodity}_ma_{ma_period}'
                        market_data[ma_col] = market_data[commodity].rolling(window=ma_period).mean()
                        df_features[f'{commodity}_rel_ma_{ma_period}'] = market_data[commodity] / market_data[ma_col] - 1
            
            # Add specific commodity relationships
            
            # Gold to Silver ratio
            if "GOLD" in market_data.columns and "SILVER" in market_data.columns:
                df_features['gold_to_silver_ratio'] = market_data["GOLD"] / market_data["SILVER"]
                df_features['gold_to_silver_ratio_z'] = (df_features['gold_to_silver_ratio'] - 
                                                      df_features['gold_to_silver_ratio'].rolling(50).mean()) / \
                                                     df_features['gold_to_silver_ratio'].rolling(50).std()
            
            # Oil price changes
            if "CRUDE_OIL" in market_data.columns:
                # Oil price momentum
                df_features['oil_momentum'] = market_data["CRUDE_OIL"].pct_change(10)
                
                # Oil volatility
                df_features['oil_volatility'] = market_data["CRUDE_OIL"].pct_change().rolling(window=10).std()
                
                # Oil price shock (large moves)
                mean_change = market_data["CRUDE_OIL"].pct_change().abs().rolling(60).mean()
                std_change = market_data["CRUDE_OIL"].pct_change().abs().rolling(60).std()
                daily_change = market_data["CRUDE_OIL"].pct_change().abs()
                
                df_features['oil_shock'] = (daily_change > (mean_change + 2 * std_change)).astype(int)
            
            # Add commodity-equity relationships
            
            # Gold vs. S&P 500 (fear indicator)
            if "GOLD" in market_data.columns and "SPX" in market_data.columns:
                df_features['gold_vs_spx'] = market_data["GOLD"].pct_change(5) - market_data["SPX"].pct_change(5)
            
            # Oil vs. Energy sector index (if available)
            if "CRUDE_OIL" in market_data.columns and "ENERGY_INDEX" in market_data.columns:
                df_features['oil_vs_energy_sector'] = market_data["CRUDE_OIL"].pct_change(5) - market_data["ENERGY_INDEX"].pct_change(5)
            
            # Commodity aggregate indices
            energy_cols = [col for col in ["CRUDE_OIL", "BRENT_OIL", "NATURAL_GAS"] if col in market_data.columns]
            metals_cols = [col for col in ["GOLD", "SILVER", "COPPER", "ALUMINUM"] if col in market_data.columns]
            agri_cols = [col for col in ["WHEAT", "CORN", "SOYBEAN"] if col in market_data.columns]
            
            # Energy index
            if len(energy_cols) >= 2:
                energy_returns = market_data[energy_cols].pct_change(5)
                df_features['energy_index_return'] = energy_returns.mean(axis=1)
            
            # Metals index
            if len(metals_cols) >= 2:
                metals_returns = market_data[metals_cols].pct_change(5)
                df_features['metals_index_return'] = metals_returns.mean(axis=1)
            
            # Agriculture index
            if len(agri_cols) >= 2:
                agri_returns = market_data[agri_cols].pct_change(5)
                df_features['agri_index_return'] = agri_returns.mean(axis=1)
            
            # Overall commodity index
            all_commodity_cols = energy_cols + metals_cols + agri_cols
            if len(all_commodity_cols) >= 3:
                commodity_returns = market_data[all_commodity_cols].pct_change(5)
                df_features['commodity_index_return'] = commodity_returns.mean(axis=1)
            
        except Exception as e:
            self.logger.error(f"Error adding commodity features: {str(e)}")
    
    def _add_bond_features(self, df_features, market_data):
        """Add features based on bond yields and interest rates"""
        try:
            # Define key bond yields and rates to track
            bonds = [
                # US bonds
                "US10Y", "US2Y", "US30Y", "US3M",
                # Indian bonds
                "IND10Y", "IND2Y", "IND5Y",
                # Other major markets
                "GER10Y", "UK10Y", "JPN10Y"
            ]
            
            # Add yield features
            for bond in bonds:
                if bond in market_data.columns:
                    # Absolute yield
                    df_features[f'{bond}_yield'] = market_data[bond]
                    
                    # Yield change
                    df_features[f'{bond}_change_1d'] = market_data[bond].diff(1)
                    df_features[f'{bond}_change_5d'] = market_data[bond].diff(5)
                    
                    # Yield volatility
                    df_features[f'{bond}_volatility_10d'] = market_data[bond].diff().rolling(window=10).std()
                    
                    # Yield z-score (compared to 60-day average)
                    mean_60d = market_data[bond].rolling(window=60).mean()
                    std_60d = market_data[bond].rolling(window=60).std()
                    df_features[f'{bond}_zscore_60d'] = (market_data[bond] - mean_60d) / std_60d
            
            # Add yield curve metrics
            # US yield curve
            if "US10Y" in market_data.columns and "US2Y" in market_data.columns:
                # 10Y-2Y spread
                df_features['us_10y_2y_spread'] = market_data["US10Y"] - market_data["US2Y"]
                
                # Yield curve steepness
                df_features['us_yield_curve_steepness'] = df_features['us_10y_2y_spread']
                
                # Yield curve inversion (binary indicator)
                df_features['us_yield_curve_inverted'] = (df_features['us_10y_2y_spread'] < 0).astype(int)
                
                # Yield curve direction
                df_features['us_yield_curve_direction'] = df_features['us_10y_2y_spread'].diff(5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            
            # Indian yield curve
            if "IND10Y" in market_data.columns and "IND2Y" in market_data.columns:
                # 10Y-2Y spread
                df_features['ind_10y_2y_spread'] = market_data["IND10Y"] - market_data["IND2Y"]
                
                # Yield curve steepness
                df_features['ind_yield_curve_steepness'] = df_features['ind_10y_2y_spread']
                
                # Yield curve inversion (binary indicator)
                df_features['ind_yield_curve_inverted'] = (df_features['ind_10y_2y_spread'] < 0).astype(int)
                
                # Yield curve direction
                df_features['ind_yield_curve_direction'] = df_features['ind_10y_2y_spread'].diff(5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            
            # Interest rate indicators
            
            # US policy rate related
            if "US_POLICY_RATE" in market_data.columns:
                df_features['us_policy_rate'] = market_data["US_POLICY_RATE"]
                df_features['us_policy_rate_change_3m'] = market_data["US_POLICY_RATE"].diff(60)  # Assuming daily data, ~60 trading days in 3 months
                
                # Policy rate direction
                df_features['us_rate_direction'] = market_data["US_POLICY_RATE"].diff(20).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            
            # Indian policy rate related
            if "IND_POLICY_RATE" in market_data.columns:
                df_features['ind_policy_rate'] = market_data["IND_POLICY_RATE"]
                df_features['ind_policy_rate_change_3m'] = market_data["IND_POLICY_RATE"].diff(60)
                
                # Policy rate direction
                df_features['ind_rate_direction'] = market_data["IND_POLICY_RATE"].diff(20).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            
            # Cross-market yield differentials
            
            # US-India yield differential
            if "US10Y" in market_data.columns and "IND10Y" in market_data.columns:
                df_features['us_ind_10y_differential'] = market_data["IND10Y"] - market_data["US10Y"]
                
                # Z-score of differential
                mean_diff = df_features['us_ind_10y_differential'].rolling(window=60).mean()
                std_diff = df_features['us_ind_10y_differential'].rolling(window=60).std()
                df_features['us_ind_10y_differential_z'] = (df_features['us_ind_10y_differential'] - mean_diff) / std_diff
                
                # Change in differential
                df_features['us_ind_10y_differential_change'] = df_features['us_ind_10y_differential'].diff(5)
            
            # Real yield features (if inflation data available)
            if "US10Y" in market_data.columns and "US_CPI" in market_data.columns:
                df_features['us_real_10y_yield'] = market_data["US10Y"] - market_data["US_CPI"]
            
            if "IND10Y" in market_data.columns and "IND_CPI" in market_data.columns:
                df_features['ind_real_10y_yield'] = market_data["IND10Y"] - market_data["IND_CPI"]
            
        except Exception as e:
            self.logger.error(f"Error adding bond features: {str(e)}")
    
    def _add_volatility_features(self, df_features, market_data):
        """Add features based on volatility indices"""
        try:
            # Define volatility indices to track
            vol_indices = ["VIX", "INDIA_VIX", "VSTOXX", "VKOSPI"]
            
            # Add volatility features
            for index in vol_indices:
                if index in market_data.columns:
                    # Absolute level
                    df_features[f'{index}_level'] = market_data[index]
                    
                    # Changes
                    df_features[f'{index}_change_1d'] = market_data[index].diff(1)
                    df_features[f'{index}_change_5d'] = market_data[index].diff(5)
                    
                    # Percentage changes
                    df_features[f'{index}_pct_change_1d'] = market_data[index].pct_change(1)
                    df_features[f'{index}_pct_change_5d'] = market_data[index].pct_change(5)
                    
                    # Z-scores
                    mean_20d = market_data[index].rolling(window=20).mean()
                    std_20d = market_data[index].rolling(window=20).std()
                    df_features[f'{index}_zscore_20d'] = (market_data[index] - mean_20d) / std_20d
                    
                    # Volatility regimes
                    low_threshold = market_data[index].rolling(window=252).quantile(0.25)
                    high_threshold = market_data[index].rolling(window=252).quantile(0.75)
                    
                    df_features[f'{index}_regime'] = 0  # Normal
                    df_features.loc[market_data[index] <= low_threshold, f'{index}_regime'] = -1  # Low volatility
                    df_features.loc[market_data[index] >= high_threshold, f'{index}_regime'] = 1  # High volatility
                    
                    # Rate of change in volatility
                    df_features[f'{index}_roc_5d'] = market_data[index].diff(5) / market_data[index].shift(5)
                    
                    # Volatility of volatility
                    df_features[f'{index}_vol_of_vol'] = market_data[index].rolling(window=10).std() / market_data[index].rolling(window=10).mean()
            
            # Add specific volatility relationships
            
            # VIX vs S&P 500 returns
            if "VIX" in market_data.columns and "SPX" in market_data.columns:
                df_features['vix_spx_correlation'] = market_data[["VIX", "SPX"]].pct_change().rolling(window=20).corr().iloc[::20]["SPX"]
            
            # VIX term structure (if available)
            if "VIX" in market_data.columns and "VIX3M" in market_data.columns:
                df_features['vix_term_structure'] = market_data["VIX"] / market_data["VIX3M"]
                
                # Contango/backwardation indicator
                df_features['vix_contango'] = (df_features['vix_term_structure'] < 1).astype(int)
            
            # India VIX vs. NIFTY correlation
            if "INDIA_VIX" in market_data.columns and "NIFTY50" in market_data.columns:
                df_features['indiavix_nifty_correlation'] = market_data[["INDIA_VIX", "NIFTY50"]].pct_change().rolling(window=20).corr().iloc[::20]["NIFTY50"]
            
            # Create a global volatility index
            vol_cols = [col for col in vol_indices if col in market_data.columns]
            if len(vol_cols) >= 2:
                # Normalize each volatility index
                normalized_vols = {}
                for col in vol_cols:
                    normalized_vols[col] = (market_data[col] - market_data[col].rolling(window=252).min()) / \
                                          (market_data[col].rolling(window=252).max() - market_data[col].rolling(window=252).min())
                
                # Average the normalized indices
                df_features['global_vol_index'] = pd.DataFrame(normalized_vols).mean(axis=1)
                
                # Volatility regime based on global index
                df_features['global_vol_regime'] = pd.cut(
                    df_features['global_vol_index'],
                    bins=[0, 0.3, 0.7, 1],
                    labels=[-1, 0, 1]  # Low, Normal, High
                ).astype(int)
            
            # Volatility risk premium (if implied and realized volatility available)
            if "VIX" in market_data.columns and "SPX_REALIZED_VOL" in market_data.columns:
                df_features['vol_risk_premium'] = market_data["VIX"] - market_data["SPX_REALIZED_VOL"]
            
            if "INDIA_VIX" in market_data.columns and "NIFTY_REALIZED_VOL" in market_data.columns:
                df_features['india_vol_risk_premium'] = market_data["INDIA_VIX"] - market_data["NIFTY_REALIZED_VOL"]
            
        except Exception as e:
            self.logger.error(f"Error adding volatility features: {str(e)}")
    
    def _add_correlation_features(self, df_features, market_data):
        """Add cross-asset correlation features"""
        try:
            # Define key asset pairs to analyze
            asset_pairs = [
                # Equity index correlations
                ("SPX", "NIFTY50"),
                ("SPX", "SHANGHAI"),
                ("NIFTY50", "SENSEX"),
                ("NIFTY50", "NIFTYBANK"),
                # Equity vs. other asset classes
                ("SPX", "US10Y"),
                ("NIFTY50", "IND10Y"),
                ("SPX", "GOLD"),
                ("NIFTY50", "GOLD"),
                ("SPX", "CRUDE_OIL"),
                ("NIFTY50", "CRUDE_OIL"),
                # Currency vs. equity
                ("USDINR", "NIFTY50"),
                ("EURUSD", "SPX"),
                # Bond vs. currency
                ("US10Y", "EURUSD"),
                ("IND10Y", "USDINR")
            ]
            
            # Calculate rolling correlations
            for period in [20, 60]:
                for asset1, asset2 in asset_pairs:
                    if asset1 in market_data.columns and asset2 in market_data.columns:
                        # Calculate correlation
                        correlation = market_data[[asset1, asset2]].pct_change().rolling(window=period).corr()
                        # Extract the correlation values (every period rows, to avoid duplicate entries)
                        corr_values = correlation.iloc[period-1::period][asset2][asset1]
                        # Extend to daily values by forward filling
                        corr_series = corr_values.reindex(market_data.index).ffill()
                        
                        # Add to features
                        df_features[f'{asset1}_{asset2}_corr_{period}d'] = corr_series
            
            # Calculate cross-asset correlation index (average pairwise correlation)
            main_indices = [idx for idx in ["SPX", "NIFTY50", "DAX", "FTSE", "NIKKEI", "HSI"] if idx in market_data.columns]
            
            if len(main_indices) >= 3:
                # Calculate pairwise correlations
                corr_matrix = market_data[main_indices].pct_change().rolling(window=20).corr()
                
                # Extract unique pairs
                avg_correlations = []
                for i in range(len(main_indices)):
                    for j in range(i+1, len(main_indices)):
                        idx1, idx2 = main_indices[i], main_indices[j]
                        # Get correlation series for this pair
                        pair_corr = corr_matrix.xs(idx2, level=1)[idx1].iloc[19::20]
                        # Reindex to all dates and forward fill
                        pair_corr = pair_corr.reindex(market_data.index).ffill()
                        avg_correlations.append(pair_corr)
                
                # Calculate average correlation
                if avg_correlations:
                    df_features['global_market_correlation'] = pd.concat(avg_correlations, axis=1).mean(axis=1)
                    
                    # Correlation regime
                    df_features['correlation_regime'] = pd.cut(
                        df_features['global_market_correlation'],
                        bins=[-1, 0.3, 0.7, 1],
                        labels=[0, 1, 2]  # Low, Medium, High
                    ).astype(int)
            
            # Correlation between stocks and bonds
            if "SPX" in market_data.columns and "US10Y" in market_data.columns:
                stock_bond_corr = market_data[["SPX", "US10Y"]].pct_change().rolling(window=60).corr()
                stock_bond_corr = stock_bond_corr.iloc[59::60]["US10Y"]["SPX"]
                df_features['us_stock_bond_corr'] = stock_bond_corr.reindex(market_data.index).ffill()
            
            if "NIFTY50" in market_data.columns and "IND10Y" in market_data.columns:
                india_stock_bond_corr = market_data[["NIFTY50", "IND10Y"]].pct_change().rolling(window=60).corr()
                india_stock_bond_corr = india_stock_bond_corr.iloc[59::60]["IND10Y"]["NIFTY50"]
                df_features['india_stock_bond_corr'] = india_stock_bond_corr.reindex(market_data.index).ffill()
            
            # Correlation between developed and emerging markets
            dev_indices = [idx for idx in ["SPX", "DAX", "FTSE", "NIKKEI"] if idx in market_data.columns]
            em_indices = [idx for idx in ["NIFTY50", "SHANGHAI", "HSI", "KOSPI"] if idx in market_data.columns]
            
            if len(dev_indices) >= 2 and len(em_indices) >= 2:
                # Calculate average return for developed and emerging markets
                dev_returns = market_data[dev_indices].pct_change().mean(axis=1)
                em_returns = market_data[em_indices].pct_change().mean(axis=1)
                
                # Calculate correlation
                dev_em_corr = pd.concat([dev_returns, em_returns], axis=1).rolling(window=60).corr()
                dev_em_corr = dev_em_corr.iloc[59::60][1][0]  # Extract correlation values
                df_features['dev_em_correlation'] = dev_em_corr.reindex(market_data.index).ffill()
            
            # Add sector correlation features (if sector indices available)
            sector_indices = [col for col in market_data.columns if col.endswith('_SECTOR')]
            if len(sector_indices) >= 3:
                # Calculate average pairwise correlation
                sector_corr_matrix = market_data[sector_indices].pct_change().rolling(window=20).corr()
                
                sector_avg_correlations = []
                for i in range(len(sector_indices)):
                    for j in range(i+1, len(sector_indices)):
                        sector1, sector2 = sector_indices[i], sector_indices[j]
                        pair_corr = sector_corr_matrix.xs(sector2, level=1)[sector1].iloc[19::20]
                        pair_corr = pair_corr.reindex(market_data.index).ffill()
                        sector_avg_correlations.append(pair_corr)
                
                if sector_avg_correlations:
                    df_features['sector_correlation'] = pd.concat(sector_avg_correlations, axis=1).mean(axis=1)
                    
                    # Sector correlation regime
                    df_features['sector_correlation_regime'] = pd.cut(
                        df_features['sector_correlation'],
                        bins=[-1, 0.3, 0.7, 1],
                        labels=[0, 1, 2]  # Low, Medium, High
                    ).astype(int)
            
            # Normalize correlation features
            corr_cols = [col for col in df_features.columns if 'corr' in col]
            if corr_cols:
                df_features[corr_cols] = self.corr_scaler.fit_transform(df_features[corr_cols].fillna(0))
            
        except Exception as e:
            self.logger.error(f"Error adding correlation features: {str(e)}")
    
    def _add_target_correlation_features(self, df_features, market_data, symbol, exchange, timeframe):
        """Add correlation features specific to the target symbol"""
        try:
            # Get target symbol historical data
            target_data = self._get_target_data(symbol, exchange, timeframe)
            if target_data is None or len(target_data) < 20:
                return
            
            # Align indices
            common_index = market_data.index.intersection(target_data.index)
            if len(common_index) < 20:
                return
            
            market_subset = market_data.loc[common_index]
            target_subset = target_data.loc[common_index]
            
            # Define key assets to check correlation with
            key_assets = [
                # Major indices
                "SPX", "NIFTY50", "SENSEX", "NIFTYBANK",
                # Currencies
                "USDINR", "EURUSD",
                # Commodities
                "CRUDE_OIL", "GOLD",
                # Bonds
                "US10Y", "IND10Y",
                # Volatility
                "VIX", "INDIA_VIX"
            ]
            
            # Calculate correlation with key assets
            for asset in key_assets:
                if asset in market_subset.columns:
                    # Calculate correlation for different periods
                    for period in [20, 60]:
                        if len(common_index) < period:
                            continue
                        
                        # Get correlation
                        corr_series = pd.concat([target_subset["close"], market_subset[asset]], axis=1)
                        corr_series.columns = ["target", "asset"]
                        
                        correlation = corr_series.pct_change().rolling(window=period).corr()
                        corr_values = correlation.iloc[period-1::period]["asset"]["target"]
                        
                        # Reindex and forward fill
                        corr_series = corr_values.reindex(df_features.index).ffill()
                        
                        # Add to features
                        df_features[f'target_{asset}_corr_{period}d'] = corr_series
            
            # Calculate correlation with sector indices (if available)
            target_sector = self._get_target_sector(symbol, exchange)
            if target_sector:
                sector_index = f"{target_sector}_SECTOR"
                if sector_index in market_subset.columns:
                    # Calculate correlation for different periods
                    for period in [20, 60]:
                        if len(common_index) < period:
                            continue
                        
                        # Get correlation
                        corr_series = pd.concat([target_subset["close"], market_subset[sector_index]], axis=1)
                        corr_series.columns = ["target", "sector"]
                        
                        correlation = corr_series.pct_change().rolling(window=period).corr()
                        corr_values = correlation.iloc[period-1::period]["sector"]["target"]
                        
                        # Reindex and forward fill
                        corr_series = corr_values.reindex(df_features.index).ffill()
                        
                        # Add to features
                        df_features[f'target_sector_corr_{period}d'] = corr_series
                
                # Calculate relative performance vs. sector
                if sector_index in market_subset.columns:
                    # 1-month relative performance
                    target_return = target_subset["close"].pct_change(20)
                    sector_return = market_subset[sector_index].pct_change(20)
                    
                    rel_performance = target_return - sector_return
                    df_features['target_sector_rel_perf'] = rel_performance.reindex(df_features.index).ffill()
            
            # Calculate beta (sensitivity to market)
            if "NIFTY50" in market_subset.columns:
                # Calculate 60-day rolling beta
                target_returns = target_subset["close"].pct_change().dropna()
                market_returns = market_subset["NIFTY50"].pct_change().dropna()
                
                if len(target_returns) >= 60:
                    # Calculate beta for rolling 60-day windows
                    betas = []
                    dates = []
                    
                    for i in range(60, len(target_returns), 20):
                        end_idx = min(i, len(target_returns))
                        start_idx = end_idx - 60
                        
                        x = market_returns.iloc[start_idx:end_idx]
                        y = target_returns.iloc[start_idx:end_idx]
                        
                        # Calculate covariance and market variance
                        covariance = np.cov(x, y)[0, 1]
                        market_variance = np.var(x)
                        
                        # Calculate beta
                        if market_variance > 0:
                            beta = covariance / market_variance
                        else:
                            beta = 1.0
                        
                        betas.append(beta)
                        dates.append(target_returns.index[end_idx-1])
                    
                    # Create beta series
                    beta_series = pd.Series(betas, index=dates)
                    
                    # Reindex and forward fill
                    df_features['target_beta'] = beta_series.reindex(df_features.index, method='ffill')
                    
                    # Beta regime
                    df_features['target_beta_regime'] = pd.cut(
                        df_features['target_beta'],
                        bins=[-float('inf'), 0.7, 1.3, float('inf')],
                        labels=[0, 1, 2]  # Defensive, Neutral, Aggressive
                    ).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error adding target correlation features: {str(e)}")
    
    def _get_target_data(self, symbol, exchange, timeframe):
        """Get historical data for target symbol"""
        try:
            # Query for target data
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }
            
            # Get historical data
            data = list(self.db.market_data_collection.find(
                query,
                {"timestamp": 1, "close": 1}
            ))
            
            if not data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Set timestamp as index
            df = df.set_index("timestamp")
            
            # Sort by timestamp
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting target data: {str(e)}")
            return None
    
    def _get_target_sector(self, symbol, exchange):
        """Get sector for target symbol"""
        try:
            # Query for symbol metadata
            query = {
                "symbol": symbol,
                "exchange": exchange
            }
            
            # Get symbol metadata
            metadata = self.db.stock_metadata_collection.find_one(query)
            
            if metadata and "sector" in metadata:
                return metadata["sector"]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting target sector: {str(e)}")
            return None
    
    def _add_economic_features(self, df_features, for_date=None):
        """Add economic indicator features"""
        try:
            # Get economic data
            economic_data = self._get_economic_data(for_date)
            if not economic_data:
                return
            
            # Add GDP growth rate
            if "US_GDP_GROWTH" in economic_data:
                df_features['us_gdp_growth'] = economic_data["US_GDP_GROWTH"]
                
                # GDP growth momentum
                df_features['us_gdp_momentum'] = economic_data.get("US_GDP_MOMENTUM", 0)  # Acceleration/deceleration
                
                # GDP growth regime
                df_features['us_gdp_regime'] = pd.cut(
                    df_features['us_gdp_growth'],
                    bins=[-float('inf'), 1.0, 3.0, float('inf')],
                    labels=[0, 1, 2]  # Slow, Moderate, Fast
                ).astype(int)
            
            if "INDIA_GDP_GROWTH" in economic_data:
                df_features['india_gdp_growth'] = economic_data["INDIA_GDP_GROWTH"]
                
                # GDP growth momentum
                df_features['india_gdp_momentum'] = economic_data.get("INDIA_GDP_MOMENTUM", 0)
                
                # GDP growth regime
                df_features['india_gdp_regime'] = pd.cut(
                    df_features['india_gdp_growth'],
                    bins=[-float('inf'), 5.0, 7.0, float('inf')],
                    labels=[0, 1, 2]  # Slow, Moderate, Fast
                ).astype(int)
            
            # Add inflation rate
            if "US_CPI" in economic_data:
                df_features['us_inflation'] = economic_data["US_CPI"]
                
                # Inflation momentum
                df_features['us_inflation_momentum'] = economic_data.get("US_CPI_MOMENTUM", 0)
                
                # Inflation regime
                df_features['us_inflation_regime'] = pd.cut(
                    df_features['us_inflation'],
                    bins=[-float('inf'), 2.0, 4.0, float('inf')],
                    labels=[0, 1, 2]  # Low, Target, High
                ).astype(int)
            
            if "INDIA_CPI" in economic_data:
                df_features['india_inflation'] = economic_data["INDIA_CPI"]
                
                # Inflation momentum
                df_features['india_inflation_momentum'] = economic_data.get("INDIA_CPI_MOMENTUM", 0)
                
                # Inflation regime
                df_features['india_inflation_regime'] = pd.cut(
                    df_features['india_inflation'],
                    bins=[-float('inf'), 4.0, 6.0, float('inf')],
                    labels=[0, 1, 2]  # Low, Target, High
                ).astype(int)
            
            # Add unemployment rate
            if "US_UNEMPLOYMENT" in economic_data:
                df_features['us_unemployment'] = economic_data["US_UNEMPLOYMENT"]
                
                # Unemployment momentum
                df_features['us_unemployment_momentum'] = economic_data.get("US_UNEMPLOYMENT_MOMENTUM", 0)
                
                # Unemployment regime
                df_features['us_unemployment_regime'] = pd.cut(
                    df_features['us_unemployment'],
                    bins=[-float('inf'), 4.0, 6.0, float('inf')],
                    labels=[0, 1, 2]  # Low, Moderate, High
                ).astype(int)
            
            if "INDIA_UNEMPLOYMENT" in economic_data:
                df_features['india_unemployment'] = economic_data["INDIA_UNEMPLOYMENT"]
                
                # Unemployment momentum
                df_features['india_unemployment_momentum'] = economic_data.get("INDIA_UNEMPLOYMENT_MOMENTUM", 0)
            
            # Add manufacturing PMI
            if "US_PMI" in economic_data:
                df_features['us_pmi'] = economic_data["US_PMI"]
                
                # PMI momentum
                df_features['us_pmi_momentum'] = economic_data.get("US_PMI_MOMENTUM", 0)
                
                # PMI regime (expansion/contraction)
                df_features['us_pmi_regime'] = (df_features['us_pmi'] > 50).astype(int)
            
            if "INDIA_PMI" in economic_data:
                df_features['india_pmi'] = economic_data["INDIA_PMI"]
                
                # PMI momentum
                df_features['india_pmi_momentum'] = economic_data.get("INDIA_PMI_MOMENTUM", 0)
                
                # PMI regime (expansion/contraction)
                df_features['india_pmi_regime'] = (df_features['india_pmi'] > 50).astype(int)
            
            # Add retail sales growth
            if "US_RETAIL_SALES" in economic_data:
                df_features['us_retail_sales_growth'] = economic_data["US_RETAIL_SALES"]
                
                # Retail sales momentum
                df_features['us_retail_sales_momentum'] = economic_data.get("US_RETAIL_SALES_MOMENTUM", 0)
            
            if "INDIA_RETAIL_SALES" in economic_data:
                df_features['india_retail_sales_growth'] = economic_data["INDIA_RETAIL_SALES"]
                
                # Retail sales momentum
                df_features['india_retail_sales_momentum'] = economic_data.get("INDIA_RETAIL_SALES_MOMENTUM", 0)
            
            # Add consumer sentiment
            if "US_CONSUMER_SENTIMENT" in economic_data:
                df_features['us_consumer_sentiment'] = economic_data["US_CONSUMER_SENTIMENT"]
                
                # Consumer sentiment momentum
                df_features['us_consumer_sentiment_momentum'] = economic_data.get("US_CONSUMER_SENTIMENT_MOMENTUM", 0)
            
            if "INDIA_CONSUMER_SENTIMENT" in economic_data:
                df_features['india_consumer_sentiment'] = economic_data["INDIA_CONSUMER_SENTIMENT"]
                
                # Consumer sentiment momentum
                df_features['india_consumer_sentiment_momentum'] = economic_data.get("INDIA_CONSUMER_SENTIMENT_MOMENTUM", 0)
            
            # Add composite economic indicator
            if "US_ECONOMIC_SURPRISE" in economic_data:
                df_features['us_economic_surprise'] = economic_data["US_ECONOMIC_SURPRISE"]
                
                # Economic surprise regime
                df_features['us_economic_surprise_regime'] = pd.cut(
                    df_features['us_economic_surprise'],
                    bins=[-float('inf'), -0.5, 0.5, float('inf')],
                    labels=[0, 1, 2]  # Negative, Neutral, Positive
                ).astype(int)
            
            if "INDIA_ECONOMIC_SURPRISE" in economic_data:
                df_features['india_economic_surprise'] = economic_data["INDIA_ECONOMIC_SURPRISE"]
                
                # Economic surprise regime
                df_features['india_economic_surprise_regime'] = pd.cut(
                    df_features['india_economic_surprise'],
                    bins=[-float('inf'), -0.5, 0.5, float('inf')],
                    labels=[0, 1, 2]  # Negative, Neutral, Positive
                ).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error adding economic features: {str(e)}")
    
    def _get_economic_data(self, for_date=None):
        """Get economic indicator data"""
        try:
            # Base query for economic data
            query = {
                "data_type": "economic_indicators"
            }
            
            # Add date filter if specified
            if for_date:
                query["timestamp"] = {"$lte": for_date}
            
            # Get latest economic data
            economic_data = self.db.economic_data_collection.find_one(
                query,
                sort=[("timestamp", -1)]
            )
            
            return economic_data
            
        except Exception as e:
            self.logger.error(f"Error getting economic data: {str(e)}")
            return None
    
    def _add_seasonal_features(self, df_features):
        """Add seasonal indicator features"""
        try:
            # Ensure index is datetime
            if not isinstance(df_features.index, pd.DatetimeIndex):
                return
            
            # Extract date components
            df_features['month'] = df_features.index.month
            df_features['day_of_month'] = df_features.index.day
            df_features['day_of_week'] = df_features.index.dayofweek
            df_features['week_of_year'] = df_features.index.isocalendar().week
            df_features['quarter'] = df_features.index.quarter
            
            # Create month-end indicator
            df_features['month_end'] = df_features.index.is_month_end.astype(int)
            
            # Create quarter-end indicator
            df_features['quarter_end'] = df_features.index.is_quarter_end.astype(int)
            
            # Create year-end indicator
            df_features['year_end'] = df_features.index.is_year_end.astype(int)
            
            # Create day-of-week indicators
            for day in range(5):  # 0=Monday, 4=Friday
                df_features[f'day_{day}'] = (df_features['day_of_week'] == day).astype(int)
            
            # Create month indicators
            for month in range(1, 13):
                df_features[f'month_{month}'] = (df_features['month'] == month).astype(int)
            
            # Create quarter indicators
            for quarter in range(1, 5):
                df_features[f'quarter_{quarter}'] = (df_features['quarter'] == quarter).astype(int)
            
            # Add cyclic encoding for month and day of week
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
            
            df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
            
            # Add special period indicators
            
            # Tax season
            df_features['tax_season'] = ((df_features['month'] == 3) & (df_features['day_of_month'] >= 15)) | \
                                       (df_features['month'] == 4)
            
            # Budget month (for India)
            df_features['budget_month'] = (df_features['month'] == 2)
            
            # Earnings seasons (approximate)
            df_features['earnings_season'] = ((df_features['month'] == 1) & (df_features['day_of_month'] >= 15)) | \
                                           ((df_features['month'] == 2) & (df_features['day_of_month'] <= 15)) | \
                                           ((df_features['month'] == 4) & (df_features['day_of_month'] >= 15)) | \
                                           ((df_features['month'] == 5) & (df_features['day_of_month'] <= 15)) | \
                                           ((df_features['month'] == 7) & (df_features['day_of_month'] >= 15)) | \
                                           ((df_features['month'] == 8) & (df_features['day_of_month'] <= 15)) | \
                                           ((df_features['month'] == 10) & (df_features['day_of_month'] >= 15)) | \
                                           ((df_features['month'] == 11) & (df_features['day_of_month'] <= 15))
            
            # Convert boolean to int
            df_features['tax_season'] = df_features['tax_season'].astype(int)
            df_features['budget_month'] = df_features['budget_month'].astype(int)
            df_features['earnings_season'] = df_features['earnings_season'].astype(int)
            
            # Drop intermediate columns
            df_features = df_features.drop(columns=['month', 'day_of_month', 'day_of_week', 'week_of_year', 'quarter'])
            
        except Exception as e:
            self.logger.error(f"Error adding seasonal features: {str(e)}")
    
    def _add_market_breadth_features(self, df_features, market_data):
        """Add market breadth indicator features"""
        try:
            # Check if breadth indicators are available
            breadth_cols = [col for col in market_data.columns if col.endswith('_BREADTH')]
            
            if not breadth_cols:
                return
            
            # Add breadth indicators
            for col in breadth_cols:
                # Extract market name
                market_name = col.split('_')[0]
                
                # Absolute value
                df_features[f'{market_name}_breadth'] = market_data[col]
                
                # Change over different periods
                for period in [1, 5, 10]:
                    df_features[f'{market_name}_breadth_change_{period}d'] = market_data[col].diff(period)
                
                # Z-score
                mean_20d = market_data[col].rolling(window=20).mean()
                std_20d = market_data[col].rolling(window=20).std()
                df_features[f'{market_name}_breadth_zscore'] = (market_data[col] - mean_20d) / std_20d
                
                # Breadth regime
                df_features[f'{market_name}_breadth_regime'] = pd.cut(
                    market_data[col],
                    bins=[-1, -0.5, 0.5, 1],
                    labels=[0, 1, 2]  # Negative, Neutral, Positive
                ).astype(int)
            
            # Add specific breadth components (if available)
            
            # Advance-decline ratio
            if "ADV_DECL_RATIO" in market_data.columns:
                df_features['adv_decl_ratio'] = market_data["ADV_DECL_RATIO"]
                
                # Regime based on ratio
                df_features['adv_decl_regime'] = pd.cut(
                    df_features['adv_decl_ratio'],
                    bins=[0, 0.7, 1.3, float('inf')],
                    labels=[0, 1, 2]  # Negative, Neutral, Positive
                ).astype(int)
            
            # New highs minus new lows
            if "NEW_HIGHS_MINUS_LOWS" in market_data.columns:
                df_features['new_highs_minus_lows'] = market_data["NEW_HIGHS_MINUS_LOWS"]
                
                # Z-score of new highs minus lows
                mean_20d = market_data["NEW_HIGHS_MINUS_LOWS"].rolling(window=20).mean()
                std_20d = market_data["NEW_HIGHS_MINUS_LOWS"].rolling(window=20).std()
                df_features['new_highs_minus_lows_zscore'] = (market_data["NEW_HIGHS_MINUS_LOWS"] - mean_20d) / std_20d
            
            # Percentage of stocks above moving averages
            for ma_period in [50, 200]:
                col_name = f"PCT_ABOVE_MA{ma_period}"
                if col_name in market_data.columns:
                    df_features[f'pct_above_ma{ma_period}'] = market_data[col_name]
                    
                    # Change over different periods
                    for period in [1, 5, 10]:
                        df_features[f'pct_above_ma{ma_period}_change_{period}d'] = market_data[col_name].diff(period)
                    
                    # Regime based on percentage
                    df_features[f'pct_above_ma{ma_period}_regime'] = pd.cut(
                        market_data[col_name],
                        bins=[0, 0.2, 0.8, 1],
                        labels=[0, 1, 2]  # Weak, Neutral, Strong
                    ).astype(int)
            
            # McClellan Oscillator (if available)
            if "MCCLELLAN_OSC" in market_data.columns:
                df_features['mcclellan_osc'] = market_data["MCCLELLAN_OSC"]
                
                # Regime based on oscillator
                df_features['mcclellan_regime'] = pd.cut(
                    df_features['mcclellan_osc'],
                    bins=[-100, -20, 20, 100],
                    labels=[0, 1, 2]  # Oversold, Neutral, Overbought
                ).astype(int)
            
            # Bullish Percent Index (if available)
            if "BULLISH_PERCENT" in market_data.columns:
                df_features['bullish_percent'] = market_data["BULLISH_PERCENT"]
                
                # Regime based on percentage
                df_features['bullish_percent_regime'] = pd.cut(
                    df_features['bullish_percent'],
                    bins=[0, 0.3, 0.7, 1],
                    labels=[0, 1, 2]  # Bearish, Neutral, Bullish
                ).astype(int)
                
                # Change over different periods
                for period in [1, 5, 10]:
                    df_features[f'bullish_percent_change_{period}d'] = market_data["BULLISH_PERCENT"].diff(period)
            
        except Exception as e:
            self.logger.error(f"Error adding market breadth features: {str(e)}")
    
    def _add_market_regime_features(self, df_features, market_data):
        """Add market regime and momentum features"""
        try:
            # Check if major indices are available
            major_indices = [idx for idx in ["SPX", "NIFTY50"] if idx in market_data.columns]
            
            if not major_indices:
                return
            
            # Add momentum features for major indices
            for index in major_indices:
                # Momentum over different timeframes
                for period in [10, 20, 50, 200]:
                    df_features[f'{index}_momentum_{period}d'] = market_data[index].pct_change(period)
                
                # Moving average crosses
                for short_period, long_period in [(5, 20), (20, 50), (50, 200)]:
                    short_ma = market_data[index].rolling(window=short_period).mean()
                    long_ma = market_data[index].rolling(window=long_period).mean()
                    
                    # MA cross indicator (1 if short > long, 0 if short < long)
                    df_features[f'{index}_ma{short_period}_{long_period}_cross'] = (short_ma > long_ma).astype(int)
                    
                    # Distance between MAs (normalized by price)
                    df_features[f'{index}_ma{short_period}_{long_period}_diff'] = (short_ma - long_ma) / market_data[index]
                
                # Trend strength
                if f"{index}_TRENDSTRENGTH" in market_data.columns:
                    df_features[f'{index}_trend_strength'] = market_data[f"{index}_TRENDSTRENGTH"]
                else:
                    # Calculate simple trend strength based on directional movement
                    returns = market_data[index].pct_change()
                    up_days = (returns > 0).rolling(window=20).sum()
                    down_days = (returns < 0).rolling(window=20).sum()
                    
                    # Normalized trend strength (-1 to +1)
                    df_features[f'{index}_trend_strength'] = (up_days - down_days) / 20
                
                # Market regime based on trend and volatility
                if "VIX" in market_data.columns and index == "SPX":
                    # Trend component (based on 50-day momentum)
                    trend = market_data[index].pct_change(50)
                    trend_z = (trend - trend.rolling(252).mean()) / trend.rolling(252).std()
                    
                    # Volatility component
                    vol_z = (market_data["VIX"] - market_data["VIX"].rolling(252).mean()) / market_data["VIX"].rolling(252).std()
                    
                    # Regime calculation
                    # High trend, low vol = Bullish trend
                    # High trend, high vol = Bullish volatile
                    # Low trend, low vol = Bearish quiet
                    # Low trend, high vol = Bearish volatile
                    
                    df_features['spx_trend_component'] = pd.cut(
                        trend_z,
                        bins=[-float('inf'), -0.5, 0.5, float('inf')],
                        labels=[-1, 0, 1]  # Bearish, Neutral, Bullish
                    ).astype(int)
                    
                    df_features['vix_component'] = pd.cut(
                        vol_z,
                        bins=[-float('inf'), -0.5, 0.5, float('inf')],
                        labels=[-1, 0, 1]  # Low Vol, Normal Vol, High Vol
                    ).astype(int)
                    
                    # Combined regime
                    # Create a mapping from (trend, vol) pairs to regime
                    regime_map = {
                        (1, -1): 4,    # Strong Bull (high trend, low vol)
                        (1, 0): 3,     # Bull (high trend, normal vol)
                        (1, 1): 2,     # Volatile Bull (high trend, high vol)
                        (0, -1): 3,    # Quiet Neutral (neutral trend, low vol)
                        (0, 0): 2,     # Neutral (neutral trend, normal vol)
                        (0, 1): 1,     # Volatile Neutral (neutral trend, high vol)
                        (-1, -1): 2,   # Quiet Bear (low trend, low vol)
                        (-1, 0): 1,    # Bear (low trend, normal vol)
                        (-1, 1): 0     # Crisis (low trend, high vol)
                    }
                    
                    # Apply the mapping
                    df_features['market_regime'] = df_features.apply(
                        lambda row: regime_map.get((row['spx_trend_component'], row['vix_component']), 2),
                        axis=1
                    )
                
                elif "INDIA_VIX" in market_data.columns and index == "NIFTY50":
                    # Similar regime calculation for Indian market
                    trend = market_data[index].pct_change(50)
                    trend_z = (trend - trend.rolling(252).mean()) / trend.rolling(252).std()
                    
                    vol_z = (market_data["INDIA_VIX"] - market_data["INDIA_VIX"].rolling(252).mean()) / market_data["INDIA_VIX"].rolling(252).std()
                    
                    df_features['nifty_trend_component'] = pd.cut(
                        trend_z,
                        bins=[-float('inf'), -0.5, 0.5, float('inf')],
                        labels=[-1, 0, 1]
                    ).astype(int)
                    
                    df_features['india_vix_component'] = pd.cut(
                        vol_z,
                        bins=[-float('inf'), -0.5, 0.5, float('inf')],
                        labels=[-1, 0, 1]
                    ).astype(int)
                    
                    # Same regime mapping as above
                    regime_map = {
                        (1, -1): 4,    # Strong Bull
                        (1, 0): 3,     # Bull
                        (1, 1): 2,     # Volatile Bull
                        (0, -1): 3,    # Quiet Neutral
                        (0, 0): 2,     # Neutral
                        (0, 1): 1,     # Volatile Neutral
                        (-1, -1): 2,   # Quiet Bear
                        (-1, 0): 1,    # Bear
                        (-1, 1): 0     # Crisis
                    }
                    
                    df_features['india_market_regime'] = df_features.apply(
                        lambda row: regime_map.get((row['nifty_trend_component'], row['india_vix_component']), 2),
                        axis=1
                    )
            
            # Add risk appetite indicators
            
            # Credit spread as risk appetite measure
            if "US_HIGH_YIELD_SPREAD" in market_data.columns:
                df_features['credit_spread'] = market_data["US_HIGH_YIELD_SPREAD"]
                
                # Z-score of credit spread
                mean_60d = market_data["US_HIGH_YIELD_SPREAD"].rolling(window=60).mean()
                std_60d = market_data["US_HIGH_YIELD_SPREAD"].rolling(window=60).std()
                df_features['credit_spread_zscore'] = (market_data["US_HIGH_YIELD_SPREAD"] - mean_60d) / std_60d
                
                # Risk appetite based on credit spread
                df_features['credit_risk_appetite'] = pd.cut(
                    df_features['credit_spread_zscore'],
                    bins=[-float('inf'), -0.5, 0.5, float('inf')],
                    labels=[2, 1, 0]  # High, Neutral, Low risk appetite
                ).astype(int)
            
            # Growth vs. Value performance
            if "GROWTH_INDEX" in market_data.columns and "VALUE_INDEX" in market_data.columns:
                # 50-day relative performance
                growth_returns = market_data["GROWTH_INDEX"].pct_change(50)
                value_returns = market_data["VALUE_INDEX"].pct_change(50)
                
                df_features['growth_vs_value'] = growth_returns - value_returns
                
                # Risk appetite based on Growth vs. Value
                df_features['growth_value_risk_appetite'] = pd.cut(
                    df_features['growth_vs_value'],
                    bins=[-float('inf'), -0.03, 0.03, float('inf')],
                    labels=[0, 1, 2]  # Value leading (risk-off), Neutral, Growth leading (risk-on)
                ).astype(int)
            
            # Small cap vs. Large cap performance
            if "SMALL_CAP_INDEX" in market_data.columns and "LARGE_CAP_INDEX" in market_data.columns:
                # 50-day relative performance
                small_returns = market_data["SMALL_CAP_INDEX"].pct_change(50)
                large_returns = market_data["LARGE_CAP_INDEX"].pct_change(50)
                
                df_features['small_vs_large'] = small_returns - large_returns
                
                # Risk appetite based on Small vs. Large
                df_features['cap_risk_appetite'] = pd.cut(
                    df_features['small_vs_large'],
                    bins=[-float('inf'), -0.03, 0.03, float('inf')],
                    labels=[0, 1, 2]  # Large leading (risk-off), Neutral, Small leading (risk-on)
                ).astype(int)
            
            # Combined risk appetite indicator
            risk_appetite_cols = [col for col in df_features.columns if 'risk_appetite' in col]
            if len(risk_appetite_cols) >= 2:
                df_features['combined_risk_appetite'] = df_features[risk_appetite_cols].mean(axis=1)
                
                # Overall risk regime
                df_features['risk_regime'] = pd.cut(
                    df_features['combined_risk_appetite'],
                    bins=[-float('inf'), 0.7, 1.3, float('inf')],
                    labels=[0, 1, 2]  # Risk-off, Neutral, Risk-on
                ).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error adding market regime features: {str(e)}")
    
    def _add_target_variables(self, df_features, symbol, exchange, timeframe, for_date=None):
        """Add target variables for prediction"""
        try:
            # Get future returns data
            future_returns = self._get_future_returns(symbol, exchange, timeframe, for_date)
            if future_returns is None:
                return
            
            # Add future returns for different horizons
            for horizon in [1, 3, 5, 10, 20]:
                if f"{horizon}d" in future_returns:
                    # Return target
                    df_features[f'target_return_{horizon}d'] = future_returns[f"{horizon}d"]
                    
                    # Direction target (binary)
                    df_features[f'target_direction_{horizon}d'] = (future_returns[f"{horizon}d"] > 0).astype(int)
                    
                    # Significant move target
                    threshold = 0.01  # 1% threshold
                    df_features[f'target_significant_{horizon}d'] = pd.cut(
                        future_returns[f"{horizon}d"],
                        bins=[-float('inf'), -threshold, threshold, float('inf')],
                        labels=[-1, 0, 1]  # Down, Flat, Up
                    ).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error adding target variables: {str(e)}")
    
    def _get_future_returns(self, symbol, exchange, timeframe, for_date=None):
        """Get future returns for target symbol"""
        try:
            # Query constraints
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }
            
            # Add date constraint if specified
            if for_date:
                query["timestamp"] = {"$lte": for_date}
            
            # Get historical price data
            data = list(self.db.market_data_collection.find(
                query,
                {"timestamp": 1, "close": 1}
            ).sort("timestamp", 1))
            
            if not data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df = df.set_index("timestamp")
            
            # Calculate future returns for different horizons
            future_returns = {}
            
            for horizon in [1, 3, 5, 10, 20]:
                # Calculate future return
                df[f"{horizon}d"] = df["close"].shift(-horizon) / df["close"] - 1
            
            # Extract future returns
            future_returns = df[[f"{horizon}d" for horizon in [1, 3, 5, 10, 20]]]
            
            return future_returns
            
        except Exception as e:
            self.logger.error(f"Error getting future returns: {str(e)}")
            return None
    
    def get_feature_importance(self, symbol, exchange, target='target_return_5d', timeframe="day"):
        """
        Calculate feature importance for global features
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange (e.g., NSE)
        - target: Target variable to predict
        - timeframe: Data timeframe

        Returns:
        - DataFrame with feature importances
        """
        try:
            # Generate features with target variable
            df = self.generate_features(symbol, exchange, timeframe=timeframe, include_target=True)
            if df is None or len(df) < 30:
                return None
            
            # Check if target exists
            if target not in df.columns:
                self.logger.error(f"Target {target} not found in dataframe")
                return None
            
            # Get feature columns
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            
            # Drop rows with missing values in target
            df = df.dropna(subset=[target])
            
            # Drop rows with any missing values in features
            df = df.dropna(subset=feature_cols)
            
            if len(df) < 30:
                return None
            
            # Calculate feature importance using Random Forest
            try:
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
                
                # Prepare data
                X = df[feature_cols].values
                y = df[target].values
                
                # Check if classification or regression
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
                
                # Get feature importance
                rf_importance = model.feature_importances_
                
                # Calculate mutual information
                mi_importance = mutual_info_func(X, y, random_state=42)
                
                # Create importance dataframe
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
                    