# sentiment_features.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class SentimentFeatureGenerator:
    def __init__(self, db_connector):
        """Initialize the sentiment feature generator"""
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        
        # Initialize scalers
        self.score_scaler = StandardScaler()
        self.volume_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Initialize vectorizers
        self.count_vectorizer = CountVectorizer(max_features=100, stop_words='english')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    def generate_features(self, symbol=None, exchange=None, timeframe="day", 
                        lookback=30, for_date=None, include_target=False):
        """
        Generate features from sentiment and news data
        
        Parameters:
        - symbol: Symbol for target-specific sentiment (optional)
        - exchange: Exchange for the symbol (optional)
        - timeframe: Data timeframe
        - lookback: Number of historical periods to include
        - for_date: Optional specific date to generate features for
        - include_target: Whether to include target variables
        
        Returns:
        - DataFrame with sentiment features
        """
        try:
            # Get news and sentiment data
            sentiment_data = self._get_sentiment_data(symbol, exchange, timeframe, lookback, for_date)
            if sentiment_data is None or len(sentiment_data) < 5:
                return None
            
            # Create a new DataFrame for features
            df_features = pd.DataFrame(index=sentiment_data.index)
            
            # Add sentiment score features
            self._add_sentiment_score_features(df_features, sentiment_data)
            
            # Add sentiment volume features
            self._add_sentiment_volume_features(df_features, sentiment_data)
            
            # Add sentiment topic features
            self._add_sentiment_topic_features(df_features, sentiment_data)
            
            # Add market-wide sentiment features
            self._add_market_sentiment_features(df_features, sentiment_data)
            
            # Add target-specific sentiment features if a symbol is provided
            if symbol and exchange:
                self._add_target_sentiment_features(df_features, sentiment_data, symbol, exchange)
            
            # Add sentiment contrast features
            self._add_sentiment_contrast_features(df_features, sentiment_data)
            
            # Add social media specific features
            self._add_social_media_features(df_features, sentiment_data)
            
            # Add news-specific features
            self._add_news_features(df_features, sentiment_data)
            
            # Add text features (word frequencies)
            self._add_text_features(df_features, sentiment_data)
            
            # Add sentiment divergence features
            self._add_sentiment_divergence_features(df_features, sentiment_data)
            
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
            self.logger.error(f"Error generating sentiment features: {str(e)}")
            return None
    
    def _get_sentiment_data(self, symbol, exchange, timeframe, lookback, for_date=None):
        """Get sentiment and news data from database"""
        try:
            # Base query for sentiment data
            if symbol and exchange:
                # Get sentiment data for specific symbol
                query = {
                    "data_type": "sentiment",
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe
                }
            else:
                # Get market-wide sentiment data
                query = {
                    "data_type": "sentiment",
                    "sentiment_type": "market",
                    "timeframe": timeframe
                }
            
            # Add date filter if specified
            if for_date:
                query["timestamp"] = {"$lte": for_date}
            
            # Get data from database
            data = list(self.db.sentiment_collection.find(
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
            self.logger.error(f"Error getting sentiment data: {str(e)}")
            return None
    
    def _add_sentiment_score_features(self, df_features, sentiment_data):
        """Add features based on sentiment scores"""
        try:
            # Check if sentiment scores exist
            if "sentiment_score" not in sentiment_data.columns:
                return
            
            # Basic sentiment score
            df_features['sentiment_score'] = sentiment_data["sentiment_score"]
            
            # Rolling statistics for sentiment scores
            for window in [3, 5, 10]:
                # Rolling mean
                df_features[f'sentiment_ma_{window}'] = sentiment_data["sentiment_score"].rolling(window=window).mean()
                
                # Rolling standard deviation (volatility)
                df_features[f'sentiment_std_{window}'] = sentiment_data["sentiment_score"].rolling(window=window).std()
                
                # Exponential moving average
                df_features[f'sentiment_ema_{window}'] = sentiment_data["sentiment_score"].ewm(span=window).mean()
            
            # Sentiment momentum (change over time)
            for period in [1, 3, 5]:
                df_features[f'sentiment_change_{period}d'] = sentiment_data["sentiment_score"].diff(period)
            
            # Z-score (normalized sentiment)
            rolling_mean = sentiment_data["sentiment_score"].rolling(window=20).mean()
            rolling_std = sentiment_data["sentiment_score"].rolling(window=20).std()
            df_features['sentiment_zscore'] = (sentiment_data["sentiment_score"] - rolling_mean) / rolling_std
            
            # Sentiment regime
            df_features['sentiment_regime'] = pd.cut(
                df_features['sentiment_zscore'],
                bins=[-float('inf'), -1.0, 1.0, float('inf')],
                labels=[0, 1, 2]  # Negative, Neutral, Positive
            ).astype(int)
            
            # Extreme sentiment indicators
            df_features['extreme_positive'] = (df_features['sentiment_zscore'] > 2).astype(int)
            df_features['extreme_negative'] = (df_features['sentiment_zscore'] < -2).astype(int)
            
            # Sentiment reversal signals
            df_features['sentiment_reversal_up'] = ((df_features['sentiment_change_3d'] > 0) & 
                                                 (df_features['sentiment_zscore'] < -1)).astype(int)
            df_features['sentiment_reversal_down'] = ((df_features['sentiment_change_3d'] < 0) & 
                                                   (df_features['sentiment_zscore'] > 1)).astype(int)
            
            # Check if source-specific sentiment scores exist
            source_cols = [col for col in sentiment_data.columns if col.endswith('_sentiment')]
            
            # Add source-specific sentiment features
            for col in source_cols:
                source_name = col.replace('_sentiment', '')
                
                # Basic sentiment score
                df_features[f'{source_name}_sentiment'] = sentiment_data[col]
                
                # 5-day moving average
                df_features[f'{source_name}_sentiment_ma_5'] = sentiment_data[col].rolling(window=5).mean()
                
                # Sentiment change
                df_features[f'{source_name}_sentiment_change'] = sentiment_data[col].diff(3)
            
            # Calculate sentiment disagreement across sources
            if len(source_cols) >= 2:
                # Extract the scores for each period
                source_scores = sentiment_data[source_cols]
                
                # Calculate standard deviation across sources
                df_features['sentiment_disagreement'] = source_scores.std(axis=1)
                
                # Calculate maximum difference between sources
                df_features['sentiment_max_divergence'] = source_scores.max(axis=1) - source_scores.min(axis=1)
            
        except Exception as e:
            self.logger.error(f"Error adding sentiment score features: {str(e)}")
    
    def _add_sentiment_volume_features(self, df_features, sentiment_data):
        """Add features based on sentiment volume/activity"""
        try:
            # Check if sentiment volume exists
            volume_cols = [col for col in sentiment_data.columns if 'volume' in col or 'count' in col or 'activity' in col]
            
            if not volume_cols:
                return
            
            # Total sentiment volume across all sources
            if 'total_volume' in sentiment_data.columns:
                df_features['sentiment_volume'] = sentiment_data['total_volume']
            elif len(volume_cols) > 0:
                # Sum up all volume columns
                df_features['sentiment_volume'] = sentiment_data[volume_cols].sum(axis=1)
            
            # Rolling statistics for sentiment volume
            for window in [3, 5, 10]:
                # Rolling mean
                df_features[f'sentiment_volume_ma_{window}'] = df_features['sentiment_volume'].rolling(window=window).mean()
                
                # Rolling standard deviation
                df_features[f'sentiment_volume_std_{window}'] = df_features['sentiment_volume'].rolling(window=window).std()
            
            # Volume momentum (change over time)
            for period in [1, 3, 5]:
                df_features[f'sentiment_volume_change_{period}d'] = df_features['sentiment_volume'].pct_change(period)
            
            # Z-score for volume
            rolling_mean = df_features['sentiment_volume'].rolling(window=20).mean()
            rolling_std = df_features['sentiment_volume'].rolling(window=20).std()
            df_features['sentiment_volume_zscore'] = (df_features['sentiment_volume'] - rolling_mean) / rolling_std
            
            # Volume regime
            df_features['sentiment_volume_regime'] = pd.cut(
                df_features['sentiment_volume_zscore'],
                bins=[-float('inf'), -1.0, 1.0, float('inf')],
                labels=[0, 1, 2]  # Low, Normal, High
            ).astype(int)
            
            # Abnormal volume indicator
            df_features['abnormal_volume'] = (abs(df_features['sentiment_volume_zscore']) > 2).astype(int)
            
            # Add source-specific volume features
            for col in volume_cols:
                if col != 'total_volume':
                    source_name = col.replace('_volume', '').replace('_count', '').replace('_activity', '')
                    
                    # Basic volume
                    df_features[f'{source_name}_volume'] = sentiment_data[col]
                    
                    # 5-day moving average
                    df_features[f'{source_name}_volume_ma_5'] = sentiment_data[col].rolling(window=5).mean()
                    
                    # Volume change
                    df_features[f'{source_name}_volume_change'] = sentiment_data[col].pct_change(3)
                    
                    # Relative volume (compared to total)
                    if 'sentiment_volume' in df_features.columns and df_features['sentiment_volume'].mean() > 0:
                        df_features[f'{source_name}_rel_volume'] = sentiment_data[col] / df_features['sentiment_volume']
            
            # Positive and negative sentiment volumes (if available)
            if 'positive_volume' in sentiment_data.columns and 'negative_volume' in sentiment_data.columns:
                # Positive-to-negative ratio
                df_features['pos_neg_ratio'] = sentiment_data['positive_volume'] / sentiment_data['negative_volume']
                
                # Bullish sentiment percentage
                total = sentiment_data['positive_volume'] + sentiment_data['negative_volume']
                df_features['bullish_percent'] = sentiment_data['positive_volume'] / total
                
                # Bullish-bearish spread
                df_features['bull_bear_spread'] = (sentiment_data['positive_volume'] - sentiment_data['negative_volume']) / total
                
                # Bull-bear spread z-score
                rolling_mean = df_features['bull_bear_spread'].rolling(window=20).mean()
                rolling_std = df_features['bull_bear_spread'].rolling(window=20).std()
                df_features['bull_bear_zscore'] = (df_features['bull_bear_spread'] - rolling_mean) / rolling_std
                
                # Sentiment trend
                df_features['sentiment_trend'] = df_features['bull_bear_spread'].diff(5)
            
            # Normalize volume features
            volume_feat_cols = [col for col in df_features.columns if 'volume' in col and 'regime' not in col and 'zscore' not in col]
            if volume_feat_cols:
                df_features[volume_feat_cols] = self.volume_scaler.fit_transform(df_features[volume_feat_cols].fillna(0))
            
        except Exception as e:
            self.logger.error(f"Error adding sentiment volume features: {str(e)}")
    
    def _add_sentiment_topic_features(self, df_features, sentiment_data):
        """Add features based on sentiment topics"""
        try:
            # Check if topic columns exist
            topic_cols = [col for col in sentiment_data.columns if col.startswith('topic_')]
            
            if not topic_cols:
                return
            
            # Add features for each topic
            for col in topic_cols:
                topic_name = col.replace('topic_', '')
                
                # Topic prevalence
                df_features[f'topic_{topic_name}'] = sentiment_data[col]
                
                # 5-day moving average
                df_features[f'topic_{topic_name}_ma_5'] = sentiment_data[col].rolling(window=5).mean()
                
                # Topic change
                df_features[f'topic_{topic_name}_change'] = sentiment_data[col].diff(3)
                
                # Topic z-score
                rolling_mean = sentiment_data[col].rolling(window=20).mean()
                rolling_std = sentiment_data[col].rolling(window=20).std()
                df_features[f'topic_{topic_name}_zscore'] = (sentiment_data[col] - rolling_mean) / rolling_std
                
                # Abnormal topic activity
                df_features[f'topic_{topic_name}_abnormal'] = (abs(df_features[f'topic_{topic_name}_zscore']) > 2).astype(int)
            
            # Top topics
            if len(topic_cols) >= 3:
                # Get top 3 topics for each period
                topic_data = sentiment_data[topic_cols]
                
                # Calculate top topic
                top_topic_idx = topic_data.apply(lambda x: np.argmax(x), axis=1)
                top_topic_names = [topic_cols[i].replace('topic_', '') for i in top_topic_idx]
                df_features['top_topic'] = top_topic_names
                
                # One-hot encode top topics
                for topic in set(top_topic_names):
                    df_features[f'top_topic_{topic}'] = (df_features['top_topic'] == topic).astype(int)
            
            # Topic sentiment (if available)
            topic_sentiment_cols = [col for col in sentiment_data.columns if col.startswith('topic_') and col.endswith('_sentiment')]
            
            for col in topic_sentiment_cols:
                topic_name = col.replace('topic_', '').replace('_sentiment', '')
                
                # Topic sentiment
                df_features[f'topic_{topic_name}_sentiment'] = sentiment_data[col]
                
                # 5-day moving average
                df_features[f'topic_{topic_name}_sentiment_ma_5'] = sentiment_data[col].rolling(window=5).mean()
                
                # Sentiment change
                df_features[f'topic_{topic_name}_sentiment_change'] = sentiment_data[col].diff(3)
            
            # Drop the string column
            if 'top_topic' in df_features.columns:
                df_features = df_features.drop('top_topic', axis=1)
            
        except Exception as e:
            self.logger.error(f"Error adding sentiment topic features: {str(e)}")
    
    def _add_market_sentiment_features(self, df_features, sentiment_data):
        """Add market-wide sentiment features"""
        try:
            # Check if market sentiment columns exist
            market_cols = [col for col in sentiment_data.columns if 'market' in col]
            
            if not market_cols:
                return
            
            # Add features for market sentiment
            if 'market_sentiment' in sentiment_data.columns:
                # Basic market sentiment
                df_features['market_sentiment'] = sentiment_data['market_sentiment']
                
                # 5-day moving average
                df_features['market_sentiment_ma_5'] = sentiment_data['market_sentiment'].rolling(window=5).mean()
                
                # Market sentiment change
                df_features['market_sentiment_change'] = sentiment_data['market_sentiment'].diff(3)
                
                # Market sentiment z-score
                rolling_mean = sentiment_data['market_sentiment'].rolling(window=20).mean()
                rolling_std = sentiment_data['market_sentiment'].rolling(window=20).std()
                df_features['market_sentiment_zscore'] = (sentiment_data['market_sentiment'] - rolling_mean) / rolling_std
                
                # Market sentiment regime
                df_features['market_sentiment_regime'] = pd.cut(
                    df_features['market_sentiment_zscore'],
                    bins=[-float('inf'), -1.0, 1.0, float('inf')],
                    labels=[0, 1, 2]  # Negative, Neutral, Positive
                ).astype(int)
                
                # Extreme market sentiment indicators
                df_features['extreme_market_positive'] = (df_features['market_sentiment_zscore'] > 2).astype(int)
                df_features['extreme_market_negative'] = (df_features['market_sentiment_zscore'] < -2).astype(int)
                
                # Contrarian indicator (extreme sentiment often precedes reversals)
                df_features['contrarian_signal'] = ((df_features['market_sentiment_zscore'] > 2) | 
                                                (df_features['market_sentiment_zscore'] < -2)).astype(int)
            
            # Market sentiment by sector (if available)
            sector_sentiment_cols = [col for col in sentiment_data.columns if col.startswith('sector_') and col.endswith('_sentiment')]
            
            for col in sector_sentiment_cols:
                sector_name = col.replace('sector_', '').replace('_sentiment', '')
                
                # Sector sentiment
                df_features[f'sector_{sector_name}_sentiment'] = sentiment_data[col]
                
                # 5-day moving average
                df_features[f'sector_{sector_name}_sentiment_ma_5'] = sentiment_data[col].rolling(window=5).mean()
                
                # Sentiment change
                df_features[f'sector_{sector_name}_sentiment_change'] = sentiment_data[col].diff(3)
            
            # Sector relative sentiment (if sectors available)
            if len(sector_sentiment_cols) >= 2:
                # Get sector sentiment data
                sector_data = sentiment_data[sector_sentiment_cols]
                
                # Calculate average sector sentiment
                avg_sector_sentiment = sector_data.mean(axis=1)
                
                # Calculate relative sentiment for each sector
                for col in sector_sentiment_cols:
                    sector_name = col.replace('sector_', '').replace('_sentiment', '')
                    df_features[f'sector_{sector_name}_rel_sentiment'] = sentiment_data[col] - avg_sector_sentiment
            
            # Fear & Greed index (if available)
            if 'fear_greed_index' in sentiment_data.columns:
                # Basic Fear & Greed index
                df_features['fear_greed_index'] = sentiment_data['fear_greed_index']
                
                # 5-day moving average
                df_features['fear_greed_ma_5'] = sentiment_data['fear_greed_index'].rolling(window=5).mean()
                
                # Fear & Greed change
                df_features['fear_greed_change'] = sentiment_data['fear_greed_index'].diff(3)
                
                # Fear & Greed regime
                df_features['fear_greed_regime'] = pd.cut(
                    df_features['fear_greed_index'],
                    bins=[0, 25, 45, 55, 75, 100],
                    labels=[0, 1, 2, 3, 4]  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
                ).astype(int)
                
                # Extreme sentiment signals
                df_features['extreme_fear'] = (df_features['fear_greed_index'] < 20).astype(int)
                df_features['extreme_greed'] = (df_features['fear_greed_index'] > 80).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error adding market sentiment features: {str(e)}")
    
    def _add_target_sentiment_features(self, df_features, sentiment_data, symbol, exchange):
        """Add sentiment features specific to the target symbol"""
        try:
            # Check if symbol-specific sentiment exists
            if ('symbol' in sentiment_data.columns and 
                symbol in sentiment_data['symbol'].values and 
                'sentiment_score' in sentiment_data.columns):
                
                # Basic sentiment score
                df_features['target_sentiment'] = sentiment_data['sentiment_score']
                
                # 5-day moving average
                df_features['target_sentiment_ma_5'] = sentiment_data['sentiment_score'].rolling(window=5).mean()
                
                # Sentiment change
                df_features['target_sentiment_change'] = sentiment_data['sentiment_score'].diff(3)
                
                # Sentiment z-score
                rolling_mean = sentiment_data['sentiment_score'].rolling(window=20).mean()
                rolling_std = sentiment_data['sentiment_score'].rolling(window=20).std()
                df_features['target_sentiment_zscore'] = (sentiment_data['sentiment_score'] - rolling_mean) / rolling_std
            
            # Add sentiment relative to sector and market
            if ('target_sentiment' in df_features.columns and 
                'market_sentiment' in df_features.columns):
                
                # Relative to market
                df_features['target_vs_market'] = df_features['target_sentiment'] - df_features['market_sentiment']
                
                # Z-score of relative sentiment
                mean_diff = df_features['target_vs_market'].rolling(window=20).mean()
                std_diff = df_features['target_vs_market'].rolling(window=20).std()
                df_features['target_vs_market_zscore'] = (df_features['target_vs_market'] - mean_diff) / std_diff
            
            # Symbol vs. sector sentiment
            target_sector = self._get_target_sector(symbol, exchange)
            sector_col = f'sector_{target_sector}_sentiment'
            
            if (target_sector and 'target_sentiment' in df_features.columns and
                sector_col in df_features.columns):
                
                # Relative to sector
                df_features['target_vs_sector'] = df_features['target_sentiment'] - df_features[sector_col]
                
                # Z-score of relative sentiment
                mean_diff = df_features['target_vs_sector'].rolling(window=20).mean()
                std_diff = df_features['target_vs_sector'].rolling(window=20).std()
                df_features['target_vs_sector_zscore'] = (df_features['target_vs_sector'] - mean_diff) / std_diff
            
            # Check sentiment volume for target
            if 'sentiment_volume' in df_features.columns:
                # Target sentiment volume z-score
                mean_vol = df_features['sentiment_volume'].rolling(window=20).mean()
                std_vol = df_features['sentiment_volume'].rolling(window=20).std()
                df_features['target_volume_zscore'] = (df_features['sentiment_volume'] - mean_vol) / std_vol
                
                # Abnormal volume indicator
                df_features['target_abnormal_volume'] = (abs(df_features['target_volume_zscore']) > 2).astype(int)
            
            # Add sentiment topics related to target (if available)
            target_topic_cols = [col for col in sentiment_data.columns if col.startswith(f'topic_{symbol.lower()}_')]
            
            for col in target_topic_cols:
                topic_suffix = col.replace(f'topic_{symbol.lower()}_', '')
                
                # Topic feature
                df_features[f'target_topic_{topic_suffix}'] = sentiment_data[col]
                
                # 5-day moving average
                df_features[f'target_topic_{topic_suffix}_ma_5'] = sentiment_data[col].rolling(window=5).mean()
                
                # Change
                df_features[f'target_topic_{topic_suffix}_change'] = sentiment_data[col].diff(3)
            
        except Exception as e:
            self.logger.error(f"Error adding target sentiment features: {str(e)}")
    
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
                return metadata["sector"].lower()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting target sector: {str(e)}")
            return None
    
    def _add_sentiment_contrast_features(self, df_features, sentiment_data):
        """Add features based on contrasting sentiment from different sources"""
        try:
            # Check if multiple sentiment sources exist
            source_cols = [col for col in sentiment_data.columns if col.endswith('_sentiment') and not col.startswith('topic_')]
            
            if len(source_cols) < 2:
                return
            
            # Calculate sentiment divergence between sources
            source_pairs = []
            for i in range(len(source_cols)):
                for j in range(i+1, len(source_cols)):
                    source1 = source_cols[i].replace('_sentiment', '')
                    source2 = source_cols[j].replace('_sentiment', '')
                    
                    # Skip if same source
                    if source1 == source2:
                        continue
                    
                    # Calculate sentiment difference
                    diff_col = f'{source1}_vs_{source2}_diff'
                    df_features[diff_col] = sentiment_data[source_cols[i]] - sentiment_data[source_cols[j]]
                    
                    # Add pair to list
                    source_pairs.append((source1, source2, diff_col))
            
            # Identify contrarian signals (when sources disagree significantly)
            for source1, source2, diff_col in source_pairs:
                # Z-score of difference
                mean_diff = df_features[diff_col].rolling(window=20).mean()
                std_diff = df_features[diff_col].rolling(window=20).std()
                
                if std_diff.min() > 0:  # Avoid division by zero
                    z_col = f'{source1}_vs_{source2}_zscore'
                    df_features[z_col] = (df_features[diff_col] - mean_diff) / std_diff
                    
                    # Significant disagreement indicator
                    df_features[f'{source1}_vs_{source2}_contrast'] = (abs(df_features[z_col]) > 2).astype(int)
            
            # Calculate overall sentiment divergence
            if len(source_cols) >= 3:
                # Standard deviation across sources
                df_features['sentiment_source_divergence'] = sentiment_data[source_cols].std(axis=1)
                
                # Z-score of divergence
                mean_div = df_features['sentiment_source_divergence'].rolling(window=20).mean()
                std_div = df_features['sentiment_source_divergence'].rolling(window=20).std()
                
                if std_div.min() > 0:  # Avoid division by zero
                    df_features['sentiment_divergence_zscore'] = (df_features['sentiment_source_divergence'] - mean_div) / std_div
                    
                    # High divergence indicator
                    df_features['high_sentiment_divergence'] = (df_features['sentiment_divergence_zscore'] > 2).astype(int)
            
            # Smart money vs. retail sentiment (if available)
            if 'institutional_sentiment' in sentiment_data.columns and 'retail_sentiment' in sentiment_data.columns:
                # Calculate difference
                df_features['smart_vs_retail_diff'] = sentiment_data['institutional_sentiment'] - sentiment_data['retail_sentiment']
                
                # Z-score of difference
                mean_diff = df_features['smart_vs_retail_diff'].rolling(window=20).mean()
                std_diff = df_features['smart_vs_retail_diff'].rolling(window=20).std()
                
                if std_diff.min() > 0:  # Avoid division by zero
                    df_features['smart_vs_retail_zscore'] = (df_features['smart_vs_retail_diff'] - mean_diff) / std_diff
                    
                    # Contrarian signal (when smart and retail disagree significantly)
                    df_features['smart_retail_contrast'] = (abs(df_features['smart_vs_retail_zscore']) > 2).astype(int)
                    
                    # Smart money signal (positive when smart money is more bullish)
                    df_features['smart_money_signal'] = (df_features['smart_vs_retail_zscore'] > 1).astype(int)
                    
                    # Dumb money signal (positive when retail is more bullish)
                    df_features['dumb_money_signal'] = (df_features['smart_vs_retail_zscore'] < -1).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error adding sentiment contrast features: {str(e)}")
    
    def _add_social_media_features(self, df_features, sentiment_data):
        """Add features specific to social media sentiment"""
        try:
            # Check if social media sentiment columns exist
            social_cols = [col for col in sentiment_data.columns if col.startswith(('twitter_', 'reddit_', 'stocktwits_'))]
            
            if not social_cols:
                return
            
            # Add features for each social media platform
            platforms = set([col.split('_')[0] for col in social_cols])
            
            for platform in platforms:
                # Sentiment score
                if f'{platform}_sentiment' in sentiment_data.columns:
                    df_features[f'{platform}_sentiment'] = sentiment_data[f'{platform}_sentiment']
                    
                    # 5-day moving average
                    df_features[f'{platform}_sentiment_ma_5'] = sentiment_data[f'{platform}_sentiment'].rolling(window=5).mean()
                    
                    # Sentiment change
                    df_features[f'{platform}_sentiment_change'] = sentiment_data[f'{platform}_sentiment'].diff(3)
                
                # Message volume
                volume_col = next((col for col in social_cols if col.startswith(f'{platform}_') and 'volume' in col), None)
                if volume_col:
                    df_features[f'{platform}_volume'] = sentiment_data[volume_col]
                    
                    # 5-day moving average
                    df_features[f'{platform}_volume_ma_5'] = sentiment_data[volume_col].rolling(window=5).mean()
                    
                    # Volume change
                    df_features[f'{platform}_volume_change'] = sentiment_data[volume_col].pct_change(3)
                    
                    # Volume z-score
                    mean_vol = sentiment_data[volume_col].rolling(window=20).mean()
                    std_vol = sentiment_data[volume_col].rolling(window=20).std()
                    
                    if std_vol.min() > 0:  # Avoid division by zero
                        df_features[f'{platform}_volume_zscore'] = (sentiment_data[volume_col] - mean_vol) / std_vol
                        
                        # Abnormal volume indicator
                        df_features[f'{platform}_abnormal_volume'] = (df_features[f'{platform}_volume_zscore'] > 2).astype(int)
                
                # Message count
                count_col = next((col for col in social_cols if col.startswith(f'{platform}_') and 'count' in col), None)
                if count_col:
                    df_features[f'{platform}_count'] = sentiment_data[count_col]
                    
                    # 5-day moving average
                    df_features[f'{platform}_count_ma_5'] = sentiment_data[count_col].rolling(window=5).mean()
                    
                    # Count change
                    df_features[f'{platform}_count_change'] = sentiment_data[count_col].pct_change(3)
                
                # Bullish/bearish percentage
                if f'{platform}_bullish_percent' in sentiment_data.columns:
                    df_features[f'{platform}_bullish_percent'] = sentiment_data[f'{platform}_bullish_percent']
                    
                    # 5-day moving average
                    df_features[f'{platform}_bullish_ma_5'] = sentiment_data[f'{platform}_bullish_percent'].rolling(window=5).mean()
                    
                    # Percentage change
                    df_features[f'{platform}_bullish_change'] = sentiment_data[f'{platform}_bullish_percent'].diff(3)
                    
                    # Extreme bullishness indicator
                    df_features[f'{platform}_extreme_bullish'] = (sentiment_data[f'{platform}_bullish_percent'] > 0.75).astype(int)
                    
                    # Extreme bearishness indicator
                    df_features[f'{platform}_extreme_bearish'] = (sentiment_data[f'{platform}_bullish_percent'] < 0.25).astype(int)
                
                # Engagement metrics
                engagement_col = next((col for col in social_cols if col.startswith(f'{platform}_') and 'engagement' in col), None)
                if engagement_col:
                    df_features[f'{platform}_engagement'] = sentiment_data[engagement_col]
                    
                    # 5-day moving average
                    df_features[f'{platform}_engagement_ma_5'] = sentiment_data[engagement_col].rolling(window=5).mean()
                    
                    # Engagement change
                    df_features[f'{platform}_engagement_change'] = sentiment_data[engagement_col].pct_change(3)
                    
                    # Engagement z-score
                    mean_eng = sentiment_data[engagement_col].rolling(window=20).mean()
                    std_eng = sentiment_data[engagement_col].rolling(window=20).std()
                    
                    if std_eng.min() > 0:  # Avoid division by zero
                        df_features[f'{platform}_engagement_zscore'] = (sentiment_data[engagement_col] - mean_eng) / std_eng
                        
                        # Abnormal engagement indicator
                        df_features[f'{platform}_viral'] = (df_features[f'{platform}_engagement_zscore'] > 2).astype(int)
            
            # Add inter-platform contrasts
            if len(platforms) >= 2:
                platforms_list = list(platforms)
                
                for i in range(len(platforms_list)):
                    for j in range(i+1, len(platforms_list)):
                        platform1 = platforms_list[i]
                        platform2 = platforms_list[j]
                        
                        # Skip if sentiment not available for both
                        if f'{platform1}_sentiment' not in sentiment_data.columns or f'{platform2}_sentiment' not in sentiment_data.columns:
                            continue
                        
                        # Calculate sentiment difference
                        diff_col = f'{platform1}_vs_{platform2}_diff'
                        df_features[diff_col] = sentiment_data[f'{platform1}_sentiment'] - sentiment_data[f'{platform2}_sentiment']
                        
                        # Z-score of difference
                        mean_diff = df_features[diff_col].rolling(window=20).mean()
                        std_diff = df_features[diff_col].rolling(window=20).std()
                        
                        if std_diff.min() > 0:  # Avoid division by zero
                            df_features[f'{platform1}_vs_{platform2}_zscore'] = (df_features[diff_col] - mean_diff) / std_diff
                            
                            # Significant disagreement indicator
                            df_features[f'{platform1}_vs_{platform2}_contrast'] = (abs(df_features[f'{platform1}_vs_{platform2}_zscore']) > 2).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error adding social media features: {str(e)}")
    
    def _add_news_features(self, df_features, sentiment_data):
        """Add features specific to news sentiment"""
        try:
            # Check if news sentiment columns exist
            news_cols = [col for col in sentiment_data.columns if 'news' in col]
            
            if not news_cols:
                return
            
            # Add news sentiment features
            if 'news_sentiment' in sentiment_data.columns:
                df_features['news_sentiment'] = sentiment_data['news_sentiment']
                
                # 5-day moving average
                df_features['news_sentiment_ma_5'] = sentiment_data['news_sentiment'].rolling(window=5).mean()
                
                # Sentiment change
                df_features['news_sentiment_change'] = sentiment_data['news_sentiment'].diff(3)
                
                # Sentiment z-score
                mean_sent = sentiment_data['news_sentiment'].rolling(window=20).mean()
                std_sent = sentiment_data['news_sentiment'].rolling(window=20).std()
                
                if std_sent.min() > 0:  # Avoid division by zero
                    df_features['news_sentiment_zscore'] = (sentiment_data['news_sentiment'] - mean_sent) / std_sent
                    
                    # Extreme news sentiment indicators
                    df_features['extreme_positive_news'] = (df_features['news_sentiment_zscore'] > 2).astype(int)
                    df_features['extreme_negative_news'] = (df_features['news_sentiment_zscore'] < -2).astype(int)
            
            # News volume/count
            volume_col = next((col for col in news_cols if 'volume' in col or 'count' in col), None)
            if volume_col:
                df_features['news_volume'] = sentiment_data[volume_col]
                
                # 5-day moving average
                df_features['news_volume_ma_5'] = sentiment_data[volume_col].rolling(window=5).mean()
                
                # Volume change
                df_features['news_volume_change'] = sentiment_data[volume_col].pct_change(3)
                
                # Volume z-score
                mean_vol = sentiment_data[volume_col].rolling(window=20).mean()
                std_vol = sentiment_data[volume_col].rolling(window=20).std()
                
                if std_vol.min() > 0:  # Avoid division by zero
                    df_features['news_volume_zscore'] = (sentiment_data[volume_col] - mean_vol) / std_vol
                    
                    # Abnormal volume indicator
                    df_features['news_spike'] = (df_features['news_volume_zscore'] > 2).astype(int)
            
            # News impact/relevance
            impact_col = next((col for col in news_cols if 'impact' in col or 'relevance' in col), None)
            if impact_col:
                df_features['news_impact'] = sentiment_data[impact_col]
                
                # 5-day moving average
                df_features['news_impact_ma_5'] = sentiment_data[impact_col].rolling(window=5).mean()
                
                # Impact change
                df_features['news_impact_change'] = sentiment_data[impact_col].diff(3)
                
                # Impact z-score
                mean_imp = sentiment_data[impact_col].rolling(window=20).mean()
                std_imp = sentiment_data[impact_col].rolling(window=20).std()
                
                if std_imp.min() > 0:  # Avoid division by zero
                    df_features['news_impact_zscore'] = (sentiment_data[impact_col] - mean_imp) / std_imp
                    
                    # High impact news indicator
                    df_features['high_impact_news'] = (df_features['news_impact_zscore'] > 2).astype(int)
            
            # News sources
            source_cols = [col for col in news_cols if 'source' in col and col.endswith('_sentiment')]
            
            for col in source_cols:
                source_name = col.replace('_news_sentiment', '').replace('news_source_', '')
                
                # Source sentiment
                df_features[f'{source_name}_news_sentiment'] = sentiment_data[col]
                
                # 5-day moving average
                df_features[f'{source_name}_news_sentiment_ma_5'] = sentiment_data[col].rolling(window=5).mean()
                
                # Sentiment change
                df_features[f'{source_name}_news_sentiment_change'] = sentiment_data[col].diff(3)
            
            # News topic sentiment
            topic_cols = [col for col in news_cols if 'topic' in col and col.endswith('_sentiment')]
            
            for col in topic_cols:
                topic_name = col.replace('_sentiment', '').replace('news_topic_', '')
                
                # Topic sentiment
                df_features[f'news_topic_{topic_name}_sentiment'] = sentiment_data[col]
                
                # 5-day moving average
                df_features[f'news_topic_{topic_name}_sentiment_ma_5'] = sentiment_data[col].rolling(window=5).mean()
                
                # Sentiment change
                df_features[f'news_topic_{topic_name}_sentiment_change'] = sentiment_data[col].diff(3)
            
            # Compare news sentiment with social media sentiment
            if 'news_sentiment' in sentiment_data.columns:
                # Compare with Twitter
                if 'twitter_sentiment' in sentiment_data.columns:
                    df_features['news_vs_twitter'] = sentiment_data['news_sentiment'] - sentiment_data['twitter_sentiment']
                    
                    # Z-score of difference
                    mean_diff = df_features['news_vs_twitter'].rolling(window=20).mean()
                    std_diff = df_features['news_vs_twitter'].rolling(window=20).std()
                    
                    if std_diff.min() > 0:  # Avoid division by zero
                        df_features['news_vs_twitter_zscore'] = (df_features['news_vs_twitter'] - mean_diff) / std_diff
                        
                        # Significant contrast indicator
                        df_features['news_twitter_contrast'] = (abs(df_features['news_vs_twitter_zscore']) > 2).astype(int)
                
                # Compare with Reddit
                if 'reddit_sentiment' in sentiment_data.columns:
                    df_features['news_vs_reddit'] = sentiment_data['news_sentiment'] - sentiment_data['reddit_sentiment']
                    
                    # Z-score of difference
                    mean_diff = df_features['news_vs_reddit'].rolling(window=20).mean()
                    std_diff = df_features['news_vs_reddit'].rolling(window=20).std()
                    
                    if std_diff.min() > 0:  # Avoid division by zero
                        df_features['news_vs_reddit_zscore'] = (df_features['news_vs_reddit'] - mean_diff) / std_diff
                        
                        # Significant contrast indicator
                        df_features['news_reddit_contrast'] = (abs(df_features['news_vs_reddit_zscore']) > 2).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error adding news features: {str(e)}")
    
    def _add_text_features(self, df_features, sentiment_data):
        """Add features based on text content (word frequencies)"""
        try:
            # Check if text content is available
            if 'news_headlines' not in sentiment_data.columns and 'social_texts' not in sentiment_data.columns:
                return
            
            # Combine available text
            text_data = []
            
            if 'news_headlines' in sentiment_data.columns:
                for headlines in sentiment_data['news_headlines'].dropna():
                    if isinstance(headlines, list):
                        text_data.extend(headlines)
                    elif isinstance(headlines, str):
                        text_data.append(headlines)
            
            if 'social_texts' in sentiment_data.columns:
                for texts in sentiment_data['social_texts'].dropna():
                    if isinstance(texts, list):
                        text_data.extend(texts)
                    elif isinstance(texts, str):
                        text_data.append(texts)
            
            if not text_data:
                return
            
            # Calculate word frequencies using Count Vectorizer
            # Group by date to get daily text corpus
            text_by_date = {}
            
            if 'news_headlines' in sentiment_data.columns:
                for date, headlines in zip(sentiment_data.index, sentiment_data['news_headlines']):
                    date_str = str(date.date())
                    if date_str not in text_by_date:
                        text_by_date[date_str] = []
                    
                    if isinstance(headlines, list):
                        text_by_date[date_str].extend(headlines)
                    elif isinstance(headlines, str):
                        text_by_date[date_str].append(headlines)
            
            if 'social_texts' in sentiment_data.columns:
                for date, texts in zip(sentiment_data.index, sentiment_data['social_texts']):
                    date_str = str(date.date())
                    if date_str not in text_by_date:
                        text_by_date[date_str] = []
                    
                    if isinstance(texts, list):
                        text_by_date[date_str].extend(texts)
                    elif isinstance(texts, str):
                        text_by_date[date_str].append(texts)
            
            # Create corpus
            dates = []
            corpus = []
            
            for date, texts in text_by_date.items():
                dates.append(date)
                corpus.append(' '.join(texts))
            
            if not corpus:
                return
            
            # Fit count vectorizer
            self.count_vectorizer = CountVectorizer(max_features=20, stop_words='english')
            word_counts = self.count_vectorizer.fit_transform(corpus)
            
            # Get feature names
            feature_names = self.count_vectorizer.get_feature_names_out()
            
            # Create DataFrame with word frequencies
            word_freq_df = pd.DataFrame(word_counts.toarray(), columns=feature_names, index=dates)
            
            # Map word frequencies to original dates
            for word in feature_names:
                df_features[f'word_{word}'] = np.nan
                
                for date in df_features.index:
                    date_str = str(date.date())
                    if date_str in word_freq_df.index:
                        df_features.loc[date, f'word_{word}'] = word_freq_df.loc[date_str, word]
            
            # Forward fill missing values
            word_cols = [col for col in df_features.columns if col.startswith('word_')]
            df_features[word_cols] = df_features[word_cols].fillna(method='ffill')
            
            # Calculate word frequency change
            for word in feature_names:
                df_features[f'word_{word}_change'] = df_features[f'word_{word}'].diff(3)
                
                # Z-score of word frequency
                mean_freq = df_features[f'word_{word}'].rolling(window=10).mean()
                std_freq = df_features[f'word_{word}'].rolling(window=10).std()
                
                if std_freq.min() > 0:  # Avoid division by zero
                    df_features[f'word_{word}_zscore'] = (df_features[f'word_{word}'] - mean_freq) / std_freq
                    
                    # Abnormal frequency indicator
                    df_features[f'word_{word}_spike'] = (df_features[f'word_{word}_zscore'] > 2).astype(int)
            
            # Add topic analysis based on word clusters
            # Define topic keywords
            topics = {
                'bullish': ['buy', 'bull', 'long', 'upside', 'growth', 'positive', 'up'],
                'bearish': ['sell', 'bear', 'short', 'downside', 'decline', 'negative', 'down'],
                'financial': ['earnings', 'revenue', 'profit', 'eps', 'quarter', 'guidance'],
                'economic': ['economy', 'inflation', 'fed', 'rates', 'economic', 'gdp'],
                'risk': ['risk', 'volatility', 'uncertainty', 'fear', 'crisis', 'crash']
            }
            
            # Calculate topic scores (sum of word frequencies for each topic)
            for topic, keywords in topics.items():
                # Get matching word columns
                matching_cols = []
                for word in keywords:
                    matching_cols.extend([col for col in word_cols if word in col])
                
                if matching_cols:
                    df_features[f'topic_{topic}_score'] = df_features[matching_cols].sum(axis=1)
                    
                    # Z-score of topic score
                    mean_score = df_features[f'topic_{topic}_score'].rolling(window=10).mean()
                    std_score = df_features[f'topic_{topic}_score'].rolling(window=10).std()
                    
                    if std_score.min() > 0:  # Avoid division by zero
                        df_features[f'topic_{topic}_zscore'] = (df_features[f'topic_{topic}_score'] - mean_score) / std_score
                        
                        # Topic spike indicator
                        df_features[f'topic_{topic}_spike'] = (df_features[f'topic_{topic}_zscore'] > 2).astype(int)
            
            # Calculate bullish-bearish ratio from topics
            if 'topic_bullish_score' in df_features.columns and 'topic_bearish_score' in df_features.columns:
                bull_bear_sum = df_features['topic_bullish_score'] + df_features['topic_bearish_score']
                
                if bull_bear_sum.min() > 0:  # Avoid division by zero
                    df_features['topic_bull_bear_ratio'] = df_features['topic_bullish_score'] / bull_bear_sum
                    
                    # Z-score of ratio
                    mean_ratio = df_features['topic_bull_bear_ratio'].rolling(window=10).mean()
                    std_ratio = df_features['topic_bull_bear_ratio'].rolling(window=10).std()
                    
                    if std_ratio.min() > 0:  # Avoid division by zero
                        df_features['topic_bull_bear_zscore'] = (df_features['topic_bull_bear_ratio'] - mean_ratio) / std_ratio
                        
                        # Extreme sentiment indicators
                        df_features['topic_extreme_bullish'] = (df_features['topic_bull_bear_zscore'] > 2).astype(int)
                        df_features['topic_extreme_bearish'] = (df_features['topic_bull_bear_zscore'] < -2).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error adding text features: {str(e)}")
    
    def _add_sentiment_divergence_features(self, df_features, sentiment_data):
        """Add features based on sentiment divergence from price movement"""
        try:
            # Get price data (if available)
            price_data = None
            
            if 'symbol' in sentiment_data.columns and 'exchange' in sentiment_data.columns:
                symbol = sentiment_data['symbol'].iloc[0] if 'symbol' in sentiment_data.columns else None
                exchange = sentiment_data['exchange'].iloc[0] if 'exchange' in sentiment_data.columns else None
                
                if symbol and exchange:
                    price_data = self._get_price_data(symbol, exchange, sentiment_data.index)
            
            if price_data is None:
                # Try to get market index data for general sentiment
                price_data = self._get_market_index_data(sentiment_data.index)
            
            if price_data is None:
                return
            
            # Align price data with sentiment data
            common_index = sentiment_data.index.intersection(price_data.index)
            if len(common_index) < 5:
                return
            
            sentiment_subset = sentiment_data.loc[common_index]
            price_subset = price_data.loc[common_index]
            
            # Get sentiment score (if available)
            sentiment_score = None
            
            if 'sentiment_score' in sentiment_subset.columns:
                sentiment_score = sentiment_subset['sentiment_score']
            elif 'market_sentiment' in sentiment_subset.columns:
                sentiment_score = sentiment_subset['market_sentiment']
            
            if sentiment_score is None:
                return
            
            # Calculate price returns
            price_returns = price_subset['close'].pct_change()
            
            # Calculate correlation between sentiment and returns
            for window in [5, 10, 20]:
                if len(common_index) >= window:
                    # Calculate rolling correlation
                    corr = pd.Series(
                        [
                            sentiment_score.iloc[i-window+1:i+1].corr(price_returns.iloc[i-window+1:i+1])
                            for i in range(window-1, len(common_index))
                        ],
                        index=common_index[window-1:]
                    )
                    
                    # Add to features
                    df_features[f'sentiment_price_corr_{window}d'] = np.nan
                    df_features.loc[corr.index, f'sentiment_price_corr_{window}d'] = corr
                    
                    # Forward fill to avoid gaps
                    df_features[f'sentiment_price_corr_{window}d'] = df_features[f'sentiment_price_corr_{window}d'].fillna(method='ffill')
                    
                    # Correlation regime
                    df_features[f'sentiment_price_corr_{window}d_regime'] = pd.cut(
                        df_features[f'sentiment_price_corr_{window}d'],
                        bins=[-1.01, -0.3, 0.3, 1.01],
                        labels=[0, 1, 2]  # Negative, Neutral, Positive
                    ).astype(int)
            
            # Calculate sentiment divergence from price
            # 1. Normalize both sentiment and price to z-scores
            sentiment_z = (sentiment_score - sentiment_score.rolling(window=20).mean()) / sentiment_score.rolling(window=20).std()
            price_z = (price_subset['close'] - price_subset['close'].rolling(window=20).mean()) / price_subset['close'].rolling(window=20).std()
            
            # 2. Calculate divergence
            divergence = sentiment_z - price_z
            
            # 3. Add to features
            df_features['sentiment_price_divergence'] = divergence
            
            # 4. Divergence z-score
            divergence_z = (divergence - divergence.rolling(window=10).mean()) / divergence.rolling(window=10).std()
            df_features['sentiment_price_divergence_z'] = divergence_z
            
            # 5. Significant divergence indicators
            df_features['sentiment_more_bullish'] = (divergence_z > 2).astype(int)
            df_features['sentiment_more_bearish'] = (divergence_z < -2).astype(int)
            
            # 6. Contrarian signal (extreme divergence often precedes mean reversion)
            df_features['sentiment_contrarian_signal'] = ((divergence_z > 2) | (divergence_z < -2)).astype(int)
            
            # Calculate lead-lag relationship
            # 1. Correlation with future returns
            for lead in [1, 3, 5]:
                if len(common_index) >= 20 + lead:
                    # Future returns
                    future_returns = price_returns.shift(-lead)
                    
                    # Calculate correlation with future returns
                    lead_corr = pd.Series(
                        [
                            sentiment_score.iloc[i-20+1:i+1].corr(future_returns.iloc[i-20+1:i+1])
                            for i in range(20-1, len(common_index)-lead)
                        ],
                        index=common_index[20-1:-lead]
                    )
                    
                    # Add to features
                    df_features[f'sentiment_future_corr_{lead}d'] = np.nan
                    df_features.loc[lead_corr.index, f'sentiment_future_corr_{lead}d'] = lead_corr
                    
                    # Forward fill to avoid gaps
                    df_features[f'sentiment_future_corr_{lead}d'] = df_features[f'sentiment_future_corr_{lead}d'].fillna(method='ffill')
                    
                    # Predictive power indicator
                    df_features[f'sentiment_predictive_{lead}d'] = (abs(df_features[f'sentiment_future_corr_{lead}d']) > 0.5).astype(int)
            
            # Calculate sentiment momentum divergence
            # 1. Sentiment momentum
            sentiment_momentum = sentiment_score.diff(5)
            
            # 2. Price momentum
            price_momentum = price_subset['close'].pct_change(5)
            
            # 3. Normalize both to z-scores
            sentiment_momentum_z = (sentiment_momentum - sentiment_momentum.rolling(window=20).mean()) / sentiment_momentum.rolling(window=20).std()
            price_momentum_z = (price_momentum - price_momentum.rolling(window=20).mean()) / price_momentum.rolling(window=20).std()
            
            # 4. Calculate momentum divergence
            momentum_divergence = sentiment_momentum_z - price_momentum_z
            
            # 5. Add to features
            df_features['sentiment_momentum_divergence'] = momentum_divergence
            
            # 6. Divergence z-score
            momentum_divergence_z = (momentum_divergence - momentum_divergence.rolling(window=10).mean()) / momentum_divergence.rolling(window=10).std()
            df_features['sentiment_momentum_divergence_z'] = momentum_divergence_z
            
            # 7. Significant momentum divergence indicators
            df_features['sentiment_momentum_bullish'] = (momentum_divergence_z > 2).astype(int)
            df_features['sentiment_momentum_bearish'] = (momentum_divergence_z < -2).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error adding sentiment divergence features: {str(e)}")
    
    def _get_price_data(self, symbol, exchange, dates):
        """Get price data for a symbol"""
        try:
            # Get start and end dates
            start_date = min(dates) - timedelta(days=30)  # Add buffer for calculations
            end_date = max(dates)
            
            # Query for price data
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": {"$gte": start_date, "$lte": end_date}
            }
            
            # Get price data
            data = list(self.db.market_data_collection.find(
                query,
                {"timestamp": 1, "close": 1}
            ))
            
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
            self.logger.error(f"Error getting price data: {str(e)}")
            return None
    
    def _get_market_index_data(self, dates):
        """Get market index data"""
        try:
            # Get start and end dates
            start_date = min(dates) - timedelta(days=30)  # Add buffer for calculations
            end_date = max(dates)
            
            # Try common market indices
            indices = ["NIFTY50", "SENSEX", "SPX", "DJIA"]
            
            for index in indices:
                # Query for index data
                query = {
                    "symbol": index,
                    "timestamp": {"$gte": start_date, "$lte": end_date}
                }
                
                # Get index data
                data = list(self.db.market_data_collection.find(
                    query,
                    {"timestamp": 1, "close": 1}
                ))
                
                if data:
                    # Convert to pandas DataFrame
                    df = pd.DataFrame(data)
                    
                    # Set timestamp as index
                    df = df.set_index("timestamp")
                    
                    # Sort by timestamp
                    df = df.sort_index()
                    
                    return df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting market index data: {str(e)}")
            return None
    
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
            
            # Add volatility target
            volatility_target = self._get_future_volatility(symbol, exchange, timeframe, for_date)
            if volatility_target is not None:
                df_features['target_volatility_5d'] = volatility_target
            
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
    
    def _get_future_volatility(self, symbol, exchange, timeframe, for_date=None):
        """Get future volatility for target symbol"""
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
            
            # Calculate daily returns
            df["returns"] = df["close"].pct_change()
            
            # Calculate 5-day forward volatility
            df["future_volatility"] = df["returns"].rolling(window=5).std().shift(-5)
            
            # Extract future volatility
            return df["future_volatility"]
            
        except Exception as e:
            self.logger.error(f"Error getting future volatility: {str(e)}")
            return None
    
    def get_feature_importance(self, symbol, exchange, target='target_return_5d', timeframe="day"):
        """
        Calculate feature importance for sentiment features
        
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