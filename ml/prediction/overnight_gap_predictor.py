# ml/prediction/overnight_gap_predictor.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import base64
from scipy.stats import norm

class OvernightGapPredictor:
    """Predict overnight price gaps for stocks."""
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the overnight gap predictor.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'training_days': 252,  # One year of trading days
            'test_size': 0.2,
            'use_time_series_split': True,
            'model_type': 'gradient_boosting',
            'min_samples': 60,  # Minimum samples needed for training
            'gap_threshold': 0.01,  # Threshold for significant gap (1%)
            'confidence_threshold': 0.7,  # Minimum confidence for trading signals
            'feature_groups': ['technical', 'global', 'sentiment', 'news']
        }
        
        # Model
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
    
    def set_config(self, config):
        """
        Set predictor configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated overnight gap predictor configuration: {self.config}")
    
    def get_overnight_gaps(self, symbol, exchange, days=None):
        """
        Get historical overnight price gaps.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            days (int): Number of days of data to retrieve
            
        Returns:
            DataFrame: Overnight gap data
        """
        days = days or self.config['training_days'] * 2  # Double to account for weekends/holidays
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query database
        query = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': 'day',
            'timestamp': {
                '$gte': start_date,
                '$lte': end_date
            }
        }
        
        # Sort by timestamp
        cursor = self.db.market_data_collection.find(query).sort('timestamp', 1)
        
        # Convert to DataFrame
        market_data = pd.DataFrame(list(cursor))
        
        if len(market_data) < 10:  # Minimum for gap calculation
            self.logger.error(f"Insufficient market data for {symbol} {exchange}")
            return None
        
        # Calculate overnight gaps
        gaps = []
        
        for i in range(1, len(market_data)):
            prev_day = market_data.iloc[i-1]
            curr_day = market_data.iloc[i]
            
            prev_close = prev_day.get('close')
            curr_open = curr_day.get('open')
            
            # Check if days are consecutive trading days
            prev_date = prev_day.get('timestamp')
            curr_date = curr_day.get('timestamp')
            
            date_diff = (curr_date - prev_date).days
            
            # Only consider gaps between consecutive trading days or over weekends (1-3 days)
            if 1 <= date_diff <= 3:
                gap_percent = (curr_open - prev_close) / prev_close
                
                gap_data = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'prev_date': prev_date,
                    'curr_date': curr_date,
                    'date_diff': date_diff,
                    'prev_close': prev_close,
                    'curr_open': curr_open,
                    'gap_percent': gap_percent,
                    'gap_direction': 'up' if gap_percent > 0 else 'down',
                    'is_significant': abs(gap_percent) >= self.config['gap_threshold']
                }
                
                gaps.append(gap_data)
        
        if not gaps:
            self.logger.warning(f"No overnight gaps found for {symbol} {exchange}")
            return None
            
        gap_df = pd.DataFrame(gaps)
        self.logger.info(f"Found {len(gap_df)} overnight gaps for {symbol} {exchange}")
        
        return gap_df
    
    def generate_features(self, symbol, exchange, gap_data=None):
        """
        Generate features for predicting overnight gaps.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            gap_data (DataFrame): Gap data (optional, will be generated if not provided)
            
        Returns:
            tuple: (feature DataFrame, target Series)
        """
        self.logger.info(f"Generating overnight gap features for {symbol} {exchange}")
        
        # Get gap data if not provided
        if gap_data is None:
            gap_data = self.get_overnight_gaps(symbol, exchange)
            
        if gap_data is None or len(gap_data) == 0:
            self.logger.error(f"No gap data available for {symbol} {exchange}")
            return None, None
        
        # Load feature generators
        try:
            from ml.features.technical_features import TechnicalFeatureGenerator
            from ml.features.global_features import GlobalFeatureGenerator
            from ml.features.sentiment_features import SentimentFeatureGenerator
            
            technical_features = TechnicalFeatureGenerator(self.db)
            global_features = GlobalFeatureGenerator(self.db)
            sentiment_features = SentimentFeatureGenerator(self.db)
            
        except Exception as e:
            self.logger.error(f"Error loading feature generators: {e}")
            return None, None
        
        # Create feature DataFrame
        features = pd.DataFrame(index=gap_data.index)
        
        # Add basic gap features
        features['date_diff'] = gap_data['date_diff']
        features['is_monday'] = gap_data['curr_date'].dt.dayofweek == 0
        features['is_after_holiday'] = gap_data['date_diff'] > 1
        
        # Generate technical features using previous day's data
        feature_groups = self.config['feature_groups']
        
        if 'technical' in feature_groups:
            # Get market data
            start_date = gap_data['prev_date'].min() - timedelta(days=60)  # Buffer for indicators
            end_date = gap_data['curr_date'].max()
            
            query = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': 'day',
                'timestamp': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            cursor = self.db.market_data_collection.find(query).sort('timestamp', 1)
            market_data = pd.DataFrame(list(cursor))
            
            if len(market_data) > 30:  # Minimum for technical indicators
                market_data.set_index('timestamp', inplace=True)
                
                # Generate technical features
                tech_features = technical_features.generate_all_features(
                    symbol, exchange, market_data
                )
                
                if tech_features is not None:
                    # Join on previous day's date
                    for i, row in gap_data.iterrows():
                        prev_date = row['prev_date']
                        
                        if prev_date in tech_features.index:
                            tech_row = tech_features.loc[prev_date]
                            
                            for col in tech_features.columns:
                                features.at[i, f"tech_{col}"] = tech_row[col]
        
        # Add global market features
        if 'global' in feature_groups:
            for i, row in gap_data.iterrows():
                prev_date = row['prev_date']
                
                try:
                    # Get global data for previous day
                    global_data = global_features.generate_features(
                        prev_date, prev_date + timedelta(minutes=1)
                    )
                    
                    if global_data is not None and len(global_data) > 0:
                        global_row = global_data.iloc[0]
                        
                        for col in global_data.columns:
                            features.at[i, f"global_{col}"] = global_row[col]
                            
                except Exception as e:
                    self.logger.warning(f"Error generating global features: {e}")
        
        # Add sentiment features
        if 'sentiment' in feature_groups:
            for i, row in gap_data.iterrows():
                prev_date = row['prev_date']
                
                try:
                    # Get sentiment for previous day
                    sentiment_data = sentiment_features.generate_features(
                        symbol, exchange, prev_date, prev_date + timedelta(minutes=1)
                    )
                    
                    if sentiment_data is not None and len(sentiment_data) > 0:
                        sentiment_row = sentiment_data.iloc[0]
                        
                        for col in sentiment_data.columns:
                            features.at[i, f"sentiment_{col}"] = sentiment_row[col]
                            
                except Exception as e:
                    self.logger.warning(f"Error generating sentiment features: {e}")
        
        # Add news features
        if 'news' in feature_groups:
            for i, row in gap_data.iterrows():
                prev_date = row['prev_date']
                
                try:
                    # Count recent news articles
                    news_query = {
                        'symbols': symbol,
                        'published_date': {
                            '$gte': prev_date - timedelta(days=1),
                            '$lte': prev_date
                        }
                    }
                    
                    news_count = self.db.news_collection.count_documents(news_query)
                    features.at[i, 'news_count_24h'] = news_count
                    
                    # Check for news sentiment
                    if news_count > 0:
                        news_cursor = self.db.news_collection.find(news_query)
                        news_items = list(news_cursor)
                        
                        # Calculate average sentiment
                        sentiments = [n.get('sentiment_score', 0) for n in news_items if 'sentiment_score' in n]
                        if sentiments:
                            features.at[i, 'news_avg_sentiment'] = np.mean(sentiments)
                            features.at[i, 'news_sentiment_std'] = np.std(sentiments)
                            
                except Exception as e:
                    self.logger.warning(f"Error generating news features: {e}")
        
        # Add previous gaps (auto-correlation)
        for i, row in gap_data.iterrows():
            if i >= 5:  # Need at least 5 previous gaps
                # Last 5 gaps
                last_gaps = gap_data.iloc[i-5:i]['gap_percent'].values
                
                features.at[i, 'gap_prev_1'] = last_gaps[-1]
                features.at[i, 'gap_prev_2'] = last_gaps[-2]
                features.at[i, 'gap_prev_3'] = last_gaps[-3]
                features.at[i, 'gap_prev_5_mean'] = np.mean(last_gaps)
                features.at[i, 'gap_prev_5_std'] = np.std(last_gaps)
        
        # Handle missing values
        features = features.fillna(0)
        
        # Create target variable
        target = gap_data['gap_percent']
        
        self.logger.info(f"Generated {len(features.columns)} features for overnight gap prediction")
        
        return features, target
    
    def train_model(self, symbol, exchange, features=None, target=None):
        """
        Train a model to predict overnight gaps.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            features (DataFrame): Feature data
            target (Series): Target data
            
        Returns:
            dict: Training results
        """
        self.logger.info(f"Training overnight gap model for {symbol} {exchange}")
        
        # Generate features and target if not provided
        if features is None or target is None:
            features, target = self.generate_features(symbol, exchange)
            
        if features is None or len(features) == 0 or target is None:
            self.logger.error(f"No features available for {symbol} {exchange}")
            return None
        
        # Check if we have enough samples
        if len(features) < self.config['min_samples']:
            self.logger.warning(f"Insufficient samples for {symbol} {exchange}: {len(features)} < {self.config['min_samples']}")
            return None
        
        # Scale features
        X = self.scaler.fit_transform(features)
        y = target.values
        
        # Split data
        if self.config['use_time_series_split']:
            # Use the last test_size percent for testing
            split_idx = int(len(X) * (1 - self.config['test_size']))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], shuffle=False
            )
            
        # Build model
        if self.config['model_type'] == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:  # default to gradient boosting
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy
        direction_actual = np.sign(y_test)
        direction_pred = np.sign(y_pred)
        directional_accuracy = np.mean(direction_actual == direction_pred)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_names = features.columns.tolist()
            importance = self.model.feature_importances_
            
            indices = np.argsort(importance)[::-1]
            
            self.feature_importance = [{
                'feature': feature_names[i],
                'importance': float(importance[i])
            } for i in indices]
        
        self.logger.info(f"Model training complete. RMSE: {rmse:.4f}, Dir. Acc: {directional_accuracy:.4f}")
        
        # Save model
        model_id = self.save_model(symbol, exchange)
        
        return {
            'symbol': symbol,
            'exchange': exchange,
            'model_type': self.config['model_type'],
            'samples': len(features),
            'features': len(features.columns),
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy
            },
            'model_id': model_id,
            'feature_importance': self.feature_importance[:10] if self.feature_importance else None
        }
    
    def save_model(self, symbol, exchange):
        """
        Save the trained model to the database.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            str: Model ID
        """
        if self.model is None:
            self.logger.error("No model to save. Train the model first.")
            return None
            
        try:
            # Serialize model
            model_bytes = pickle.dumps(self.model)
            model_base64 = base64.b64encode(model_bytes).decode('utf-8')
            
            # Serialize scaler
            scaler_bytes = pickle.dumps(self.scaler)
            scaler_base64 = base64.b64encode(scaler_bytes).decode('utf-8')
            
            # Create model document
            model_doc = {
                'symbol': symbol,
                'exchange': exchange,
                'model_name': f"{symbol}_{exchange}_overnight_gap",
                'model_type': self.config['model_type'],
                'prediction_type': 'overnight_gap',
                'creation_date': datetime.now(),
                'model_data': model_base64,
                'scaler_data': scaler_base64,
                'config': self.config,
                'feature_importance': self.feature_importance
            }
            
            # Insert into database
            result = self.db.models_collection.insert_one(model_doc)
            model_id = str(result.inserted_id)
            
            self.logger.info(f"Saved overnight gap model with ID: {model_id}")
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return None
    
    def load_model(self, symbol, exchange):
        """
        Load a model from the database.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            bool: Success/failure
        """
        try:
            # Query database
            query = {
                'symbol': symbol,
                'exchange': exchange,
                'prediction_type': 'overnight_gap'
            }
            
            model_doc = self.db.models_collection.find_one(query, sort=[('creation_date', -1)])
            
            if not model_doc:
                self.logger.warning(f"No overnight gap model found for {symbol} {exchange}")
                return False
                
            # Deserialize model
            model_base64 = model_doc['model_data']
            model_bytes = base64.b64decode(model_base64)
            self.model = pickle.loads(model_bytes)
            
            # Deserialize scaler
            scaler_base64 = model_doc['scaler_data']
            scaler_bytes = base64.b64decode(scaler_base64)
            self.scaler = pickle.loads(scaler_bytes)
            
            # Load feature importance
            self.feature_importance = model_doc.get('feature_importance')
            
            # Load config
            if 'config' in model_doc:
                self.config.update(model_doc['config'])
                
            self.logger.info(f"Loaded overnight gap model for {symbol} {exchange}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def predict_gap(self, symbol, exchange, date=None):
        """
        Predict overnight gap for the next trading day.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            date (datetime): Date to predict from (default: today)
            
        Returns:
            dict: Prediction result
        """
        self.logger.info(f"Predicting overnight gap for {symbol} {exchange}")
        
        # Load model if not already loaded
        if self.model is None:
            if not self.load_model(symbol, exchange):
                self.logger.error(f"Failed to load model for {symbol} {exchange}")
                return None
        
        # Set date to today if not provided
        date = date or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        try:
            # Load feature generators
            from ml.features.technical_features import TechnicalFeatureGenerator
            from ml.features.global_features import GlobalFeatureGenerator
            from ml.features.sentiment_features import SentimentFeatureGenerator
            
            technical_features = TechnicalFeatureGenerator(self.db)
            global_features = GlobalFeatureGenerator(self.db)
            sentiment_features = SentimentFeatureGenerator(self.db)
            
            # Get market data for feature generation
            start_date = date - timedelta(days=60)  # Buffer for indicators
            
            query = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': 'day',
                'timestamp': {
                    '$gte': start_date,
                    '$lte': date
                }
            }
            
            cursor = self.db.market_data_collection.find(query).sort('timestamp', 1)
            market_data = pd.DataFrame(list(cursor))
            
            if len(market_data) < 30:
                self.logger.error(f"Insufficient market data for {symbol} {exchange}")
                return None
                
            # Get latest market data
            latest_data = market_data.iloc[-1]
            latest_date = latest_data.get('timestamp')
            latest_close = latest_data.get('close')
            
            # Calculate date diff to next trading day
            today_weekday = date.weekday()
            
            if today_weekday == 4:  # Friday
                date_diff = 3  # Next trading day is Monday
            elif today_weekday == 5:  # Saturday
                date_diff = 2  # Next trading day is Monday
            else:
                date_diff = 1  # Next trading day is tomorrow
                
            # Generate features for prediction
            features = {}
            
            # Basic gap features
            features['date_diff'] = date_diff
            features['is_monday'] = (today_weekday + date_diff) % 7 == 0
            features['is_after_holiday'] = date_diff > 1
            
            # Technical features
            market_data.set_index('timestamp', inplace=True)
            tech_features = technical_features.generate_all_features(
                symbol, exchange, market_data
            )
            
            if tech_features is not None and latest_date in tech_features.index:
                tech_row = tech_features.loc[latest_date]
                
                for col in tech_features.columns:
                    features[f"tech_{col}"] = tech_row[col]
            
            # Global features
            global_data = global_features.generate_features(
                latest_date, latest_date + timedelta(minutes=1)
            )
            
            if global_data is not None and len(global_data) > 0:
                global_row = global_data.iloc[0]
                
                for col in global_data.columns:
                    features[f"global_{col}"] = global_row[col]
            
            # Sentiment features
            sentiment_data = sentiment_features.generate_features(
                symbol, exchange, latest_date, latest_date + timedelta(minutes=1)
            )
            
            if sentiment_data is not None and len(sentiment_data) > 0:
                sentiment_row = sentiment_data.iloc[0]
                
                for col in sentiment_data.columns:
                    features[f"sentiment_{col}"] = sentiment_row[col]
            
            # News features
            news_query = {
                'symbols': symbol,
                'published_date': {
                    '$gte': latest_date - timedelta(days=1),
                    '$lte': latest_date
                }
            }
            
            news_count = self.db.news_collection.count_documents(news_query)
            features['news_count_24h'] = news_count
            
            if news_count > 0:
                news_cursor = self.db.news_collection.find(news_query)
                news_items = list(news_cursor)
                
                sentiments = [n.get('sentiment_score', 0) for n in news_items if 'sentiment_score' in n]
                if sentiments:
                    features['news_avg_sentiment'] = np.mean(sentiments)
                    features['news_sentiment_std'] = np.std(sentiments)
            
            # Previous gaps
            gaps = self.get_overnight_gaps(symbol, exchange, 10)
            
            if gaps is not None and len(gaps) > 0:
                last_gaps = gaps['gap_percent'].values[-5:]
                
                if len(last_gaps) >= 3:
                    features['gap_prev_1'] = last_gaps[-1]
                    features['gap_prev_2'] = last_gaps[-2]
                    features['gap_prev_3'] = last_gaps[-3]
                    features['gap_prev_5_mean'] = np.mean(last_gaps)
                    features['gap_prev_5_std'] = np.std(last_gaps)
            
            # Convert features to DataFrame
            features_df = pd.DataFrame([features])
            
            # Fill missing values
            all_columns = self.scaler.feature_names_in_
            for col in all_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
                    
            # Keep only columns used during training
            features_df = features_df[all_columns]
            
            # Scale features
            X = self.scaler.transform(features_df)
            
            # Make prediction
            gap_pred = float(self.model.predict(X)[0])
            
            # Calculate prediction interval
            # Use historical RMSE to estimate 95% confidence interval
            rmse = 0.01  # Default 1% RMSE if not available
            
            # Get historical performance
            perf_query = {
                'symbol': symbol,
                'exchange': exchange,
                'prediction_type': 'overnight_gap',
                'metrics.rmse': {'$exists': True}
            }
            
            perf_doc = self.db.model_evaluations_collection.find_one(perf_query, sort=[('timestamp', -1)])
            
            if perf_doc and 'metrics' in perf_doc and 'rmse' in perf_doc['metrics']:
                rmse = perf_doc['metrics']['rmse']
                
            # Calculate prediction interval
            confidence = 0.95
            z_value = norm.ppf((1 + confidence) / 2)
            prediction_interval = z_value * rmse
            
            # Calculate expected price
            expected_open = latest_close * (1 + gap_pred)
            
            # Determine if gap is significant
            is_significant = abs(gap_pred) >= self.config['gap_threshold']
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(gap_pred, prediction_interval)
            
            # Create prediction document
            prediction = {
                'symbol': symbol,
                'exchange': exchange,
                'prediction_date': date,
                'for_date': date + timedelta(days=date_diff),
                'prediction_type': 'overnight_gap',
                'close_price': latest_close,
                'expected_open': expected_open,
                'gap_percent': gap_pred,
                'gap_direction': 'up' if gap_pred > 0 else 'down',
                'is_significant': is_significant,
                'prediction_interval': prediction_interval,
                'lower_bound': gap_pred - prediction_interval,
                'upper_bound': gap_pred + prediction_interval,
                'confidence': confidence_score,
                'features_used': len(features),
                'date_diff': date_diff,
                'next_trading_day': 'Monday' if features['is_monday'] else 'Weekday'
            }
            
            # Add trading signal if confidence is high enough
            if confidence_score >= self.config['confidence_threshold'] and is_significant:
                prediction['trading_signal'] = 'buy' if gap_pred > 0 else 'sell'
                
                # Add target price and stop loss
                if gap_pred > 0:  # Expected up gap
                    prediction['target_price'] = expected_open * 1.005  # 0.5% above gap
                    prediction['stop_loss'] = latest_close * 0.997  # 0.3% below close
                else:  # Expected down gap
                    prediction['target_price'] = expected_open * 0.995  # 0.5% below gap
                    prediction['stop_loss'] = latest_close * 1.003  # 0.3% above close
            
            # Save prediction
            self._save_prediction(prediction)
            
            self.logger.info(f"Predicted {gap_pred:.2%} overnight gap for {symbol} {exchange}")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting gap: {e}")
            return None
    
    def _calculate_confidence(self, gap_pred, prediction_interval):
        """
        Calculate confidence score for gap prediction.
        
        Args:
            gap_pred (float): Predicted gap
            prediction_interval (float): Prediction interval
            
        Returns:
            float: Confidence score (0-1)
        """
        # Confidence is higher when:
        # 1. Gap size is larger
        # 2. Prediction interval is smaller
        
        # Normalize gap size
        gap_conf = min(abs(gap_pred) / 0.02, 1.0)  # Cap at 2% gap
        
        # Normalize prediction interval (inversely related to confidence)
        interval_conf = max(0, 1 - (prediction_interval / 0.02))  # Lower interval = higher confidence
        
        # Combine (weighted average)
        confidence = 0.7 * gap_conf + 0.3 * interval_conf
        
        return min(max(confidence, 0.5), 0.95)  # Bound between 0.5 and 0.95
    
    def _save_prediction(self, prediction):
        """
        Save prediction to database.
        
        Args:
            prediction (dict): Prediction data
            
        Returns:
            str: Prediction ID
        """
        try:
            # Add timestamp
            prediction['timestamp'] = datetime.now()
            
            # Insert into database
            result = self.db.predictions_collection.insert_one(prediction)
            prediction_id = str(result.inserted_id)
            
            self.logger.info(f"Saved overnight gap prediction with ID: {prediction_id}")
            
            return prediction_id
            
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
            return None
    
    def evaluate_predictions(self, symbol, exchange, start_date=None, end_date=None):
        """
        Evaluate historical overnight gap predictions.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            start_date (datetime): Start date for evaluation
            end_date (datetime): End date for evaluation
            
        Returns:
            dict: Evaluation results
        """
        # Set default date range (last 30 days)
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))
        
        self.logger.info(f"Evaluating overnight gap predictions for {symbol} {exchange}")
        
        # Get predictions
        # Get predictions
        pred_query = {
            'symbol': symbol,
            'exchange': exchange,
            'prediction_type': 'overnight_gap',
            'prediction_date': {
                '$gte': start_date,
                '$lte': end_date
            }
        }
        
        cursor = self.db.predictions_collection.find(pred_query)
        predictions = list(cursor)
        
        if not predictions:
            self.logger.warning(f"No predictions found for evaluation")
            return {
                'symbol': symbol,
                'exchange': exchange,
                'start_date': start_date,
                'end_date': end_date,
                'count': 0,
                'message': 'No predictions available for evaluation'
            }
        
        # Get actual data for evaluation
        results = []
        
        for pred in predictions:
            try:
                pred_date = pred.get('prediction_date')
                for_date = pred.get('for_date')
                
                if not pred_date or not for_date:
                    continue
                    
                # Get actual data for prediction date and following date
                query = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': 'day',
                    'timestamp': {
                        '$gte': pred_date,
                        '$lte': for_date + timedelta(days=1)
                    }
                }
                
                cursor = self.db.market_data_collection.find(query).sort('timestamp', 1)
                market_data = list(cursor)
                
                if len(market_data) < 2:
                    continue
                    
                # Find data points for prediction date and target date
                pred_day_data = None
                target_day_data = None
                
                for data in market_data:
                    data_date = data.get('timestamp')
                    if data_date.date() == pred_date.date():
                        pred_day_data = data
                    elif data_date.date() == for_date.date():
                        target_day_data = data
                
                if not pred_day_data or not target_day_data:
                    continue
                    
                # Calculate actual gap
                pred_close = pred_day_data.get('close')
                target_open = target_day_data.get('open')
                
                actual_gap = (target_open - pred_close) / pred_close
                actual_direction = 'up' if actual_gap > 0 else 'down'
                
                # Compare with prediction
                pred_gap = pred.get('gap_percent')
                pred_direction = pred.get('gap_direction')
                
                is_direction_correct = pred_direction == actual_direction
                gap_error = actual_gap - pred_gap
                
                # Calculate other metrics
                in_prediction_interval = (
                    pred.get('lower_bound', -float('inf')) <= actual_gap <= 
                    pred.get('upper_bound', float('inf'))
                )
                
                # For trading signal evaluation
                trading_result = None
                
                if 'trading_signal' in pred:
                    signal = pred.get('trading_signal')
                    target_price = pred.get('target_price')
                    stop_loss = pred.get('stop_loss')
                    
                    if signal and target_price and stop_loss:
                        # Get high and low for the day
                        day_high = target_day_data.get('high')
                        day_low = target_day_data.get('low')
                        
                        # Determine result
                        if signal == 'buy':
                            if day_high >= target_price:
                                trading_result = 'win'
                            elif day_low <= stop_loss:
                                trading_result = 'loss'
                            else:
                                trading_result = 'neutral'
                        else:  # sell
                            if day_low <= target_price:
                                trading_result = 'win'
                            elif day_high >= stop_loss:
                                trading_result = 'loss'
                            else:
                                trading_result = 'neutral'
                
                # Add results
                result = {
                    'prediction_id': pred.get('_id'),
                    'prediction_date': pred_date,
                    'target_date': for_date,
                    'predicted_gap': pred_gap,
                    'actual_gap': actual_gap,
                    'gap_error': gap_error,
                    'predicted_direction': pred_direction,
                    'actual_direction': actual_direction,
                    'is_direction_correct': is_direction_correct,
                    'in_prediction_interval': in_prediction_interval,
                    'confidence': pred.get('confidence'),
                    'trading_signal': pred.get('trading_signal'),
                    'trading_result': trading_result
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Error evaluating prediction: {e}")
        
        if not results:
            self.logger.warning(f"No predictions could be evaluated")
            return {
                'symbol': symbol,
                'exchange': exchange,
                'start_date': start_date,
                'end_date': end_date,
                'count': len(predictions),
                'evaluated': 0,
                'message': 'No predictions could be evaluated'
            }
        
        # Calculate evaluation metrics
        count = len(results)
        
        # Direction accuracy
        direction_correct = sum(1 for r in results if r['is_direction_correct'])
        direction_accuracy = direction_correct / count
        
        # RMSE of gap prediction
        gap_errors = [r['gap_error'] for r in results]
        rmse = np.sqrt(np.mean(np.square(gap_errors)))
        mae = np.mean(np.abs(gap_errors))
        
        # Prediction interval coverage
        interval_coverage = sum(1 for r in results if r['in_prediction_interval']) / count
        
        # Trading signal performance
        signals = [r for r in results if 'trading_signal' in r and r['trading_signal']]
        
        trading_metrics = None
        
        if signals:
            wins = sum(1 for r in signals if r['trading_result'] == 'win')
            losses = sum(1 for r in signals if r['trading_result'] == 'loss')
            neutral = sum(1 for r in signals if r['trading_result'] == 'neutral')
            
            win_rate = wins / len(signals) if signals else 0
            
            trading_metrics = {
                'signals': len(signals),
                'wins': wins,
                'losses': losses,
                'neutral': neutral,
                'win_rate': win_rate
            }
        
        # Create evaluation document
        evaluation = {
            'symbol': symbol,
            'exchange': exchange,
            'start_date': start_date,
            'end_date': end_date,
            'timestamp': datetime.now(),
            'count': count,
            'metrics': {
                'direction_accuracy': direction_accuracy,
                'rmse': rmse,
                'mae': mae,
                'interval_coverage': interval_coverage
            },
            'trading': trading_metrics,
            'results': results
        }
        
        # Save evaluation
        try:
            result = self.db.model_evaluations_collection.insert_one(evaluation)
            evaluation_id = str(result.inserted_id)
            
            self.logger.info(f"Saved overnight gap evaluation with ID: {evaluation_id}")
            evaluation['_id'] = evaluation_id
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation: {e}")
        
        return evaluation
    
    def batch_predict(self, symbols_list=None, date=None):
        """
        Generate overnight gap predictions for multiple symbols.
        
        Args:
            symbols_list (list): List of (symbol, exchange) tuples
            date (datetime): Date to predict from (default: today)
            
        Returns:
            dict: Batch prediction results
        """
        if symbols_list is None:
            # Get active symbols from portfolio
            cursor = self.db.portfolio_collection.find({
                'status': 'active',
                'trading_config.enabled': True
            })
            
            symbols_list = [(doc['symbol'], doc['exchange']) for doc in cursor]
        
        date = date or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        self.logger.info(f"Generating overnight gap predictions for {len(symbols_list)} symbols")
        
        results = {}
        significant_gaps = []
        trading_signals = []
        
        for symbol, exchange in symbols_list:
            try:
                # Check if prediction already exists
                existing = self.db.predictions_collection.find_one({
                    'symbol': symbol,
                    'exchange': exchange,
                    'prediction_type': 'overnight_gap',
                    'prediction_date': {'$gte': date, '$lt': date + timedelta(days=1)}
                })
                
                if existing:
                    self.logger.info(f"Prediction already exists for {symbol} {exchange}")
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'exists',
                        'prediction': existing
                    }
                    
                    # Add to significant gaps if applicable
                    if existing.get('is_significant', False):
                        significant_gaps.append(existing)
                        
                    # Add to trading signals if applicable
                    if 'trading_signal' in existing:
                        trading_signals.append(existing)
                        
                    continue
                
                # Generate prediction
                prediction = self.predict_gap(symbol, exchange, date)
                
                if prediction:
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'success',
                        'prediction': prediction
                    }
                    
                    # Add to significant gaps if applicable
                    if prediction.get('is_significant', False):
                        significant_gaps.append(prediction)
                        
                    # Add to trading signals if applicable
                    if 'trading_signal' in prediction:
                        trading_signals.append(prediction)
                        
                else:
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'error',
                        'message': 'Failed to generate prediction'
                    }
                    
            except Exception as e:
                self.logger.error(f"Error predicting for {symbol} {exchange}: {e}")
                results[f"{symbol}_{exchange}"] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        # Create summary
        summary = {
            'date': date,
            'total': len(symbols_list),
            'success': sum(1 for r in results.values() if r['status'] in ['success', 'exists']),
            'significant_gaps': len(significant_gaps),
            'trading_signals': len(trading_signals),
            'up_gaps': sum(1 for p in significant_gaps if p['gap_direction'] == 'up'),
            'down_gaps': sum(1 for p in significant_gaps if p['gap_direction'] == 'down')
        }
        
        # Generate report
        report = self._generate_batch_report(date, summary, significant_gaps, trading_signals)
        
        return {
            'summary': summary,
            'report': report,
            'results': results
        }
    
    def _generate_batch_report(self, date, summary, significant_gaps, trading_signals):
        """
        Generate a report from batch predictions.
        
        Args:
            date (datetime): Prediction date
            summary (dict): Summary metrics
            significant_gaps (list): Significant gap predictions
            trading_signals (list): Trading signal predictions
            
        Returns:
            dict: Report data
        """
        # Sort significant gaps and signals
        significant_gaps = sorted(
            significant_gaps,
            key=lambda x: abs(x.get('gap_percent', 0)),
            reverse=True
        )
        
        trading_signals = sorted(
            trading_signals,
            key=lambda x: x.get('confidence', 0),
            reverse=True
        )
        
        # Create report
        report = {
            'date': date,
            'generation_time': datetime.now(),
            'for_date': significant_gaps[0]['for_date'] if significant_gaps else None,
            'summary': summary,
            'significant_gaps': [{
                'symbol': p['symbol'],
                'exchange': p['exchange'],
                'gap_percent': p['gap_percent'],
                'gap_direction': p['gap_direction'],
                'confidence': p['confidence'],
                'close_price': p['close_price'],
                'expected_open': p['expected_open']
            } for p in significant_gaps[:20]],  # Limit to top 20
            'trading_signals': [{
                'symbol': p['symbol'],
                'exchange': p['exchange'],
                'gap_percent': p['gap_percent'],
                'trading_signal': p['trading_signal'],
                'confidence': p['confidence'],
                'target_price': p.get('target_price'),
                'stop_loss': p.get('stop_loss')
            } for p in trading_signals[:10]]  # Limit to top 10
        }
        
        # Save report
        try:
            result = self.db.prediction_reports_collection.insert_one(report)
            report_id = str(result.inserted_id)
            
            self.logger.info(f"Saved overnight gap report with ID: {report_id}")
            report['_id'] = report_id
            
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
        
        return report