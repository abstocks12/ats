# ml/prediction/daily_predictor.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import os
import time
import json
from bson.objectid import ObjectId

class DailyPredictor:
    """Daily price prediction system."""
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the daily price predictor.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'prediction_horizon': 1,  # Days ahead to predict
            'feature_window': 120,    # Days of data for feature calculation
            'model_types': ['ensemble', 'random_forest', 'xgboost', 'gradient_boosting'],
            'model_class': 'classifier',  # 'classifier' or 'regressor'
            'confidence_threshold': 0.65,  # Minimum confidence to generate signals
            'retry_attempts': 3,      # Number of attempts to load models or generate predictions
            'timeout': 300,           # Maximum time (seconds) to wait for prediction generation
            'default_market_open': '09:15:00',  # Default market open time
            'default_market_close': '15:30:00', # Default market close time
            'save_features': True     # Whether to save generated features to DB
        }
        
        # Load feature generators
        self._load_feature_generators()
    
    def set_config(self, config):
        """
        Set predictor configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated predictor configuration: {self.config}")
    
    def _load_feature_generators(self):
        """Load feature generators."""
        try:
            from ml.features.technical_features import TechnicalFeatureGenerator
            from ml.features.fundamental_features import FundamentalFeatureGenerator
            from ml.features.global_features import GlobalFeatureGenerator
            from ml.features.sentiment_features import SentimentFeatureGenerator
            
            self.technical_features = TechnicalFeatureGenerator(self.db)
            self.fundamental_features = FundamentalFeatureGenerator(self.db)
            self.global_features = GlobalFeatureGenerator(self.db)
            self.sentiment_features = SentimentFeatureGenerator(self.db)
            
            self.logger.info("Feature generators loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading feature generators: {e}")
            raise
    
    def get_market_data(self, symbol, exchange, days=None):
        """
        Get recent market data for prediction.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            days (int): Number of days of data to retrieve
            
        Returns:
            DataFrame: Market data
        """
        days = days or self.config['feature_window']
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 2)  # Request more days to account for holidays/weekends
        
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
        
        if len(market_data) == 0:
            self.logger.error(f"No market data found for {symbol} {exchange}")
            return None
        
        # Set timestamp as index
        market_data.set_index('timestamp', inplace=True)
        
        # Ensure we have enough data
        if len(market_data) < days * 0.7:  # At least 70% of requested days
            self.logger.warning(f"Insufficient market data for {symbol} {exchange}: {len(market_data)} days")
            return None
        
        # Take only the last 'days' days
        if len(market_data) > days:
            market_data = market_data.iloc[-days:]
        
        self.logger.info(f"Retrieved {len(market_data)} days of market data for {symbol} {exchange}")
        
        return market_data
    
    def generate_features(self, symbol, exchange, market_data=None):
        """
        Generate features for prediction.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            market_data (DataFrame): Market data (optional, will be fetched if not provided)
            
        Returns:
            DataFrame: Feature data
        """
        self.logger.info(f"Generating prediction features for {symbol} {exchange}")
        
        # Get market data if not provided
        if market_data is None:
            market_data = self.get_market_data(symbol, exchange)
            
        if market_data is None or len(market_data) == 0:
            self.logger.error(f"No market data available for {symbol} {exchange}")
            return None
        
        # Create feature DataFrame with market data index
        features_data = pd.DataFrame(index=market_data.index)
        
        # Generate technical features
        tech_features = self.technical_features.generate_features(
            symbol, exchange, market_data
        )
        
        if tech_features is not None:
            features_data = features_data.join(tech_features)
        
        # Get start and end dates for other feature types
        start_date = market_data.index.min()
        end_date = market_data.index.max()
        
        # Generate fundamental features
        try:
            fund_features = self.fundamental_features.generate_features(
                symbol, exchange, start_date, end_date
            )
            
            if fund_features is not None:
                # Join on date
                features_data = features_data.join(fund_features)
                
        except Exception as e:
            self.logger.warning(f"Error generating fundamental features: {e}")
        
        # Generate global features
        try:
            global_features = self.global_features.generate_features(
                start_date, end_date
            )
            
            if global_features is not None:
                # Join on date
                features_data = features_data.join(global_features)
                
        except Exception as e:
            self.logger.warning(f"Error generating global features: {e}")
        
        # Generate sentiment features
        try:
            sentiment_features = self.sentiment_features.generate_features(
                symbol, exchange, start_date, end_date
            )
            
            if sentiment_features is not None:
                # Join on date
                features_data = features_data.join(sentiment_features)
                
        except Exception as e:
            self.logger.warning(f"Error generating sentiment features: {e}")
        
        # Remove rows with NaN values
        features_data = features_data.dropna()
        
        # Save features if configured
        if self.config['save_features']:
            self._save_features(symbol, exchange, features_data)
        
        self.logger.info(f"Generated {len(features_data.columns)} features for {symbol} {exchange}")
        
        return features_data
    
    def _save_features(self, symbol, exchange, features_data):
        """
        Save generated features to database.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            features_data (DataFrame): Feature data
        """
        try:
            # Convert DataFrame to records
            features_records = []
            
            for date, row in features_data.iterrows():
                record = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'date': date,
                    'features': row.to_dict()
                }
                features_records.append(record)
            
            # Delete existing records for today
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            self.db.features_collection.delete_many({
                'symbol': symbol,
                'exchange': exchange,
                'date': {'$gte': today}
            })
            
            # Insert new records
            if features_records:
                self.db.features_collection.insert_many(features_records)
                
            self.logger.info(f"Saved {len(features_records)} feature records to database")
            
        except Exception as e:
            self.logger.error(f"Error saving features: {e}")
    
    def load_models(self, symbol, exchange):
        """
        Load prediction models for a symbol.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            dict: Loaded models
        """
        self.logger.info(f"Loading prediction models for {symbol} {exchange}")
        
        models = {}
        model_types = self.config['model_types']
        model_class = self.config['model_class']
        
        for model_type in model_types:
            for attempt in range(self.config['retry_attempts']):
                try:
                    if model_type == 'ensemble':
                        # Load ensemble model
                        from ml.models.ensemble_predictor import EnsemblePredictor
                        model = EnsemblePredictor(self.db)
                        
                        model_name = f"{symbol}_{exchange}_ensemble_{model_class}r"
                        if model_class == 'classifier':
                            model_name = f"{symbol}_{exchange}_ensemble_classifier"
                        else:
                            model_name = f"{symbol}_{exchange}_ensemble_regressor"
                        
                        if model.load_models(symbol=symbol, exchange=exchange, ensemble_name=model_name):
                            models['ensemble'] = model
                            self.logger.info(f"Loaded ensemble model for {symbol} {exchange}")
                            break
                        
                    else:
                        # Load individual model
                        if model_class == 'classifier':
                            from ml.models.classifier import MarketClassifier
                            model = MarketClassifier(self.db)
                            model_name = f"{symbol}_{exchange}_{model_type}_classifier"
                        else:
                            from ml.models.regressor import MarketRegressor
                            model = MarketRegressor(self.db)
                            model_name = f"{symbol}_{exchange}_{model_type}_regressor"
                        
                        if model.load_model(symbol=symbol, exchange=exchange, model_name=model_name):
                            models[model_type] = model
                            self.logger.info(f"Loaded {model_type} model for {symbol} {exchange}")
                            break
                    
                except Exception as e:
                    self.logger.warning(f"Error loading {model_type} model (attempt {attempt+1}): {e}")
                    time.sleep(1)  # Short delay before retry
        
        if not models:
            self.logger.error(f"Failed to load any models for {symbol} {exchange}")
            
        return models
    
    def generate_prediction(self, symbol, exchange, features_data=None, models=None):
        """
        Generate prediction for a symbol.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            features_data (DataFrame): Feature data (optional, will be generated if not provided)
            models (dict): Loaded models (optional, will be loaded if not provided)
            
        Returns:
            dict: Prediction results
        """
        self.logger.info(f"Generating prediction for {symbol} {exchange}")
        
        # Get features if not provided
        if features_data is None:
            features_data = self.generate_features(symbol, exchange)
            
        if features_data is None or len(features_data) == 0:
            self.logger.error(f"No feature data available for {symbol} {exchange}")
            return None
        
        # Get latest features
        latest_features = features_data.iloc[-1:].copy()
        
        # Load models if not provided
        if models is None:
            models = self.load_models(symbol, exchange)
            
        if not models:
            self.logger.error(f"No models available for {symbol} {exchange}")
            return None
        
        # Generate predictions from each model
        predictions = {}
        model_class = self.config['model_class']
        
        for model_type, model in models.items():
            try:
                if model_type == 'ensemble':
                    # Generate ensemble prediction
                    prediction = model.generate_market_prediction(
                        symbol, exchange, latest_features, save_prediction=False
                    )
                else:
                    # Generate individual model prediction
                    if model_class == 'classifier':
                        prediction = model.generate_market_prediction(
                            symbol, exchange, latest_features, save_prediction=False
                        )
                    else:
                        prediction = model.generate_price_prediction(
                            symbol, exchange, latest_features, save_prediction=False
                        )
                
                if prediction:
                    predictions[model_type] = prediction
                    self.logger.info(f"{model_type} prediction: {prediction['prediction']} ({prediction.get('confidence', 0):.2f})")
                    
            except Exception as e:
                self.logger.error(f"Error generating {model_type} prediction: {e}")
        
        if not predictions:
            self.logger.error(f"Failed to generate any predictions for {symbol} {exchange}")
            return None
        
        # Combine predictions
        combined_prediction = self._combine_predictions(predictions)
        
        # Add market context
        combined_prediction = self._add_market_context(symbol, exchange, combined_prediction)
        
        # Save combined prediction
        self._save_prediction(combined_prediction)
        
        return combined_prediction
    
    def _combine_predictions(self, predictions):
        """
        Combine predictions from multiple models.
        
        Args:
            predictions (dict): Predictions from different models
            
        Returns:
            dict: Combined prediction
        """
        # Prioritize ensemble if available
        if 'ensemble' in predictions:
            combined = predictions['ensemble'].copy()
            combined['sources'] = predictions.keys()
            combined['prediction_type'] = 'combined'
            return combined
        
        # Alternatively, use weighted average of individual predictions
        if predictions:
            # Default weights
            weights = {
                'random_forest': 1.0,
                'xgboost': 1.0,
                'gradient_boosting': 1.0,
                'lightgbm': 1.0
            }
            
            # Extract all predictions and confidences
            pred_values = []
            confidences = []
            model_types = []
            
            for model_type, prediction in predictions.items():
                pred_values.append(prediction['prediction'])
                confidences.append(prediction.get('confidence', 0.5))
                model_types.append(model_type)
            
            # Count predictions for each class
            pred_counts = {}
            for pred, conf, model in zip(pred_values, confidences, model_types):
                if pred not in pred_counts:
                    pred_counts[pred] = {'count': 0, 'weighted_count': 0, 'models': []}
                
                weight = weights.get(model, 1.0)
                pred_counts[pred]['count'] += 1
                pred_counts[pred]['weighted_count'] += conf * weight
                pred_counts[pred]['models'].append(model)
            
            # Find prediction with highest weighted count
            best_pred = max(pred_counts.items(), key=lambda x: x[1]['weighted_count'])
            pred_class = best_pred[0]
            pred_info = best_pred[1]
            
            # Use the first model's prediction as a template
            first_model = list(predictions.values())[0]
            combined = first_model.copy()
            
            # Update with combined results
            combined['prediction'] = pred_class
            combined['confidence'] = pred_info['weighted_count'] / sum(confidences)
            combined['prediction_type'] = 'combined'
            combined['sources'] = model_types
            combined['agreement'] = {
                'total_models': len(predictions),
                'agreeing_models': pred_info['count'],
                'agreeing_models_list': pred_info['models']
            }
            
            return combined
        
        return None
    
    def _add_market_context(self, symbol, exchange, prediction):
        """
        Add market context to prediction.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            prediction (dict): Prediction data
            
        Returns:
            dict: Prediction with market context
        """
        if prediction is None:
            return None
        
        try:
            # Get latest market data
            latest_data = self.db.market_data_collection.find_one({
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': 'day'
            }, sort=[('timestamp', -1)])
            
            if latest_data:
                prediction['current_price'] = latest_data.get('close')
                prediction['trading_date'] = latest_data.get('timestamp')
                
                # Calculate price levels
                if 'current_price' in prediction:
                    current_price = prediction['current_price']
                    
                    # Support/resistance levels
                    prediction['price_levels'] = {
                        'support_1': current_price * 0.99,
                        'resistance_1': current_price * 1.01
                    }
                    
                    # Target price
                    if 'prediction' in prediction:
                        if prediction['prediction'] == 'up':
                            prediction['target_price'] = current_price * 1.01
                            prediction['stop_loss'] = current_price * 0.99
                        elif prediction['prediction'] == 'down':
                            prediction['target_price'] = current_price * 0.99
                            prediction['stop_loss'] = current_price * 1.01
            
            # Get market hours
            prediction['market_hours'] = {
                'open': self.config['default_market_open'],
                'close': self.config['default_market_close']
            }
            
            # Get sector performance
            sector = self._get_sector(symbol, exchange)
            if sector:
                prediction['sector'] = sector
                prediction['sector_performance'] = self._get_sector_performance(sector)
            
            # Add market regime
            prediction['market_regime'] = self._get_market_regime(symbol, exchange)
            
        except Exception as e:
            self.logger.error(f"Error adding market context: {e}")
        
        return prediction
    
    def _get_sector(self, symbol, exchange):
        """
        Get sector for a symbol.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            str: Sector name
        """
        try:
            # Check portfolio collection
            portfolio_item = self.db.portfolio_collection.find_one({
                'symbol': symbol,
                'exchange': exchange
            })
            
            if portfolio_item and 'sector' in portfolio_item:
                return portfolio_item['sector']
            
            # If not in portfolio, check financial data
            financial_data = self.db.financial_data_collection.find_one({
                'symbol': symbol,
                'exchange': exchange
            }, sort=[('report_date', -1)])
            
            if financial_data and 'sector' in financial_data:
                return financial_data['sector']
            
        except Exception as e:
            self.logger.warning(f"Error getting sector: {e}")
        
        return None
    
    def _get_sector_performance(self, sector):
        """
        Get sector performance.
        
        Args:
            sector (str): Sector name
            
        Returns:
            dict: Sector performance data
        """
        try:
            # Get today's date
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday = today - timedelta(days=1)
            
            # Get sector performance from database
            sector_data = self.db.sector_performance_collection.find_one({
                'sector': sector,
                'date': {'$gte': yesterday}
            })
            
            if sector_data:
                return {
                    'daily_change': sector_data.get('daily_change'),
                    'weekly_change': sector_data.get('weekly_change'),
                    'monthly_change': sector_data.get('monthly_change'),
                    'relative_strength': sector_data.get('relative_strength')
                }
            
        except Exception as e:
            self.logger.warning(f"Error getting sector performance: {e}")
        
        return None
    
    def _get_market_regime(self, symbol, exchange):
        """
        Get current market regime.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            dict: Market regime data
        """
        try:
            # Get today's date
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = today - timedelta(days=7)
            
            # Get latest market analysis
            analysis = self.db.market_analysis_collection.find_one({
                'symbol': symbol,
                'exchange': exchange,
                'date': {'$gte': week_ago}
            }, sort=[('date', -1)])
            
            if analysis and 'regime' in analysis:
                return {
                    'regime': analysis['regime'],
                    'volatility': analysis.get('volatility_regime'),
                    'trend': analysis.get('trend_strength'),
                    'momentum': analysis.get('momentum_regime')
                }
            
        except Exception as e:
            self.logger.warning(f"Error getting market regime: {e}")
        
        return None
    
    def _save_prediction(self, prediction):
        """
        Save prediction to database.
        
        Args:
            prediction (dict): Prediction data
            
        Returns:
            str: Prediction ID
        """
        if prediction is None:
            return None
        
        try:
            # Add timestamp if not present
            if 'date' not in prediction:
                prediction['date'] = datetime.now()
            
            # Add for_date if not present
            if 'for_date' not in prediction:
                prediction['for_date'] = prediction['date'] + timedelta(days=self.config['prediction_horizon'])
            
            # Insert into database
            result = self.db.predictions_collection.insert_one(prediction)
            prediction_id = str(result.inserted_id)
            
            self.logger.info(f"Saved prediction for {prediction.get('symbol')} {prediction.get('exchange')}")
            
            return prediction_id
            
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
            return None
    
    def generate_daily_predictions(self, symbols_list=None):
        """
        Generate daily predictions for a list of symbols.
        
        Args:
            symbols_list (list): List of (symbol, exchange) tuples
            
        Returns:
            dict: Prediction results by symbol
        """
        if symbols_list is None:
            # Get all active symbols from portfolio
            cursor = self.db.portfolio_collection.find({
                'status': 'active',
                'trading_config.enabled': True
            })
            
            symbols_list = [(doc['symbol'], doc['exchange']) for doc in cursor]
        
        self.logger.info(f"Generating daily predictions for {len(symbols_list)} symbols")
        
        results = {}
        success_count = 0
        fail_count = 0
        
        # Generate predictions for each symbol
        for symbol, exchange in symbols_list:
            try:
                # Check if prediction already exists for today
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                existing = self.db.predictions_collection.find_one({
                    'symbol': symbol,
                    'exchange': exchange,
                    'date': {'$gte': today}
                })
                
                if existing:
                    self.logger.info(f"Prediction already exists for {symbol} {exchange}")
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'exists',
                        'prediction': existing
                    }
                    success_count += 1
                    continue
                
                # Generate new prediction
                prediction = self.generate_prediction(symbol, exchange)
                
                if prediction:
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'success',
                        'prediction': prediction
                    }
                    success_count += 1
                else:
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'error',
                        'message': 'Failed to generate prediction'
                    }
                    fail_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error generating prediction for {symbol} {exchange}: {e}")
                results[f"{symbol}_{exchange}"] = {
                    'status': 'error',
                    'message': str(e)
                }
                fail_count += 1
        
        self.logger.info(f"Daily prediction generation complete: {success_count} success, {fail_count} failed")
        
        return {
            'total': len(symbols_list),
            'success': success_count,
            'failed': fail_count,
            'results': results
        }
    
    def generate_prediction_report(self, date=None):
        """
        Generate a summary report of daily predictions.
        
        Args:
            date (datetime): Date to generate report for (default: today)
            
        Returns:
            dict: Report data
        """
        date = date or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        next_day = date + timedelta(days=1)
        
        self.logger.info(f"Generating prediction report for {date.strftime('%Y-%m-%d')}")
        
        # Get all predictions for the date
        cursor = self.db.predictions_collection.find({
            'date': {'$gte': date, '$lt': next_day}
        })
        
        predictions = list(cursor)
        
        if not predictions:
            self.logger.warning(f"No predictions found for {date.strftime('%Y-%m-%d')}")
            return {
                'date': date,
                'count': 0,
                'message': 'No predictions found'
            }
        
        # Count predictions by type
        prediction_counts = {
            'up': 0,
            'down': 0,
            'neutral': 0,
            'total': len(predictions)
        }
        
        # Group by sector
        sector_predictions = {}
        
        # Group by confidence
        confidence_groups = {
            'high': [],   # >= 0.8
            'medium': [], # 0.65-0.8
            'low': []     # < 0.65
        }
        
        # Extract top predictions
        top_predictions = []
        
        for pred in predictions:
            # Count by prediction type
            pred_type = pred.get('prediction', 'unknown')
            if pred_type in prediction_counts:
                prediction_counts[pred_type] += 1
            
            # Group by sector
            sector = pred.get('sector')
            if sector:
                if sector not in sector_predictions:
                    sector_predictions[sector] = {
                        'up': 0,
                        'down': 0,
                        'neutral': 0,
                        'total': 0
                    }
                    
                sector_predictions[sector]['total'] += 1
                if pred_type in sector_predictions[sector]:
                    sector_predictions[sector][pred_type] += 1
            
            # Group by confidence
            confidence = pred.get('confidence', 0)
            
            if confidence >= 0.8:
                confidence_groups['high'].append(pred)
            elif confidence >= 0.65:
                confidence_groups['medium'].append(pred)
            else:
                confidence_groups['low'].append(pred)
            
            # Add to top predictions if confidence is high
            if confidence >= self.config['confidence_threshold']:
                top_predictions.append(pred)
        
        # Sort top predictions by confidence
        top_predictions = sorted(
            top_predictions, 
            key=lambda x: x.get('confidence', 0), 
            reverse=True
        )[:20]  # Limit to top 20
        
        # Calculate market sentiment
        bull_bear_ratio = prediction_counts['up'] / max(prediction_counts['down'], 1)
        market_sentiment = 'bullish' if bull_bear_ratio > 1.2 else 'bearish' if bull_bear_ratio < 0.8 else 'neutral'
        
        # Generate report
        report = {
            'date': date,
            'generation_time': datetime.now(),
            'count': {
                'total': prediction_counts['total'],
                'up': prediction_counts['up'],
                'down': prediction_counts['down'],
                'neutral': prediction_counts['neutral'],
                'high_confidence': len(confidence_groups['high']),
                'medium_confidence': len(confidence_groups['medium']),
                'low_confidence': len(confidence_groups['low'])
            },
            'market_sentiment': {
                'bull_bear_ratio': bull_bear_ratio,
                'sentiment': market_sentiment
            },
            'sector_analysis': sector_predictions,
            'top_predictions': [{
                'symbol': p.get('symbol'),
                'exchange': p.get('exchange'),
                'prediction': p.get('prediction'),
                'confidence': p.get('confidence'),
                'current_price': p.get('current_price'),
                'target_price': p.get('target_price'),
                'sector': p.get('sector')
            } for p in top_predictions]
        }
        
        # Save report
        try:
            # Check if report already exists
            existing = self.db.prediction_reports_collection.find_one({
                'date': date
            })
            
            if existing:
                # Update existing report
                self.db.prediction_reports_collection.update_one(
                    {'_id': existing['_id']},
                    {'$set': report}
                )
                report_id = str(existing['_id'])
            else:
                # Insert new report
                result = self.db.prediction_reports_collection.insert_one(report)
                report_id = str(result.inserted_id)
                
            self.logger.info(f"Saved prediction report with ID: {report_id}")
            report['_id'] = report_id
            
        except Exception as e:
            self.logger.error(f"Error saving prediction report: {e}")
        
        return report
    
    def get_prediction(self, symbol, exchange, date=None, id=None):
        """
        Get a prediction for a symbol.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            date (datetime): Date to get prediction for (default: today)
            id (str): Prediction ID
            
        Returns:
            dict: Prediction data
        """
        if id:
            # Get prediction by ID
            try:
                prediction = self.db.predictions_collection.find_one({
                    '_id': ObjectId(id)
                })
                return prediction
            except Exception as e:
                self.logger.error(f"Error getting prediction by ID: {e}")
                return None
        
        date = date or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        next_day = date + timedelta(days=1)
        
        # Get prediction
        # Get prediction by symbol, exchange, and date
        prediction = self.db.predictions_collection.find_one({
            'symbol': symbol,
            'exchange': exchange,
            'date': {'$gte': date, '$lt': next_day}
        }, sort=[('date', -1)])
        
        return prediction
    
    def update_prediction_performance(self, prediction_id=None, date=None):
        """
        Update prediction performance with actual results.
        
        Args:
            prediction_id (str): Prediction ID
            date (datetime): Date to update predictions for (default: yesterday)
            
        Returns:
            int: Number of predictions updated
        """
        # Determine which predictions to update
        if prediction_id:
            # Update specific prediction
            query = {'_id': ObjectId(prediction_id)}
        else:
            # Update predictions for a date
            date = date or (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            next_day = date + timedelta(days=1)
            
            query = {
                'date': {'$gte': date, '$lt': next_day},
                'performance': {'$exists': False}
            }
        
        self.logger.info(f"Updating prediction performance for {prediction_id or date.strftime('%Y-%m-%d')}")
        
        # Get predictions to update
        cursor = self.db.predictions_collection.find(query)
        predictions = list(cursor)
        
        if not predictions:
            self.logger.warning(f"No predictions found to update")
            return 0
        
        updated_count = 0
        
        for prediction in predictions:
            symbol = prediction.get('symbol')
            exchange = prediction.get('exchange')
            pred_date = prediction.get('date')
            
            if not all([symbol, exchange, pred_date]):
                continue
            
            # Get actual price data for the prediction date and the next day
            next_date = pred_date + timedelta(days=1)
            
            market_data = self.db.market_data_collection.find({
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': 'day',
                'timestamp': {'$gte': pred_date, '$lte': next_date + timedelta(days=1)}
            }).sort('timestamp', 1)
            
            market_data = market_data.to_dict('records')
            
            if len(market_data) < 2:
                self.logger.warning(f"Insufficient market data to evaluate prediction for {symbol} {exchange}")
                continue
            
            # Get predicted day's close and next day's close
            pred_day_data = None
            next_day_data = None
            
            for data in market_data:
                data_date = data.get('timestamp')
                if data_date.date() == pred_date.date():
                    pred_day_data = data
                elif data_date.date() == next_date.date():
                    next_day_data = data
            
            if not pred_day_data or not next_day_data:
                self.logger.warning(f"Missing market data for prediction or next day")
                continue
            
            # Calculate actual price change
            pred_close = pred_day_data.get('close')
            next_close = next_day_data.get('close')
            
            if not pred_close or not next_close:
                continue
                
            actual_change = (next_close - pred_close) / pred_close
            actual_direction = 'up' if actual_change > 0 else 'down' if actual_change < 0 else 'neutral'
            
            # Compare with prediction
            pred_direction = prediction.get('prediction')
            is_correct = pred_direction == actual_direction
            
            # Update prediction document
            performance = {
                'actual_change': actual_change,
                'actual_direction': actual_direction,
                'is_correct': is_correct,
                'pred_price': pred_close,
                'actual_price': next_close,
                'updated_at': datetime.now()
            }
            
            # Add reward/loss calculation if target price was specified
            if 'target_price' in prediction and 'stop_loss' in prediction:
                target = prediction.get('target_price')
                stop = prediction.get('stop_loss')
                
                # Calculate if reached target or stop
                day_high = next_day_data.get('high', next_close)
                day_low = next_day_data.get('low', next_close)
                
                if pred_direction == 'up':
                    reached_target = day_high >= target
                    reached_stop = day_low <= stop
                else:  # down
                    reached_target = day_low <= target
                    reached_stop = day_high >= stop
                
                performance['reached_target'] = reached_target
                performance['reached_stop'] = reached_stop
                
                # Calculate reward/risk ratio
                if reached_target:
                    result = (target - pred_close) / pred_close
                elif reached_stop:
                    result = (stop - pred_close) / pred_close
                else:
                    result = actual_change
                    
                performance['trade_result'] = result
            
            # Update prediction document
            self.db.predictions_collection.update_one(
                {'_id': prediction['_id']},
                {'$set': {'performance': performance}}
            )
            
            updated_count += 1
            
        self.logger.info(f"Updated performance for {updated_count} predictions")
        
        return updated_count
    
    def generate_performance_report(self, start_date=None, end_date=None, symbols=None):
        """
        Generate a report on prediction performance.
        
        Args:
            start_date (datetime): Start date for report
            end_date (datetime): End date for report
            symbols (list): List of symbols to include (optional)
            
        Returns:
            dict: Performance report
        """
        # Set default date range (last 30 days)
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))
        
        self.logger.info(f"Generating performance report from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Build query
        query = {
            'date': {'$gte': start_date, '$lte': end_date},
            'performance': {'$exists': True}
        }
        
        if symbols:
            query['symbol'] = {'$in': symbols if isinstance(symbols, list) else [symbols]}
        
        # Get predictions with performance data
        cursor = self.db.predictions_collection.find(query)
        predictions = list(cursor)
        
        if not predictions:
            self.logger.warning("No predictions with performance data found")
            return {
                'start_date': start_date,
                'end_date': end_date,
                'count': 0,
                'message': 'No performance data available'
            }
        
        # Aggregate performance metrics
        total_count = len(predictions)
        correct_count = sum(1 for p in predictions if p.get('performance', {}).get('is_correct', False))
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        # Count by prediction type
        up_predictions = [p for p in predictions if p.get('prediction') == 'up']
        down_predictions = [p for p in predictions if p.get('prediction') == 'down']
        
        up_correct = sum(1 for p in up_predictions if p.get('performance', {}).get('is_correct', False))
        down_correct = sum(1 for p in down_predictions if p.get('performance', {}).get('is_correct', False))
        
        up_accuracy = up_correct / len(up_predictions) if len(up_predictions) > 0 else 0
        down_accuracy = down_correct / len(down_predictions) if len(down_predictions) > 0 else 0
        
        # Calculate average return
        avg_return = np.mean([p.get('performance', {}).get('actual_change', 0) for p in predictions])
        
        # Calculate returns for correct and incorrect predictions
        correct_returns = [p.get('performance', {}).get('actual_change', 0) 
                          for p in predictions if p.get('performance', {}).get('is_correct', False)]
        incorrect_returns = [p.get('performance', {}).get('actual_change', 0) 
                            for p in predictions if not p.get('performance', {}).get('is_correct', False)]
        
        avg_correct_return = np.mean(correct_returns) if correct_returns else 0
        avg_incorrect_return = np.mean(incorrect_returns) if incorrect_returns else 0
        
        # Calculate by confidence level
        high_conf = [p for p in predictions if p.get('confidence', 0) >= 0.8]
        med_conf = [p for p in predictions if 0.65 <= p.get('confidence', 0) < 0.8]
        low_conf = [p for p in predictions if p.get('confidence', 0) < 0.65]
        
        high_conf_accuracy = sum(1 for p in high_conf if p.get('performance', {}).get('is_correct', False)) / len(high_conf) if high_conf else 0
        med_conf_accuracy = sum(1 for p in med_conf if p.get('performance', {}).get('is_correct', False)) / len(med_conf) if med_conf else 0
        low_conf_accuracy = sum(1 for p in low_conf if p.get('performance', {}).get('is_correct', False)) / len(low_conf) if low_conf else 0
        
        # Calculate by model type
        model_performance = {}
        for p in predictions:
            sources = p.get('sources', [])
            for source in sources:
                if source not in model_performance:
                    model_performance[source] = {
                        'count': 0,
                        'correct': 0,
                        'accuracy': 0,
                        'returns': []
                    }
                
                model_performance[source]['count'] += 1
                
                if p.get('performance', {}).get('is_correct', False):
                    model_performance[source]['correct'] += 1
                    
                model_performance[source]['returns'].append(
                    p.get('performance', {}).get('actual_change', 0)
                )
        
        for source, data in model_performance.items():
            data['accuracy'] = data['correct'] / data['count'] if data['count'] > 0 else 0
            data['avg_return'] = np.mean(data['returns']) if data['returns'] else 0
        
        # Generate report
        report = {
            'start_date': start_date,
            'end_date': end_date,
            'generation_time': datetime.now(),
            'overall': {
                'total_predictions': total_count,
                'correct_predictions': correct_count,
                'accuracy': accuracy,
                'avg_return': avg_return
            },
            'by_direction': {
                'up': {
                    'count': len(up_predictions),
                    'correct': up_correct,
                    'accuracy': up_accuracy
                },
                'down': {
                    'count': len(down_predictions),
                    'correct': down_correct,
                    'accuracy': down_accuracy
                }
            },
            'by_confidence': {
                'high': {
                    'count': len(high_conf),
                    'accuracy': high_conf_accuracy
                },
                'medium': {
                    'count': len(med_conf),
                    'accuracy': med_conf_accuracy
                },
                'low': {
                    'count': len(low_conf),
                    'accuracy': low_conf_accuracy
                }
            },
            'returns': {
                'avg_return': avg_return,
                'avg_correct_return': avg_correct_return,
                'avg_incorrect_return': avg_incorrect_return
            },
            'by_model': model_performance
        }
        
        # Calculate profit/loss for trading simulation
        trading_results = self._simulate_trading_performance(predictions)
        if trading_results:
            report['trading_simulation'] = trading_results
        
        # Save report
        try:
            result = self.db.performance_reports_collection.insert_one(report)
            report_id = str(result.inserted_id)
            
            self.logger.info(f"Saved performance report with ID: {report_id}")
            report['_id'] = report_id
            
        except Exception as e:
            self.logger.error(f"Error saving performance report: {e}")
        
        return report
    
    def _simulate_trading_performance(self, predictions):
        """
        Simulate trading performance based on predictions.
        
        Args:
            predictions (list): List of predictions with performance data
            
        Returns:
            dict: Trading simulation results
        """
        if not predictions:
            return None
        
        # Sort predictions by date
        predictions = sorted(predictions, key=lambda x: x.get('date', datetime.min))
        
        # Initialize simulation
        initial_capital = 10000
        capital = initial_capital
        position = 0
        transaction_cost = 0.001  # 0.1% per trade
        
        # Results tracking
        equity_curve = [capital]
        positions = [position]
        trades = []
        
        # Execute simulation
        for pred in predictions:
            performance = pred.get('performance', {})
            if not performance:
                continue
                
            pred_direction = pred.get('prediction')
            confidence = pred.get('confidence', 0)
            actual_change = performance.get('actual_change', 0)
            
            # Skip if no clear direction or low confidence
            if not pred_direction or pred_direction == 'neutral' or confidence < 0.65:
                continue
                
            # Determine position based on prediction
            new_position = 1 if pred_direction == 'up' else -1 if pred_direction == 'down' else 0
            
            # Calculate returns
            if position != 0:
                # Close existing position
                trade_return = position * actual_change
                capital *= (1 + trade_return)
                
                # Apply transaction cost
                cost = capital * transaction_cost
                capital -= cost
                
                trades.append({
                    'symbol': pred.get('symbol'),
                    'entry_date': pred.get('date'),
                    'exit_date': performance.get('updated_at'),
                    'direction': 'long' if position > 0 else 'short',
                    'return': trade_return,
                    'is_correct': performance.get('is_correct', False)
                })
            
            # Open new position
            position = new_position
            
            # Apply transaction cost for new position
            if position != 0:
                cost = capital * transaction_cost
                capital -= cost
            
            # Update equity curve
            equity_curve.append(capital)
            positions.append(position)
        
        # Calculate performance metrics
        total_return = (capital / initial_capital - 1) * 100
        win_count = sum(1 for t in trades if t['return'] > 0)
        loss_count = sum(1 for t in trades if t['return'] <= 0)
        win_rate = win_count / len(trades) if trades else 0
        
        # Calculate average win and loss
        wins = [t['return'] for t in trades if t['return'] > 0]
        losses = [t['return'] for t in trades if t['return'] <= 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Calculate profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return_percent': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown_percent': max_drawdown,
            'trade_count': len(trades),
            'win_count': win_count,
            'loss_count': loss_count,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }