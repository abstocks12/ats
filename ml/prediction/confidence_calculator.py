# ml/prediction/confidence_calculator.py
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

class ConfidenceCalculator:
    """Calculate confidence scores for predictions."""
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the confidence calculator.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'min_confidence': 0.5,
            'max_confidence': 0.95,
            'backtest_days': 30,
            'accuracy_weight': 0.4,
            'model_weight': 0.2,
            'market_weight': 0.2,
            'volatility_weight': 0.1,
            'consistency_weight': 0.1
        }
    
    def set_config(self, config):
        """
        Set calculator configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated confidence calculator configuration: {self.config}")
    
    def calculate_confidence(self, prediction, model_confidence=None, features=None):
        """
        Calculate an adjusted confidence score for a prediction.
        
        Args:
            prediction (dict): Prediction data
            model_confidence (float): Raw model confidence (optional)
            features (dict): Feature values (optional)
            
        Returns:
            float: Adjusted confidence score
        """
        self.logger.info(f"Calculating confidence for {prediction.get('symbol')} prediction")
        
        # Use model confidence if provided, otherwise get from prediction
        if model_confidence is None:
            model_confidence = prediction.get('confidence', 0.5)
            
        symbol = prediction.get('symbol')
        exchange = prediction.get('exchange')
        direction = prediction.get('prediction')
        
        if not all([symbol, exchange, direction]):
            return model_confidence
        
        # Get weights
        weights = {
            'model': self.config['model_weight'],
            'accuracy': self.config['accuracy_weight'],
            'market': self.config['market_weight'],
            'volatility': self.config['volatility_weight'],
            'consistency': self.config['consistency_weight']
        }
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate individual confidence components
        confidence_components = {}
        
        # 1. Model confidence (provided or from prediction)
        confidence_components['model'] = model_confidence
        
        # 2. Historical accuracy confidence
        confidence_components['accuracy'] = self._calculate_historical_accuracy(symbol, exchange, direction)
        
        # 3. Market regime confidence
        confidence_components['market'] = self._calculate_market_confidence(symbol, exchange, direction)
        
        # 4. Volatility confidence
        confidence_components['volatility'] = self._calculate_volatility_confidence(symbol, exchange)
        
        # 5. Model consistency confidence
        confidence_components['consistency'] = self._calculate_consistency_confidence(symbol, exchange, direction)
        
        # Calculate weighted confidence
        confidence = 0
        for component, value in confidence_components.items():
            if value is not None:
                confidence += value * weights.get(component, 0)
        
        # Ensure confidence is within bounds
        confidence = max(min(confidence, self.config['max_confidence']), self.config['min_confidence'])
        
        self.logger.info(f"Calculated confidence: {confidence:.2f}")
        
        return confidence
    
    def _calculate_historical_accuracy(self, symbol, exchange, direction):
        """
        Calculate confidence based on historical prediction accuracy.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            direction (str): Prediction direction
            
        Returns:
            float: Historical accuracy confidence
        """
        try:
            # Get prediction history with performance data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config['backtest_days'] * 2)  # Double to ensure enough data
            
            query = {
                'symbol': symbol,
                'exchange': exchange,
                'prediction': direction,
                'date': {'$gte': start_date, '$lte': end_date},
                'performance': {'$exists': True}
            }
            
            cursor = self.db.predictions_collection.find(query)
            predictions = list(cursor)
            
            if len(predictions) < 5:  # Minimum for statistical relevance
                # Try broader query without direction constraint
                query.pop('prediction')
                cursor = self.db.predictions_collection.find(query)
                predictions = list(cursor)
                
                if len(predictions) < 5:
                    return 0.5  # Default neutral confidence
            
            # Calculate accuracy
            correct_count = sum(1 for p in predictions if p.get('performance', {}).get('is_correct', False))
            accuracy = correct_count / len(predictions)
            
            # Apply sigmoid transformation to convert accuracy to confidence
            # Accuracy of 0.5 (random) should give confidence of 0.5, higher accuracy gives higher confidence
            confidence = 1 / (1 + np.exp(-10 * (accuracy - 0.5)))
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating historical accuracy: {e}")
            return 0.5
    
    def _calculate_market_confidence(self, symbol, exchange, direction):
        """
        Calculate confidence based on market regime and conditions.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            direction (str): Prediction direction
            
        Returns:
            float: Market confidence
        """
        try:
            # Get current market regime
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = today - timedelta(days=7)
            
            regime_query = {
                'symbol': symbol,
                'exchange': exchange,
                'date': {'$gte': week_ago}
            }
            
            regime_data = self.db.market_analysis_collection.find_one(
                regime_query, sort=[('date', -1)]
            )
            
            if not regime_data:
                return 0.6  # Default slightly above neutral
            
            # Extract regime information
            regime = regime_data.get('regime', 'unknown')
            trend = regime_data.get('trend_strength', 0.5)
            momentum = regime_data.get('momentum_regime', 'neutral')
            
            # Calculate confidence based on alignment with current regime
            confidence = 0.5  # Start neutral
            
            # Adjust based on regime
            if direction == 'up':
                if regime == 'bullish':
                    confidence += 0.2
                elif regime == 'bearish':
                    confidence -= 0.2
                elif regime == 'neutral':
                    confidence += 0.0
                
                # Adjust based on trend and momentum
                confidence += (trend - 0.5) * 0.2
                
                if momentum == 'positive':
                    confidence += 0.1
                elif momentum == 'negative':
                    confidence -= 0.1
                
            elif direction == 'down':
                if regime == 'bullish':
                    confidence -= 0.2
                elif regime == 'bearish':
                    confidence += 0.2
                elif regime == 'neutral':
                    confidence += 0.0
                
                # Adjust based on trend and momentum (inverted)
                confidence -= (trend - 0.5) * 0.2
                
                if momentum == 'positive':
                    confidence -= 0.1
                elif momentum == 'negative':
                    confidence += 0.1
            
            # Ensure in range [0, 1]
            confidence = max(0, min(1, confidence))
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating market confidence: {e}")
            return 0.5
    
    def _calculate_volatility_confidence(self, symbol, exchange):
        """
        Calculate confidence based on current volatility.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            float: Volatility confidence
        """
        try:
            # Get recent market data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 30 days for volatility calculation
            
            query = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': 'day',
                'timestamp': {'$gte': start_date, '$lte': end_date}
            }
            
            cursor = self.db.market_data_collection.find(query).sort('timestamp', 1)
            market_data = list(cursor)
            
            if len(market_data) < 10:  # Minimum for volatility calculation
                return 0.5  # Default neutral confidence
            
            # Calculate daily returns
            closes = [data.get('close', 0) for data in market_data]
            returns = np.diff(closes) / closes[:-1]
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns)
            
            # Normalize volatility to confidence
            # Higher volatility -> lower confidence
            # Use historical volatility distribution to calibrate
            avg_volatility = 0.015  # Approximate average daily volatility (1.5%)
            std_volatility = 0.01  # Standard deviation of volatility
            
            # Calculate z-score of current volatility
            volatility_z = (volatility - avg_volatility) / std_volatility
            
            # Convert to confidence using sigmoid function
            # Higher volatility (positive z) -> lower confidence
            confidence = 1 / (1 + np.exp(volatility_z))
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility confidence: {e}")
            return 0.5
    
    def _calculate_consistency_confidence(self, symbol, exchange, direction):
        """
        Calculate confidence based on prediction consistency across models.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            direction (str): Prediction direction
            
        Returns:
            float: Consistency confidence
        """
        try:
            # Get recent predictions from different models
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday = today - timedelta(days=1)
            
            query = {
                'symbol': symbol,
                'exchange': exchange,
                'date': {'$gte': yesterday},
                'sources': {'$exists': True}
            }
            
            cursor = self.db.predictions_collection.find(query)
            predictions = list(cursor)
            
            if not predictions:
                return 0.5  # Default neutral confidence
            
            # Get prediction counts by direction
            direction_counts = {}
            total_predictions = 0
            
            for pred in predictions:
                pred_direction = pred.get('prediction')
                if pred_direction not in direction_counts:
                    direction_counts[pred_direction] = 0
                    
                direction_counts[pred_direction] += 1
                total_predictions += 1
            
            # Calculate agreement ratio
            if direction in direction_counts and total_predictions > 0:
                agreement_ratio = direction_counts[direction] / total_predictions
                
                # Apply sigmoid transformation for consistency confidence
                consistency = 1 / (1 + np.exp(-10 * (agreement_ratio - 0.5)))
                
                return consistency
            else:
                return 0.5  # Default if no agreement data
            
        except Exception as e:
            self.logger.error(f"Error calculating consistency confidence: {e}")
            return 0.5
    
    def adjust_prediction_confidence(self, prediction):
        """
        Adjust the confidence of an existing prediction.
        
        Args:
            prediction (dict): Prediction to adjust
            
        Returns:
            dict: Updated prediction with adjusted confidence
        """
        if not prediction:
            return None
            
        original_confidence = prediction.get('confidence', 0.5)
        
        # Calculate adjusted confidence
        adjusted_confidence = self.calculate_confidence(prediction, original_confidence)
        
        # Update prediction with new confidence
        updated_prediction = prediction.copy()
        updated_prediction['confidence'] = adjusted_confidence
        updated_prediction['confidence_components'] = {
            'original': original_confidence,
            'adjusted': adjusted_confidence,
            'adjustment_time': datetime.now()
        }
        
        # Update in database if has ID
        if '_id' in prediction:
            try:
                self.db.predictions_collection.update_one(
                    {'_id': prediction['_id']},
                    {'$set': {
                        'confidence': adjusted_confidence,
                        'confidence_components': updated_prediction['confidence_components']
                    }}
                )
                
                self.logger.info(f"Updated confidence for prediction {prediction['_id']}")
                
            except Exception as e:
                self.logger.error(f"Error updating prediction confidence: {e}")
        
        return updated_prediction
    
    def batch_adjust_confidence(self, date=None):
        """
        Batch adjust confidence for all predictions on a date.
        
        Args:
            date (datetime): Date to process (default: today)
            
        Returns:
            int: Number of predictions adjusted
        """
        date = date or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        next_day = date + timedelta(days=1)
        
        self.logger.info(f"Batch adjusting confidence for predictions on {date.strftime('%Y-%m-%d')}")
        
        # Get predictions for the date
        query = {
            'date': {'$gte': date, '$lt': next_day},
            'confidence_components': {'$exists': False}  # Only ones not already adjusted
        }
        
        cursor = self.db.predictions_collection.find(query)
        predictions = list(cursor)
        
        if not predictions:
            self.logger.info("No predictions found for adjustment")
            return 0
        
        adjusted_count = 0
        
        for prediction in predictions:
            try:
                self.adjust_prediction_confidence(prediction)
                adjusted_count += 1
            except Exception as e:
                self.logger.error(f"Error adjusting prediction {prediction.get('_id')}: {e}")
        
        self.logger.info(f"Adjusted confidence for {adjusted_count} predictions")
        
        return adjusted_count