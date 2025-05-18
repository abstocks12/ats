# automation/prediction_validator.py (Session 45: End-of-Day Analysis)

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class PredictionValidator:
    """
    Validates prediction accuracy and performance.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the prediction validator.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("Prediction validator initialized")
    
    def validate_daily_predictions(self):
        """
        Validate yesterday's predictions for today.
        
        Returns:
            dict: Validation results
        """
        try:
            self.logger.info("Validating daily predictions")
            
            # Get dates
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday = today - timedelta(days=1)
            
            # Get predictions made yesterday for today
            predictions = self._get_predictions(
                prediction_date=yesterday,
                target_date=today,
                prediction_type="daily"
            )
            
            if not predictions:
                self.logger.info("No daily predictions to validate")
                return {"status": "no_predictions", "count": 0}
            
            self.logger.info(f"Found {len(predictions)} daily predictions to validate")
            
            # Validate predictions
            results = self._validate_predictions(predictions)
            
            # Log results
            accuracy = results.get('accuracy', 0)
            correct_count = results.get('correct_count', 0)
            total_count = results.get('total_count', 0)
            
            self.logger.info(f"Daily prediction accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating daily predictions: {e}")
            return {"status": "error", "error": str(e)}
    
    def validate_overnight_gap_predictions(self):
        """
        Validate yesterday's overnight gap predictions.
        
        Returns:
            dict: Validation results
        """
        try:
            self.logger.info("Validating overnight gap predictions")
            
            # Get dates
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday = today - timedelta(days=1)
            
            # Get predictions made yesterday for today's open
            predictions = self._get_predictions(
                prediction_date=yesterday,
                target_date=today,
                prediction_type="overnight_gap"
            )
            
            if not predictions:
                self.logger.info("No overnight gap predictions to validate")
                return {"status": "no_predictions", "count": 0}
            
            self.logger.info(f"Found {len(predictions)} overnight gap predictions to validate")
            
            # Validate predictions
            results = self._validate_predictions(predictions)
            
            # Log results
            accuracy = results.get('accuracy', 0)
            correct_count = results.get('correct_count', 0)
            total_count = results.get('total_count', 0)
            
            self.logger.info(f"Overnight gap prediction accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating overnight gap predictions: {e}")
            return {"status": "error", "error": str(e)}
    
    def validate_intraday_predictions(self):
        """
        Validate today's intraday predictions.
        
        Returns:
            dict: Validation results
        """
        try:
            self.logger.info("Validating intraday predictions")
            
            # Get dates
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            # Get predictions made today with intraday timeframe
            predictions = self._get_predictions(
                prediction_date=today,
                target_date=today,
                prediction_type="intraday"
            )
            
            if not predictions:
                self.logger.info("No intraday predictions to validate")
                return {"status": "no_predictions", "count": 0}
            
            self.logger.info(f"Found {len(predictions)} intraday predictions to validate")
            
            # Validate predictions
            results = self._validate_predictions(predictions)
            
            # Log results
            accuracy = results.get('accuracy', 0)
            correct_count = results.get('correct_count', 0)
            total_count = results.get('total_count', 0)
            
            self.logger.info(f"Intraday prediction accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating intraday predictions: {e}")
            return {"status": "error", "error": str(e)}
    
    def analyze_prediction_performance(self, days=30):
        """
        Analyze prediction performance over specified period.
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            dict: Performance analysis
        """
        try:
            self.logger.info(f"Analyzing prediction performance over {days} days")
            
            # Get dates
            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = end_date - timedelta(days=days)
            
            # Get validated predictions
            validated_predictions = self._get_validated_predictions(
                start_date=start_date,
                end_date=end_date
            )
            
            if not validated_predictions:
                self.logger.info("No validated predictions found for analysis")
                return {"status": "no_data", "count": 0}
            
            self.logger.info(f"Found {len(validated_predictions)} validated predictions for analysis")
            
            # Calculate overall accuracy
            correct_predictions = [p for p in validated_predictions if p.get('correct', False)]
            total_predictions = len(validated_predictions)
            correct_count = len(correct_predictions)
            
            if total_predictions > 0:
                overall_accuracy = (correct_count / total_predictions) * 100
            else:
                overall_accuracy = 0
            
            # Analyze by prediction type
            prediction_types = {}
            
            for p_type in ['daily', 'overnight_gap', 'intraday']:
                type_predictions = [p for p in validated_predictions if p.get('prediction_type') == p_type]
                type_correct = [p for p in type_predictions if p.get('correct', False)]
                
                prediction_types[p_type] = {
                    'count': len(type_predictions),
                    'correct': len(type_correct),
                    'accuracy': (len(type_correct) / len(type_predictions) * 100) if len(type_predictions) > 0 else 0
                }
            
            # Analyze by direction
            directions = {}
            
            for direction in ['up', 'down']:
                dir_predictions = [p for p in validated_predictions if p.get('prediction') == direction]
                dir_correct = [p for p in dir_predictions if p.get('correct', False)]
                
                directions[direction] = {
                    'count': len(dir_predictions),
                    'correct': len(dir_correct),
                    'accuracy': (len(dir_correct) / len(dir_predictions) * 100) if len(dir_predictions) > 0 else 0
                }
            
            # Analyze by confidence level
            confidence_levels = {
                'high': {'min': 0.7, 'max': 1.0},
                'medium': {'min': 0.5, 'max': 0.7},
                'low': {'min': 0.0, 'max': 0.5}
            }
            
            confidence_analysis = {}
            
            for level, thresholds in confidence_levels.items():
                level_predictions = [
                    p for p in validated_predictions 
                    if thresholds['min'] <= p.get('confidence', 0) < thresholds['max']
                ]
                level_correct = [p for p in level_predictions if p.get('correct', False)]
                
                confidence_analysis[level] = {
                    'count': len(level_predictions),
                    'correct': len(level_correct),
                    'accuracy': (len(level_correct) / len(level_predictions) * 100) if len(level_predictions) > 0 else 0
                }
            
            # Analyze trends over time
            # Group by date
            date_groups = {}
            
            for prediction in validated_predictions:
                date_str = prediction.get('for_date', datetime.now()).strftime('%Y-%m-%d')
                
                if date_str not in date_groups:
                    date_groups[date_str] = []
                
                date_groups[date_str].append(prediction)
            
            # Calculate daily accuracy
            daily_accuracy = []
            
            for date_str, predictions in sorted(date_groups.items()):
                correct = sum(1 for p in predictions if p.get('correct', False))
                total = len(predictions)
                
                if total > 0:
                    accuracy = (correct / total) * 100
                else:
                    accuracy = 0
                
                daily_accuracy.append({
                    'date': date_str,
                    'total': total,
                    'correct': correct,
                    'accuracy': accuracy
                })
            
            # Calculate top performing symbols
            symbols = {}
            
            for prediction in validated_predictions:
                symbol = prediction.get('symbol', 'unknown')
                
                if symbol not in symbols:
                    symbols[symbol] = {
                        'total': 0,
                        'correct': 0
                    }
                
                symbols[symbol]['total'] += 1
                
                if prediction.get('correct', False):
                    symbols[symbol]['correct'] += 1
            
            # Calculate accuracy and filter symbols with at least 5 predictions
            symbol_accuracy = []
            
            for symbol, data in symbols.items():
                if data['total'] >= 5:
                    accuracy = (data['correct'] / data['total']) * 100
                    
                    symbol_accuracy.append({
                        'symbol': symbol,
                        'total': data['total'],
                        'correct': data['correct'],
                        'accuracy': accuracy
                    })
            
            # Sort by accuracy (descending)
            symbol_accuracy.sort(key=lambda x: x['accuracy'], reverse=True)
            
            # Top 10 symbols
            top_symbols = symbol_accuracy[:10]
            
            # Bottom 10 symbols
            bottom_symbols = symbol_accuracy[-10:] if len(symbol_accuracy) > 10 else []
            
            # Create performance analysis
            performance_analysis = {
                'period': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'days': days
                },
                'overall': {
                    'total_predictions': total_predictions,
                    'correct_predictions': correct_count,
                    'accuracy': overall_accuracy
                },
                'by_type': prediction_types,
                'by_direction': directions,
                'by_confidence': confidence_analysis,
                'daily_trend': daily_accuracy,
                'top_symbols': top_symbols,
                'bottom_symbols': bottom_symbols
            }
            
            # Save analysis to database
            if self.db:
                analysis_id = f"prediction_performance_{end_date.strftime('%Y%m%d')}"
                
                self.db.prediction_analysis.update_one(
                    {'analysis_id': analysis_id},
                    {'$set': {
                        'analysis_id': analysis_id,
                        'date': end_date,
                        'period_days': days,
                        'results': performance_analysis
                    }},
                    upsert=True
                )
            
            return performance_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing prediction performance: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_confidence_adjustments(self):
        """
        Generate confidence adjustment factors based on historical performance.
        
        Returns:
            dict: Confidence adjustment factors
        """
        try:
            self.logger.info("Generating confidence adjustment factors")
            
            # Get 60 days of validated predictions
            performance = self.analyze_prediction_performance(days=60)
            
            if not performance or performance.get('status') == 'no_data':
                self.logger.info("Insufficient data to generate confidence adjustments")
                return {
                    'status': 'insufficient_data',
                    'default_factor': 1.0,
                    'adjustments': {}
                }
            
            # Calculate adjustment factors
            
            # 1. Type-based adjustments
            type_adjustments = {}
            overall_accuracy = performance['overall']['accuracy'] / 100.0  # Convert to 0-1 scale
            
            if overall_accuracy <= 0:
                overall_accuracy = 0.5  # Default to 50% if no data
            
            for p_type, data in performance['by_type'].items():
                type_accuracy = data['accuracy'] / 100.0  # Convert to 0-1 scale
                
                if type_accuracy <= 0:
                    type_accuracy = 0.5  # Default to 50% if no data
                
                # Calculate adjustment factor (normalize around overall accuracy)
                if overall_accuracy > 0:
                    adjustment = type_accuracy / overall_accuracy
                else:
                    adjustment = 1.0
                
                # Cap adjustment between 0.5 and 1.5
                adjustment = max(0.5, min(1.5, adjustment))
                
                type_adjustments[p_type] = adjustment
            
            # 2. Direction-based adjustments
            direction_adjustments = {}
            
            for direction, data in performance['by_direction'].items():
                dir_accuracy = data['accuracy'] / 100.0  # Convert to 0-1 scale
                
                if dir_accuracy <= 0:
                    dir_accuracy = 0.5  # Default to 50% if no data
                
                # Calculate adjustment factor (normalize around overall accuracy)
                if overall_accuracy > 0:
                    adjustment = dir_accuracy / overall_accuracy
                else:
                    adjustment = 1.0
                
                # Cap adjustment between 0.5 and 1.5
                adjustment = max(0.5, min(1.5, adjustment))
                
                direction_adjustments[direction] = adjustment
            
            # 3. Symbol-based adjustments
            symbol_adjustments = {}
            
            for symbol_data in performance['top_symbols'] + performance['bottom_symbols']:
                symbol = symbol_data['symbol']
                sym_accuracy = symbol_data['accuracy'] / 100.0  # Convert to 0-1 scale
                
                if sym_accuracy <= 0:
                    sym_accuracy = 0.5  # Default to 50% if no data
                
                # Calculate adjustment factor (normalize around overall accuracy)
                if overall_accuracy > 0:
                    adjustment = sym_accuracy / overall_accuracy
                else:
                    adjustment = 1.0
                
                # Cap adjustment between 0.5 and 1.5
                adjustment = max(0.5, min(1.5, adjustment))
                
                symbol_adjustments[symbol] = adjustment
            
            # Create adjustment factors
            adjustment_factors = {
                'status': 'success',
                'default_factor': 1.0,
                'overall_accuracy': overall_accuracy,
                'adjustments': {
                    'by_type': type_adjustments,
                    'by_direction': direction_adjustments,
                    'by_symbol': symbol_adjustments
                }
            }
            
            # Save to database
            if self.db:
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                self.db.confidence_adjustments.update_one(
                    {'date': today},
                    {'$set': {
                        'date': today,
                        'factors': adjustment_factors
                    }},
                    upsert=True
                )
            
            return adjustment_factors
            
        except Exception as e:
            self.logger.error(f"Error generating confidence adjustments: {e}")
            return {"status": "error", "error": str(e)}
    
    def _get_predictions(self, prediction_date, target_date, prediction_type):
        """
        Get predictions from database.
        
        Args:
            prediction_date (datetime): Date when prediction was made
            target_date (datetime): Date for which prediction was made
            prediction_type (str): Type of prediction
            
        Returns:
            list: List of predictions
        """
        try:
            if not self.db:
                return []
            
            next_date = prediction_date + timedelta(days=1)
            next_target = target_date + timedelta(days=1)
            
            # Query database
            cursor = self.db.predictions.find({
                'date': {'$gte': prediction_date, '$lt': next_date},
                'for_date': {'$gte': target_date, '$lt': next_target},
                'prediction_type': prediction_type,
                'validated': {'$ne': True}  # Only get unvalidated predictions
            })
            
            predictions = list(cursor)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error getting predictions: {e}")
            return []
    
    def _get_validated_predictions(self, start_date, end_date):
        """
        Get validated predictions from database.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            list: List of validated predictions
        """
        try:
            if not self.db:
                return []
            
            # Query database for validated predictions
            cursor = self.db.predictions.find({
                'for_date': {'$gte': start_date, '$lt': end_date},
                'validated': True
            })
            
            predictions = list(cursor)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error getting validated predictions: {e}")
            return []
    
    def _validate_predictions(self, predictions):
        """
        Validate a list of predictions.
        
        Args:
            predictions (list): List of predictions to validate
            
        Returns:
            dict: Validation results
        """
        try:
            if not predictions:
                return {
                    'status': 'no_predictions',
                    'total_count': 0,
                    'correct_count': 0,
                    'accuracy': 0
                }
            
            # Initialize counters
            total_count = len(predictions)
            correct_count = 0
            validation_details = []
            
            # Validate each prediction
            for prediction in predictions:
                result = self._validate_single_prediction(prediction)
                validation_details.append(result)
                
                if result.get('correct', False):
                    correct_count += 1
            
            # Calculate accuracy
            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            
            # Create validation results
            validation_results = {
                'status': 'success',
                'total_count': total_count,
                'correct_count': correct_count,
                'accuracy': accuracy,
                'details': validation_details
            }
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating predictions: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'total_count': len(predictions) if predictions else 0,
                'correct_count': 0,
                'accuracy': 0
            }
    
    def _validate_single_prediction(self, prediction):
        """
        Validate a single prediction.
        
        Args:
            prediction (dict): Prediction to validate
            
        Returns:
            dict: Validation result
        """
        try:
            prediction_id = prediction.get('_id')
            symbol = prediction.get('symbol')
            exchange = prediction.get('exchange')
            prediction_type = prediction.get('prediction_type', 'daily')
            predicted_direction = prediction.get('prediction', 'unknown')
            target_date = prediction.get('for_date', datetime.now())
            
            # Get actual market data
            actual_data = self._get_market_data(symbol, exchange, target_date)
            
            if not actual_data:
                self.logger.warning(f"No market data found for {symbol}/{exchange} on {target_date}")
                
                if self.db:
                    # Mark as invalid
                    self.db.predictions.update_one(
                        {'_id': prediction_id},
                        {'$set': {
                            'validated': True,
                            'validation_status': 'no_data',
                            'validated_at': datetime.now()
                        }}
                    )
                
                return {
                    'prediction_id': prediction_id,
                    'symbol': symbol,
                    'exchange': exchange,
                    'predicted': predicted_direction,
                    'actual': 'unknown',
                    'correct': False,
                    'validation_status': 'no_data'
                }
            
            # Determine actual direction based on prediction type
            if prediction_type == 'daily':
                # Compare close to previous close
                if actual_data['close'] > actual_data['prev_close']:
                    actual_direction = 'up'
                else:
                    actual_direction = 'down'
                
                # Calculate actual change
                if actual_data['prev_close'] > 0:
                    actual_change = ((actual_data['close'] - actual_data['prev_close']) / actual_data['prev_close']) * 100
                else:
                    actual_change = 0
                
            elif prediction_type == 'overnight_gap':
                # Compare open to previous close
                if actual_data['open'] > actual_data['prev_close']:
                    actual_direction = 'up'
                else:
                    actual_direction = 'down'
                
                # Calculate actual change
                if actual_data['prev_close'] > 0:
                    actual_change = ((actual_data['open'] - actual_data['prev_close']) / actual_data['prev_close']) * 100
                else:
                    actual_change = 0
                
            elif prediction_type == 'intraday':
                # Compare close to open
                if actual_data['close'] > actual_data['open']:
                    actual_direction = 'up'
                else:
                    actual_direction = 'down'
                
                # Calculate actual change
                if actual_data['open'] > 0:
                    actual_change = ((actual_data['close'] - actual_data['open']) / actual_data['open']) * 100
                else:
                    actual_change = 0
                
            else:
                # Default: compare close to previous close
                if actual_data['close'] > actual_data['prev_close']:
                    actual_direction = 'up'
                else:
                    actual_direction = 'down'
                
                # Calculate actual change
                if actual_data['prev_close'] > 0:
                    actual_change = ((actual_data['close'] - actual_data['prev_close']) / actual_data['prev_close']) * 100
                else:
                    actual_change = 0
            
            # Check if prediction was correct
            is_correct = (predicted_direction == actual_direction)
            
            # Update prediction in database
            if self.db:
                self.db.predictions.update_one(
                    {'_id': prediction_id},
                    {'$set': {
                        'validated': True,
                        'actual': actual_direction,
                        'actual_change': actual_change,
                        'correct': is_correct,
                        'validated_at': datetime.now()
                    }}
                )
            
            # Create validation result
            validation_result = {
                'prediction_id': prediction_id,
                'symbol': symbol,
                'exchange': exchange,
                'predicted': predicted_direction,
                'actual': actual_direction,
                'actual_change': actual_change,
                'correct': is_correct,
                'validation_status': 'validated'
            }
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating prediction: {e}")
            
            # Try to mark as failed in database
            try:
                if self.db and prediction.get('_id'):
                    self.db.predictions.update_one(
                        {'_id': prediction.get('_id')},
                        {'$set': {
                            'validated': True,
                            'validation_status': 'error',
                            'validation_error': str(e),
                            'validated_at': datetime.now()
                        }}
                    )
            except:
                pass
            
            return {
                'prediction_id': prediction.get('_id', 'unknown'),
                'symbol': prediction.get('symbol', 'unknown'),
                'exchange': prediction.get('exchange', 'unknown'),
                'predicted': prediction.get('prediction', 'unknown'),
                'actual': 'unknown',
                'correct': False,
                'validation_status': 'error',
                'error': str(e)
            }
    
    def _get_market_data(self, symbol, exchange, date):
        """
        Get market data for a specific date.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            date (datetime): Date
            
        Returns:
            dict: Market data
        """
        try:
            if not self.db:
                return None
            
            next_date = date + timedelta(days=1)
            
            # Get today's market data
            data = self.db.market_data.find_one({
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': 'day',
                'timestamp': {'$gte': date, '$lt': next_date}
            })
            
            if not data:
                return None
            
            # Get previous day's market data
            prev_date = date - timedelta(days=1)
            
            prev_data = self.db.market_data.find_one({
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': 'day',
                'timestamp': {'$gte': prev_date, '$lt': date}
            })
            
            prev_close = prev_data.get('close', 0) if prev_data else 0
            
            # Create market data
            market_data = {
                'open': data.get('open', 0),
                'high': data.get('high', 0),
                'low': data.get('low', 0),
                'close': data.get('close', 0),
                'volume': data.get('volume', 0),
                'prev_close': prev_close
            }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}/{exchange} on {date}: {e}")
            return None