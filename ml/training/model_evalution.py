# ml/training/model_evaluation.py (Session 46: Report Generation System)

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class ModelEvaluator:
    """
    Evaluates machine learning model performance.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the model evaluator.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("Model evaluator initialized")
    
    def evaluate_all_models(self, start_date=None, end_date=None):
        """
        Evaluate all active models.
        
        Args:
            start_date (datetime): Start date for evaluation
            end_date (datetime): End date for evaluation
            
        Returns:
            dict: Evaluation results
        """
        try:
            self.logger.info("Evaluating all models")
            
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now()
            
            if not start_date:
                # Use previous month
                start_date = end_date - timedelta(days=30)
            
            self.logger.info(f"Evaluation period: {start_date.date()} to {end_date.date()}")
            
            # Get all active models
            active_models = self._get_active_models()
            
            if not active_models:
                self.logger.info("No active models found")
                return {"status": "no_models", "count": 0}
            
            self.logger.info(f"Found {len(active_models)} active models")
            
            # Evaluate each model
            results = []
            
            for model in active_models:
                try:
                    model_id = model.get('model_id')
                    symbol = model.get('symbol')
                    exchange = model.get('exchange')
                    model_type = model.get('model_type')
                    
                    self.logger.info(f"Evaluating model {model_id} ({symbol}/{exchange}, {model_type})")
                    
                    # Evaluate model
                    evaluation = self.evaluate_model(
                        symbol=symbol,
                        exchange=exchange,
                        model_type=model_type,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Add to results
                    results.append({
                        'model_id': model_id,
                        'symbol': symbol,
                        'exchange': exchange,
                        'model_type': model_type,
                        'evaluation': evaluation
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating model {model.get('model_id', 'unknown')}: {e}")
                    
                    # Add error to results
                    results.append({
                        'model_id': model.get('model_id', 'unknown'),
                        'symbol': model.get('symbol', 'unknown'),
                        'exchange': model.get('exchange', 'unknown'),
                        'model_type': model.get('model_type', 'unknown'),
                        'evaluation': {
                            'status': 'error',
                            'error': str(e)
                        }
                    })
            
            # Create evaluation results
            evaluation_results = {
                'status': 'success',
                'count': len(results),
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                'results': results
            }
            
            # Save to database
            # Save to database
            if self.db:
                evaluation_id = f"model_evaluation_{end_date.strftime('%Y%m%d')}"
                
                self.db.model_evaluations.update_one(
                    {'evaluation_id': evaluation_id},
                    {'$set': {
                        'evaluation_id': evaluation_id,
                        'date': end_date,
                        'results': evaluation_results
                    }},
                    upsert=True
                )
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating models: {e}")
            return {"status": "error", "error": str(e)}
    
    def evaluate_model(self, symbol, exchange, model_type, start_date=None, end_date=None):
        """
        Evaluate a specific model.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            model_type (str): Model type
            start_date (datetime): Start date for evaluation
            end_date (datetime): End date for evaluation
            
        Returns:
            dict: Evaluation results
        """
        try:
            self.logger.info(f"Evaluating {model_type} model for {symbol}/{exchange}")
            
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now()
            
            if not start_date:
                # Use previous month
                start_date = end_date - timedelta(days=30)
            
            # Get predictions and actuals
            predictions = self._get_model_predictions(
                symbol=symbol,
                exchange=exchange,
                model_type=model_type,
                start_date=start_date,
                end_date=end_date
            )
            
            if not predictions:
                self.logger.info(f"No validated predictions found for {symbol}/{exchange} {model_type}")
                return {
                    "status": "no_predictions",
                    "count": 0
                }
            
            # Calculate performance metrics
            correct_predictions = [p for p in predictions if p.get('correct', False)]
            total_predictions = len(predictions)
            correct_count = len(correct_predictions)
            
            if total_predictions > 0:
                accuracy = (correct_count / total_predictions) * 100
            else:
                accuracy = 0
            
            # Analyze by direction
            up_predictions = [p for p in predictions if p.get('prediction') == 'up']
            down_predictions = [p for p in predictions if p.get('prediction') == 'down']
            
            up_correct = [p for p in up_predictions if p.get('correct', False)]
            down_correct = [p for p in down_predictions if p.get('correct', False)]
            
            up_accuracy = (len(up_correct) / len(up_predictions) * 100) if up_predictions else 0
            down_accuracy = (len(down_correct) / len(down_predictions) * 100) if down_predictions else 0
            
            # Analyze by confidence
            high_conf = [p for p in predictions if p.get('confidence', 0) >= 0.7]
            med_conf = [p for p in predictions if 0.5 <= p.get('confidence', 0) < 0.7]
            low_conf = [p for p in predictions if p.get('confidence', 0) < 0.5]
            
            high_correct = [p for p in high_conf if p.get('correct', False)]
            med_correct = [p for p in med_conf if p.get('correct', False)]
            low_correct = [p for p in low_conf if p.get('correct', False)]
            
            high_accuracy = (len(high_correct) / len(high_conf) * 100) if high_conf else 0
            med_accuracy = (len(med_correct) / len(med_conf) * 100) if med_conf else 0
            low_accuracy = (len(low_correct) / len(low_conf) * 100) if low_conf else 0
            
            # Calculate profit/loss if used for trading
            pnl = self._calculate_trading_pnl(predictions)
            
            # Create evaluation
            evaluation = {
                'status': 'success',
                'model_id': f"{symbol}_{exchange}_{model_type}",
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                'predictions': {
                    'total': total_predictions,
                    'correct': correct_count,
                    'accuracy': accuracy
                },
                'by_direction': {
                    'up': {
                        'total': len(up_predictions),
                        'correct': len(up_correct),
                        'accuracy': up_accuracy
                    },
                    'down': {
                        'total': len(down_predictions),
                        'correct': len(down_correct),
                        'accuracy': down_accuracy
                    }
                },
                'by_confidence': {
                    'high': {
                        'total': len(high_conf),
                        'correct': len(high_correct),
                        'accuracy': high_accuracy
                    },
                    'medium': {
                        'total': len(med_conf),
                        'correct': len(med_correct),
                        'accuracy': med_accuracy
                    },
                    'low': {
                        'total': len(low_conf),
                        'correct': len(low_correct),
                        'accuracy': low_accuracy
                    }
                },
                'trading_simulation': pnl
            }
            
            # Save to database
            if self.db:
                model_id = f"{symbol}_{exchange}_{model_type}"
                evaluation_id = f"{model_id}_{end_date.strftime('%Y%m%d')}"
                
                self.db.model_performance.update_one(
                    {'evaluation_id': evaluation_id},
                    {'$set': {
                        'evaluation_id': evaluation_id,
                        'model_id': model_id,
                        'symbol': symbol,
                        'exchange': exchange,
                        'model_type': model_type,
                        'date': end_date,
                        'results': evaluation
                    }},
                    upsert=True
                )
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return {"status": "error", "error": str(e)}
    
    def check_model_drift(self, period_months=3):
        """
        Check for model drift by comparing recent performance to historical.
        
        Args:
            period_months (int): Number of months for comparison
            
        Returns:
            dict: Drift analysis results
        """
        try:
            self.logger.info(f"Checking model drift over {period_months} months")
            
            # Get dates
            now = datetime.now()
            recent_start = now - timedelta(days=30)  # Last month
            historical_start = now - timedelta(days=30 * period_months)  # Last N months
            
            # Get all active models
            active_models = self._get_active_models()
            
            if not active_models:
                self.logger.info("No active models found")
                return {"status": "no_models", "count": 0}
            
            # Check drift for each model
            drift_results = []
            
            for model in active_models:
                try:
                    model_id = model.get('model_id')
                    symbol = model.get('symbol')
                    exchange = model.get('exchange')
                    model_type = model.get('model_type')
                    
                    # Get recent performance
                    recent_eval = self.evaluate_model(
                        symbol=symbol,
                        exchange=exchange,
                        model_type=model_type,
                        start_date=recent_start,
                        end_date=now
                    )
                    
                    if recent_eval.get('status') != 'success' or recent_eval.get('predictions', {}).get('total', 0) < 10:
                        # Skip if not enough recent predictions
                        continue
                    
                    # Get historical performance
                    historical_eval = self.evaluate_model(
                        symbol=symbol,
                        exchange=exchange,
                        model_type=model_type,
                        start_date=historical_start,
                        end_date=recent_start
                    )
                    
                    if historical_eval.get('status') != 'success' or historical_eval.get('predictions', {}).get('total', 0) < 10:
                        # Skip if not enough historical predictions
                        continue
                    
                    # Calculate accuracy drift
                    recent_accuracy = recent_eval.get('predictions', {}).get('accuracy', 0)
                    historical_accuracy = historical_eval.get('predictions', {}).get('accuracy', 0)
                    
                    accuracy_drift = recent_accuracy - historical_accuracy
                    
                    # Calculate relative drift
                    if historical_accuracy > 0:
                        relative_drift = (accuracy_drift / historical_accuracy) * 100
                    else:
                        relative_drift = 0
                    
                    # Determine drift status
                    if abs(accuracy_drift) < 5:
                        drift_status = 'stable'
                    elif accuracy_drift <= -10:
                        drift_status = 'severe_degradation'
                    elif accuracy_drift < 0:
                        drift_status = 'mild_degradation'
                    elif accuracy_drift >= 10:
                        drift_status = 'significant_improvement'
                    else:
                        drift_status = 'mild_improvement'
                    
                    # Create drift result
                    drift_result = {
                        'model_id': model_id,
                        'symbol': symbol,
                        'exchange': exchange,
                        'model_type': model_type,
                        'recent': {
                            'period': {
                                'start': recent_start,
                                'end': now
                            },
                            'predictions': recent_eval.get('predictions', {}).get('total', 0),
                            'accuracy': recent_accuracy
                        },
                        'historical': {
                            'period': {
                                'start': historical_start,
                                'end': recent_start
                            },
                            'predictions': historical_eval.get('predictions', {}).get('total', 0),
                            'accuracy': historical_accuracy
                        },
                        'drift': {
                            'accuracy_change': accuracy_drift,
                            'relative_change': relative_drift,
                            'status': drift_status
                        }
                    }
                    
                    drift_results.append(drift_result)
                    
                except Exception as e:
                    self.logger.error(f"Error checking drift for model {model.get('model_id', 'unknown')}: {e}")
            
            # Create drift analysis
            drift_analysis = {
                'status': 'success',
                'count': len(drift_results),
                'period_months': period_months,
                'results': drift_results
            }
            
            # Save to database
            if self.db:
                drift_id = f"model_drift_{now.strftime('%Y%m%d')}"
                
                self.db.model_drift.update_one(
                    {'drift_id': drift_id},
                    {'$set': {
                        'drift_id': drift_id,
                        'date': now,
                        'period_months': period_months,
                        'results': drift_analysis
                    }},
                    upsert=True
                )
            
            return drift_analysis
            
        except Exception as e:
            self.logger.error(f"Error checking model drift: {e}")
            return {"status": "error", "error": str(e)}
    
    def identify_retraining_candidates(self, accuracy_threshold=0.55, drift_threshold=0.1):
        """
        Identify models that need retraining.
        
        Args:
            accuracy_threshold (float): Minimum acceptable accuracy (0-1)
            drift_threshold (float): Maximum acceptable negative drift
            
        Returns:
            list: Models that need retraining
        """
        try:
            self.logger.info(f"Identifying retraining candidates (accuracy < {accuracy_threshold*100}% or drift < {-drift_threshold*100}%)")
            
            # Check model drift
            drift_analysis = self.check_model_drift(period_months=3)
            
            if drift_analysis.get('status') != 'success':
                return []
            
            # Identify candidates
            candidates = []
            
            for result in drift_analysis.get('results', []):
                model_id = result.get('model_id')
                symbol = result.get('symbol')
                exchange = result.get('exchange')
                model_type = result.get('model_type')
                
                recent_accuracy = result.get('recent', {}).get('accuracy', 0) / 100  # Convert to 0-1 scale
                accuracy_drift = result.get('drift', {}).get('accuracy_change', 0) / 100  # Convert to 0-1 scale
                
                # Check if model needs retraining
                needs_retraining = False
                reason = []
                
                if recent_accuracy < accuracy_threshold:
                    needs_retraining = True
                    reason.append(f"Low accuracy ({recent_accuracy:.2%})")
                
                if accuracy_drift < -drift_threshold:
                    needs_retraining = True
                    reason.append(f"Significant drift ({accuracy_drift:.2%})")
                
                if needs_retraining:
                    candidates.append({
                        'model_id': model_id,
                        'symbol': symbol,
                        'exchange': exchange,
                        'model_type': model_type,
                        'current_accuracy': recent_accuracy,
                        'accuracy_drift': accuracy_drift,
                        'reason': ', '.join(reason)
                    })
            
            self.logger.info(f"Identified {len(candidates)} models for retraining")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error identifying retraining candidates: {e}")
            return []
    
    def _get_active_models(self):
        """
        Get all active models.
        
        Returns:
            list: Active models
        """
        try:
            if not self.db:
                return []
            
            # Query database for active models
            cursor = self.db.ml_models.find({
                'status': 'active'
            })
            
            models = list(cursor)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error getting active models: {e}")
            return []
    
    def _get_model_predictions(self, symbol, exchange, model_type, start_date, end_date):
        """
        Get validated predictions for a model.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            model_type (str): Model type
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            list: Validated predictions
        """
        try:
            if not self.db:
                return []
            
            # Query database for validated predictions
            cursor = self.db.predictions.find({
                'symbol': symbol,
                'exchange': exchange,
                'prediction_type': model_type,
                'for_date': {'$gte': start_date, '$lt': end_date},
                'validated': True
            })
            
            predictions = list(cursor)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error getting model predictions: {e}")
            return []
    
    def _calculate_trading_pnl(self, predictions):
        """
        Calculate P&L from trading based on predictions.
        
        Args:
            predictions (list): List of validated predictions
            
        Returns:
            dict: P&L calculations
        """
        try:
            if not predictions:
                return {
                    'trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'avg_return': 0,
                    'profit_factor': 0
                }
            
            # Sort predictions by date
            predictions.sort(key=lambda x: x.get('for_date', datetime.now()))
            
            # Initialize variables
            initial_capital = 100000  # 1 lakh rupees
            capital = initial_capital
            position_size = 0.05  # 5% of capital
            
            trades = []
            total_return = 0
            winning_trades = 0
            losing_trades = 0
            total_profit = 0
            total_loss = 0
            
            # Simulate trading
            for prediction in predictions:
                # Only trade high confidence predictions
                if prediction.get('confidence', 0) < 0.65:
                    continue
                
                # Get prediction details
                direction = prediction.get('prediction')
                actual_change = prediction.get('actual_change', 0)
                
                # Calculate trade return
                trade_size = capital * position_size
                
                # Long trades for 'up' predictions, short for 'down'
                if direction == 'up':
                    trade_return = actual_change * trade_size / 100
                else:  # 'down'
                    trade_return = -actual_change * trade_size / 100
                
                # Apply slippage and commission
                slippage = trade_size * 0.001  # 0.1% slippage
                commission = trade_size * 0.0005  # 0.05% commission
                
                net_return = trade_return - slippage - commission
                
                # Update capital
                capital += net_return
                
                # Record trade
                trade = {
                    'date': prediction.get('for_date'),
                    'symbol': prediction.get('symbol'),
                    'direction': direction,
                    'confidence': prediction.get('confidence'),
                    'actual_change': actual_change,
                    'trade_return': trade_return,
                    'net_return': net_return,
                    'return_pct': (net_return / trade_size) * 100 if trade_size > 0 else 0
                }
                
                trades.append(trade)
                
                # Update statistics
                total_return += net_return
                
                if net_return > 0:
                    winning_trades += 1
                    total_profit += net_return
                else:
                    losing_trades += 1
                    total_loss += abs(net_return)
            
            # Calculate performance metrics
            num_trades = len(trades)
            
            if num_trades > 0:
                win_rate = (winning_trades / num_trades) * 100
                avg_return = total_return / num_trades
            else:
                win_rate = 0
                avg_return = 0
            
            if total_loss > 0:
                profit_factor = total_profit / total_loss
            else:
                profit_factor = float('inf') if total_profit > 0 else 0
            
            final_return_pct = ((capital - initial_capital) / initial_capital) * 100
            
            # Create trading simulation results
            simulation_results = {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'trades': num_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'total_return_pct': final_return_pct,
                'avg_return': avg_return,
                'profit_factor': profit_factor
            }
            
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"Error calculating trading P&L: {e}")
            return {
                'trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'avg_return': 0,
                'profit_factor': 0,
                'error': str(e)
            }