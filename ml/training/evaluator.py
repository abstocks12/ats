# ml/training/evaluator.py
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

class ModelEvaluator:
    """Model evaluation and validation."""
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the model evaluator.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'test_size': 0.2,
            'walk_forward_window': 30,  # Days for walk-forward validation
            'walk_forward_step': 5,  # Step size for walk-forward windows
            'baseline_strategy': 'buy_and_hold',  # Baseline for comparison
            'trading_costs': 0.0001,  # Transaction cost for simulation
            'capital': 10000,  # Initial capital for simulation
            'risk_free_rate': 0.02  # Annual risk-free rate for Sharpe ratio
        }
    
    def set_config(self, config):
        """
        Set evaluation configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated evaluation configuration: {self.config}")
    
    def evaluate_classifier(self, model, X_test, y_test):
        """
        Evaluate a classification model.
        
        Args:
            model: Classifier instance
            X_test (DataFrame): Test features
            y_test (Series): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        self.logger.info("Evaluating classifier")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            
            # For binary classification
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
        else:
            y_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate ROC AUC if applicable
        if y_proba is not None and len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = None
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        self.logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }
    
    def evaluate_regressor(self, model, X_test, y_test):
        """
        Evaluate a regression model.
        
        Args:
            model: Regressor instance
            X_test (DataFrame): Test features
            y_test (Series): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        self.logger.info("Evaluating regressor")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy
        direction_actual = np.sign(y_test)
        direction_pred = np.sign(y_pred)
        directional_accuracy = np.mean(direction_actual == direction_pred)
        
        self.logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Directional Accuracy: {directional_accuracy:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def walk_forward_validation(self, model, X, y, window_size=None, step_size=None):
        """
        Perform walk-forward validation.
        
        Args:
            model: Model instance (classifier or regressor)
            X (DataFrame): Features
            y (Series): Target
            window_size (int): Size of each validation window
            step_size (int): Steps between windows
            
        Returns:
            dict: Validation results
        """
        window_size = window_size or self.config['walk_forward_window']
        step_size = step_size or self.config['walk_forward_step']
        
        self.logger.info(f"Running walk-forward validation with window size {window_size}, step size {step_size}")
        
        # Determine if classifier or regressor
        is_classifier = hasattr(model, 'classes_')
        
        # Initialize results
        results = {
            'windows': [],
            'metrics': {
                'window_metrics': []
            }
        }
        
        # Calculate number of windows
        n_samples = len(X)
        n_windows = (n_samples - window_size) // step_size + 1
        
        # Minimum training size
        min_train_size = min(252, n_samples // 3)  # At least 1 year or 1/3 of data
        
        for i in range(n_windows):
            train_end = min_train_size + i * step_size
            test_start = train_end
            test_end = min(test_start + window_size, n_samples)
            
            # Skip if test window is too small
            if test_end - test_start < 5:
                continue
            
            # Split data
            X_train, X_test = X.iloc[:train_end], X.iloc[test_start:test_end]
            y_train, y_test = y.iloc[:train_end], y.iloc[test_start:test_end]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            window_result = {
                'window': i,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_end_idx': train_end,
                'test_start_idx': test_start,
                'test_end_idx': test_end
            }
            
            if is_classifier:
                metrics = self.evaluate_classifier(model, X_test, y_test)
            else:
                metrics = self.evaluate_regressor(model, X_test, y_test)
            
            window_result.update(metrics)
            results['windows'].append(window_result)
        
        # Calculate average metrics
        if is_classifier:
            avg_metrics = {
                'avg_accuracy': np.mean([w.get('accuracy', 0) for w in results['windows']]),
                'avg_precision': np.mean([w.get('precision', 0) for w in results['windows']]),
                'avg_recall': np.mean([w.get('recall', 0) for w in results['windows']]),
                'avg_f1': np.mean([w.get('f1', 0) for w in results['windows']])
            }
            
            if all('roc_auc' in w for w in results['windows']):
                avg_metrics['avg_roc_auc'] = np.mean([w.get('roc_auc', 0) for w in results['windows']])
        else:
            avg_metrics = {
                'avg_mse': np.mean([w.get('mse', 0) for w in results['windows']]),
                'avg_rmse': np.mean([w.get('rmse', 0) for w in results['windows']]),
                'avg_mae': np.mean([w.get('mae', 0) for w in results['windows']]),
                'avg_r2': np.mean([w.get('r2', 0) for w in results['windows']]),
                'avg_directional_accuracy': np.mean([w.get('directional_accuracy', 0) for w in results['windows']])
            }
        
        results['metrics']['average'] = avg_metrics
        
        self.logger.info(f"Walk-forward validation complete: {n_windows} windows")
        
        return results
    
    def simulate_trading(self, model, X, y, prices, is_classifier=True):
        """
        Simulate trading based on model predictions.
        
        Args:
            model: Model instance
            X (DataFrame): Features
            y (Series): Target
            prices (Series): Price series aligned with X
            is_classifier (bool): Whether model is a classifier
            
        Returns:
            dict: Simulation results
        """
        self.logger.info("Simulating trading based on model predictions")
        
        # Generate predictions
        if is_classifier:
            y_pred = model.predict(X)
            signals = (y_pred > 0.5).astype(int) * 2 - 1  # Convert to -1, 1
        else:
            y_pred = model.predict(X)
            signals = np.sign(y_pred)  # -1, 0, 1
        
        # Initialize simulation
        capital = self.config['capital']
        initial_capital = capital
        position = 0
        transaction_cost = self.config['trading_costs']
        
        # Results tracking
        equity_curve = [capital]
        positions = [position]
        returns = [0]
        transactions = []
        
        # Run simulation
        for i in range(1, len(signals)):
            current_signal = signals[i-1]  # Previous day's signal
            current_price = prices.iloc[i]
            prev_price = prices.iloc[i-1]
            
            # Calculate price change
            price_change = current_price / prev_price - 1
            
            # Calculate return based on position
            position_return = position * price_change
            
            # Update capital (before any new transactions)
            capital *= (1 + position_return)
            
            # Determine new position
            new_position = current_signal
            
            # If position changes, apply transaction costs
            if new_position != position:
                # Close old position if any
                if position != 0:
                    cost = capital * transaction_cost
                    capital -= cost
                    transactions.append({
                        'day': i,
                        'type': 'sell' if position > 0 else 'cover',
                        'price': current_price,
                        'cost': cost,
                        'capital': capital
                    })
                
                # Open new position if any
                if new_position != 0:
                    cost = capital * transaction_cost
                    capital -= cost
                    transactions.append({
                        'day': i,
                        'type': 'buy' if new_position > 0 else 'short',
                        'price': current_price,
                        'cost': cost,
                        'capital': capital
                    })
                
                # Update position
                position = new_position
            
            # Record state
            equity_curve.append(capital)
            positions.append(position)
            returns.append(position_return)
        
        # Calculate metrics
        total_return = (capital / initial_capital - 1) * 100
        daily_returns = np.array(returns[1:])  # Skip initial 0
        annual_return = (1 + total_return / 100) ** (252 / len(daily_returns)) - 1
        annual_return *= 100  # Convert to percentage
        
        # Risk metrics
        std_dev = np.std(daily_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = (annual_return / 100 - self.config['risk_free_rate']) / std_dev
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Transaction metrics
        n_trades = len(transactions)
        avg_trade_return = total_return / n_trades if n_trades > 0 else 0
        win_rate = np.sum(daily_returns > 0) / len(daily_returns)
        
        self.logger.info(f"Simulation complete: Return: {total_return:.2f}%, Sharpe: {sharpe_ratio:.2f}, MaxDD: {max_drawdown:.2f}%")
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return_percent': total_return,
            'annual_return_percent': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_percent': max_drawdown,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'avg_trade_return': avg_trade_return,
            'equity_curve': equity_curve,
            'positions': positions,
            'returns': returns,
            'transactions': transactions
        }
    
    def _calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve (list): Equity curve
            
        Returns:
            float: Maximum drawdown as percentage
        """
        # Convert to numpy array
        equity = np.array(equity_curve)
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        
        return np.max(drawdown)
    
    def compare_to_baseline(self, model_results, X, y, prices, baseline_strategy=None):
        """
        Compare model results to a baseline strategy.
        
        Args:
            model_results (dict): Results from simulated trading
            X (DataFrame): Features
            y (Series): Target
            prices (Series): Price series aligned with X
            baseline_strategy (str): Baseline strategy name
            
        Returns:
            dict: Comparison results
        """
        baseline_strategy = baseline_strategy or self.config['baseline_strategy']
        
        self.logger.info(f"Comparing model to baseline strategy: {baseline_strategy}")
        
        # Generate signals based on baseline strategy
        if baseline_strategy == 'buy_and_hold':
            signals = np.ones(len(X))
        elif baseline_strategy == 'random':
            np.random.seed(42)
            signals = np.random.choice([-1, 0, 1], size=len(X))
        elif baseline_strategy == 'momentum':
            # Simple momentum: positive if price > 20-day moving average
            ma_20 = prices.rolling(window=20).mean()
            signals = np.zeros(len(X))
            signals[prices > ma_20] = 1
            signals[prices < ma_20] = -1
        else:
            self.logger.error(f"Unknown baseline strategy: {baseline_strategy}")
            return None
        
        # Initialize simulation
        capital = self.config['capital']
        initial_capital = capital
        position = 0
        transaction_cost = self.config['trading_costs']
        
        # Results tracking
        equity_curve = [capital]
        positions = [position]
        returns = [0]
        transactions = []
        
        # Run simulation
        for i in range(1, len(signals)):
            current_signal = signals[i-1]  # Previous day's signal
            current_price = prices.iloc[i]
            prev_price = prices.iloc[i-1]
            
            # Calculate price change
            price_change = current_price / prev_price - 1
            
            # Calculate return based on position
            position_return = position * price_change
            
            # Update capital (before any new transactions)
            capital *= (1 + position_return)
            
            # Determine new position
            new_position = current_signal
            
            # If position changes, apply transaction costs
            if new_position != position:
                # Close old position if any
                if position != 0:
                    cost = capital * transaction_cost
                    capital -= cost
                    transactions.append({
                        'day': i,
                        'type': 'sell' if position > 0 else 'cover',
                        'price': current_price,
                        'cost': cost,
                        'capital': capital
                    })
                
                # Open new position if any
                if new_position != 0:
                    cost = capital * transaction_cost
                    capital -= cost
                    transactions.append({
                        'day': i,
                        'type': 'buy' if new_position > 0 else 'short',
                        'price': current_price,
                        'cost': cost,
                        'capital': capital
                    })
                
                # Update position
                position = new_position
            
            # Record state
            equity_curve.append(capital)
            positions.append(position)
            returns.append(position_return)
        
        # Calculate metrics
        total_return = (capital / initial_capital - 1) * 100
        daily_returns = np.array(returns[1:])  # Skip initial 0
        annual_return = (1 + total_return / 100) ** (252 / len(daily_returns)) - 1
        annual_return *= 100  # Convert to percentage
        
        # Risk metrics
        std_dev = np.std(daily_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = (annual_return / 100 - self.config['risk_free_rate']) / std_dev
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Transaction metrics
        n_trades = len(transactions)
        avg_trade_return = total_return / n_trades if n_trades > 0 else 0
        win_rate = np.sum(daily_returns > 0) / len(daily_returns)
        
        baseline_results = {
            'strategy': baseline_strategy,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return_percent': total_return,
            'annual_return_percent': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_percent': max_drawdown,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'avg_trade_return': avg_trade_return,
            'equity_curve': equity_curve
        }
        
        # Comparison
        comparison = {
            'model': {
                'total_return_percent': model_results['total_return_percent'],
                'annual_return_percent': model_results['annual_return_percent'],
                'sharpe_ratio': model_results['sharpe_ratio'],
                'max_drawdown_percent': model_results['max_drawdown_percent'],
                'win_rate': model_results['win_rate']
            },
            'baseline': {
                'total_return_percent': baseline_results['total_return_percent'],
                'annual_return_percent': baseline_results['annual_return_percent'],
                'sharpe_ratio': baseline_results['sharpe_ratio'],
                'max_drawdown_percent': baseline_results['max_drawdown_percent'],
                'win_rate': baseline_results['win_rate']
            },
            'difference': {
                'total_return_percent': model_results['total_return_percent'] - baseline_results['total_return_percent'],
                'annual_return_percent': model_results['annual_return_percent'] - baseline_results['annual_return_percent'],
                'sharpe_ratio': model_results['sharpe_ratio'] - baseline_results['sharpe_ratio'],
                'max_drawdown_percent': model_results['max_drawdown_percent'] - baseline_results['max_drawdown_percent'],
                'win_rate': model_results['win_rate'] - baseline_results['win_rate']
            }
        }
        
        self.logger.info(f"Baseline {baseline_strategy}: Return: {total_return:.2f}%, Sharpe: {sharpe_ratio:.2f}")
        self.logger.info(f"Model outperforms baseline by {comparison['difference']['total_return_percent']:.2f}% return")
        
        return {
            'baseline_results': baseline_results,
            'comparison': comparison
        }
    
    def feature_importance_analysis(self, model, feature_names):
        """
        Analyze feature importance.
        
        Args:
            model: Trained model
            feature_names (list): Feature names
            
        Returns:
            dict: Feature importance analysis
        """
        self.logger.info("Analyzing feature importance")
        
        importance = None
        
        # Extract feature importance if available
        if hasattr(model, 'feature_importances_'):
            # Tree-based models (Random Forest, Gradient Boosting, etc.)
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models (Linear Regression, Logistic Regression, etc.)
            importance = np.abs(model.coef_)
            
            # Handle multi-class case
            if len(importance.shape) > 1 and importance.shape[0] > 1:
                importance = np.mean(importance, axis=0)
        
        if importance is None:
            self.logger.warning("Model does not provide feature importance")
            return None
        
        # Convert to list if numpy array
        if isinstance(importance, np.ndarray):
            importance = importance.tolist()
        
        # Ensure feature_names is the right length
        if len(feature_names) != len(importance):
            self.logger.error(f"Feature names length ({len(feature_names)}) does not match importance length ({len(importance)})")
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        # Create importance dict
        importance_dict = dict(zip(feature_names, importance))
        
        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Group by feature type
        feature_groups = {}
        for feature, imp in sorted_importance:
            # Extract feature group from name
            # Assuming format like "technical_rsi_14", "fundamental_pe_ratio", etc.
            parts = feature.split('_')
            if len(parts) > 1:
                group = parts[0]
                if group not in feature_groups:
                    feature_groups[group] = []
                feature_groups[group].append((feature, imp))
        
        # Calculate group importance
        group_importance = {}
        for group, features in feature_groups.items():
            group_importance[group] = sum(imp for _, imp in features)
        
        # Normalize group importance
        total_importance = sum(group_importance.values())
        if total_importance > 0:
            group_importance = {k: v / total_importance for k, v in group_importance.items()}
        
        # Results
        results = {
            'feature_importance': sorted_importance[:50],  # Top 50 features
            'group_importance': sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
        }
        
        self.logger.info(f"Feature importance analysis complete. Top feature: {sorted_importance[0]}")
        
        return results
    
    def visualize_confusion_matrix(self, confusion_matrix, classes=None):
        """
        Visualize confusion matrix and return as base64 image.
        
        Args:
            confusion_matrix (array): Confusion matrix
            classes (list): Class names
            
        Returns:
            str: Base64 encoded image
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Encode as base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
    
    def visualize_equity_curve(self, equity_curve, baseline_equity_curve=None):
        """
        Visualize equity curve and return as base64 image.
        
        Args:
            equity_curve (list): Equity curve
            baseline_equity_curve (list): Baseline equity curve for comparison
            
        Returns:
            str: Base64 encoded image
        """
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve, label='Model')
        
        if baseline_equity_curve is not None:
            plt.plot(baseline_equity_curve, label='Baseline', linestyle='--')
        
        plt.xlabel('Trading Day')
        plt.ylabel('Portfolio Value')
        plt.title('Equity Curve')
        plt.legend()
        plt.grid(True)
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Encode as base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
    
    def visualize_feature_importance(self, feature_importance, top_n=20):
        """
        Visualize feature importance and return as base64 image.
        
        Args:
            feature_importance (list): List of (feature, importance) tuples
            top_n (int): Number of top features to display
            
        Returns:
            str: Base64 encoded image
        """
        top_features = feature_importance[:top_n]
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), importances, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()  # Highest importance at the top
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Encode as base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
    
    def visualize_group_importance(self, group_importance):
        """
        Visualize feature group importance as a pie chart and return as base64 image.
        
        Args:
            group_importance (list): List of (group, importance) tuples
            
        Returns:
            str: Base64 encoded image
        """
        groups, importances = zip(*group_importance)
        
        plt.figure(figsize=(10, 8))
        plt.pie(importances, labels=groups, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Feature Group Importance')
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Encode as base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
    
    def save_evaluation_results(self, symbol, exchange, results, model_id=None):
        """
        Save evaluation results to database.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            results (dict): Evaluation results
            model_id (str): Model ID
            
        Returns:
            str: Document ID
        """
        # Create document
        doc = {
            'symbol': symbol,
            'exchange': exchange,
            'model_id': model_id,
            'results': results,
            'config': self.config,
            'timestamp': datetime.now()
        }
        
        # Insert into database
        result = self.db.model_evaluations_collection.insert_one(doc)
        
        self.logger.info(f"Saved evaluation results for {symbol} {exchange}")
        
        return str(result.inserted_id)
    
    def generate_evaluation_report(self, symbol, exchange, evaluation_id=None):
        """
        Generate an evaluation report with visualizations.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            evaluation_id (str): Evaluation ID
            
        Returns:
            dict: Report with visualizations
        """
        # Query database
        if evaluation_id:
            from bson.objectid import ObjectId
            query = {'_id': ObjectId(evaluation_id)}
        else:
            query = {
                'symbol': symbol,
                'exchange': exchange
            }
        
        evaluation = self.db.model_evaluations_collection.find_one(query, sort=[('timestamp', -1)])
        
        if not evaluation:
            self.logger.error(f"No evaluation found for {symbol} {exchange}")
            return None
        
        results = evaluation['results']
        
        # Generate visualizations
        visualizations = {}
        
        # Confusion matrix
        if 'confusion_matrix' in results:
            cm = np.array(results['confusion_matrix'])
            classes = ['Down', 'Up'] if cm.shape[0] == 2 else None
            visualizations['confusion_matrix'] = self.visualize_confusion_matrix(cm, classes)
        
        # Equity curve
        if 'equity_curve' in results:
            baseline_equity = None
            if 'baseline_results' in results and 'equity_curve' in results['baseline_results']:
                baseline_equity = results['baseline_results']['equity_curve']
                
            visualizations['equity_curve'] = self.visualize_equity_curve(
                results['equity_curve'], baseline_equity
            )
        
        # Feature importance
        if 'feature_importance' in results:
            visualizations['feature_importance'] = self.visualize_feature_importance(
                results['feature_importance']
            )
            
            if 'group_importance' in results:
                visualizations['group_importance'] = self.visualize_group_importance(
                    results['group_importance']
                )
        
        # Create report
        report = {
            'symbol': symbol,
            'exchange': exchange,
            'model_id': evaluation.get('model_id'),
            'timestamp': evaluation['timestamp'],
            'metrics': {
                'classification': {
                    'accuracy': results.get('accuracy'),
                    'precision': results.get('precision'),
                    'recall': results.get('recall'),
                    'f1': results.get('f1')
                } if 'accuracy' in results else None,
                'regression': {
                    'rmse': results.get('rmse'),
                    'mae': results.get('mae'),
                    'r2': results.get('r2'),
                    'directional_accuracy': results.get('directional_accuracy')
                } if 'rmse' in results else None,
                'trading': {
                    'total_return_percent': results.get('total_return_percent'),
                    'annual_return_percent': results.get('annual_return_percent'),
                    'sharpe_ratio': results.get('sharpe_ratio'),
                    'max_drawdown_percent': results.get('max_drawdown_percent'),
                    'win_rate': results.get('win_rate'),
                    'n_trades': results.get('n_trades')
                } if 'total_return_percent' in results else None,
                'baseline_comparison': results.get('comparison', {})
            },
            'visualizations': visualizations
        }
        
        self.logger.info(f"Generated evaluation report for {symbol} {exchange}")
        
        return report