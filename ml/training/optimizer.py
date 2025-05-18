# ml/training/optimizer.py
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

class ModelOptimizer:
    """Optimizer for model hyperparameters."""
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the model optimizer.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Optimization config
        self.config = {
            'optimization_method': 'grid',  # 'grid', 'random', or 'genetic'
            'cv_folds': 5,
            'scoring': 'f1',  # For classifiers
            'reg_scoring': 'neg_mean_squared_error',  # For regressors
            'n_jobs': -1,
            'timeout': 3600,  # Max time in seconds for optimization
            'random_iterations': 30,  # For random search
            'genetic_generations': 10,  # For genetic algorithm
            'genetic_population': 20  # For genetic algorithm
        }
        
        # Default parameter grids
        self.classifier_param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, -1],
                'num_leaves': [31, 63, 127],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        }
        
        self.regressor_param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2]
            },
            'linear': {
                'fit_intercept': [True, False],
                'normalize': [True, False],
                'copy_X': [True]
            },
            'ridge': {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            },
            'lasso': {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'selection': ['cyclic', 'random']
            }
        }
    
    def set_config(self, config):
        """
        Set optimization configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated optimization configuration: {self.config}")
    
    def set_param_grid(self, model_type, param_grid, model_class='classifier'):
        """
        Set custom parameter grid for a model type.
        
        Args:
            model_type (str): Type of model
            param_grid (dict): Parameter grid
            model_class (str): 'classifier' or 'regressor'
        """
        if model_class == 'classifier':
            self.classifier_param_grids[model_type] = param_grid
        else:
            self.regressor_param_grids[model_type] = param_grid
        
        self.logger.info(f"Updated parameter grid for {model_class} {model_type}")
    
    def optimize_classifier(self, model, X, y, param_grid=None, model_type=None):
        """
        Optimize hyperparameters for a classifier.
        
        Args:
            model: Classifier instance
            X (DataFrame): Feature data
            y (Series): Target data
            param_grid (dict): Parameter grid (optional)
            model_type (str): Type of model (optional)
            
        Returns:
            dict: Optimization results
        """
        if model_type and not param_grid:
            param_grid = self.classifier_param_grids.get(model_type)
        
        if not param_grid:
            self.logger.error("No parameter grid provided")
            return None
        
        self.logger.info(f"Optimizing classifier with {self.config['optimization_method']} search")
        
        # Configure cross-validation
        if self.config['cv_folds'] > 1:
            cv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
        else:
            cv = 5  # Default
        
        try:
            # Start timer
            start_time = time.time()
            
            # Choose optimization method
            if self.config['optimization_method'] == 'grid':
                # Grid search
                search = GridSearchCV(
                    model,
                    param_grid,
                    scoring=self.config['scoring'],
                    cv=cv,
                    n_jobs=self.config['n_jobs'],
                    verbose=1
                )
            elif self.config['optimization_method'] == 'random':
                # Random search
                search = RandomizedSearchCV(
                    model,
                    param_grid,
                    n_iter=self.config['random_iterations'],
                    scoring=self.config['scoring'],
                    cv=cv,
                    n_jobs=self.config['n_jobs'],
                    verbose=1
                )
            elif self.config['optimization_method'] == 'genetic':
                # Genetic algorithm (custom implementation)
                return self._genetic_optimize(
                    model, X, y, param_grid, 
                    model_class='classifier',
                    generations=self.config['genetic_generations'],
                    population_size=self.config['genetic_population']
                )
            else:
                self.logger.error(f"Unknown optimization method: {self.config['optimization_method']}")
                return None
            
            # Execute search with timeout
            max_time = self.config['timeout']
            
            with ProcessPoolExecutor(max_workers=self.config['n_jobs'] if self.config['n_jobs'] > 0 else None) as executor:
                future = executor.submit(search.fit, X, y)
                
                try:
                    search = future.result(timeout=max_time)
                except TimeoutError:
                    self.logger.warning(f"Optimization timed out after {max_time} seconds")
                    return {
                        'status': 'timeout',
                        'best_params': None,
                        'best_score': None,
                        'cv_results': None,
                        'elapsed_time': max_time
                    }
            
            # Get results
            elapsed_time = time.time() - start_time
            
            self.logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"Best parameters: {search.best_params_}")
            self.logger.info(f"Best score: {search.best_score_:.4f}")
            
            return {
                'status': 'success',
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_,
                'elapsed_time': elapsed_time
            }
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def optimize_regressor(self, model, X, y, param_grid=None, model_type=None):
        """
        Optimize hyperparameters for a regressor.
        
        Args:
            model: Regressor instance
            X (DataFrame): Feature data
            y (Series): Target data
            param_grid (dict): Parameter grid (optional)
            model_type (str): Type of model (optional)
            
        Returns:
            dict: Optimization results
        """
        if model_type and not param_grid:
            param_grid = self.regressor_param_grids.get(model_type)
        
        if not param_grid:
            self.logger.error("No parameter grid provided")
            return None
        
        self.logger.info(f"Optimizing regressor with {self.config['optimization_method']} search")
        
        # Configure cross-validation
        if self.config['cv_folds'] > 1:
            cv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
        else:
            cv = 5  # Default
        
        try:
            # Start timer
            start_time = time.time()
            
            # Choose optimization method
            if self.config['optimization_method'] == 'grid':
                # Grid search
                search = GridSearchCV(
                    model,
                    param_grid,
                    scoring=self.config['reg_scoring'],
                    cv=cv,
                    n_jobs=self.config['n_jobs'],
                    verbose=1
                )
            elif self.config['optimization_method'] == 'random':
                # Random search
                search = RandomizedSearchCV(
                    model,
                    param_grid,
                    n_iter=self.config['random_iterations'],
                    scoring=self.config['reg_scoring'],
                    cv=cv,
                    n_jobs=self.config['n_jobs'],
                    verbose=1
                )
            elif self.config['optimization_method'] == 'genetic':
                # Genetic algorithm (custom implementation)
                return self._genetic_optimize(
                    model, X, y, param_grid, 
                    model_class='regressor',
                    generations=self.config['genetic_generations'],
                    population_size=self.config['genetic_population']
                )
            else:
                self.logger.error(f"Unknown optimization method: {self.config['optimization_method']}")
                return None
            
            # Execute search with timeout
            max_time = self.config['timeout']
            
            with ProcessPoolExecutor(max_workers=self.config['n_jobs'] if self.config['n_jobs'] > 0 else None) as executor:
                future = executor.submit(search.fit, X, y)
                
                try:
                    search = future.result(timeout=max_time)
                except TimeoutError:
                    self.logger.warning(f"Optimization timed out after {max_time} seconds")
                    return {
                        'status': 'timeout',
                        'best_params': None,
                        'best_score': None,
                        'cv_results': None,
                        'elapsed_time': max_time
                    }
            
            # Get results
            elapsed_time = time.time() - start_time
            
            self.logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"Best parameters: {search.best_params_}")
            self.logger.info(f"Best score: {search.best_score_:.4f}")
            
            return {
                'status': 'success',
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_,
                'elapsed_time': elapsed_time
            }
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _genetic_optimize(self, model, X, y, param_grid, model_class='classifier', 
                         generations=10, population_size=20, mutation_rate=0.2):
        """
        Optimize hyperparameters using a genetic algorithm.
        
        Args:
            model: Model instance
            X (DataFrame): Feature data
            y (Series): Target data
            param_grid (dict): Parameter grid
            model_class (str): 'classifier' or 'regressor'
            generations (int): Number of generations
            population_size (int): Size of population
            mutation_rate (float): Mutation rate
            
        Returns:
            dict: Optimization results
        """
        self.logger.info(f"Optimizing model with genetic algorithm: {generations} generations, {population_size} population")
        
        # Start timer
        start_time = time.time()
        
        # Configure cross-validation
        if self.config['cv_folds'] > 1:
            cv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
        else:
            cv = 5  # Default
        
        # Scoring function
        if model_class == 'classifier':
            scoring = self.config['scoring']
        else:
            scoring = self.config['reg_scoring']
        
        # Generate initial population
        population = []
        
        for _ in range(population_size):
            # Random selection of parameters
            params = {}
            for param, values in param_grid.items():
                params[param] = random.choice(values)
            
            population.append(params)
        
        # Best solution tracking
        best_score = float('-inf')
        best_params = None
        
        # Evolution
        for generation in range(generations):
            self.logger.info(f"Generation {generation+1}/{generations}")
            
            # Evaluate population
            scores = []
            
            for params in population:
                # Set parameters
                model.set_params(**params)
                
                # Cross-validate
                try:
                    from sklearn.model_selection import cross_val_score
                    score = np.mean(cross_val_score(model, X, y, cv=cv, scoring=scoring))
                    scores.append(score)
                    
                    # Update best solution
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        self.logger.info(f"New best: {best_score:.4f} - {best_params}")
                        
                except Exception as e:
                    self.logger.warning(f"Error evaluating parameters {params}: {e}")
                    scores.append(float('-inf'))
            
            # Check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > self.config['timeout']:
                self.logger.warning(f"Genetic optimization timed out after {elapsed_time:.2f} seconds")
                break
            
            # Select parents for next generation
            parents_indices = np.argsort(scores)[-population_size//2:]
            parents = [population[idx] for idx in parents_indices]
            
            # Create next generation through crossover
            next_generation = parents.copy()  # Elitism
            
            while len(next_generation) < population_size:
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)
                
                # Crossover
                child = {}
                for param in param_grid:
                    # Either take from parent1 or parent2
                    child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
                    
                    # Mutation
                    if random.random() < mutation_rate:
                        child[param] = random.choice(param_grid[param])
                
                next_generation.append(child)
            
            # Replace population
            population = next_generation
        
        # Final evaluation with best parameters
        model.set_params(**best_params)
        model.fit(X, y)
        
        # Evaluate on training data
        if model_class == 'classifier':
            y_pred = model.predict(X)
            training_accuracy = accuracy_score(y, y_pred)
            training_precision = precision_score(y, y_pred, average='weighted')
            training_recall = recall_score(y, y_pred, average='weighted')
            training_f1 = f1_score(y, y_pred, average='weighted')
            
            final_metrics = {
                'accuracy': training_accuracy,
                'precision': training_precision,
                'recall': training_recall,
                'f1': training_f1
            }
        else:
            y_pred = model.predict(X)
            training_mse = mean_squared_error(y, y_pred)
            training_rmse = np.sqrt(training_mse)
            training_mae = mean_absolute_error(y, y_pred)
            training_r2 = r2_score(y, y_pred)
            
            final_metrics = {
                'mse': training_mse,
                'rmse': training_rmse,
                'mae': training_mae,
                'r2': training_r2
            }
        
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"Genetic optimization completed in {elapsed_time:.2f} seconds")
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best score: {best_score:.4f}")
        
        return {
            'status': 'success',
            'best_params': best_params,
            'best_score': best_score,
            'training_metrics': final_metrics,
            'elapsed_time': elapsed_time,
            'generations_completed': min(generation + 1, generations)
        }
    
    def run_optimization(self, model_types, X, y, model_class='classifier'):
        """
        Run optimization for multiple model types.
        
        Args:
            model_types (list): List of model types to optimize
            X (DataFrame): Feature data
            y (Series): Target data
            model_class (str): 'classifier' or 'regressor'
            
        Returns:
            dict: Optimization results for each model type
        """
        results = {}
        
        for model_type in model_types:
            self.logger.info(f"Optimizing {model_class} {model_type}")
            
            # Instantiate model
            if model_class == 'classifier':
                from ml.models.classifier import MarketClassifier
                model = MarketClassifier(self.db)
                model.build_model(model_type=model_type)
                param_grid = self.classifier_param_grids.get(model_type)
                
                if param_grid:
                    opt_result = self.optimize_classifier(
                        model.model, X, y, param_grid, model_type
                    )
                    results[model_type] = opt_result
                else:
                    self.logger.error(f"No parameter grid for classifier {model_type}")
            else:
                from ml.models.regressor import MarketRegressor
                model = MarketRegressor(self.db)
                model.build_model(model_type=model_type)
                param_grid = self.regressor_param_grids.get(model_type)
                
                if param_grid:
                    opt_result = self.optimize_regressor(
                        model.model, X, y, param_grid, model_type
                    )
                    results[model_type] = opt_result
                else:
                    self.logger.error(f"No parameter grid for regressor {model_type}")
        
        return results
    
    def save_optimization_results(self, symbol, exchange, results, model_class='classifier'):
        """
        Save optimization results to database.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            results (dict): Optimization results
            model_class (str): 'classifier' or 'regressor'
            
        Returns:
            str: Document ID
        """
        # Create document
        doc = {
            'symbol': symbol,
            'exchange': exchange,
            'model_class': model_class,
            'results': results,
            'config': self.config,
            'timestamp': datetime.now()
        }
        
        # Insert into database
        result = self.db.optimization_results_collection.insert_one(doc)
        
        self.logger.info(f"Saved optimization results for {symbol} {exchange} {model_class}")
        
        return str(result.inserted_id)