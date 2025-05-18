# backtesting/optimizer.py
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import multiprocessing
import itertools
import time
import json
from tqdm import tqdm
import random

class StrategyOptimizer:
    """
    Optimizer for backtesting parameters using various methods.
    """
    
    def __init__(self, db_connector, engine=None, logger=None):
        """
        Initialize the strategy optimizer.
        
        Args:
            db_connector: MongoDB connector
            engine: Backtesting engine instance
            logger: Logger instance
        """
        self.db = db_connector
        self.engine = engine
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'max_workers': multiprocessing.cpu_count() - 1,
            'optimization_method': 'grid',  # 'grid', 'random', 'genetic'
            'max_iterations': 100,
            'random_samples': 50,
            'genetic_population': 20,
            'genetic_generations': 10,
            'genetic_mutation_rate': 0.1,
            'genetic_crossover_rate': 0.7,
            'metric': 'sharpe_ratio',  # Optimization target metric
            'timeout_seconds': 3600,  # 1 hour timeout
            'store_all_results': True  # Store all optimization results or just the best
        }
    
    def set_config(self, config):
        """
        Set optimizer configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated optimizer configuration: {self.config}")
    
    def get_backtest_engine(self):
        """
        Get the backtesting engine instance.
        
        Returns:
            BacktestEngine: Backtesting engine
        """
        if self.engine:
            return self.engine
            
        try:
            # Dynamically import to avoid circular imports
            from backtesting.engine import BacktestEngine
            self.engine = BacktestEngine(self.db)
            return self.engine
        except Exception as e:
            self.logger.error(f"Error getting backtest engine: {e}")
            return None
    
    def optimize(self, strategy_class, parameter_space, symbols, start_date, end_date, 
                initial_capital=100000, timeframe='day', method=None):
        """
        Optimize strategy parameters using the specified method.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_space (dict): Parameters and their possible values
            symbols (list): List of symbols to backtest
            start_date (str): Start date for backtest
            end_date (str): End date for backtest
            initial_capital (float): Initial capital
            timeframe (str): Timeframe for backtest
            method (str): Optimization method (if None, use default from config)
            
        Returns:
            dict: Optimization results
        """
        method = method or self.config['optimization_method']
        
        self.logger.info(f"Starting strategy optimization using {method} method")
        self.logger.info(f"Parameter space: {parameter_space}")
        
        # Convert dates to datetime if strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Record start time
        start_time = time.time()
        
        try:
            # Choose optimization method
            if method == 'grid':
                results = self._grid_search(strategy_class, parameter_space, symbols, 
                                          start_date, end_date, initial_capital, timeframe)
            elif method == 'random':
                results = self._random_search(strategy_class, parameter_space, symbols, 
                                            start_date, end_date, initial_capital, timeframe)
            elif method == 'genetic':
                results = self._genetic_algorithm(strategy_class, parameter_space, symbols, 
                                                start_date, end_date, initial_capital, timeframe)
            else:
                self.logger.error(f"Unsupported optimization method: {method}")
                return None
                
            # Record total time
            total_time = time.time() - start_time
            
            # Create optimization record
            optimization_record = {
                'strategy': strategy_class.__name__,
                'method': method,
                'symbols': symbols,
                'parameter_space': parameter_space,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'timeframe': timeframe,
                'metric': self.config['metric'],
                'total_iterations': len(results['all_results']),
                'best_parameters': results['best_parameters'],
                'best_performance': results['best_performance'],
                'total_time_seconds': total_time,
                'timestamp': datetime.now()
            }
            
            # Store only top results if configured
            if not self.config['store_all_results']:
                # Sort by target metric and keep top 10
                top_results = sorted(results['all_results'], 
                                    key=lambda x: x['performance'][self.config['metric']], 
                                    reverse=True)[:10]
                optimization_record['all_results'] = top_results
            else:
                optimization_record['all_results'] = results['all_results']
            
            # Save to database
            self._save_optimization_results(optimization_record)
            
            self.logger.info(f"Optimization completed in {total_time:.2f} seconds")
            self.logger.info(f"Best parameters: {results['best_parameters']}")
            self.logger.info(f"Best performance: {results['best_performance']}")
            
            return optimization_record
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            return None
    
    def _grid_search(self, strategy_class, parameter_space, symbols, start_date, end_date, 
                    initial_capital, timeframe):
        """
        Perform grid search optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_space (dict): Parameters and their possible values
            symbols (list): List of symbols to backtest
            start_date (datetime): Start date for backtest
            end_date (datetime): End date for backtest
            initial_capital (float): Initial capital
            timeframe (str): Timeframe for backtest
            
        Returns:
            dict: Optimization results
        """
        # Generate all parameter combinations
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"Grid search: {len(combinations)} parameter combinations to evaluate")
        
        # Prepare for parallel execution
        engine = self.get_backtest_engine()
        if not engine:
            return None
        
        # Define evaluation function for a single parameter set
        def evaluate_parameters(params):
            try:
                # Create parameter dictionary
                param_dict = dict(zip(param_names, params))
                
                # Create strategy instance
                strategy = strategy_class(**param_dict)
                
                # Run backtest
                backtest_results = engine.run_backtest(
                    strategy=strategy,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    timeframe=timeframe
                )
                
                # Extract performance metrics
                performance = backtest_results.get('statistics', {})
                
                return {
                    'parameters': param_dict,
                    'performance': performance,
                    'backtest_id': backtest_results.get('_id')
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating parameters {params}: {e}")
                return None
        
        # Run evaluations with progress bar
        results = []
        timeout = self.config['timeout_seconds']
        start_time = time.time()
        
        with multiprocessing.Pool(processes=self.config['max_workers']) as pool:
            # Use tqdm to show progress
            for result in tqdm(
                pool.imap_unordered(evaluate_parameters, combinations),
                total=len(combinations),
                desc="Grid Search Progress"
            ):
                # Check for timeout
                if time.time() - start_time > timeout:
                    self.logger.warning(f"Optimization timeout after {timeout} seconds")
                    pool.terminate()
                    break
                    
                if result:
                    results.append(result)
        
        # Find best result based on target metric
        metric = self.config['metric']
        valid_results = [r for r in results if r and metric in r['performance']]
        
        if not valid_results:
            self.logger.error("No valid results found during optimization")
            return {
                'best_parameters': None,
                'best_performance': None,
                'all_results': results
            }
        
        # Sort by target metric (assuming higher is better)
        best_result = max(valid_results, key=lambda x: x['performance'][metric])
        
        return {
            'best_parameters': best_result['parameters'],
            'best_performance': best_result['performance'],
            'all_results': results
        }
    
    def _random_search(self, strategy_class, parameter_space, symbols, start_date, end_date, 
                      initial_capital, timeframe):
        """
        Perform random search optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_space (dict): Parameters and their possible values
            symbols (list): List of symbols to backtest
            start_date (datetime): Start date for backtest
            end_date (datetime): End date for backtest
            initial_capital (float): Initial capital
            timeframe (str): Timeframe for backtest
            
        Returns:
            dict: Optimization results
        """
        num_samples = self.config['random_samples']
        self.logger.info(f"Random search: {num_samples} parameter combinations to evaluate")
        
        # Prepare for parallel execution
        engine = self.get_backtest_engine()
        if not engine:
            return None
        
        # Generate random parameter combinations
        param_names = list(parameter_space.keys())
        random_combinations = []
        
        for _ in range(num_samples):
            combination = []
            for values in parameter_space.values():
                # Handle different parameter types
                if isinstance(values, list):
                    combination.append(random.choice(values))
                elif isinstance(values, range):
                    combination.append(random.choice(list(values)))
                elif isinstance(values, tuple) and len(values) == 3:
                    # Assume (min, max, step) format
                    min_val, max_val, step = values
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer range
                        val = random.randrange(min_val, max_val, step)
                    else:
                        # Float range
                        steps = int((max_val - min_val) / step)
                        val = min_val + random.randint(0, steps) * step
                    combination.append(val)
                else:
                    combination.append(values[0] if values else None)
            
            random_combinations.append(tuple(combination))
        
        # Define evaluation function for a single parameter set
        def evaluate_parameters(params):
            try:
                # Create parameter dictionary
                param_dict = dict(zip(param_names, params))
                
                # Create strategy instance
                strategy = strategy_class(**param_dict)
                
                # Run backtest
                backtest_results = engine.run_backtest(
                    strategy=strategy,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    timeframe=timeframe
                )
                
                # Extract performance metrics
                performance = backtest_results.get('statistics', {})
                
                return {
                    'parameters': param_dict,
                    'performance': performance,
                    'backtest_id': backtest_results.get('_id')
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating parameters {params}: {e}")
                return None
        
        # Run evaluations with progress bar
        results = []
        timeout = self.config['timeout_seconds']
        start_time = time.time()
        
        with multiprocessing.Pool(processes=self.config['max_workers']) as pool:
            # Use tqdm to show progress
            for result in tqdm(
                pool.imap_unordered(evaluate_parameters, random_combinations),
                total=len(random_combinations),
                desc="Random Search Progress"
            ):
                # Check for timeout
                if time.time() - start_time > timeout:
                    self.logger.warning(f"Optimization timeout after {timeout} seconds")
                    pool.terminate()
                    break
                    
                if result:
                    results.append(result)
        
        # Find best result based on target metric
        metric = self.config['metric']
        valid_results = [r for r in results if r and metric in r['performance']]
        
        if not valid_results:
            self.logger.error("No valid results found during optimization")
            return {
                'best_parameters': None,
                'best_performance': None,
                'all_results': results
            }
        
        # Sort by target metric (assuming higher is better)
        best_result = max(valid_results, key=lambda x: x['performance'][metric])
        
        return {
            'best_parameters': best_result['parameters'],
            'best_performance': best_result['performance'],
            'all_results': results
        }
    
    def _genetic_algorithm(self, strategy_class, parameter_space, symbols, start_date, end_date, 
                          initial_capital, timeframe):
        """
        Perform genetic algorithm optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_space (dict): Parameters and their possible values
            symbols (list): List of symbols to backtest
            start_date (datetime): Start date for backtest
            end_date (datetime): End date for backtest
            initial_capital (float): Initial capital
            timeframe (str): Timeframe for backtest
            
        Returns:
            dict: Optimization results
        """
        population_size = self.config['genetic_population']
        generations = self.config['genetic_generations']
        mutation_rate = self.config['genetic_mutation_rate']
        crossover_rate = self.config['genetic_crossover_rate']
        
        self.logger.info(f"Genetic algorithm: population={population_size}, generations={generations}")
        
        # Prepare for execution
        engine = self.get_backtest_engine()
        if not engine:
            return None
        
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        
        # Helper functions for genetic algorithm
        def create_individual():
            """Create a random individual (parameter combination)"""
            combination = []
            for values in param_values:
                # Handle different parameter types
                if isinstance(values, list):
                    combination.append(random.choice(values))
                elif isinstance(values, range):
                    combination.append(random.choice(list(values)))
                elif isinstance(values, tuple) and len(values) == 3:
                    # Assume (min, max, step) format
                    min_val, max_val, step = values
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer range
                        val = random.randrange(min_val, max_val, step)
                    else:
                        # Float range
                        steps = int((max_val - min_val) / step)
                        val = min_val + random.randint(0, steps) * step
                    combination.append(val)
                else:
                    combination.append(values[0] if values else None)
            
            return tuple(combination)
        
        def evaluate_individual(individual):
            """Evaluate an individual's fitness"""
            try:
                # Create parameter dictionary
                param_dict = dict(zip(param_names, individual))
                
                # Create strategy instance
                strategy = strategy_class(**param_dict)
                
                # Run backtest
                backtest_results = engine.run_backtest(
                    strategy=strategy,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    timeframe=timeframe
                )
                
                # Extract performance metrics
                performance = backtest_results.get('statistics', {})
                
                # Get fitness value (target metric)
                metric = self.config['metric']
                fitness = performance.get(metric, 0)
                
                return {
                    'individual': individual,
                    'parameters': param_dict,
                    'performance': performance,
                    'fitness': fitness,
                    'backtest_id': backtest_results.get('_id')
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating individual {individual}: {e}")
                return {
                    'individual': individual,
                    'parameters': dict(zip(param_names, individual)),
                    'performance': {},
                    'fitness': -9999,  # Very low fitness
                    'backtest_id': None
                }
        
        def crossover(parent1, parent2):
            """Perform crossover between two parents"""
            if random.random() > crossover_rate:
                return parent1, parent2
                
            # Single-point crossover
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            
            return child1, child2
        
        def mutate(individual):
            """Mutate an individual"""
            mutated = list(individual)
            
            for i in range(len(mutated)):
                if random.random() < mutation_rate:
                    values = param_values[i]
                    
                    # Handle different parameter types
                    if isinstance(values, list):
                        mutated[i] = random.choice(values)
                    elif isinstance(values, range):
                        mutated[i] = random.choice(list(values))
                    elif isinstance(values, tuple) and len(values) == 3:
                        # Assume (min, max, step) format
                        min_val, max_val, step = values
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            # Integer range
                            mutated[i] = random.randrange(min_val, max_val, step)
                        else:
                            # Float range
                            steps = int((max_val - min_val) / step)
                            mutated[i] = min_val + random.randint(0, steps) * step
            
            return tuple(mutated)
        
        def select_parents(population, tournament_size=3):
            """Select parents using tournament selection"""
            tournament = random.sample(population, tournament_size)
            return max(tournament, key=lambda x: x['fitness'])['individual']
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = create_individual()
            population.append(evaluate_individual(individual))
        
        # Sort initial population by fitness
        population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Keep track of all evaluated individuals
        all_evaluations = population.copy()
        
        # Run genetic algorithm
        for generation in range(generations):
            self.logger.info(f"Generation {generation+1}/{generations}")
            
            # Create new generation
            new_population = []
            
            # Elitism: Keep best individuals
            elite_count = max(1, int(population_size * 0.1))
            new_population.extend(population[:elite_count])
            
            # Fill the rest with crossover and mutation
            while len(new_population) < population_size:
                # Select parents
                parent1 = select_parents(population)
                parent2 = select_parents(population)
                
                # Crossover
                child1, child2 = crossover(parent1, parent2)
                
                # Mutation
                child1 = mutate(child1)
                child2 = mutate(child2)
                
                # Evaluate children
                child1_eval = evaluate_individual(child1)
                child2_eval = evaluate_individual(child2)
                
                # Add to new population
                new_population.append(child1_eval)
                all_evaluations.append(child1_eval)
                
                if len(new_population) < population_size:
                    new_population.append(child2_eval)
                    all_evaluations.append(child2_eval)
            
            # Sort new population by fitness
            new_population.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Replace old population
            population = new_population
            
            # Log progress
            best_fitness = population[0]['fitness']
            avg_fitness = sum(p['fitness'] for p in population) / len(population)
            self.logger.info(f"Generation {generation+1}: Best fitness={best_fitness:.4f}, Avg fitness={avg_fitness:.4f}")
        
        # Get best result
        best_result = max(all_evaluations, key=lambda x: x['fitness'])
        
        return {
            'best_parameters': best_result['parameters'],
            'best_performance': best_result['performance'],
            'all_results': all_evaluations
        }
    
    def _save_optimization_results(self, optimization_record):
        """
        Save optimization results to database.
        
        Args:
            optimization_record (dict): Optimization record
            
        Returns:
            str: Optimization ID
        """
        try:
            # Insert into database
            result = self.db.optimization_results_collection.insert_one(optimization_record)
            optimization_id = str(result.inserted_id)
            
            self.logger.info(f"Saved optimization results to database with ID: {optimization_id}")
            
            return optimization_id
            
        except Exception as e:
            self.logger.error(f"Error saving optimization results to database: {e}")
            return None
    
    def get_optimization_results(self, optimization_id):
        """
        Get optimization results from database.
        
        Args:
            optimization_id (str): Optimization ID
            
        Returns:
            dict: Optimization results
        """
        try:
            from bson.objectid import ObjectId
            
            # Query database
            optimization = self.db.optimization_results_collection.find_one({
                '_id': ObjectId(optimization_id)
            })
            
            if not optimization:
                self.logger.error(f"Optimization {optimization_id} not found")
                return None
                
            return optimization
            
        except Exception as e:
            self.logger.error(f"Error getting optimization results: {e}")
            return None
    
    def optimize_walkforward(self, strategy_class, parameter_space, symbols, start_date, end_date, 
                            train_period=180, test_period=60, initial_capital=100000, 
                            timeframe='day', method=None):
        """
        Perform walk-forward optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_space (dict): Parameters and their possible values
            symbols (list): List of symbols to backtest
            start_date (str): Start date for backtest
            end_date (str): End date for backtest
            train_period (int): Training period in days
            test_period (int): Testing period in days
            initial_capital (float): Initial capital
            timeframe (str): Timeframe for backtest
            method (str): Optimization method (if None, use default from config)
            
        Returns:
            dict: Walk-forward optimization results
        """
        method = method or self.config['optimization_method']
        
        self.logger.info(f"Starting walk-forward optimization using {method} method")
        self.logger.info(f"Parameter space: {parameter_space}")
        
        # Convert dates to datetime if strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Calculate number of windows
        total_days = (end_date - start_date).days
        window_count = max(1, total_days // test_period)
        
        self.logger.info(f"Walk-forward optimization: {window_count} windows")
        
        # Initialize result storage
        window_results = []
        combined_equity_curve = []
        combined_trades = []
        
        # Walk forward through time
        current_date = start_date
        
        for window in range(window_count):
            # Calculate window dates
            train_start = current_date
            train_end = train_start + timedelta(days=train_period)
            test_start = train_end
            test_end = test_start + timedelta(days=test_period)
            
            # Ensure we don't go beyond end_date
            test_end = min(test_end, end_date)
            
            self.logger.info(f"Window {window+1}/{window_count}:")
            self.logger.info(f"  Train: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
            self.logger.info(f"  Test: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            
            # Skip if training period too short
            if (train_end - train_start).days < 30:
                self.logger.warning(f"Training period too short for window {window+1}, skipping")
                current_date = test_end
                continue
            
            try:
                # Optimize on training period
                optimization_results = self.optimize(
                    strategy_class=strategy_class,
                    parameter_space=parameter_space,
                    symbols=symbols,
                    start_date=train_start,
                    end_date=train_end,
                    initial_capital=initial_capital,
                    timeframe=timeframe,
                    method=method
                )
                
                if not optimization_results or not optimization_results['best_parameters']:
                    self.logger.warning(f"No optimal parameters found for window {window+1}, skipping")
                    current_date = test_end
                    continue
                
                # Get best parameters
                best_params = optimization_results['best_parameters']
                
                # Run backtest with best parameters on test period
                engine = self.get_backtest_engine()
                if not engine:
                    return None
                
                strategy = strategy_class(**best_params)
                
                test_results = engine.run_backtest(
                    strategy=strategy,
                    symbols=symbols,
                    start_date=test_start,
                    end_date=test_end,
                    initial_capital=initial_capital,
                    timeframe=timeframe
                )
                
                # Store window results
                window_result = {
                    'window': window + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'best_parameters': best_params,
                    'train_performance': optimization_results['best_performance'],
                    'test_performance': test_results.get('statistics', {}),
                    'test_backtest_id': test_results.get('_id')
                }
                
                window_results.append(window_result)
                
                # Append to combined equity curve and trades
                equity_curve = test_results.get('equity_curve', [])
                equity_timestamps = test_results.get('equity_timestamps', [])
                
                for i, value in enumerate(equity_curve):
                    if i < len(equity_timestamps):
                        combined_equity_curve.append({
                            'timestamp': equity_timestamps[i],
                            'equity': value,
                            'window': window + 1
                        })
                
                trades = test_results.get('trades', [])
                for trade in trades:
                    trade['window'] = window + 1
                    combined_trades.append(trade)
                
            except Exception as e:
                self.logger.error(f"Error in walk-forward window {window+1}: {e}")
            
            # Move to next window
            current_date = test_end
        
        # Calculate overall performance
        overall_performance = self._calculate_walkforward_performance(combined_equity_curve, combined_trades)
        
        # Create walk-forward optimization record
        walkforward_record = {
            'strategy': strategy_class.__name__,
            'method': method,
            'symbols': symbols,
            'parameter_space': parameter_space,
            'start_date': start_date,
            'end_date': end_date,
            'train_period': train_period,
            'test_period': test_period,
            'initial_capital': initial_capital,
            'timeframe': timeframe,
            'window_count': window_count,
            'windows': window_results,
            'overall_performance': overall_performance,
            'combined_equity_curve': combined_equity_curve,
            'combined_trades': combined_trades,
            'timestamp': datetime.now()
        }
        
        # Save to database
        result = self.db.walkforward_results_collection.insert_one(walkforward_record)
        walkforward_id = str(result.inserted_id)
        
        self.logger.info(f"Walk-forward optimization completed and saved with ID: {walkforward_id}")
        
        return {
            'walkforward_id': walkforward_id,
            'overall_performance': overall_performance,
            'windows': window_results
        }
    
   
    def _calculate_walkforward_performance(self, equity_curve, trades):
        """
        Calculate overall performance from walk-forward test results.
        
        Args:
            equity_curve (list): Combined equity curve
            trades (list): Combined trades list
            
        Returns:
            dict: Performance statistics
        """
        if not equity_curve:
            return {}
        
        try:
            # Convert to DataFrame
            equity_df = pd.DataFrame(equity_curve)
            
            # Sort by timestamp
            equity_df.sort_values('timestamp', inplace=True)
            
            # Calculate returns
            equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
            
            # Calculate key metrics
            initial_equity = equity_df['equity'].iloc[0]
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity / initial_equity - 1) * 100
            
            # Calculate annualized return
            days = (equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).total_seconds() / (60*60*24)
            annual_return = (((final_equity / initial_equity) ** (365 / days)) - 1) * 100 if days > 0 else 0
            
            # Calculate drawdowns
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak'] * 100
            max_drawdown = equity_df['drawdown'].max()
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.02  # 2% annual risk-free rate
            daily_risk_free = (1 + risk_free_rate) ** (1/365) - 1
            excess_returns = equity_df['returns'] - daily_risk_free
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5) if excess_returns.std() > 0 else 0
            
            # Calculate trade statistics
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            
            if not trades_df.empty:
                # Calculate win rate
                win_rate = len(trades_df[trades_df['profit_loss'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0
                
                # Calculate profit factor
                gross_profit = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].sum()
                gross_loss = abs(trades_df[trades_df['profit_loss'] < 0]['profit_loss'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
                
                # Calculate average trade
                avg_trade = trades_df['profit_loss'].mean()
                
                # Calculate average win/loss
                avg_win = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].mean() if len(trades_df[trades_df['profit_loss'] > 0]) > 0 else 0
                avg_loss = trades_df[trades_df['profit_loss'] < 0]['profit_loss'].mean() if len(trades_df[trades_df['profit_loss'] < 0]) > 0 else 0
                
                # Calculate max consecutive wins/losses
                trades_df['win'] = trades_df['profit_loss'] > 0
                
                # Initialize counters
                current_streak = 1
                max_win_streak = 0
                max_loss_streak = 0
                prev_win = None
                
                for win in trades_df['win']:
                    if prev_win is None:
                        prev_win = win
                    elif win == prev_win:
                        current_streak += 1
                    else:
                        if prev_win:
                            max_win_streak = max(max_win_streak, current_streak)
                        else:
                            max_loss_streak = max(max_loss_streak, current_streak)
                        current_streak = 1
                        prev_win = win
                
                # Check final streak
                if prev_win:
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)
            else:
                win_rate = 0
                profit_factor = 0
                avg_trade = 0
                avg_win = 0
                avg_loss = 0
                max_win_streak = 0
                max_loss_streak = 0
            
            # Calculate recovery factor
            recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf')
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'recovery_factor': recovery_factor,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade': avg_trade,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                'total_trades': len(trades_df),
                'training_windows': len(set(equity_df['window'])) if 'window' in equity_df.columns else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating walk-forward performance: {e}")
            return {}
    
    def parameter_sensitivity(self, strategy_class, base_parameters, parameter_to_test, test_values,
                            symbols, start_date, end_date, initial_capital=100000, timeframe='day'):
        """
        Analyze the sensitivity of strategy performance to changes in a specific parameter.
        
        Args:
            strategy_class: Strategy class to test
            base_parameters (dict): Base parameter values
            parameter_to_test (str): Name of parameter to test
            test_values (list): Values to test for the parameter
            symbols (list): List of symbols to backtest
            start_date (str/datetime): Start date for backtest
            end_date (str/datetime): End date for backtest
            initial_capital (float): Initial capital
            timeframe (str): Timeframe for backtest
            
        Returns:
            dict: Sensitivity analysis results
        """
        self.logger.info(f"Starting parameter sensitivity analysis for {parameter_to_test}")
        self.logger.info(f"Base parameters: {base_parameters}")
        self.logger.info(f"Test values: {test_values}")
        
        # Convert dates to datetime if strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get backtesting engine
        engine = self.get_backtest_engine()
        if not engine:
            return None
        
        # Run backtests with different parameter values
        results = []
        
        for value in test_values:
            try:
                # Create parameter set
                params = base_parameters.copy()
                params[parameter_to_test] = value
                
                # Create strategy instance
                strategy = strategy_class(**params)
                
                # Run backtest
                backtest_results = engine.run_backtest(
                    strategy=strategy,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    timeframe=timeframe
                )
                
                # Extract performance metrics
                performance = backtest_results.get('statistics', {})
                
                results.append({
                    'parameter_value': value,
                    'performance': performance,
                    'backtest_id': backtest_results.get('_id')
                })
                
                self.logger.info(f"Tested {parameter_to_test}={value}: {self.config['metric']}={performance.get(self.config['metric'], 0)}")
                
            except Exception as e:
                self.logger.error(f"Error testing {parameter_to_test}={value}: {e}")
        
        # Create sensitivity analysis record
        sensitivity_record = {
            'strategy': strategy_class.__name__,
            'parameter': parameter_to_test,
            'base_parameters': base_parameters,
            'test_values': test_values,
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'timeframe': timeframe,
            'results': results,
            'timestamp': datetime.now()
        }
        
        # Save to database
        result = self.db.sensitivity_results_collection.insert_one(sensitivity_record)
        sensitivity_id = str(result.inserted_id)
        
        self.logger.info(f"Parameter sensitivity analysis completed and saved with ID: {sensitivity_id}")
        
        return {
            'sensitivity_id': sensitivity_id,
            'parameter': parameter_to_test,
            'results': results
        }
    
    def monte_carlo_analysis(self, backtest_id, simulations=1000, resample_method='bootstrap'):
        """
        Perform Monte Carlo simulation on a backtest to analyze risk and return distribution.
        
        Args:
            backtest_id (str): Backtest ID
            simulations (int): Number of Monte Carlo simulations
            resample_method (str): Method for resampling ('bootstrap', 'block', 'random_sequence')
            
        Returns:
            dict: Monte Carlo analysis results
        """
        self.logger.info(f"Starting Monte Carlo analysis for backtest {backtest_id}")
        self.logger.info(f"Simulations: {simulations}, Method: {resample_method}")
        
        try:
            # Get backtest results
            from bson.objectid import ObjectId
            
            backtest = self.db.backtest_results_collection.find_one({
                '_id': ObjectId(backtest_id)
            })
            
            if not backtest:
                self.logger.error(f"Backtest {backtest_id} not found")
                return None
            
            # Extract trades from backtest
            trades = backtest.get('trades', [])
            
            if not trades:
                self.logger.error("No trades found in backtest")
                return None
            
            # Convert to DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Calculate trade returns
            trades_df['return'] = trades_df['profit_loss'] / backtest.get('initial_capital', 100000)
            
            # Run Monte Carlo simulations
            simulation_results = []
            
            for i in range(simulations):
                # Resample trades based on method
                resampled_returns = self._resample_trades(trades_df['return'].values, method=resample_method)
                
                # Calculate equity curve
                equity = np.zeros(len(resampled_returns) + 1)
                equity[0] = 1.0  # Starting with normalized equity of 1.0
                
                for j, ret in enumerate(resampled_returns):
                    equity[j+1] = equity[j] * (1 + ret)
                
                # Calculate key metrics
                total_return = (equity[-1] - 1) * 100
                max_drawdown = self._calculate_max_drawdown(equity)
                
                simulation_results.append({
                    'simulation': i + 1,
                    'final_equity': equity[-1],
                    'total_return': total_return,
                    'max_drawdown': max_drawdown,
                    'equity_curve': equity.tolist()
                })
            
            # Calculate statistics across simulations
            returns = [sim['total_return'] for sim in simulation_results]
            drawdowns = [sim['max_drawdown'] for sim in simulation_results]
            final_equities = [sim['final_equity'] for sim in simulation_results]
            
            # Calculate percentiles
            percentiles = [5, 25, 50, 75, 95]
            return_percentiles = np.percentile(returns, percentiles).tolist()
            drawdown_percentiles = np.percentile(drawdowns, percentiles).tolist()
            equity_percentiles = np.percentile(final_equities, percentiles).tolist()
            
            # Create Monte Carlo analysis record
            monte_carlo_record = {
                'backtest_id': backtest_id,
                'strategy': backtest.get('strategy'),
                'simulations': simulations,
                'resample_method': resample_method,
                'original_return': backtest.get('statistics', {}).get('total_return', 0),
                'original_max_drawdown': backtest.get('statistics', {}).get('max_drawdown', 0),
                'return_percentiles': {str(p): v for p, v in zip(percentiles, return_percentiles)},
                'drawdown_percentiles': {str(p): v for p, v in zip(percentiles, drawdown_percentiles)},
                'equity_percentiles': {str(p): v for p, v in zip(percentiles, equity_percentiles)},
                'worst_case': {
                    'return': min(returns),
                    'drawdown': max(drawdowns)
                },
                'best_case': {
                    'return': max(returns),
                    'drawdown': min(drawdowns)
                },
                'simulation_results': simulation_results[:100],  # Store only first 100 simulations to reduce DB size
                'timestamp': datetime.now()
            }
            
            # Save to database
            result = self.db.monte_carlo_results_collection.insert_one(monte_carlo_record)
            monte_carlo_id = str(result.inserted_id)
            
            self.logger.info(f"Monte Carlo analysis completed and saved with ID: {monte_carlo_id}")
            
            return {
                'monte_carlo_id': monte_carlo_id,
                'return_percentiles': monte_carlo_record['return_percentiles'],
                'drawdown_percentiles': monte_carlo_record['drawdown_percentiles'],
                'worst_case': monte_carlo_record['worst_case'],
                'best_case': monte_carlo_record['best_case']
            }
            
        except Exception as e:
            self.logger.error(f"Error performing Monte Carlo analysis: {e}")
            return None
    
    def _resample_trades(self, returns, method='bootstrap', block_size=20):
        """
        Resample trade returns for Monte Carlo simulation.
        
        Args:
            returns (array): Array of trade returns
            method (str): Resampling method
            block_size (int): Block size for block bootstrap
            
        Returns:
            array: Resampled returns
        """
        n = len(returns)
        
        if method == 'bootstrap':
            # Simple bootstrap resampling (sampling with replacement)
            indices = np.random.randint(0, n, size=n)
            return returns[indices]
        
        elif method == 'block':
            # Block bootstrap (preserves some autocorrelation)
            resampled = []
            blocks = []
            
            # Create blocks
            for i in range(0, n - block_size + 1):
                blocks.append(returns[i:i+block_size])
            
            # Randomly sample blocks
            num_blocks = n // block_size + (1 if n % block_size > 0 else 0)
            
            for _ in range(num_blocks):
                block_idx = np.random.randint(0, len(blocks))
                resampled.extend(blocks[block_idx])
            
            # Trim to original length
            return np.array(resampled[:n])
        
        elif method == 'random_sequence':
            # Randomly shuffle the sequence of returns
            indices = np.random.permutation(n)
            return returns[indices]
        
        else:
            self.logger.warning(f"Unknown resampling method: {method}, using bootstrap")
            indices = np.random.randint(0, n, size=n)
            return returns[indices]
    
    def _calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve (array): Equity curve
            
        Returns:
            float: Maximum drawdown percentage
        """
        # Running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdowns
        drawdowns = (running_max - equity_curve) / running_max * 100
        
        # Maximum drawdown
        max_drawdown = np.max(drawdowns)
        
        return max_drawdown