# backtesting/validator.py
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import math
import random

class BacktestValidator:
    """
    Validation tools for backtesting results.
    """
    
    def __init__(self, db_connector, engine=None, logger=None):
        """
        Initialize the backtest validator.
        
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
            'min_trades': 30,  # Minimum number of trades for statistical significance
            'out_of_sample_ratio': 0.3,  # Out-of-sample data ratio
            'monte_carlo_simulations': 1000,  # Number of Monte Carlo simulations
            'significance_level': 0.05  # Statistical significance level
        }
    
    def set_config(self, config):
        """
        Set validator configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated validator configuration: {self.config}")
    
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
    
    def validate_strategy(self, strategy, symbols, start_date, end_date, validation_methods=None, 
                         initial_capital=100000, timeframe='day'):
        """
        Validate a trading strategy using multiple validation methods.
        
        Args:
            strategy: Strategy instance to validate
            symbols (list): List of symbols to backtest
            start_date (str/datetime): Start date for backtest
            end_date (str/datetime): End date for backtest
            validation_methods (list): List of validation methods to use
            initial_capital (float): Initial capital
            timeframe (str): Timeframe for backtest
            
        Returns:
            dict: Validation results
        """
        self.logger.info(f"Starting strategy validation for {strategy.__class__.__name__}")
        
        # Default validation methods
        if validation_methods is None:
            validation_methods = [
                'in_sample_out_of_sample',
                'robustness_tests',
                'monte_carlo',
                'statistical_significance'
            ]
        
        # Convert dates to datetime if strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get backtesting engine
        engine = self.get_backtest_engine()
        if not engine:
            return None
        
        # Run baseline backtest
        self.logger.info("Running baseline backtest")
        baseline_results = engine.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            timeframe=timeframe
        )
        
        if not baseline_results:
            self.logger.error("Baseline backtest failed")
            return None
        
        # Store validation results
        validation_results = {
            'strategy': strategy.__class__.__name__,
            'symbols': symbols,
            'timeframe': timeframe,
            'baseline_backtest_id': baseline_results.get('_id'),
            'baseline_performance': baseline_results.get('statistics', {}),
            'validation_results': {}
        }
        
        # Run selected validation methods
        for method in validation_methods:
            self.logger.info(f"Running validation method: {method}")
            
            try:
                if method == 'in_sample_out_of_sample':
                    validation_results['validation_results'][method] = self.validate_in_out_of_sample(
                        strategy, symbols, start_date, end_date, initial_capital, timeframe
                    )
                elif method == 'robustness_tests':
                    validation_results['validation_results'][method] = self.validate_robustness(
                        strategy, symbols, start_date, end_date, initial_capital, timeframe
                    )
                elif method == 'monte_carlo':
                    validation_results['validation_results'][method] = self.validate_monte_carlo(
                        baseline_results.get('_id')
                    )
                elif method == 'statistical_significance':
                    validation_results['validation_results'][method] = self.validate_statistical_significance(
                        baseline_results.get('trades', [])
                    )
                else:
                    self.logger.warning(f"Unknown validation method: {method}")
            except Exception as e:
                self.logger.error(f"Error running validation method {method}: {e}")
                validation_results['validation_results'][method] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Calculate overall validation score
        validation_results['overall_score'] = self._calculate_validation_score(validation_results)
        
        # Save validation results to database
        validation_id = self._save_validation_results(validation_results)
        validation_results['validation_id'] = validation_id
        
        self.logger.info(f"Strategy validation completed with score: {validation_results['overall_score']}")
        
        return validation_results
    
    def validate_in_out_of_sample(self, strategy, symbols, start_date, end_date, 
                                 initial_capital=100000, timeframe='day'):
        """
        Validate strategy performance on in-sample and out-of-sample data.
        
        Args:
            strategy: Strategy instance to validate
            symbols (list): List of symbols to backtest
            start_date (datetime): Start date for backtest
            end_date (datetime): End date for backtest
            initial_capital (float): Initial capital
            timeframe (str): Timeframe for backtest
            
        Returns:
            dict: Validation results
        """
        # Calculate in-sample and out-of-sample periods
        total_days = (end_date - start_date).days
        in_sample_days = int(total_days * (1 - self.config['out_of_sample_ratio']))
        
        in_sample_end = start_date + timedelta(days=in_sample_days)
        out_sample_start = in_sample_end + timedelta(days=1)
        
        self.logger.info(f"In-sample period: {start_date.strftime('%Y-%m-%d')} to {in_sample_end.strftime('%Y-%m-%d')}")
        self.logger.info(f"Out-of-sample period: {out_sample_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get backtesting engine
        engine = self.get_backtest_engine()
        if not engine:
            return None
        
        # Run in-sample backtest
        in_sample_results = engine.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=in_sample_end,
            initial_capital=initial_capital,
            timeframe=timeframe
        )
        
        # Run out-of-sample backtest
        out_sample_results = engine.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=out_sample_start,
            end_date=end_date,
            initial_capital=initial_capital,
            timeframe=timeframe
        )
        
        # Compare performance metrics
        in_sample_performance = in_sample_results.get('statistics', {})
        out_sample_performance = out_sample_results.get('statistics', {})
        
        # Calculate key metric differences
        metric_diffs = {}
        for metric in ['total_return', 'annual_return', 'sharpe_ratio', 'win_rate', 'profit_factor']:
            if metric in in_sample_performance and metric in out_sample_performance:
                in_val = in_sample_performance.get(metric, 0)
                out_val = out_sample_performance.get(metric, 0)
                diff = out_val - in_val
                pct_diff = (diff / in_val * 100) if in_val != 0 else float('inf')
                
                metric_diffs[metric] = {
                    'in_sample': in_val,
                    'out_sample': out_val,
                    'difference': diff,
                    'percent_difference': pct_diff
                }
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(metric_diffs)
        
        return {
            'in_sample_period': {
                'start_date': start_date,
                'end_date': in_sample_end
            },
            'out_sample_period': {
                'start_date': out_sample_start,
                'end_date': end_date
            },
            'in_sample_performance': in_sample_performance,
            'out_sample_performance': out_sample_performance,
            'metric_differences': metric_diffs,
            'consistency_score': consistency_score,
            'in_sample_backtest_id': in_sample_results.get('_id'),
            'out_sample_backtest_id': out_sample_results.get('_id')
        }
    
    def _calculate_consistency_score(self, metric_diffs):
        """
        Calculate consistency score between in-sample and out-of-sample performance.
        
        Args:
            metric_diffs (dict): Dictionary of metric differences
            
        Returns:
            float: Consistency score (0-100)
        """
        # Weights for different metrics
        weights = {
            'total_return': 0.2,
            'annual_return': 0.3,
            'sharpe_ratio': 0.3,
            'win_rate': 0.1,
            'profit_factor': 0.1
        }
        
        # Calculate weighted score
        score = 100  # Start with perfect score
        
        for metric, weight in weights.items():
            if metric in metric_diffs:
                # Penalize based on percent difference
                pct_diff = abs(metric_diffs[metric]['percent_difference'])
                
                # Cap at 100% difference
                pct_diff = min(pct_diff, 100)
                
                # Reduce score proportionally
                score -= weight * pct_diff
        
        # Ensure score is between 0 and 100
        return max(0, min(100, score))
    
    def validate_robustness(self, strategy, symbols, start_date, end_date, 
                           initial_capital=100000, timeframe='day'):
        """
        Validate strategy robustness through various stress tests.
        
        Args:
            strategy: Strategy instance to validate
            symbols (list): List of symbols to backtest
            start_date (datetime): Start date for backtest
            end_date (datetime): End date for backtest
            initial_capital (float): Initial capital
            timeframe (str): Timeframe for backtest
            
        Returns:
            dict: Validation results
        """
        self.logger.info("Running robustness tests")
        
        # Get backtesting engine
        engine = self.get_backtest_engine()
        if not engine:
            return None
        
        # Define stress tests
        stress_tests = [
            {
                'name': 'slippage_impact',
                'description': 'Increased slippage to 0.5%',
                'parameters': {'slippage_pct': 0.5}
            },
            {
                'name': 'commission_impact',
                'description': 'Increased commission to 0.2%',
                'parameters': {'commission_pct': 0.2}
            },
            {
                'name': 'delayed_execution',
                'description': 'Delayed execution by 1 bar',
                'parameters': {'execution_delay': 1}
            },
            {
                'name': 'reduced_capital',
                'description': 'Reduced initial capital by 50%',
                'parameters': {'initial_capital': initial_capital * 0.5}
            },
            {
                'name': 'volatile_period',
                'description': 'Test on highly volatile period',
                'parameters': {'volatility_factor': 1.5}
            }
        ]
        
        # Run baseline backtest
        baseline_results = engine.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            timeframe=timeframe
        )
        
        baseline_performance = baseline_results.get('statistics', {})
        
        # Run stress tests
        stress_test_results = []
        
        for test in stress_tests:
            try:
                self.logger.info(f"Running stress test: {test['name']}")
                
                # Apply test parameters
                test_params = {}
                for key, value in test['parameters'].items():
                    if key == 'initial_capital':
                        test_params[key] = value
                    elif key == 'slippage_pct':
                        test_params['slippage'] = value
                    elif key == 'commission_pct':
                        test_params['commission'] = value
                    elif key == 'execution_delay':
                        test_params['delay'] = value
                    elif key == 'volatility_factor':
                        # Implement selective period testing based on volatility
                        # This would need a more complex implementation
                        continue
                
                # Run stress test backtest
                test_results = engine.run_backtest(
                    strategy=strategy,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    **test_params
                )
                
                test_performance = test_results.get('statistics', {})
                
                # Calculate performance degradation
                degradation = {}
                for metric in ['total_return', 'sharpe_ratio', 'max_drawdown']:
                    if metric in baseline_performance and metric in test_performance:
                        baseline_val = baseline_performance.get(metric, 0)
                        test_val = test_performance.get(metric, 0)
                        
                        if metric == 'max_drawdown':
                            # For drawdown, higher is worse
                            pct_change = ((test_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else float('inf')
                        else:
                            # For other metrics, lower is worse
                            pct_change = ((baseline_val - test_val) / baseline_val * 100) if baseline_val != 0 else float('inf')
                        
                        degradation[metric] = {
                            'baseline': baseline_val,
                            'stress_test': test_val,
                            'percent_change': pct_change
                        }
                
                # Calculate impact score (0-100, lower is better)
                impact_score = self._calculate_impact_score(degradation)
                
                stress_test_results.append({
                    'test_name': test['name'],
                    'description': test['description'],
                    'parameters': test['parameters'],
                    'performance': test_performance,
                    'degradation': degradation,
                    'impact_score': impact_score,
                    'backtest_id': test_results.get('_id')
                })
                
            except Exception as e:
                self.logger.error(f"Error running stress test {test['name']}: {e}")
                stress_test_results.append({
                    'test_name': test['name'],
                    'description': test['description'],
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Calculate overall robustness score
        robustness_score = self._calculate_robustness_score(stress_test_results)
        
        return {
            'baseline_performance': baseline_performance,
            'stress_tests': stress_test_results,
            'robustness_score': robustness_score
        }
    
    def _calculate_impact_score(self, degradation):
        """
        Calculate impact score for a stress test.
        
        Args:
            degradation (dict): Dictionary of metric degradation
            
        Returns:
            float: Impact score (0-100, lower is better)
        """
        # Weights for different metrics
        weights = {
            'total_return': 0.4,
            'sharpe_ratio': 0.4,
            'max_drawdown': 0.2
        }
        
        # Calculate weighted score
        score = 0
        
        for metric, weight in weights.items():
            if metric in degradation:
                # Calculate score based on percent change
                pct_change = abs(degradation[metric]['percent_change'])
                
                # Cap at 100% change
                pct_change = min(pct_change, 100)
                
                # Add to score proportionally
                score += weight * pct_change
        
        return score
    
    def _calculate_robustness_score(self, stress_test_results):
        """
        Calculate overall robustness score based on stress test results.
        
        Args:
            stress_test_results (list): List of stress test results
            
        Returns:
            float: Robustness score (0-100, higher is better)
        """
        # Filter out failed tests
        valid_tests = [test for test in stress_test_results if 'impact_score' in test]
        
        if not valid_tests:
            return 0
        
        # Average impact score (lower is better)
        avg_impact = sum(test['impact_score'] for test in valid_tests) / len(valid_tests)
        
        # Convert to robustness score (higher is better)
        robustness_score = 100 - avg_impact
        
        # Ensure score is between a minimum of 0 and maximum of 100
        return max(0, min(100, robustness_score))
    
    def validate_monte_carlo(self, backtest_id, simulations=None):
        """
        Validate strategy using Monte Carlo simulation.
        
        Args:
            backtest_id (str): Backtest ID
            simulations (int): Number of Monte Carlo simulations
            
        Returns:
            dict: Validation results
        """
        simulations = simulations or self.config['monte_carlo_simulations']
        
        self.logger.info(f"Running Monte Carlo validation with {simulations} simulations")
        
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
                # Resample trades with replacement (bootstrap)
                resampled_returns = np.random.choice(
                    trades_df['return'].values,
                    size=len(trades_df),
                    replace=True
                )
                
                # Calculate equity curve
                equity = np.zeros(len(resampled_returns) + 1)
                equity[0] = 1.0  # Starting with normalized equity of 1.0
                
                for j, ret in enumerate(resampled_returns):
                    equity[j+1] = equity[j] * (1 + ret)
                
                # Calculate key metrics
                total_return = (equity[-1] - 1) * 100
                
                # Calculate maximum drawdown
                rolling_max = np.maximum.accumulate(equity)
                drawdown = (rolling_max - equity) / rolling_max * 100
                max_drawdown = np.max(drawdown)
                
                simulation_results.append({
                    'total_return': total_return,
                    'max_drawdown': max_drawdown
                })
            
            # Convert to DataFrame for easier analysis
            sim_df = pd.DataFrame(simulation_results)
            
            # Calculate percentiles
            return_percentiles = {
                '5th': np.percentile(sim_df['total_return'], 5),
                '25th': np.percentile(sim_df['total_return'], 25),
                '50th': np.percentile(sim_df['total_return'], 50),
                '75th': np.percentile(sim_df['total_return'], 75),
                '95th': np.percentile(sim_df['total_return'], 95)
            }
            
            drawdown_percentiles = {
                '5th': np.percentile(sim_df['max_drawdown'], 5),
                '25th': np.percentile(sim_df['max_drawdown'], 25),
                '50th': np.percentile(sim_df['max_drawdown'], 50),
                '75th': np.percentile(sim_df['max_drawdown'], 75),
                '95th': np.percentile(sim_df['max_drawdown'], 95)
            }
            
            # Calculate probabilities
            prob_positive_return = (sim_df['total_return'] > 0).mean() * 100
            prob_better_than_market = 0  # This would need market return as input
            prob_extreme_drawdown = (sim_df['max_drawdown'] > 25).mean() * 100  # Assuming 25% is extreme
            
            # Calculate Monte Carlo score
            monte_carlo_score = self._calculate_monte_carlo_score(
                sim_df, 
                backtest.get('statistics', {})
            )
            
            return {
                'simulations': simulations,
                'return_percentiles': return_percentiles,
                'drawdown_percentiles': drawdown_percentiles,
                'probabilities': {
                    'positive_return': prob_positive_return,
                    'better_than_market': prob_better_than_market,
                    'extreme_drawdown': prob_extreme_drawdown
                },
                'original_performance': {
                    'total_return': backtest.get('statistics', {}).get('total_return', 0),
                    'max_drawdown': backtest.get('statistics', {}).get('max_drawdown', 0)
                },
                'monte_carlo_score': monte_carlo_score
            }
            
        except Exception as e:
            self.logger.error(f"Error performing Monte Carlo validation: {e}")
            return None
    
    def _calculate_monte_carlo_score(self, sim_df, original_stats):
        """
        Calculate Monte Carlo score based on simulation results.
        
        Args:
            sim_df (DataFrame): Simulation results
            original_stats (dict): Original backtest statistics
            
        Returns:
            float: Monte Carlo score (0-100, higher is better)
        """
        # Get original performance metrics
        original_return = original_stats.get('total_return', 0)
        original_drawdown = original_stats.get('max_drawdown', 0)
        
        # Calculate percentile ranks of original performance in Monte Carlo results
        return_rank = (sim_df['total_return'] < original_return).mean() * 100
        drawdown_rank = (sim_df['max_drawdown'] > original_drawdown).mean() * 100
        
        # Calculate probabilities
        prob_positive_return = (sim_df['total_return'] > 0).mean() * 100
        prob_low_drawdown = (sim_df['max_drawdown'] < 20).mean() * 100  # Assuming 20% is acceptable
        
        # Calculate score components
        return_score = min(100, return_rank)  # Higher is better
        drawdown_score = min(100, drawdown_rank)  # Higher is better
        prob_positive_score = prob_positive_return
        prob_low_dd_score = prob_low_drawdown
        
        # Weighted average of components
        score = (
            0.3 * return_score +
            0.3 * drawdown_score +
            0.2 * prob_positive_score +
            0.2 * prob_low_dd_score
        )
        
        return score
    
    def validate_statistical_significance(self, trades, confidence_level=None):
        """
        Validate statistical significance of trading strategy.
        
        Args:
            trades (list): List of trade records
            confidence_level (float): Statistical confidence level
            
        Returns:
            dict: Validation results
        """
        confidence_level = confidence_level or (1 - self.config['significance_level'])
        
        self.logger.info(f"Running statistical significance tests at {confidence_level*100}% confidence level")
        
        if not trades:
            self.logger.error("No trades provided for statistical significance testing")
            return {
                'status': 'failed',
                'error': 'No trades provided'
            }
        
        # Check minimum number of trades
        if len(trades) < self.config['min_trades']:
            self.logger.warning(f"Insufficient trades for statistical significance: {len(trades)} < {self.config['min_trades']}")
            return {
                'status': 'insufficient_data',
                'message': f"Need at least {self.config['min_trades']} trades for statistical significance",
                'trade_count': len(trades)
            }
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate trade returns
        if 'profit_loss_pct' in trades_df.columns:
            trades_df['return'] = trades_df['profit_loss_pct']
        else:
            trades_df['return'] = trades_df['profit_loss'] / trades_df['entry_price'] / trades_df['quantity']
        
        # Separate wins and losses
        wins = trades_df[trades_df['profit_loss'] > 0]
        losses = trades_df[trades_df['profit_loss'] <= 0]
        
        # Calculate key statistics
        win_count = len(wins)
        loss_count = len(losses)
        total_trades = win_count + loss_count
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        avg_win = wins['profit_loss'].mean() if win_count > 0 else 0
        avg_loss = losses['profit_loss'].mean() if loss_count > 0 else 0
        
        # T-test for returns
        from scipy import stats
        
        # One-sample t-test against zero mean
        t_stat, p_value = stats.ttest_1samp(trades_df['return'], 0)
        
        # Calculate z-score for win rate (binomial test)
        expected_win_rate = 0.5  # Null hypothesis: random 50/50 chance
        z_score = (win_rate - expected_win_rate) / math.sqrt(expected_win_rate * (1 - expected_win_rate) / total_trades)
        
        # P-value from z-score
        p_value_winrate = 1 - stats.norm.cdf(abs(z_score))
        
        # Run length test (for trade sequence randomness)
        run_length_p_value = self._run_length_test(trades_df['profit_loss'] > 0)
        
        # Calculate statistical significance
        t_test_significant = p_value < (1 - confidence_level)
        winrate_significant = p_value_winrate < (1 - confidence_level)
        run_length_significant = run_length_p_value >= (1 - confidence_level)  # Not significant is good here
        
        # Calculate statistical score
        statistical_score = self._calculate_statistical_score(
            t_test_significant, 
            winrate_significant, 
            run_length_significant,
            p_value,
            p_value_winrate,
            run_length_p_value,
            win_rate,
            avg_win,
            avg_loss,
            total_trades
        )
        
        return {
            'trade_count': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            't_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': t_test_significant
            },
            'win_rate_test': {
                'z_score': z_score,
                'p_value': p_value_winrate,
                'significant': winrate_significant
            },
            'run_length_test': {
                'p_value': run_length_p_value,
                'random_sequence': run_length_significant
            },
            'statistical_score': statistical_score
        }
    
    def _run_length_test(self, win_loss_series):
        """
        Perform runs test for randomness on win/loss sequence.
        
        Args:
            win_loss_series (Series): Boolean series of wins (True) and losses (False)
            
        Returns:
            float: P-value for runs test
        """
        from scipy import stats
        
        # Convert to numpy array
        sequence = win_loss_series.values
        
        # Count runs
        runs = 1
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                runs += 1
        
        # Count wins and losses
        n1 = sum(sequence)
        n2 = len(sequence) - n1
        
        # Expected number of runs
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        
        # Standard deviation of runs
        std_runs = math.sqrt(
            (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / 
            ((n1 + n2)**2 * (n1 + n2 - 1))
        )
        
        # Z-statistic
        z = (runs - expected_runs) / std_runs
        
        # P-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return p_value
    
    def _calculate_statistical_score(self, t_test_significant, winrate_significant, 
                                   run_length_significant, p_value, p_value_winrate, 
                                   run_length_p_value, win_rate, avg_win, avg_loss, total_trades):
        """
        Calculate statistical significance score.
        
        Args:
            t_test_significant (bool): T-test significance
            winrate_significant (bool): Win rate significance
            run_length_significant (bool): Run length test significance
            p_value (float): T-test p-value
            p_value_winrate (float): Win rate p-value
            run_length_p_value (float): Run length p-value
            win_rate (float): Win rate
            avg_win (float): Average win
            avg_loss (float): Average loss
            total_trades (int): Total number of trades
            
        Returns:
            float: Statistical score (0-100, higher is better)
        """
        # Base score components
        t_test_score = 100 * (1 - p_value)
        winrate_score = 100 * (1 - p_value_winrate)
        run_length_score = 100 * run_length_p_value  # Higher p-value is better for run test
        
        # Bonus for profit factor
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss) * win_rate / (1 - win_rate)
            profit_factor_score = min(100, profit_factor * 20)  # Scale to max 100
        else:
            profit_factor_score = 100  # No losses = perfect score
        
        # Bonus for number of trades (more is better)
        trade_count_score = min(100, total_trades / 100 * 100)  # 100+ trades = max score
        
        # Weighted average of components
        score = (
            0.3 * t_test_score +
            0.25 * winrate_score +
            0.15 * run_length_score +
            0.2 * profit_factor_score +
            0.1 * trade_count_score
        )
        
        # Cap at 0-100 range
        return max(0, min(100, score))
    
    def _calculate_validation_score(self, validation_results):
        """
        Calculate overall validation score based on all validation methods.
        
        Args:
            validation_results (dict): Validation results
            
        Returns:
            float: Overall validation score (0-100, higher is better)
        """
        scores = []
        weights = []
        
        # Get individual validation scores
        validation_methods = validation_results.get('validation_results', {})
        
        # In-sample/out-of-sample validation
        if 'in_sample_out_of_sample' in validation_methods:
            consistency_score = validation_methods['in_sample_out_of_sample'].get('consistency_score', 0)
            scores.append(consistency_score)
            weights.append(0.4)  # Highest weight
        
        # Robustness tests
        if 'robustness_tests' in validation_methods:
            robustness_score = validation_methods['robustness_tests'].get('robustness_score', 0)
            scores.append(robustness_score)
            weights.append(0.25)
        
        # Monte Carlo simulation
        if 'monte_carlo' in validation_methods:
            monte_carlo_score = validation_methods['monte_carlo'].get('monte_carlo_score', 0)
            scores.append(monte_carlo_score)
            weights.append(0.2)
        
        # Statistical significance
        if 'statistical_significance' in validation_methods:
            statistical_score = validation_methods['statistical_significance'].get('statistical_score', 0)
            scores.append(statistical_score)
            weights.append(0.15)
        
        # Calculate weighted average if we have scores
        if scores:
            # Normalize weights
            total_weight = sum(weights)
            norm_weights = [w / total_weight for w in weights]
            
            # Weighted average
            overall_score = sum(s * w for s, w in zip(scores, norm_weights))
        else:
            overall_score = 0
        
        return overall_score
    
    def _save_validation_results(self, validation_results):
        """
        Save validation results to database.
        
        Args:
            validation_results (dict): Validation results
            
        Returns:
            str: Validation ID
        """
        try:
            # Insert into database
            result = self.db.validation_results_collection.insert_one(validation_results)
            validation_id = str(result.inserted_id)
            
            self.logger.info(f"Saved validation results to database with ID: {validation_id}")
            
            return validation_id
            
        except Exception as e:
            self.logger.error(f"Error saving validation results to database: {e}")
            return None
    
    def compare_strategies(self, strategy_classes, symbols, start_date, end_date, 
                           initial_capital=100000, timeframe='day'):
        """
        Compare multiple trading strategies using validation metrics.
        
        Args:
            strategy_classes (list): List of strategy classes to compare
            symbols (list): List of symbols to backtest
            start_date (str/datetime): Start date for backtest
            end_date (str/datetime): End date for backtest
            initial_capital (float): Initial capital
            timeframe (str): Timeframe for backtest
            
        Returns:
            dict: Strategy comparison results
        """
        self.logger.info(f"Comparing {len(strategy_classes)} strategies")
        
        # Convert dates to datetime if strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Validate each strategy
        strategy_results = []
        
        for strategy_class in strategy_classes:
            try:
                self.logger.info(f"Validating strategy: {strategy_class.__name__}")
                
                # Create strategy instance
                strategy = strategy_class()
                
                # Run validation
                validation_results = self.validate_strategy(
                    strategy=strategy,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    timeframe=timeframe
                )
                
                if validation_results:
                    strategy_results.append({
                        'strategy_name': strategy_class.__name__,
                        'validation_id': validation_results.get('validation_id'),
                        'overall_score': validation_results.get('overall_score', 0),
                        'baseline_performance': validation_results.get('baseline_performance', {})
                    })
                
            except Exception as e:
                self.logger.error(f"Error validating strategy {strategy_class.__name__}: {e}")
        
        # Sort strategies by overall score (descending)
        strategy_results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Create comparison record
        comparison_record = {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'timeframe': timeframe,
            'strategy_count': len(strategy_classes),
            'strategy_results': strategy_results,
            'timestamp': datetime.now()
        }
        
        # Save to database
        result = self.db.strategy_comparison_collection.insert_one(comparison_record)
        comparison_id = str(result.inserted_id)
        
        self.logger.info(f"Strategy comparison completed and saved with ID: {comparison_id}")
        
        return {
            'comparison_id': comparison_id,
            'strategy_results': strategy_results
        }