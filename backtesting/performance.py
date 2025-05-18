# backtesting/performance.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import norm
import json

class PerformanceAnalyzer:
    """
    Performance metrics calculator for backtesting results.
    """
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the performance analyzer.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'benchmark_symbol': '^NSEI',  # Nifty 50 for Indian market
            'risk_free_rate': 0.04,  # 4% annual risk-free rate
            'confidence_level': 0.95,  # 95% confidence level for VaR
            'drawdown_threshold': 0.1,  # 10% drawdown threshold
        }
    
    def set_config(self, config):
        """
        Set analyzer configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated performance analyzer configuration: {self.config}")
    
    def get_backtest_results(self, backtest_id):
        """
        Get backtest results from database.
        
        Args:
            backtest_id (str): Backtest ID
            
        Returns:
            dict: Backtest results
        """
        try:
            from bson.objectid import ObjectId
            
            # Query database
            backtest = self.db.backtest_results_collection.find_one({
                '_id': ObjectId(backtest_id)
            })
            
            if not backtest:
                self.logger.error(f"Backtest {backtest_id} not found")
                return None
                
            return backtest
            
        except Exception as e:
            self.logger.error(f"Error getting backtest results: {e}")
            return None
    
    def analyze_backtest(self, backtest_id_or_results):
        """
        Analyze backtest results.
        
        Args:
            backtest_id_or_results: Backtest ID or results dictionary
            
        Returns:
            dict: Performance analysis
        """
        # Get backtest results
        if isinstance(backtest_id_or_results, str):
            backtest = self.get_backtest_results(backtest_id_or_results)
        else:
            backtest = backtest_id_or_results
        
        if not backtest:
            return None
        
        self.logger.info(f"Analyzing backtest performance for strategy {backtest.get('strategy')}")
        
        # Extract data
        equity_curve = backtest.get('equity_curve', [])
        if not equity_curve:
            self.logger.error("No equity curve data in backtest results")
            return None
            
        equity_timestamps = backtest.get('equity_timestamps', [])
        if len(equity_timestamps) != len(equity_curve):
            self.logger.error("Equity timestamps and curve have different lengths")
            return None
            
        # Convert timestamps from string to datetime if needed
        if isinstance(equity_timestamps[0], str):
            equity_timestamps = [
                datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') if ' ' in ts else datetime.strptime(ts, '%Y-%m-%d')
                for ts in equity_timestamps
            ]
        
        # Create equity DataFrame
        equity_df = pd.DataFrame({
            'date': equity_timestamps,
            'equity': equity_curve
        })
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
        
        # Get trades
        positions = backtest.get('positions', [])
        
        # Get statistics
        statistics = backtest.get('statistics', {})
        
        # Get benchmark returns
        benchmark_returns = self._get_benchmark_returns(
            min(equity_timestamps), max(equity_timestamps)
        )
        
        # Calculate performance metrics
        performance = {
            'strategy': backtest.get('strategy'),
            'timeframe': backtest.get('timeframe'),
            'start_date': backtest.get('start_date'),
            'end_date': backtest.get('end_date'),
            'initial_capital': backtest.get('initial_capital'),
            'final_capital': backtest.get('final_capital'),
            'total_return': (backtest.get('final_capital') / backtest.get('initial_capital') - 1) * 100,
            'total_trades': len(positions),
            'basic_metrics': self._calculate_basic_metrics(statistics),
            'return_metrics': self._calculate_return_metrics(equity_df, benchmark_returns),
            'risk_metrics': self._calculate_risk_metrics(equity_df, benchmark_returns),
            'drawdown_analysis': self._calculate_drawdown_analysis(equity_df),
            'trade_analysis': self._analyze_trades(positions),
            'monthly_returns': self._calculate_monthly_returns(equity_df),
            'benchmark_comparison': self._compare_to_benchmark(equity_df, benchmark_returns)
        }
        
        # Generate visualizations
        try:
            visualizations = self._generate_visualizations(equity_df, positions, benchmark_returns)
            performance['visualizations'] = visualizations
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
        
        # Save analysis
        try:
            analysis_id = self._save_performance_analysis(performance, backtest.get('_id'))
            performance['_id'] = analysis_id
        except Exception as e:
            self.logger.error(f"Error saving performance analysis: {e}")
        
        return performance
    
    def _calculate_basic_metrics(self, statistics):
        """
        Calculate basic performance metrics.
        
        Args:
            statistics (dict): Strategy statistics
            
        Returns:
            dict: Basic performance metrics
        """
        return {
            'win_rate': statistics.get('win_rate', 0) * 100,
            'profit_factor': statistics.get('profit_factor', 0),
            'avg_profit': statistics.get('avg_profit', 0),
            'avg_loss': statistics.get('avg_loss', 0),
            'avg_holding_time': statistics.get('avg_holding_time', 0),
            'return_to_drawdown': abs(statistics.get('cagr', 0) / statistics.get('max_drawdown', 1)) if statistics.get('max_drawdown', 0) > 0 else 0
        }
    
    def _calculate_return_metrics(self, equity_df, benchmark_returns):
        """
        Calculate return metrics.
        
        Args:
            equity_df (DataFrame): Equity curve data
            benchmark_returns (DataFrame): Benchmark returns
            
        Returns:
            dict: Return metrics
        """
        if len(equity_df) < 2:
            return {}
        
        # Daily returns
        daily_returns = equity_df['returns'].values
        
        # Calculate annualized metrics
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = days / 365.25
        
        # Annualized return
        total_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) - 1
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Annualized volatility
        annualized_vol = np.std(daily_returns) * np.sqrt(252)
        
        # Information ratio
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align dates
            aligned_returns = self._align_returns(equity_df, benchmark_returns)
            if len(aligned_returns) > 0:
                excess_returns = aligned_returns['returns'] - aligned_returns['benchmark_returns']
                information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
            else:
                information_ratio = 0
        else:
            information_ratio = 0
        
        return {
            'total_return': total_return * 100,
            'cagr': cagr * 100,
            'annualized_volatility': annualized_vol * 100,
            'sharpe_ratio': statistics.get('sharpe_ratio', 0),
            'sortino_ratio': statistics.get('sortino_ratio', 0),
            'information_ratio': information_ratio,
            'calmar_ratio': cagr / self._calculate_max_drawdown(equity_df) if self._calculate_max_drawdown(equity_df) > 0 else 0
        }
    
    def _calculate_risk_metrics(self, equity_df, benchmark_returns):
        """
        Calculate risk metrics.
        
        Args:
            equity_df (DataFrame): Equity curve data
            benchmark_returns (DataFrame): Benchmark returns
            
        Returns:
            dict: Risk metrics
        """
        if len(equity_df) < 2:
            return {}
        
        # Daily returns
        daily_returns = equity_df['returns'].values
        
        # Value at Risk (VaR)
        confidence_level = self.config['confidence_level']
        var = -np.percentile(daily_returns, 100 * (1 - confidence_level))
        
        # Conditional VaR (CVaR) / Expected Shortfall
        cvar = -np.mean(daily_returns[daily_returns <= -var])
        
        # Downside deviation
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(equity_df)
        
        # Beta and alpha
        beta = 0
        alpha = 0
        
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align dates
            aligned_returns = self._align_returns(equity_df, benchmark_returns)
            if len(aligned_returns) > 0:
                # Calculate beta
                cov_matrix = np.cov(aligned_returns['returns'], aligned_returns['benchmark_returns'])
                beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
                
                # Calculate alpha (annualized)
                rf_daily = self.config['risk_free_rate'] / 252
                alpha = np.mean(aligned_returns['returns'] - rf_daily - beta * (aligned_returns['benchmark_returns'] - rf_daily)) * 252
        
        return {
            'max_drawdown': max_drawdown * 100,
            'var_95': var * 100,
            'cvar_95': cvar * 100,
            'downside_deviation': downside_deviation * 100,
            'beta': beta,
            'alpha': alpha * 100
        }
    
    def _calculate_drawdown_analysis(self, equity_df):
        """
        Calculate drawdown analysis.
        
        Args:
            equity_df (DataFrame): Equity curve data
            
        Returns:
            dict: Drawdown analysis
        """
        if len(equity_df) < 2:
            return {}
        
        # Calculate running maximum
        equity_df['peak'] = equity_df['equity'].cummax()
        
        # Calculate drawdown
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak']
        
        # Find drawdown periods
        threshold = self.config['drawdown_threshold']
        
        in_drawdown = False
        drawdown_start = None
        drawdown_periods = []
        
        for i, row in equity_df.iterrows():
            if not in_drawdown and row['drawdown'] >= threshold:
                # Start of drawdown period
                in_drawdown = True
                drawdown_start = row['date']
            elif in_drawdown and row['drawdown'] < threshold:
                # End of drawdown period
                in_drawdown = False
                drawdown_periods.append({
                    'start': drawdown_start,
                    'end': row['date'],
                    'duration_days': (row['date'] - drawdown_start).days,
                    'max_drawdown': equity_df.loc[
                        (equity_df['date'] >= drawdown_start) & 
                        (equity_df['date'] <= row['date']),
                        'drawdown'
                    ].max()
                })
        
        # If still in drawdown at the end of the period
        if in_drawdown:
            drawdown_periods.append({
                'start': drawdown_start,
                'end': equity_df['date'].iloc[-1],
                'duration_days': (equity_df['date'].iloc[-1] - drawdown_start).days,
                'max_drawdown': equity_df.loc[
                    equity_df['date'] >= drawdown_start,
                    'drawdown'
                ].max()
            })
        
        # Calculate recovery time
        max_drawdown_idx = equity_df['drawdown'].idxmax()
        max_drawdown_date = equity_df.loc[max_drawdown_idx, 'date']
        
        # Find recovery date (if recovered)
        recovered = False
        recovery_date = None
        
        for i, row in equity_df.iloc[max_drawdown_idx:].iterrows():
            if row['drawdown'] == 0:
                recovered = True
                recovery_date = row['date']
                break
        
        recovery_time = (recovery_date - max_drawdown_date).days if recovered else None
        
        return {
            'max_drawdown': equity_df['drawdown'].max() * 100,
            'avg_drawdown': equity_df['drawdown'].mean() * 100,
            'max_drawdown_date': max_drawdown_date,
            'recovered': recovered,
            'recovery_time_days': recovery_time,
            'time_in_drawdown_pct': (equity_df['drawdown'] > 0).mean() * 100,
            'drawdown_periods': [
                {
                    'start': period['start'].strftime('%Y-%m-%d'),
                    'end': period['end'].strftime('%Y-%m-%d'),
                    'duration_days': period['duration_days'],
                    'max_drawdown': period['max_drawdown'] * 100
                }
                for period in drawdown_periods
            ]
        }
    
    def _analyze_trades(self, positions):
        """
        Analyze trades.
        
        Args:
            positions (list): Trade positions
            
        Returns:
            dict: Trade analysis
        """
        if not positions:
            return {}
        
        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame(positions)
        
        # Calculate trade metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        profits = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].values
        losses = trades_df[trades_df['profit_loss'] <= 0]['profit_loss'].values
        
        avg_profit = np.mean(profits) if len(profits) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        max_profit = np.max(profits) if len(profits) > 0 else 0
        max_loss = np.min(losses) if len(losses) > 0 else 0
        
        profit_factor = abs(np.sum(profits) / np.sum(losses)) if np.sum(losses) != 0 else float('inf')
        
        # Holding time analysis
        avg_holding_days = trades_df['holding_days'].mean()
        
        # Direction analysis
        long_trades = len(trades_df[trades_df['direction'] == 'long'])
        short_trades = len(trades_df[trades_df['direction'] == 'short'])
        
        long_win_rate = trades_df[trades_df['direction'] == 'long']['profit_loss'].gt(0).mean() if long_trades > 0 else 0
        short_win_rate = trades_df[trades_df['direction'] == 'short']['profit_loss'].gt(0).mean() if short_trades > 0 else 0
        
        # Exit reason analysis
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # Calculate consecutive wins/losses
        trades_df = trades_df.sort_values('exit_date')
        trades_df['is_win'] = trades_df['profit_loss'] > 0
        
        # Consecutive wins
        trades_df['win_streak'] = (trades_df['is_win'] != trades_df['is_win'].shift()).cumsum()
        win_streaks = trades_df[trades_df['is_win']].groupby('win_streak').size()
        max_cons_wins = win_streaks.max() if not win_streaks.empty else 0
        
        # Consecutive losses
        trades_df['loss_streak'] = (trades_df['is_win'] != trades_df['is_win'].shift()).cumsum()
        loss_streaks = trades_df[~trades_df['is_win']].groupby('loss_streak').size()
        max_cons_losses = loss_streaks.max() if not loss_streaks.empty else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding_days,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'long_win_rate': long_win_rate * 100,
            'short_win_rate': short_win_rate * 100,
            'exit_reasons': exit_reasons,
            'max_consecutive_wins': max_cons_wins,
            'max_consecutive_losses': max_cons_losses
        }
    
    def _calculate_monthly_returns(self, equity_df):
        """
        Calculate monthly returns.
        
        Args:
            equity_df (DataFrame): Equity curve data
            
        Returns:
            dict: Monthly returns
        """
        if len(equity_df) < 2:
            return {}
        
        # Set date as index
        equity_df = equity_df.set_index('date')
        
        # Resample to month-end
        monthly_equity = equity_df['equity'].resample('M').last()
        
        # Calculate returns
        monthly_returns = monthly_equity.pct_change().fillna(
            (monthly_equity.iloc[0] / equity_df['equity'].iloc[0]) - 1
        )
        
        # Convert to dictionary
        return {
            date.strftime('%Y-%m'): float(ret)
            for date, ret in monthly_returns.items()
        }
    
    def _compare_to_benchmark(self, equity_df, benchmark_returns):
        """
        Compare strategy to benchmark.
        
        Args:
            equity_df (DataFrame): Equity curve data
            benchmark_returns (DataFrame): Benchmark returns
            
        Returns:
            dict: Benchmark comparison
        """
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return {}
        
        # Align dates
        aligned_returns = self._align_returns(equity_df, benchmark_returns)
        
        if len(aligned_returns) == 0:
            return {}
        
        # Calculate cumulative returns
        aligned_returns['cum_returns'] = (1 + aligned_returns['returns']).cumprod() - 1
        aligned_returns['cum_benchmark'] = (1 + aligned_returns['benchmark_returns']).cumprod() - 1
        
        # Calculate outperformance
        total_return = aligned_returns['cum_returns'].iloc[-1]
        bench_return = aligned_returns['cum_benchmark'].iloc[-1]
        outperformance = total_return - bench_return
        
        # Calculate correlation
        correlation = aligned_returns['returns'].corr(aligned_returns['benchmark_returns'])
        
        # Calculate up/down capture
        up_market = aligned_returns[aligned_returns['benchmark_returns'] > 0]
        down_market = aligned_returns[aligned_returns['benchmark_returns'] < 0]
        
        up_capture = up_market['returns'].mean() / up_market['benchmark_returns'].mean() if len(up_market) > 0 and up_market['benchmark_returns'].mean() != 0 else 0
        down_capture = down_market['returns'].mean() / down_market['benchmark_returns'].mean() if len(down_market) > 0 and down_market['benchmark_returns'].mean() != 0 else 0
        
        # Calculate tracking error
        tracking_error = np.std(aligned_returns['returns'] - aligned_returns['benchmark_returns']) * np.sqrt(252)
        
        return {
            'benchmark_symbol': self.config['benchmark_symbol'],
            'strategy_return': total_return * 100,
            'benchmark_return': bench_return * 100,
            'outperformance': outperformance * 100,
            'correlation': correlation,
            'up_capture': up_capture * 100,
            'down_capture': down_capture * 100,
            'tracking_error': tracking_error * 100
        }
    
    def _get_benchmark_returns(self, start_date, end_date):
        """
        Get benchmark returns.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            DataFrame: Benchmark returns
        """
        benchmark_symbol = self.config['benchmark_symbol']
        
        try:
            # Query database
            query = {
                'symbol': benchmark_symbol,
                'timeframe': 'day',
                'timestamp': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            # Sort by timestamp
            cursor = self.db.market_data_collection.find(query).sort('timestamp', 1)
            
            # Convert to DataFrame
            benchmark_data = pd.DataFrame(list(cursor))
            
            if len(benchmark_data) == 0:
                self.logger.warning(f"No benchmark data found for {benchmark_symbol}")
                return None
            
            # Calculate returns
            benchmark_data['returns'] = benchmark_data['close'].pct_change().fillna(0)
            
            # Return DataFrame with date and returns
            return benchmark_data[['timestamp', 'returns']].rename(
                columns={'timestamp': 'date', 'returns': 'benchmark_returns'}
            )
            
        except Exception as e:
            self.logger.error(f"Error getting benchmark returns: {e}")
            return None
    
    def _align_returns(self, equity_df, benchmark_returns):
        """
        Align strategy and benchmark returns by date.
        
        Args:
            equity_df (DataFrame): Equity curve data
            benchmark_returns (DataFrame): Benchmark returns
            
        Returns:
            DataFrame: Aligned returns
        """
        if benchmark_returns is None:
            return pd.DataFrame()
        
        # Create copy of equity DataFrame
        strategy_returns = equity_df[['date', 'returns']].copy()
        
        # Merge on date
        aligned = pd.merge(
            strategy_returns,
            benchmark_returns,
            on='date',
            how='inner'
        )
        
        return aligned
    
    def _calculate_max_drawdown(self, equity_df):
        """
        Calculate maximum drawdown.
        
        Args:
            equity_df (DataFrame): Equity curve data
            
        Returns:
            float: Maximum drawdown
        """
        if len(equity_df) < 2:
            return 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_df['equity'].values)
        
        # Calculate drawdown
        drawdown = (running_max - equity_df['equity'].values) / running_max
        
        # Get maximum drawdown
        max_drawdown = np.max(drawdown)
        
        return max_drawdown
    
    def _generate_visualizations(self, equity_df, positions, benchmark_returns):
        """
        Generate visualizations.
        
        Args:
            equity_df (DataFrame): Equity curve data
            positions (list): Trade positions
            benchmark_returns (DataFrame): Benchmark returns
            
        Returns:
            dict: Visualization images as Base64
        """
        visualizations = {}
        
        # Equity curve
        equity_curve_img = self._visualize_equity_curve(equity_df, benchmark_returns)
        if equity_curve_img:
            visualizations['equity_curve'] = equity_curve_img
        
        # Drawdown chart
        drawdown_img = self._visualize_drawdowns(equity_df)
        if drawdown_img:
            visualizations['drawdowns'] = drawdown_img
        
        # Monthly returns heatmap
        monthly_returns_img = self._visualize_monthly_returns(equity_df)
        if monthly_returns_img:
            visualizations['monthly_returns'] = monthly_returns_img
        
        # Trade analysis
        if positions:
            trades_img = self._visualize_trades(positions)
            if trades_img:
                visualizations['trades'] = trades_img
        
        # Return distribution
        returns_dist_img = self._visualize_returns_distribution(equity_df)
        if returns_dist_img:
            visualizations['returns_distribution'] = returns_dist_img
        
        return visualizations
    
    def _visualize_equity_curve(self, equity_df, benchmark_returns):
        """
        Visualize equity curve.
        
        Args:
            equity_df (DataFrame): Equity curve data
            benchmark_returns (DataFrame): Benchmark returns
            
        Returns:
            str: Base64 encoded image
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot equity curve
            plt.plot(equity_df['date'], equity_df['equity'], label='Strategy', linewidth=2)
            
            # Plot benchmark if available
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                # Align dates
                aligned_returns = self._align_returns(equity_df, benchmark_returns)
                
                if len(aligned_returns) > 0:
                    # Calculate benchmark equity
                    initial_equity = equity_df['equity'].iloc[0]
                    aligned_returns['bench_equity'] = initial_equity * (1 + aligned_returns['cum_benchmark'])
                    
                    # Plot benchmark
                    plt.plot(aligned_returns['date'], aligned_returns['bench_equity'], 
                             label=self.config['benchmark_symbol'], linewidth=2, alpha=0.7, linestyle='--')
            
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format y-axis as currency
            from matplotlib.ticker import FuncFormatter
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode as Base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error visualizing equity curve: {e}")
            return None
    
    def _visualize_drawdowns(self, equity_df):
        """
        Visualize drawdowns.
        
        Args:
            equity_df (DataFrame): Equity curve data
            
        Returns:
            str: Base64 encoded image
        """
        try:
            # Calculate running maximum
            equity_df['peak'] = equity_df['equity'].cummax()
            
            # Calculate drawdown percentage
            equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak'] * 100
            
            plt.figure(figsize=(12, 6))
            
            # Plot drawdowns
            plt.fill_between(equity_df['date'], 0, -equity_df['drawdown'], color='red', alpha=0.3)
            plt.plot(equity_df['date'], -equity_df['drawdown'], color='red', linewidth=1)
            
            plt.title('Drawdowns')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            
            # Set y-axis limit
            max_drawdown = equity_df['drawdown'].max()
            plt.ylim(-max_drawdown * 1.1, 0)
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode as Base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error visualizing drawdowns: {e}")
            return None
    
    def _visualize_monthly_returns(self, equity_df):
        """
        Visualize monthly returns as a heatmap.
        
        Args:
            equity_df (DataFrame): Equity curve data
            
        Returns:
            str: Base64 encoded image
        """
        try:
            # Set date as index
            equity_df = equity_df.set_index('date')
            
            # Resample to month-end and calculate returns
            monthly_returns = equity_df['equity'].resample('M').last().pct_change().fillna(
                (equity_df['equity'].resample('M').last().iloc[0] / equity_df['equity'].iloc[0]) - 1
            )
            
            # Convert to DataFrame with year and month columns
            returns_df = pd.DataFrame({
                'returns': monthly_returns,
                'year': monthly_returns.index.year,
                'month': monthly_returns.index.month
            })
            
            # Pivot to create heatmap data
            heatmap_data = returns_df.pivot(index='year', columns='month', values='returns')
            
            # Define month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            plt.figure(figsize=(12, 6))
            
            # Create heatmap
            heatmap = plt.pcolormesh(heatmap_data.values, cmap='RdYlGn', vmin=-0.1, vmax=0.1)
            
            # Add colorbar
            cbar = plt.colorbar(heatmap)
            cbar.set_label('Returns (%)')
            
            # Configure axes
            plt.yticks(np.arange(0.5, len(heatmap_data.index)), heatmap_data.index)
            plt.xticks(np.arange(0.5, 13), month_names)
            
            # Add text annotations
            for i in range(len(heatmap_data.index)):
                for j in range(len(heatmap_data.columns)):
                    value = heatmap_data.iloc[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if abs(value) > 0.05 else 'black'
                        plt.text(j + 0.5, i + 0.5, f'{value:.1%}',
                                 ha='center', va='center', color=text_color)
            
            plt.title('Monthly Returns Heatmap')
            plt.tight_layout()
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode as Base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error visualizing monthly returns: {e}")
            return None
    
    def _visualize_trades(self, positions):
        """
        Visualize trade analysis.
        
        Args:
            positions (list): Trade positions
            
        Returns:
            str: Base64 encoded image
        """
        try:
            # Convert to DataFrame
            trades_df = pd.DataFrame(positions)
            
            plt.figure(figsize=(15, 10))
            
            # Create 2x2 subplot
            plt.subplot(2, 2, 1)
            
            # Plot trade P&L
            profits = [p['profit_loss'] for p in positions if p['profit_loss'] > 0]
            losses = [p['profit_loss'] for p in positions if p['profit_loss'] <= 0]
            
            plt.bar(['Profits', 'Losses'], [sum(profits), sum(losses)], color=['green', 'red'])
            plt.title('Total Profits vs Losses')
            plt.grid(True, alpha=0.3)
            
            # Plot trade P&L distribution
            plt.subplot(2, 2, 2)
            
            plt.hist(trades_df['profit_loss_pct'] * 100, bins=20, color='blue', alpha=0.7)
            plt.axvline(0, color='red', linestyle='--')
            plt.title('Trade P&L Distribution (%)')
            plt.xlabel('P&L %')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Plot trade durations
            plt.subplot(2, 2, 3)
            
            plt.hist(trades_df['holding_days'], bins=10, color='green', alpha=0.7)
            plt.title('Trade Duration Distribution')
            plt.xlabel('Holding Days')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Plot exit reasons
            plt.subplot(2, 2, 4)
            
            exit_reasons = trades_df['exit_reason'].value_counts()
            plt.pie(exit_reasons, labels=exit_reasons.index, autopct='%1.1f%%', startangle=90, wedgeprops={'alpha': 0.7})
            plt.title('Exit Reasons')
            
            plt.tight_layout()
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode as Base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error visualizing trades: {e}")
            return None
    
    def _visualize_returns_distribution(self, equity_df):
        """
        Visualize returns distribution.
        
        Args:
            equity_df (DataFrame): Equity curve data
            
        Returns:
            str: Base64 encoded image
        """
        try:
            # Daily returns
            returns = equity_df['returns'].values * 100  # Convert to percentage
            
            plt.figure(figsize=(12, 6))
            
            # Plot histogram
            plt.hist(returns, bins=50, alpha=0.7, color='blue', density=True)
            
            # Plot normal distribution
            x = np.linspace(min(returns), max(returns), 100)
            plt.plot(x, norm.pdf(x, np.mean(returns), np.std(returns)), 'r-', linewidth=2)
            
            # Add mean and std lines
            plt.axvline(np.mean(returns), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
            plt.axvline(0, color='black', linestyle='-', linewidth=1)
            
            # Add VaR line
            confidence_level = self.config['confidence_level']
            var = -np.percentile(returns, 100 * (1 - confidence_level))
            plt.axvline(-var, color='red', linestyle='--', linewidth=2, label=f'VaR ({confidence_level*100:.0f}%): {var:.2f}%')
            
            plt.title('Daily Returns Distribution')
            plt.xlabel('Daily Return (%)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode as Base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error visualizing returns distribution: {e}")
            return None
    
    def _save_performance_analysis(self, analysis, backtest_id):
        """
        Save performance analysis to database.
        
        Args:
            analysis (dict): Performance analysis
            backtest_id: Backtest ID
            
        Returns:
            str: Analysis ID
        """
        try:
            # Create analysis document
            analysis_doc = {
                'backtest_id': backtest_id,
                'timestamp': datetime.now(),
                'strategy': analysis['strategy'],
                'timeframe': analysis['timeframe'],
                'start_date': analysis['start_date'],
                'end_date': analysis['end_date'],
                'basic_metrics': analysis['basic_metrics'],
                'return_metrics': analysis['return_metrics'],
                'risk_metrics': analysis['risk_metrics'],
                'drawdown_analysis': analysis['drawdown_analysis'],
                'trade_analysis': analysis['trade_analysis'],
                'benchmark_comparison': analysis['benchmark_comparison']
            }
            
            # Add visualizations
            if 'visualizations' in analysis:
                analysis_doc['visualizations'] = analysis['visualizations']
            
            # Insert into database
            result = self.db.performance_analysis_collection.insert_one(analysis_doc)
            analysis_id = str(result.inserted_id)
            
            self.logger.info(f"Saved performance analysis with ID: {analysis_id}")
            
            return analysis_id
            
        except Exception as e:
            self.logger.error(f"Error saving performance analysis: {e}")
            return None
    
    def compare_strategies(self, backtest_ids):
        """
        Compare multiple backtested strategies.
        
        Args:
            backtest_ids (list): List of backtest IDs
            
        Returns:
            dict: Strategy comparison
        """
        if not backtest_ids:
            return None
        
        self.logger.info(f"Comparing {len(backtest_ids)} strategies")
        
        # Get backtest results
        backtests = []
        
        for backtest_id in backtest_ids:
            backtest = self.get_backtest_results(backtest_id)
            if backtest:
                backtests.append(backtest)
        
        if not backtests:
            self.logger.error("No valid backtest results found")
            return None
        
        # Extract key metrics
        comparison = {
            'strategies': [],
            'timeframes': [],
            'start_dates': [],
            'end_dates': [],
            'initial_capitals': [],
            'final_capitals': [],
            'total_returns': [],
            'cagrs': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'win_rates': [],
            'profit_factors': [],
            'total_trades': []
        }
        
        for backtest in backtests:
            comparison['strategies'].append(backtest.get('strategy'))
            comparison['timeframes'].append(backtest.get('timeframe'))
            comparison['start_dates'].append(backtest.get('start_date'))
            comparison['end_dates'].append(backtest.get('end_date'))
            comparison['initial_capitals'].append(backtest.get('initial_capital'))
            comparison['final_capitals'].append(backtest.get('final_capital'))
            comparison['total_returns'].append(
                (backtest.get('final_capital') / backtest.get('initial_capital') - 1) * 100
            )
            
            stats = backtest.get('statistics', {})
            comparison['cagrs'].append(stats.get('cagr', 0) * 100)
            comparison['sharpe_ratios'].append(stats.get('sharpe_ratio', 0))
            comparison['max_drawdowns'].append(stats.get('max_drawdown', 0) * 100)
            comparison['win_rates'].append(stats.get('win_rate', 0) * 100)
            comparison['profit_factors'].append(stats.get('profit_factor', 0))
            comparison['total_trades'].append(stats.get('total_trades', 0))
        
        # Generate visualizations
        try:
            visualizations = self._visualize_strategy_comparison(backtests)
            comparison['visualizations'] = visualizations
        except Exception as e:
            self.logger.error(f"Error generating comparison visualizations: {e}")
        
        # Rank strategies
        ranks = self._rank_strategies(comparison)
        comparison['ranks'] = ranks
        
        # Save comparison
        try:
            comparison_id = self._save_strategy_comparison(comparison)
            comparison['_id'] = comparison_id
        except Exception as e:
            self.logger.error(f"Error saving strategy comparison: {e}")
        
        return comparison
    
    def _visualize_strategy_comparison(self, backtests):
        """
        Visualize strategy comparison.
        
        Args:
            backtests (list): List of backtest results
            
        Returns:
            dict: Visualization images as Base64
        """
        visualizations = {}
        
        # Equity curves
        equity_img = self._visualize_equity_comparison(backtests)
        if equity_img:
            visualizations['equity_comparison'] = equity_img
        
        # Return metrics
        metrics_img = self._visualize_metrics_comparison(backtests)
        if metrics_img:
            visualizations['metrics_comparison'] = metrics_img
        
        # Drawdown comparison
        drawdown_img = self._visualize_drawdown_comparison(backtests)
        if drawdown_img:
            visualizations['drawdown_comparison'] = drawdown_img
        
        return visualizations
    
    def _visualize_equity_comparison(self, backtests):
        """
        Visualize equity curve comparison.
        
        Args:
            backtests (list): List of backtest results
            
        Returns:
            str: Base64 encoded image
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Find common date range
            min_date = None
            max_date = None
            
            for backtest in backtests:
                dates = []
                timestamps = backtest.get('equity_timestamps', [])
                
                for ts in timestamps:
                    if isinstance(ts, str):
                        try:
                            date = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') if ' ' in ts else datetime.strptime(ts, '%Y-%m-%d')
                            dates.append(date)
                        except:
                            continue
                    elif isinstance(ts, datetime):
                        dates.append(ts)
                
                if dates:
                    backtest_min = min(dates)
                    backtest_max = max(dates)
                    
                    if min_date is None or backtest_min < min_date:
                        min_date = backtest_min
                    
                    if max_date is None or backtest_max > max_date:
                        max_date = backtest_max
            
            # Plot equity curves
            for backtest in backtests:
                strategy_name = backtest.get('strategy')
                
                # Extract equity curve data
                dates = []
                equity = backtest.get('equity_curve', [])
                timestamps = backtest.get('equity_timestamps', [])
                
                if len(timestamps) != len(equity):
                    continue
                
                for i, ts in enumerate(timestamps):
                    if isinstance(ts, str):
                        try:
                            date = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') if ' ' in ts else datetime.strptime(ts, '%Y-%m-%d')
                            dates.append(date)
                        except:
                            continue
                    elif isinstance(ts, datetime):
                        dates.append(ts)
                    else:
                        continue
                
                if len(dates) != len(equity):
                    equity = equity[:len(dates)]
                
                # Normalize to percentage return
                initial_equity = equity[0]
                equity_pct = [(e / initial_equity - 1) * 100 for e in equity]
                
                # Plot
                plt.plot(dates, equity_pct, label=strategy_name, linewidth=2)
            
            plt.title('Strategy Return Comparison')
            plt.xlabel('Date')
            plt.ylabel('Return (%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode as Base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error visualizing equity comparison: {e}")
            return None
    
    def _visualize_metrics_comparison(self, backtests):
        """
        Visualize metrics comparison.
        
        Args:
            backtests (list): List of backtest results
            
        Returns:
            str: Base64 encoded image
        """
        try:
            # Extract metrics
            strategies = []
            returns = []
            sharpes = []
            drawdowns = []
            win_rates = []
            
            for backtest in backtests:
                strategies.append(backtest.get('strategy'))
                
                returns.append(
                    (backtest.get('final_capital') / backtest.get('initial_capital') - 1) * 100
                )
                
                stats = backtest.get('statistics', {})
                sharpes.append(stats.get('sharpe_ratio', 0))
                drawdowns.append(stats.get('max_drawdown', 0) * 100)
                win_rates.append(stats.get('win_rate', 0) * 100)
            
            plt.figure(figsize=(14, 10))
            
            # Return comparison
            plt.subplot(2, 2, 1)
            plt.bar(strategies, returns, color='blue', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.title('Total Return (%)')
            plt.grid(True, alpha=0.3)
            
            # Sharpe ratio comparison
            plt.subplot(2, 2, 2)
            plt.bar(strategies, sharpes, color='green', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.title('Sharpe Ratio')
            plt.grid(True, alpha=0.3)
            
            # Max drawdown comparison
            plt.subplot(2, 2, 3)
            plt.bar(strategies, drawdowns, color='red', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.title('Max Drawdown (%)')
            plt.grid(True, alpha=0.3)
            
            # Win rate comparison
            plt.subplot(2, 2, 4)
            plt.bar(strategies, win_rates, color='purple', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.title('Win Rate (%)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode as Base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error visualizing metrics comparison: {e}")
            return None
    
    def _visualize_drawdown_comparison(self, backtests):
        """
        Visualize drawdown comparison.
        
        Args:
            backtests (list): List of backtest results
            
        Returns:
            str: Base64 encoded image
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot drawdowns for each strategy
            for backtest in backtests:
                strategy_name = backtest.get('strategy')
                
                # Extract equity curve data
                dates = []
                equity = backtest.get('equity_curve', [])
                timestamps = backtest.get('equity_timestamps', [])
                
                if len(timestamps) != len(equity):
                    continue
                
                for i, ts in enumerate(timestamps):
                    if isinstance(ts, str):
                        try:
                            date = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') if ' ' in ts else datetime.strptime(ts, '%Y-%m-%d')
                            dates.append(date)
                        except:
                            continue
                    elif isinstance(ts, datetime):
                        dates.append(ts)
                    else:
                        continue
                
                if len(dates) != len(equity):
                    equity = equity[:len(dates)]
                
                # Calculate drawdowns
                equity_series = pd.Series(equity, index=dates)
                running_max = equity_series.cummax()
                drawdown = (running_max - equity_series) / running_max * 100
                
                # Plot
                plt.plot(dates, -drawdown, label=strategy_name, linewidth=2)
            
            plt.title('Drawdown Comparison')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode as Base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error visualizing drawdown comparison: {e}")
            return None
    
    def _rank_strategies(self, comparison):
        """
        Rank strategies based on multiple metrics.
        
        Args:
            comparison (dict): Strategy comparison data
            
        Returns:
            dict: Strategy rankings
        """
        # Define metrics to rank (higher is better)
        metrics = {
            'total_return': comparison['total_returns'],
            'cagr': comparison['cagrs'],
            'sharpe_ratio': comparison['sharpe_ratios'],
            'max_drawdown': [-d for d in comparison['max_drawdowns']],  # Invert so higher is better
            'win_rate': comparison['win_rates'],
            'profit_factor': comparison['profit_factors']
        }
        
        # Calculate ranks for each metric
        ranks = {}
        for metric, values in metrics.items():
            # Sort indices by values (descending)
            sorted_indices = np.argsort(values)[::-1]
            
            # Assign ranks
            metric_ranks = np.zeros(len(values))
            for rank, idx in enumerate(sorted_indices):
                metric_ranks[idx] = rank + 1
            
            ranks[metric] = metric_ranks.tolist()
        
        # Calculate overall rank (average rank across all metrics
        # Calculate overall rank (average rank across all metrics)
        overall_ranks = np.zeros(len(comparison['strategies']))
        
        for metric_ranks in ranks.values():
            overall_ranks += np.array(metric_ranks)
        
        overall_ranks = overall_ranks / len(ranks)
        
        # Add overall ranks to result
        ranks['overall'] = overall_ranks.tolist()
        
        # Add strategy names for reference
        ranks['strategies'] = comparison['strategies']
        
        return ranks
    
    def _save_strategy_comparison(self, comparison):
        """
        Save strategy comparison to database.
        
        Args:
            comparison (dict): Strategy comparison data
            
        Returns:
            str: Comparison ID
        """
        try:
            # Create comparison document
            comparison_doc = {
                'timestamp': datetime.now(),
                'strategies': comparison['strategies'],
                'timeframes': comparison['timeframes'],
                'start_dates': comparison['start_dates'],
                'end_dates': comparison['end_dates'],
                'metrics': {
                    'total_returns': comparison['total_returns'],
                    'cagrs': comparison['cagrs'],
                    'sharpe_ratios': comparison['sharpe_ratios'],
                    'max_drawdowns': comparison['max_drawdowns'],
                    'win_rates': comparison['win_rates'],
                    'profit_factors': comparison['profit_factors'],
                    'total_trades': comparison['total_trades']
                },
                'ranks': comparison['ranks']
            }
            
            # Add visualizations
            if 'visualizations' in comparison:
                comparison_doc['visualizations'] = comparison['visualizations']
            
            # Insert into database
            result = self.db.strategy_comparison_collection.insert_one(comparison_doc)
            comparison_id = str(result.inserted_id)
            
            self.logger.info(f"Saved strategy comparison with ID: {comparison_id}")
            
            return comparison_id
            
        except Exception as e:
            self.logger.error(f"Error saving strategy comparison: {e}")
            return None