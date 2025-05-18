# backtesting/visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import io
import base64
import logging
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

class BacktestVisualizer:
    """
    Visualization tools for backtest results.
    """
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the backtest visualizer.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'default_figsize': (12, 8),
            'theme': 'default',
            'benchmark_symbol': '^NSEI',  # Nifty 50 for Indian market
            'export_format': 'png',
            'interactive': True  # Use interactive plots when possible
        }
        
        # Set Matplotlib style
        plt.style.use('ggplot')
    
    def set_config(self, config):
        """
        Set visualizer configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated visualizer configuration: {self.config}")
        
        # Update Matplotlib style
        if self.config['theme'] == 'dark':
            plt.style.use('dark_background')
        else:
            plt.style.use('ggplot')
    
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
    
    def prepare_data(self, backtest_results):
        """
        Prepare backtest data for visualization.
        
        Args:
            backtest_results (dict): Backtest results
            
        Returns:
            dict: Prepared data
        """
        if not backtest_results:
            return None
        
        try:
            # Extract data
            equity_curve = backtest_results.get('equity_curve', [])
            equity_timestamps = backtest_results.get('equity_timestamps', [])
            positions = backtest_results.get('positions', [])
            
            if not equity_curve or not equity_timestamps:
                self.logger.error("No equity curve data in backtest results")
                return None
                
            # Convert timestamps from string to datetime if needed
            dates = []
            for ts in equity_timestamps:
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
            
            # Create equity DataFrame
            equity_df = pd.DataFrame({
                'date': dates,
                'equity': equity_curve
            })
            
            # Calculate returns
            equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
            
            # Calculate drawdowns
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak'] * 100
            
            # Calculate additional metrics
            equity_df['cumulative_return'] = (equity_df['equity'] / equity_df['equity'].iloc[0] - 1) * 100
            
            # Convert positions to DataFrame if available
            positions_df = pd.DataFrame(positions) if positions else pd.DataFrame()
            
            if not positions_df.empty:
                # Convert dates
                if 'entry_date' in positions_df.columns:
                    positions_df['entry_date'] = pd.to_datetime(positions_df['entry_date'])
                    
                if 'exit_date' in positions_df.columns:
                    positions_df['exit_date'] = pd.to_datetime(positions_df['exit_date'])
            
            # Get benchmark data if available
            benchmark_df = self._get_benchmark_data(
                dates[0] if dates else None,
                dates[-1] if dates else None
            )
            
            return {
                'equity_df': equity_df,
                'positions_df': positions_df,
                'benchmark_df': benchmark_df,
                'strategy': backtest_results.get('strategy'),
                'timeframe': backtest_results.get('timeframe'),
                'start_date': backtest_results.get('start_date'),
                'end_date': backtest_results.get('end_date'),
                'initial_capital': backtest_results.get('initial_capital'),
                'final_capital': backtest_results.get('final_capital'),
                'statistics': backtest_results.get('statistics', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return None
    
    def _get_benchmark_data(self, start_date, end_date):
        """
        Get benchmark data.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            DataFrame: Benchmark data
        """
        if not start_date or not end_date:
            return None
            
        try:
            benchmark_symbol = self.config['benchmark_symbol']
            
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
            
            # Set timestamp as index
            benchmark_data.set_index('timestamp', inplace=True)
            
            # Calculate returns
            benchmark_data['returns'] = benchmark_data['close'].pct_change().fillna(0)
            
            # Calculate cumulative returns
            benchmark_data['cumulative_return'] = (benchmark_data['close'] / benchmark_data['close'].iloc[0] - 1) * 100
            
            return benchmark_data
            
        except Exception as e:
            self.logger.warning(f"Error getting benchmark data: {e}")
            return None
    
    def create_equity_curve_plot(self, data, include_benchmark=True, include_drawdown=True, interactive=None):
        """
        Create equity curve plot.
        
        Args:
            data (dict): Prepared data
            include_benchmark (bool): Include benchmark in plot
            include_drawdown (bool): Include drawdown subplot
            interactive (bool): Use interactive plot
            
        Returns:
            str: Base64 encoded image or HTML
        """
        if not data or 'equity_df' not in data:
            return None
            
        interactive = interactive if interactive is not None else self.config['interactive']
        
        try:
            equity_df = data['equity_df']
            benchmark_df = data['benchmark_df'] if include_benchmark else None
            
            if interactive:
                return self._create_interactive_equity_plot(equity_df, benchmark_df, data, include_drawdown)
            else:
                return self._create_static_equity_plot(equity_df, benchmark_df, data, include_drawdown)
                
        except Exception as e:
            self.logger.error(f"Error creating equity curve plot: {e}")
            return None
    
    def _create_static_equity_plot(self, equity_df, benchmark_df, data, include_drawdown):
        """
        Create static equity curve plot.
        
        Args:
            equity_df (DataFrame): Equity curve data
            benchmark_df (DataFrame): Benchmark data
            data (dict): Full prepared data
            include_drawdown (bool): Include drawdown subplot
            
        Returns:
            str: Base64 encoded image
        """
        # Create figure
        if include_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config['default_figsize'], gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        else:
            fig, ax1 = plt.subplots(figsize=self.config['default_figsize'])
        
        # Plot equity curve
        strategy_name = data.get('strategy', 'Strategy')
        ax1.plot(equity_df['date'], equity_df['cumulative_return'], label=strategy_name, linewidth=2)
        
        # Plot benchmark if available
        if benchmark_df is not None:
            # Create date-aligned benchmark data
            benchmark_aligned = pd.DataFrame(index=equity_df['date'])
            
            # Interpolate benchmark data to match equity dates
            benchmark_returns = benchmark_df['cumulative_return'].reindex(
                benchmark_aligned.index, method='ffill'
            )
            
            if not benchmark_returns.empty:
                ax1.plot(equity_df['date'], benchmark_returns, label=self.config['benchmark_symbol'], 
                         linewidth=1.5, linestyle='--', alpha=0.7)
        
        # Format primary plot
        ax1.set_title(f"Strategy Performance: {strategy_name}")
        ax1.set_ylabel('Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Add drawdown subplot if requested
        if include_drawdown:
            # Plot drawdowns
            ax2.fill_between(equity_df['date'], 0, -equity_df['drawdown'], color='red', alpha=0.3)
            ax2.plot(equity_df['date'], -equity_df['drawdown'], color='red', linewidth=1)
            
            # Format drawdown plot
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            
            # Set y-axis limit
            max_drawdown = equity_df['drawdown'].max()
            ax2.set_ylim(-max_drawdown * 1.1, 0)
        else:
            ax1.set_xlabel('Date')
        
        plt.tight_layout()
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format=self.config['export_format'], dpi=100)
        plt.close()
        
        # Encode as Base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
    
    def _create_interactive_equity_plot(self, equity_df, benchmark_df, data, include_drawdown):
        """
        Create interactive equity curve plot.
        
        Args:
            equity_df (DataFrame): Equity curve data
            benchmark_df (DataFrame): Benchmark data
            data (dict): Full prepared data
            include_drawdown (bool): Include drawdown subplot
            
        Returns:
            str: HTML content
        """
        strategy_name = data.get('strategy', 'Strategy')
        
        if include_drawdown:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05,
                               subplot_titles=[f"Strategy Performance: {strategy_name}", "Drawdown (%)"],
                               row_heights=[0.7, 0.3])
        else:
            fig = go.Figure()
        
        # Plot equity curve
        equity_trace = go.Scatter(
            x=equity_df['date'],
            y=equity_df['cumulative_return'],
            mode='lines',
            name=strategy_name,
            line=dict(width=2)
        )
        
        if include_drawdown:
            fig.add_trace(equity_trace, row=1, col=1)
        else:
            fig.add_trace(equity_trace)
        
        # Plot benchmark if available
        if benchmark_df is not None:
            # Create date-aligned benchmark data
            benchmark_aligned = pd.DataFrame(index=equity_df['date'])
            
            # Interpolate benchmark data to match equity dates
            benchmark_returns = benchmark_df['cumulative_return'].reindex(
                benchmark_aligned.index, method='ffill'
            )
            
            if not benchmark_returns.empty:
                benchmark_trace = go.Scatter(
                    x=equity_df['date'],
                    y=benchmark_returns,
                    mode='lines',
                    name=self.config['benchmark_symbol'],
                    line=dict(width=1.5, dash='dash'),
                    opacity=0.7
                )
                
                if include_drawdown:
                    fig.add_trace(benchmark_trace, row=1, col=1)
                else:
                    fig.add_trace(benchmark_trace)
        
        # Add drawdown subplot if requested
        if include_drawdown:
            drawdown_trace = go.Scatter(
                x=equity_df['date'],
                y=-equity_df['drawdown'],
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(color='red', width=1)
            )
            
            fig.add_trace(drawdown_trace, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"Strategy Performance: {strategy_name}",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white" if self.config['theme'] == 'default' else "plotly_dark"
        )
        
        if include_drawdown:
            fig.update_yaxes(title_text="Return (%)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
        else:
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Return (%)"
            )
        
        return fig.to_html(full_html=False)
    
    def create_monthly_returns_heatmap(self, data, interactive=None):
        """
        Create monthly returns heatmap.
        
        Args:
            data (dict): Prepared data
            interactive (bool): Use interactive plot
            
        Returns:
            str: Base64 encoded image or HTML
        """
        if not data or 'equity_df' not in data:
            return None
            
        interactive = interactive if interactive is not None else self.config['interactive']
        
        try:
            equity_df = data['equity_df'].copy()
            
            # Set date as index
            equity_df.set_index('date', inplace=True)
            
            # Calculate monthly returns
            monthly_returns = equity_df['equity'].resample('M').last().pct_change().fillna(
                (equity_df['equity'].resample('M').last().iloc[0] / equity_df['equity'].iloc[0]) - 1
            )
            
            # Convert to DataFrame with year and month columns
            returns_df = pd.DataFrame({
                'returns': monthly_returns * 100,  # Convert to percentage
                'year': monthly_returns.index.year,
                'month': monthly_returns.index.month
            })
            
            # Pivot to create heatmap data
            heatmap_data = returns_df.pivot(index='year', columns='month', values='returns')
            
            if interactive:
                return self._create_interactive_monthly_heatmap(heatmap_data, data)
            else:
                return self._create_static_monthly_heatmap(heatmap_data, data)
                
        except Exception as e:
            self.logger.error(f"Error creating monthly returns heatmap: {e}")
            return None
    
    def _create_static_monthly_heatmap(self, heatmap_data, data):
        """
        Create static monthly returns heatmap.
        
        Args:
            heatmap_data (DataFrame): Monthly returns data
            data (dict): Full prepared data
            
        Returns:
            str: Base64 encoded image
        """
        # Define month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        plt.figure(figsize=self.config['default_figsize'])
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Returns (%)'})
        
        # Configure axes
        plt.yticks(np.arange(0.5, len(heatmap_data.index) + 0.5), heatmap_data.index)
        plt.xticks(np.arange(0.5, 13), month_names)
        
        strategy_name = data.get('strategy', 'Strategy')
        plt.title(f'Monthly Returns Heatmap: {strategy_name}')
        plt.tight_layout()
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format=self.config['export_format'], dpi=100)
        plt.close()
        
        # Encode as Base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
    
    def _create_interactive_monthly_heatmap(self, heatmap_data, data):
        """
        Create interactive monthly returns heatmap.
        
        Args:
            heatmap_data (DataFrame): Monthly returns data
            data (dict): Full prepared data
            
        Returns:
            str: HTML content
        """
        # Create annotated heatmap
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Prepare data for heatmap
        z_data = heatmap_data.values
        x_data = month_names[:heatmap_data.shape[1]]
        y_data = heatmap_data.index.astype(str).tolist()
        
        strategy_name = data.get('strategy', 'Strategy')
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_data,
            y=y_data,
            colorscale='RdYlGn',
            zmid=0,
            text=[[f"{val:.1f}%" if not np.isnan(val) else "" for val in row] for row in z_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Monthly Returns Heatmap: {strategy_name}',
            xaxis_title='Month',
            yaxis_title='Year',
            template="plotly_white" if self.config['theme'] == 'default' else "plotly_dark"
        )
        
        return fig.to_html(full_html=False)
    
    def create_trade_analysis_plot(self, data, interactive=None):
        """
        Create trade analysis plot.
        
        Args:
            data (dict): Prepared data
            interactive (bool): Use interactive plot
            
        Returns:
            str: Base64 encoded image or HTML
        """
        if not data or 'positions_df' not in data or data['positions_df'].empty:
            return None
            
        interactive = interactive if interactive is not None else self.config['interactive']
        
        try:
            positions_df = data['positions_df'].copy()
            
            if interactive:
                return self._create_interactive_trade_analysis(positions_df, data)
            else:
                return self._create_static_trade_analysis(positions_df, data)
                
        except Exception as e:
            self.logger.error(f"Error creating trade analysis plot: {e}")
            return None
    
    def _create_static_trade_analysis(self, positions_df, data):
        """
        Create static trade analysis plot.
        
        Args:
            positions_df (DataFrame): Trade positions data
            data (dict): Full prepared data
            
        Returns:
            str: Base64 encoded image
        """
        plt.figure(figsize=(15, 10))
        
        # Create 2x2 subplot
        plt.subplot(2, 2, 1)
        
        # Plot trade P&L
        profits = positions_df[positions_df['profit_loss'] > 0]['profit_loss'].sum()
        losses = positions_df[positions_df['profit_loss'] <= 0]['profit_loss'].sum()
        
        plt.bar(['Profits', 'Losses'], [profits, losses], color=['green', 'red'])
        plt.title('Total Profits vs Losses')
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        from matplotlib.ticker import FuncFormatter
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
        
        # Plot trade P&L distribution
        plt.subplot(2, 2, 2)
        
        plt.hist(positions_df['profit_loss_pct'] * 100, bins=20, color='blue', alpha=0.7)
        plt.axvline(0, color='red', linestyle='--')
        plt.title('Trade P&L Distribution (%)')
        plt.xlabel('P&L %')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot trade durations
        plt.subplot(2, 2, 3)
        
        plt.hist(positions_df['holding_days'], bins=10, color='green', alpha=0.7)
        plt.title('Trade Duration Distribution')
        plt.xlabel('Holding Days')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot exit reasons
        plt.subplot(2, 2, 4)
        
        exit_reasons = positions_df['exit_reason'].value_counts()
        plt.pie(exit_reasons, labels=exit_reasons.index, autopct='%1.1f%%', startangle=90, wedgeprops={'alpha': 0.7})
        plt.title('Exit Reasons')
        
        strategy_name = data.get('strategy', 'Strategy')
        plt.suptitle(f'Trade Analysis: {strategy_name}', fontsize=16)
        plt.tight_layout()
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format=self.config['export_format'], dpi=100)
        plt.close()
        
        # Encode as Base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
    
    def _create_interactive_trade_analysis(self, positions_df, data):
        """
        Create interactive trade analysis plot.
        
        Args:
            positions_df (DataFrame): Trade positions data
            data (dict): Full prepared data
            
        Returns:
            str: HTML content
        """
        strategy_name = data.get('strategy', 'Strategy')
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Total Profits vs Losses', 'Trade P&L Distribution (%)', 
                           'Trade Duration Distribution', 'Exit Reasons'],
            specs=[[{'type': 'bar'}, {'type': 'histogram'}],
                  [{'type': 'histogram'}, {'type': 'pie'}]]
        )
        
        # Plot trade P&L
        profits = positions_df[positions_df['profit_loss'] > 0]['profit_loss'].sum()
        losses = positions_df[positions_df['profit_loss'] <= 0]['profit_loss'].sum()
        
        fig.add_trace(
            go.Bar(
                x=['Profits', 'Losses'],
                y=[profits, losses],
                marker_color=['green', 'red']
            ),
            row=1, col=1
        )
        
        # Plot trade P&L distribution
        fig.add_trace(
            go.Histogram(
                x=positions_df['profit_loss_pct'] * 100,
                nbinsx=20,
                marker_color='blue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Add vertical line at zero
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=0, y1=1,
            xref="x2", yref="paper",
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=2
        )
        
        # Plot trade durations
        fig.add_trace(
            go.Histogram(
                x=positions_df['holding_days'],
                nbinsx=10,
                marker_color='green',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Plot exit reasons
        exit_reasons = positions_df['exit_reason'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=exit_reasons.index,
                values=exit_reasons.values,
                textinfo='percent',
                hoverinfo='label+percent',
                marker=dict(line=dict(color='#FFFFFF', width=1))
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Trade Analysis: {strategy_name}',
            template="plotly_white" if self.config['theme'] == 'default' else "plotly_dark",
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_yaxes(title_text="Amount", row=1, col=1)
        
        fig.update_xaxes(title_text="P&L %", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Holding Days", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        return fig.to_html(full_html=False)
    
    def create_returns_distribution_plot(self, data, interactive=None):
        """
        Create returns distribution plot.
        
        Args:
            data (dict): Prepared data
            interactive (bool): Use interactive plot
            
        Returns:
            str: Base64 encoded image or HTML
        """
        if not data or 'equity_df' not in data:
            return None
            
        interactive = interactive if interactive is not None else self.config['interactive']
        
        try:
            equity_df = data['equity_df'].copy()
            
            # Daily returns
            returns = equity_df['returns'].values * 100  # Convert to percentage
            
            if interactive:
                return self._create_interactive_returns_distribution(returns, data)
            else:
                return self._create_static_returns_distribution(returns, data)
                
        except Exception as e:
            self.logger.error(f"Error creating returns distribution plot: {e}")
            return None
    
    def _create_static_returns_distribution(self, returns, data):
        """
        Create static returns distribution plot.
        
        Args:
            returns (array): Returns data
            data (dict): Full prepared data
            
        Returns:
            str: Base64 encoded image
        """
        plt.figure(figsize=self.config['default_figsize'])
        
        # Plot histogram
        plt.hist(returns, bins=50, alpha=0.7, color='blue', density=True)
        
        # Plot normal distribution
        from scipy.stats import norm
        x = np.linspace(min(returns), max(returns), 100)
        plt.plot(x, norm.pdf(x, np.mean(returns), np.std(returns)), 'r-', linewidth=2)
        
        # Add mean and std lines
        plt.axvline(np.mean(returns), color='green', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(returns):.2f}%')
        plt.axvline(0, color='black', linestyle='-', linewidth=1)
        
        # Add VaR line
        confidence_level = 0.95
        var = -np.percentile(returns, 100 * (1 - confidence_level))
        plt.axvline(-var, color='red', linestyle='--', linewidth=2, 
                    label=f'VaR ({confidence_level*100:.0f}%): {var:.2f}%')
        
        strategy_name = data.get('strategy', 'Strategy')
        plt.title(f'Daily Returns Distribution: {strategy_name}')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format=self.config['export_format'], dpi=100)
        plt.close()
        
        # Encode as Base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
    
    def _create_interactive_returns_distribution(self, returns, data):
        """
        Create interactive returns distribution plot.
        
        Args:
            returns (array): Returns data
            data (dict): Full prepared data
            
        Returns:
            str: HTML content
        """
        from scipy.stats import norm
        
        strategy_name = data.get('strategy', 'Strategy')
        
        # Create figure
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            opacity=0.7,
            name="Returns",
            histnorm='probability density'
        ))
        
        # Add normal distribution
        x = np.linspace(min(returns), max(returns), 100)
        y = norm.pdf(x, np.mean(returns), np.std(returns))
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        # Add mean line
        mean_value = np.mean(returns)
        fig.add_vline(
            x=mean_value,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Mean: {mean_value:.2f}%",
            annotation_position="top right"
        )
        
        # Add zero line
        fig.add_vline(
            x=0,
            line_width=1,
            line_color="black"
        )
        
        # Add VaR line
        confidence_level = 0.95
        var = -np.percentile(returns, 100 * (1 - confidence_level))
        fig.add_vline(
            x=-var,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR ({confidence_level*100:.0f}%): {var:.2f}%",
            annotation_position="top left"
        )
        
        # Update layout
        fig.update_layout(
            title=f'Daily Returns Distribution: {strategy_name}',
            xaxis_title='Daily Return (%)',
            yaxis_title='Density',
            template="plotly_white" if self.config['theme'] == 'default' else "plotly_dark"
        )
        
        return fig.to_html(full_html=False)
    
    def create_comparison_report(self, backtest_ids, interactive=None):
        """
        Create a comparison report for multiple backtest strategies.
        
        Args:
            backtest_ids (list): List of backtest IDs to compare
            interactive (bool): Use interactive plots
            
        Returns:
            dict: Comparison report data
        """
        interactive = interactive if interactive is not None else self.config['interactive']
        
        try:
            if not backtest_ids or len(backtest_ids) < 2:
                self.logger.error("Need at least two backtest IDs for comparison")
                return None
                
            # Get data for each backtest
            backtests_data = []
            for backtest_id in backtest_ids:
                results = self.get_backtest_results(backtest_id)
                if results:
                    data = self.prepare_data(results)
                    if data:
                        backtests_data.append(data)
            
            if len(backtests_data) < 2:
                self.logger.error("Insufficient valid backtest data for comparison")
                return None
                
            # Create comparison visualizations
            equity_comparison = self._create_equity_comparison(backtests_data, interactive)
            stats_comparison = self._create_statistics_comparison(backtests_data, interactive)
            
            # Build report data
            report = {
                'backtest_ids': backtest_ids,
                'strategies': [data.get('strategy', f'Strategy {i+1}') for i, data in enumerate(backtests_data)],
                'charts': {
                    'equity_comparison': equity_comparison,
                    'statistics_comparison': stats_comparison
                },
                'interactive': interactive
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error creating comparison report: {e}")
            return None
        
    def _create_equity_comparison(self, backtests_data, interactive):
        """
        Create equity curve comparison chart for multiple backtests.
        
        Args:
            backtests_data (list): List of prepared backtest data
            interactive (bool): Use interactive plot
            
        Returns:
            str: Base64 encoded image or HTML
        """
        if interactive:
            fig = go.Figure()
            
            for data in backtests_data:
                equity_df = data['equity_df']
                strategy_name = data.get('strategy', 'Strategy')
                
                fig.add_trace(go.Scatter(
                    x=equity_df['date'],
                    y=equity_df['cumulative_return'],
                    mode='lines',
                    name=strategy_name
                ))
            
            fig.update_layout(
                title='Strategy Performance Comparison',
                xaxis_title='Date',
                yaxis_title='Return (%)',
                template="plotly_white" if self.config['theme'] == 'default' else "plotly_dark"
            )
            
            return fig.to_html(full_html=False)
        else:
            plt.figure(figsize=self.config['default_figsize'])
            
            for data in backtests_data:
                equity_df = data['equity_df']
                strategy_name = data.get('strategy', 'Strategy')
                
                plt.plot(equity_df['date'], equity_df['cumulative_return'], label=strategy_name)
            
            plt.title('Strategy Performance Comparison')
            plt.xlabel('Date')
            plt.ylabel('Return (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            plt.tight_layout()
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format=self.config['export_format'], dpi=100)
            plt.close()
            
            # Encode as Base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str

    def _create_statistics_comparison(self, backtests_data, interactive):
        """
        Create statistics comparison chart for multiple backtests.
        
        Args:
            backtests_data (list): List of prepared backtest data
            interactive (bool): Use interactive plot
            
        Returns:
            str: Base64 encoded image or HTML
        """
        # Extract key statistics
        strategies = []
        metrics = ['CAGR', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor']
        stats_data = []
        
        for data in backtests_data:
            strategy_name = data.get('strategy', 'Strategy')
            strategies.append(strategy_name)
            
            stats = data.get('statistics', {})
            stats_row = [
                stats.get('cagr', 0) * 100,  # Convert to percentage
                stats.get('sharpe_ratio', 0),
                stats.get('max_drawdown', 0) * 100,  # Convert to percentage
                stats.get('win_rate', 0) * 100,  # Convert to percentage
                stats.get('profit_factor', 0)
            ]
            stats_data.append(stats_row)
        
        if interactive:
            fig = go.Figure()
            
            for i, strategy in enumerate(strategies):
                fig.add_trace(go.Bar(
                    x=metrics,
                    y=stats_data[i],
                    name=strategy
                ))
            
            fig.update_layout(
                title='Strategy Statistics Comparison',
                xaxis_title='Metric',
                yaxis_title='Value',
                template="plotly_white" if self.config['theme'] == 'default' else "plotly_dark",
                barmode='group'
            )
            
            return fig.to_html(full_html=False)
        else:
            plt.figure(figsize=self.config['default_figsize'])
            
            x = np.arange(len(metrics))
            width = 0.8 / len(strategies)
            
            for i, strategy in enumerate(strategies):
                offset = (i - len(strategies) / 2 + 0.5) * width
                plt.bar(x + offset, stats_data[i], width, label=strategy)
            
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.title('Strategy Statistics Comparison')
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format=self.config['export_format'], dpi=100)
            plt.close()
            
            # Encode as Base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str

    def export_report(self, report, format='html', filename=None):
        """
        Export backtest report to file.
        
        Args:
            report (dict): Report data
            format (str): Export format (html, pdf, json)
            filename (str): Output filename (optional)
            
        Returns:
            str: Path to exported file
        """
        try:
            if not report:
                return None
                
            # Default filename
            if not filename:
                strategy = report.get('strategy', 'strategy')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"backtest_{strategy}_{timestamp}"
            
            # Export based on format
            if format == 'html':
                return self._export_html_report(report, filename)
            elif format == 'pdf':
                return self._export_pdf_report(report, filename)
            elif format == 'json':
                return self._export_json_report(report, filename)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            return None
            
    def _export_html_report(self, report, filename):
        """Export report as HTML"""
        # Implementation would depend on your HTML template system
        # This is a simplified example
        try:
            import os
            output_path = f"{filename}.html"
            
            # Simple HTML template
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Backtest Report: {report.get('strategy')}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .section {{ margin-bottom: 30px; }}
                    .chart {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>Backtest Report: {report.get('strategy')}</h1>
                <div class="section">
                    <h2>Overview</h2>
                    <p>Period: {report.get('period')}</p>
                    <p>Timeframe: {report.get('timeframe')}</p>
                </div>
                <div class="section">
                    <h2>Equity Curve</h2>
                    <div class="chart">
                        {report['charts']['equity_curve'] if report['interactive'] else f'<img src="data:image/png;base64,{report["charts"]["equity_curve"]}" />'}
                    </div>
                </div>
                <!-- Other sections... -->
            </body>
            </html>
            """
            
            with open(output_path, 'w') as f:
                f.write(html)
                
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting HTML report: {e}")
            return None
            
    def _export_pdf_report(self, report, filename):
        """Export report as PDF"""
        # Implementation would require a PDF library
        # This is a placeholder
        try:
            output_path = f"{filename}.pdf"
            self.logger.warning("PDF export not fully implemented")
            return output_path
        except Exception as e:
            self.logger.error(f"Error exporting PDF report: {e}")
            return None
            
    def _export_json_report(self, report, filename):
        """Export report as JSON"""
        try:
            import os
            import json
            
            output_path = f"{filename}.json"
            
            # Convert report to JSON-serializable format
            json_report = {**report}
            
            # Remove non-serializable data if needed
            
            with open(output_path, 'w') as f:
                json.dump(json_report, f, indent=2)
                
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting JSON report: {e}")
            return None

    def create_backtest_report(self, backtest_id, interactive=None):
        """
        Create a comprehensive backtest report with all visualizations.
        
        Args:
            backtest_id (str): Backtest ID
            interactive (bool): Use interactive plots
            
        Returns:
            dict: Report data with base64 images or HTML
        """
        interactive = interactive if interactive is not None else self.config['interactive']
        
        try:
            # Get backtest results
            backtest_results = self.get_backtest_results(backtest_id)
            if not backtest_results:
                return None
                
            # Prepare data
            data = self.prepare_data(backtest_results)
            if not data:
                return None
                
            # Create visualizations
            equity_plot = self.create_equity_curve_plot(data, interactive=interactive)
            monthly_returns = self.create_monthly_returns_heatmap(data, interactive=interactive)
            trade_analysis = self.create_trade_analysis_plot(data, interactive=interactive)
            returns_dist = self.create_returns_distribution_plot(data, interactive=interactive)
            
            # Build report data
            report = {
                'backtest_id': backtest_id,
                'strategy': data.get('strategy', 'Strategy'),
                'timeframe': data.get('timeframe', 'Unknown'),
                'period': f"{data.get('start_date', 'Unknown')} to {data.get('end_date', 'Unknown')}",
                'statistics': data.get('statistics', {}),
                'charts': {
                    'equity_curve': equity_plot,
                    'monthly_returns': monthly_returns,
                    'trade_analysis': trade_analysis,
                    'returns_distribution': returns_dist
                },
                'interactive': interactive
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error creating backtest report: {e}")
            return None


    def create_drawdown_analysis(self, data, interactive=None):
        """
        Create detailed drawdown analysis visualization.
        
        Args:
            data (dict): Prepared data
            interactive (bool): Use interactive plot
            
        Returns:
            str: Base64 encoded image or HTML
        """
        if not data or 'equity_df' not in data:
            return None
            
        interactive = interactive if interactive is not None else self.config['interactive']
        
        try:
            equity_df = data['equity_df'].copy()
            
            # Calculate drawdown periods
            drawdown_periods = self._identify_drawdown_periods(equity_df)
            
            if interactive:
                return self._create_interactive_drawdown_analysis(equity_df, drawdown_periods, data)
            else:
                return self._create_static_drawdown_analysis(equity_df, drawdown_periods, data)
                
        except Exception as e:
            self.logger.error(f"Error creating drawdown analysis: {e}")
            return None

    def _identify_drawdown_periods(self, equity_df):
        """
        Identify major drawdown periods.
        
        Args:
            equity_df (DataFrame): Equity curve data
            
        Returns:
            list: List of drawdown periods
        """
        # Find peaks in equity curve
        peak_indices = []
        in_drawdown = False
        recovery_threshold = 0.9  # 90% recovery is considered end of drawdown
        
        # Get peak and trough data
        peaks = []
        for i in range(1, len(equity_df) - 1):
            # Skip if we're in a drawdown and haven't recovered
            if in_drawdown and equity_df['equity'].iloc[i] < recovery_threshold * equity_df['equity'].iloc[peak_indices[-1]]:
                continue
                
            # Check if this is a peak
            if equity_df['equity'].iloc[i] > equity_df['equity'].iloc[i-1] and \
            equity_df['equity'].iloc[i] > equity_df['equity'].iloc[i+1]:
                peak_indices.append(i)
                in_drawdown = False
                
            # Check if we've had a significant drawdown (> 5%)
            if not in_drawdown and i > 0 and \
            equity_df['drawdown'].iloc[i] > 5.0:  # 5% drawdown threshold
                in_drawdown = True
                
        # Calculate drawdown periods
        drawdown_periods = []
        
        for i in range(len(peak_indices) - 1):
            peak_idx = peak_indices[i]
            next_peak_idx = peak_indices[i + 1]
            
            # Find maximum drawdown in this period
            period_slice = equity_df.iloc[peak_idx:next_peak_idx]
            max_dd_idx = period_slice['drawdown'].idxmax()
            
            # Only include significant drawdowns (> 5%)
            max_dd = period_slice['drawdown'].max()
            if max_dd > 5.0:  # 5% drawdown threshold
                recovery_idx = period_slice['equity'].idxmax()
                
                # Calculate recovery time (days)
                try:
                    peak_date = equity_df.iloc[peak_idx]['date']
                    trough_date = equity_df.loc[max_dd_idx]['date']
                    recovery_date = equity_df.loc[recovery_idx]['date']
                    
                    drawdown_days = (trough_date - peak_date).days
                    recovery_days = (recovery_date - trough_date).days
                    total_days = drawdown_days + recovery_days
                    
                    drawdown_periods.append({
                        'peak_idx': peak_idx,
                        'trough_idx': max_dd_idx,
                        'recovery_idx': recovery_idx,
                        'peak_date': peak_date,
                        'trough_date': trough_date,
                        'recovery_date': recovery_date,
                        'max_drawdown': max_dd,
                        'drawdown_days': drawdown_days,
                        'recovery_days': recovery_days,
                        'total_days': total_days
                    })
                except Exception as e:
                    self.logger.warning(f"Error calculating drawdown period: {e}")
        
        # Sort by drawdown size (descending)
        drawdown_periods.sort(key=lambda x: x['max_drawdown'], reverse=True)
        
        return drawdown_periods

    def _create_static_drawdown_analysis(self, equity_df, drawdown_periods, data):
        """
        Create static drawdown analysis visualization.
        
        Args:
            equity_df (DataFrame): Equity curve data
            drawdown_periods (list): List of drawdown periods
            data (dict): Full prepared data
            
        Returns:
            str: Base64 encoded image
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config['default_figsize'],
                                    gridspec_kw={'height_ratios': [2, 1]},
                                    sharex=True)
        
        # Plot equity curve
        ax1.plot(equity_df['date'], equity_df['equity'], label='Equity', color='blue')
        
        # Highlight drawdown periods
        for i, period in enumerate(drawdown_periods[:5]):  # Show top 5 drawdowns
            # Get dates
            peak_date = period['peak_date']
            trough_date = period['trough_date']
            recovery_date = period['recovery_date']
            
            # Highlight period
            ax1.axvspan(peak_date, recovery_date, alpha=0.2, color=f'C{i}')
            
            # Mark peak and trough
            peak_equity = equity_df.loc[equity_df['date'] == peak_date, 'equity'].values[0]
            trough_equity = equity_df.loc[equity_df['date'] == trough_date, 'equity'].values[0]
            
            ax1.scatter(peak_date, peak_equity, color=f'C{i}', zorder=5)
            ax1.scatter(trough_date, trough_equity, color=f'C{i}', marker='v', zorder=5)
            
            # Add label
            ax1.annotate(f"DD #{i+1}: {period['max_drawdown']:.1f}%",
                        xy=(trough_date, trough_equity),
                        xytext=(10, -10),
                        textcoords='offset points',
                        color=f'C{i}',
                        fontweight='bold')
        
        ax1.set_title('Equity Curve with Major Drawdown Periods')
        ax1.set_ylabel('Equity')
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown
        ax2.fill_between(equity_df['date'], 0, -equity_df['drawdown'], color='red', alpha=0.3)
        ax2.plot(equity_df['date'], -equity_df['drawdown'], color='red', linewidth=1)
        
        # Format drawdown plot
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Create drawdown table
        if drawdown_periods:
            table_data = []
            for i, period in enumerate(drawdown_periods[:5]):  # Top 5 drawdowns
                table_data.append([
                    f"DD #{i+1}",
                    f"{period['max_drawdown']:.1f}%",
                    period['peak_date'].strftime('%Y-%m-%d'),
                    period['trough_date'].strftime('%Y-%m-%d'),
                    f"{period['drawdown_days']} days",
                    f"{period['recovery_days']} days"
                ])
            
            table_cols = ['', 'Max DD', 'Peak Date', 'Trough Date', 'Decline', 'Recovery']
            
            plt.table(cellText=table_data,
                    colLabels=table_cols,
                    loc='bottom',
                    bbox=[0, -0.50, 1, 0.30])
            
            plt.subplots_adjust(bottom=0.30)
        
        plt.tight_layout()
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format=self.config['export_format'], dpi=100)
        plt.close()
        
        # Encode as Base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str

    def _create_interactive_drawdown_analysis(self, equity_df, drawdown_periods, data):
        """
        Create interactive drawdown analysis visualization.
        
        Args:
            equity_df (DataFrame): Equity curve data
            drawdown_periods (list): List of drawdown periods
            data (dict): Full prepared data
            
        Returns:
            str: HTML content
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3],
                        subplot_titles=['Equity Curve with Major Drawdown Periods', 'Drawdown (%)'])
        
        # Plot equity curve
        equity_trace = go.Scatter(
            x=equity_df['date'],
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        )
        
        fig.add_trace(equity_trace, row=1, col=1)
        
        # Highlight drawdown periods
        colors = px.colors.qualitative.Plotly
        
        for i, period in enumerate(drawdown_periods[:5]):  # Show top 5 drawdowns
            if i >= len(colors):
                break
                
            # Get dates
            peak_date = period['peak_date']
            trough_date = period['trough_date']
            recovery_date = period['recovery_date']
            
            # Mark peak and trough
            peak_equity = equity_df.loc[equity_df['date'] == peak_date, 'equity'].values[0]
            trough_equity = equity_df.loc[equity_df['date'] == trough_date, 'equity'].values[0]
            
            # Add peak marker
            fig.add_trace(go.Scatter(
                x=[peak_date],
                y=[peak_equity],
                mode='markers',
                marker=dict(
                    color=colors[i],
                    size=10,
                    symbol='circle'
                ),
                name=f"Peak #{i+1}",
                showlegend=False
            ), row=1, col=1)
            
            # Add trough marker
            fig.add_trace(go.Scatter(
                x=[trough_date],
                y=[trough_equity],
                mode='markers+text',
                marker=dict(
                    color=colors[i],
                    size=10,
                    symbol='triangle-down'
                ),
                text=f"DD #{i+1}: {period['max_drawdown']:.1f}%",
                textposition="bottom right",
                name=f"Trough #{i+1}",
                showlegend=False
            ), row=1, col=1)
            
            # Add shaded area
            fig.add_trace(go.Scatter(
                x=[peak_date, recovery_date],
                y=[peak_equity, peak_equity],
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ), row=1, col=1)
            
            # Highlight period with rectangle shape
            fig.add_shape(
                type="rect",
                x0=peak_date,
                x1=recovery_date,
                y0=min(equity_df['equity']),
                y1=max(equity_df['equity']),
                fillcolor=colors[i],
                opacity=0.2,
                layer="below",
                line_width=0,
                row=1, col=1
            )
        
        # Plot drawdown
        drawdown_trace = go.Scatter(
            x=equity_df['date'],
            y=-equity_df['drawdown'],
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red', width=1)
        )
        
        fig.add_trace(drawdown_trace, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title='Drawdown Analysis',
            xaxis2_title='Date',
            yaxis_title='Equity',
            yaxis2_title='Drawdown (%)',
            template="plotly_white" if self.config['theme'] == 'default' else "plotly_dark"
        )
        
        # Create drawdown table
        if drawdown_periods:
            table_data = []
            headers = ['', 'Max DD', 'Peak Date', 'Trough Date', 'Decline', 'Recovery']
            
            for i, period in enumerate(drawdown_periods[:5]):  # Top 5 drawdowns
                table_data.append([
                    f"DD #{i+1}",
                    f"{period['max_drawdown']:.1f}%",
                    period['peak_date'].strftime('%Y-%m-%d'),
                    period['trough_date'].strftime('%Y-%m-%d'),
                    f"{period['drawdown_days']} days",
                    f"{period['recovery_days']} days"
                ])
            
            fig.add_trace(go.Table(
                header=dict(
                    values=headers,
                    fill_color='lightgrey',
                    align='center',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=list(map(list, zip(*table_data))),
                    fill_color='white',
                    align='center',
                    font=dict(size=11)
                ),
                domain=dict(
                    x=[0, 1],
                    y=[0, 0.2]
                )
            ))
            
            # Adjust subplot heights to make room for table
            fig.update_layout(
                height=800,
                margin=dict(t=50, b=200)
            )
        
        return fig.to_html(full_html=False)

    def create_calendar_heatmap(self, data, interactive=None):
        """
        Create calendar heatmap showing daily returns.
        
        Args:
            data (dict): Prepared data
            interactive (bool): Use interactive plot
            
        Returns:
            str: Base64 encoded image or HTML
        """
        if not data or 'equity_df' not in data:
            return None
            
        interactive = interactive if interactive is not None else self.config['interactive']
        
        try:
            equity_df = data['equity_df'].copy()
            
            # Set date as index
            equity_df.set_index('date', inplace=True)
            
            # Resample to daily if not already daily
            if len(equity_df) > 252 * 10:  # If more than 10 years of daily data
                daily_returns = equity_df['returns'].resample('D').sum() * 100  # Convert to percentage
            else:
                daily_returns = equity_df['returns'] * 100  # Convert to percentage
            
            # Create calendar dataframe
            calendar_df = pd.DataFrame({
                'date': daily_returns.index,
                'return': daily_returns.values,
                'year': daily_returns.index.year,
                'month': daily_returns.index.month,
                'day': daily_returns.index.day,
                'weekday': daily_returns.index.weekday,
                'week': daily_returns.index.isocalendar().week
            })
            
            if interactive:
                return self._create_interactive_calendar_heatmap(calendar_df, data)
            else:
                return self._create_static_calendar_heatmap(calendar_df, data)
                
        except Exception as e:
            self.logger.error(f"Error creating calendar heatmap: {e}")
            return None

    def _create_static_calendar_heatmap(self, calendar_df, data):
        """
        Create static calendar heatmap.
        
        Args:
            calendar_df (DataFrame): Calendar data
            data (dict): Full prepared data
            
        Returns:
            str: Base64 encoded image
        """
        # Get the most recent year with complete data
        current_year = calendar_df['year'].max()
        
        # Filter for the selected year
        year_data = calendar_df[calendar_df['year'] == current_year].copy()
        
        # Create a pivot table: week x weekday
        pivot_data = year_data.pivot_table(
            values='return',
            index='week',
            columns='weekday',
            aggfunc='sum'
        )
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create the heatmap
        cmap = plt.cm.RdYlGn  # Red for negative, green for positive
        heatmap = ax.pcolormesh(pivot_data.columns, pivot_data.index, pivot_data.values, 
                            cmap=cmap, vmin=-2, vmax=2)
        
        # Add a color bar
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label('Daily Return (%)')
        
        # Configure axes
        weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ax.set_xticks(np.arange(len(weekday_labels)) + 0.5)
        ax.set_xticklabels(weekday_labels)
        
        # Set y-axis (weeks) ticks
        week_ticks = np.arange(1, 53, 4) + 0.5
        ax.set_yticks(week_ticks)
        week_labels = [str(int(w - 0.5)) for w in week_ticks]
        ax.set_yticklabels(week_labels)
        
        # Add month lines and labels
        month_weeks = []
        month_labels = []
        
        for month in range(1, 13):
            # Get the first day of each month
            month_start = year_data[year_data['month'] == month]['week'].min()
            if not np.isnan(month_start):
                month_weeks.append(month_start)
                month_labels.append(datetime(current_year, month, 1).strftime('%b'))
        
        # Plot month lines
        for week in month_weeks:
            ax.axhline(y=week, color='black', linestyle='-', alpha=0.3)
        
        # Add month labels on the right side
        for i, month in enumerate(month_labels):
            if i < len(month_weeks):
                ax.text(7.5, month_weeks[i] + 2, month, ha='center', va='center')
        
        strategy_name = data.get('strategy', 'Strategy')
        ax.set_title(f'Calendar Heatmap of Daily Returns: {strategy_name} ({current_year})')
        
        plt.tight_layout()
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format=self.config['export_format'], dpi=100)
        plt.close()
        
        # Encode as Base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str

    def _create_interactive_calendar_heatmap(self, calendar_df, data):
        """
        Create interactive calendar heatmap.
        
        Args:
            calendar_df (DataFrame): Calendar data
            data (dict): Full prepared data
            
        Returns:
            str: HTML content
        """
        # Get the most recent year with complete data
        current_year = calendar_df['year'].max()
        
        # Filter for the selected year
        year_data = calendar_df[calendar_df['year'] == current_year].copy()
        
        # Create a pivot table: week x weekday
        pivot_data = year_data.pivot_table(
            values='return',
            index='week',
            columns='weekday',
            aggfunc='sum'
        ).fillna(0)
        
        # Create data for heatmap
        z_data = pivot_data.values
        
        # Create text for hover
        text_data = []
        for week in pivot_data.index:
            week_text = []
            for weekday in pivot_data.columns:
                try:
                    # Find the date for this week and weekday
                    date_val = year_data[(year_data['week'] == week) & (year_data['weekday'] == weekday)]['date']
                    if not date_val.empty:
                        date_str = date_val.iloc[0].strftime('%Y-%m-%d')
                        return_val = year_data[(year_data['week'] == week) & (year_data['weekday'] == weekday)]['return'].iloc[0]
                        cell_text = f"Date: {date_str}<br>Return: {return_val:.2f}%"
                    else:
                        cell_text = "No data"
                except:
                    cell_text = "No data"
                    
                week_text.append(cell_text)
            text_data.append(week_text)
        
        # Convert weekday numbers to names
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=weekday_names,
            y=list(pivot_data.index),
            colorscale='RdYlGn',
            zmid=0,
            zmin=-2,
            zmax=2,
            text=text_data,
            hoverinfo='text'
        ))
        
        # Add month lines and labels
        month_weeks = []
        month_labels = []
        
        for month in range(1, 13):
            # Get the first day of each month
            month_dates = year_data[year_data['month'] == month]
            if not month_dates.empty:
                month_start = month_dates['week'].min()
                month_weeks.append(month_start)
                month_labels.append(datetime(current_year, month, 1).strftime('%b'))
        
        # Add month lines as shapes
        for week in month_weeks:
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=week,
                x1=6.5,
                y1=week,
                line=dict(color="black", width=1, dash="solid"),
                opacity=0.3
            )
        
        # Add month labels as annotations
        annotations = []
        for i, month in enumerate(month_labels):
            if i < len(month_weeks):
                annotations.append(dict(
                    x=7,
                    y=month_weeks[i] + 2,
                    xref="x",
                    yref="y",
                    text=month,
                    showarrow=False,
                    font=dict(size=12)
                ))
        
        fig.update_layout(
            annotations=annotations,
            title=f'Calendar Heatmap of Daily Returns: {data.get("strategy", "Strategy")} ({current_year})',
            xaxis_title="Day of Week",
            yaxis_title="Week of Year",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 53, 4)),
                autorange="reversed"  # To have week 1 at the top
            ),
            template="plotly_white" if self.config['theme'] == 'default' else "plotly_dark"
        )
        
        return fig.to_html(full_html=False)

    def save_to_database(self, report_data, metadata=None):
        """
        Save visualization report to database.
        
        Args:
            report_data (dict): Report data
            metadata (dict): Additional metadata
            
        Returns:
            str: Report ID
        """
        try:
            # Create report document
            report = {
                'created_at': datetime.now(),
                'report_type': metadata.get('report_type', 'backtest'),
                'strategy': metadata.get('strategy', 'Unknown'),
                'timeframe': metadata.get('timeframe', 'Unknown'),
                'report_data': report_data
            }
            
            # Add additional metadata if provided
            if metadata:
                for key, value in metadata.items():
                    if key not in report:
                        report[key] = value
            
            # Insert into database
            result = self.db.visualization_reports_collection.insert_one(report)
            report_id = str(result.inserted_id)
            
            self.logger.info(f"Saved visualization report to database with ID: {report_id}")
            
            return report_id
            
        except Exception as e:
            self.logger.error(f"Error saving visualization report to database: {e}")
            return None