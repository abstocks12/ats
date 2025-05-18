# backtesting/engine.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import time
import json
import os

class BacktestingEngine:
    """
    Event-driven backtesting engine for trading strategy evaluation.
    """
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the backtesting engine.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'initial_capital': 100000,
            'position_size': 0.02,  # 2% of capital per position
            'max_positions': 10,
            'commission_rate': 0.0005,  # 0.05% commission
            'slippage': 0.0002,  # 0.02% slippage
            'default_timeframe': 'day',
            'stop_loss_pct': 0.02,  # 2% stop loss
            'take_profit_pct': 0.04,  # 4% take profit
            'max_holding_days': 10,
            'risk_free_rate': 0.04,  # 4% annual risk-free rate
            'include_partial_positions': True
        }
        
        # Current state
        self.reset()
    
    def set_config(self, config):
        """
        Set backtesting configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated backtesting configuration: {self.config}")
        
        # Reset state with new configuration
        self.reset()
    
    def reset(self):
        """Reset the backtesting engine state."""
        # Portfolio state
        self.capital = self.config['initial_capital']
        self.equity = self.capital
        self.positions = {}  # Symbol -> Position
        self.closed_positions = []
        self.open_orders = []
        
        # Performance tracking
        self.equity_curve = [self.equity]
        self.equity_timestamps = [datetime.now()]
        self.drawdowns = []
        self.returns = []
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'cagr': 0.0,
            'avg_holding_time': 0.0
        }
        
        # Cache for market data
        self.market_data_cache = {}
        
        self.logger.info("Backtesting engine state reset")
    
    def get_market_data(self, symbol, exchange, start_date, end_date, timeframe=None):
        """
        Get historical market data for backtesting.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            start_date (datetime): Start date
            end_date (datetime): End date
            timeframe (str): Timeframe ('1min', '5min', '15min', 'hour', 'day')
            
        Returns:
            DataFrame: Market data
        """
        timeframe = timeframe or self.config['default_timeframe']
        
        # Check cache
        cache_key = f"{symbol}_{exchange}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        if cache_key in self.market_data_cache:
            return self.market_data_cache[cache_key]
        
        # Query database
        query = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
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
            self.logger.warning(f"No market data found for {symbol} {exchange} from {start_date} to {end_date}")
            return None
        
        # Set timestamp as index
        market_data.set_index('timestamp', inplace=True)
        
        # Cache data
        self.market_data_cache[cache_key] = market_data
        
        self.logger.info(f"Retrieved {len(market_data)} {timeframe} bars for {symbol} {exchange}")
        
        return market_data
    
    def backtest_strategy(self, strategy, symbols, start_date, end_date, timeframe=None):
        """
        Run a backtest for a strategy on multiple symbols.
        
        Args:
            strategy: Strategy instance
            symbols (list): List of (symbol, exchange) tuples
            start_date (datetime): Start date
            end_date (datetime): End date
            timeframe (str): Timeframe for backtesting
            
        Returns:
            dict: Backtest results
        """
        timeframe = timeframe or self.config['default_timeframe']
        
        self.logger.info(f"Starting backtest for {len(symbols)} symbols from {start_date} to {end_date}")
        start_time = time.time()
        
        # Reset state
        self.reset()
        
        # Prepare data
        all_data = {}
        
        for symbol, exchange in symbols:
            market_data = self.get_market_data(symbol, exchange, start_date, end_date, timeframe)
            if market_data is not None:
                all_data[(symbol, exchange)] = market_data
        
        if not all_data:
            self.logger.error("No market data found for any symbol")
            return None
        
        # Determine common date range
        common_dates = None
        
        for data in all_data.values():
            dates = data.index
            if common_dates is None:
                common_dates = set(dates)
            else:
                common_dates &= set(dates)
        
        common_dates = sorted(common_dates)
        
        if not common_dates:
            self.logger.error("No common dates found across symbols")
            return None
        
        # Initialize strategy
        strategy.initialize(self)
        
        # Run event loop
        for date in common_dates:
            self.equity_timestamps.append(date)
            
            # Update positions with latest prices
            self._update_positions([date], all_data)
            
            # Process any pending orders
            self._process_orders(date, all_data)
            
            # Execute strategy for this date
            for (symbol, exchange), data in all_data.items():
                if date in data.index:
                    # Get bar data
                    bar = data.loc[date].to_dict()
                    bar['timestamp'] = date
                    bar['symbol'] = symbol
                    bar['exchange'] = exchange
                    
                    # Call strategy
                    signals = strategy.on_bar(bar)
                    
                    # Process signals
                    if signals:
                        self._process_signals(signals, date, all_data)
            
            # Update equity curve
            self.equity = self.capital + self._calculate_positions_value(date, all_data)
            self.equity_curve.append(self.equity)
            
            # Calculate return
            if len(self.equity_curve) >= 2:
                ret = (self.equity_curve[-1] / self.equity_curve[-2]) - 1
                self.returns.append(ret)
            else:
                self.returns.append(0)
            
            # Update drawdowns
            self._update_drawdowns()
        
        # Close remaining positions at the end of backtest
        final_date = common_dates[-1]
        self._close_all_positions(final_date, all_data)
        
        # Calculate performance metrics
        self._calculate_performance_metrics(start_date, end_date)
        
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"Backtest completed in {elapsed_time:.2f} seconds with {self.stats['total_trades']} trades")
        
        # Create results
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'timeframe': timeframe,
            'initial_capital': self.config['initial_capital'],
            'final_capital': self.capital,
            'final_equity': self.equity,
            'return_pct': (self.equity / self.config['initial_capital'] - 1) * 100,
            'trades': len(self.closed_positions),
            'statistics': self.stats,
            'equity_curve': self.equity_curve,
            'equity_timestamps': self.equity_timestamps,
            'positions': self.closed_positions,
            'config': self.config
        }
        
        # Save backtest results
        self._save_backtest_results(results, strategy.__class__.__name__)
        
        return results
    
    def _update_positions(self, dates, all_data):
        """
        Update positions with the latest prices.
        
        Args:
            dates (list): Current trading dates
            all_data (dict): Market data for all symbols
        """
        current_date = dates[-1]
        
        for symbol, position in list(self.positions.items()):
            symbol_tuple = (position['symbol'], position['exchange'])
            
            if symbol_tuple in all_data:
                data = all_data[symbol_tuple]
                
                if current_date in data.index:
                    bar = data.loc[current_date]
                    
                    # Update position with current price
                    current_price = bar['close']
                    position['current_price'] = current_price
                    position['current_value'] = position['quantity'] * current_price
                    position['profit_loss'] = position['current_value'] - position['cost_basis']
                    position['profit_loss_pct'] = position['profit_loss'] / position['cost_basis']
                    
                    # Check stop loss and take profit
                    if position['stop_loss'] and current_price <= position['stop_loss'] and position['direction'] == 'long':
                        self._close_position(symbol, current_date, 'stop_loss', all_data)
                    elif position['stop_loss'] and current_price >= position['stop_loss'] and position['direction'] == 'short':
                        self._close_position(symbol, current_date, 'stop_loss', all_data)
                    elif position['take_profit'] and current_price >= position['take_profit'] and position['direction'] == 'long':
                        self._close_position(symbol, current_date, 'take_profit', all_data)
                    elif position['take_profit'] and current_price <= position['take_profit'] and position['direction'] == 'short':
                        self._close_position(symbol, current_date, 'take_profit', all_data)
                    
                    # Check max holding time
                    days_held = (current_date - position['entry_date']).days
                    if days_held >= self.config['max_holding_days']:
                        self._close_position(symbol, current_date, 'max_holding_time', all_data)
    
    def _process_orders(self, date, all_data):
        """
        Process pending orders.
        
        Args:
            date (datetime): Current date
            all_data (dict): Market data for all symbols
        """
        for order in list(self.open_orders):
            symbol_tuple = (order['symbol'], order['exchange'])
            
            if symbol_tuple in all_data:
                data = all_data[symbol_tuple]
                
                if date in data.index:
                    bar = data.loc[date]
                    
                    # Check if order can be executed
                    if order['type'] == 'market':
                        # Market order executes at the open price
                        execution_price = bar['open']
                        self._execute_order(order, date, execution_price)
                    elif order['type'] == 'limit':
                        # Limit buy executes if price goes below limit
                        if order['direction'] == 'buy' and bar['low'] <= order['limit_price']:
                            execution_price = order['limit_price']
                            self._execute_order(order, date, execution_price)
                        # Limit sell executes if price goes above limit
                        elif order['direction'] == 'sell' and bar['high'] >= order['limit_price']:
                            execution_price = order['limit_price']
                            self._execute_order(order, date, execution_price)
                    elif order['type'] == 'stop':
                        # Stop buy executes if price goes above stop
                        if order['direction'] == 'buy' and bar['high'] >= order['stop_price']:
                            execution_price = order['stop_price']
                            self._execute_order(order, date, execution_price)
                        # Stop sell executes if price goes below stop
                        elif order['direction'] == 'sell' and bar['low'] <= order['stop_price']:
                            execution_price = order['stop_price']
                            self._execute_order(order, date, execution_price)
    
    def _process_signals(self, signals, date, all_data):
        """
        Process trading signals from strategy.
        
        Args:
            signals (list): List of signal dictionaries
            date (datetime): Current date
            all_data (dict): Market data for all symbols
        """
        for signal in signals:
            symbol = signal.get('symbol')
            exchange = signal.get('exchange')
            direction = signal.get('direction', 'buy')
            order_type = signal.get('type', 'market')
            limit_price = signal.get('limit_price')
            stop_price = signal.get('stop_price')
            quantity = signal.get('quantity')
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            
            # Create order
            order = {
                'symbol': symbol,
                'exchange': exchange,
                'direction': direction,
                'type': order_type,
                'limit_price': limit_price,
                'stop_price': stop_price,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': date
            }
            
            # Validate order
            if not self._validate_order(order, date, all_data):
                continue
            
            # Process immediately if market order
            if order_type == 'market':
                symbol_tuple = (symbol, exchange)
                
                if symbol_tuple in all_data:
                    data = all_data[symbol_tuple]
                    
                    if date in data.index:
                        bar = data.loc[date]
                        execution_price = bar['open']
                        self._execute_order(order, date, execution_price)
            else:
                # Add to open orders
                self.open_orders.append(order)
    
    def _validate_order(self, order, date, all_data):
        """
        Validate order before execution.
        
        Args:
            order (dict): Order details
            date (datetime): Current date
            all_data (dict): Market data for all symbols
            
        Returns:
            bool: True if order is valid
        """
        symbol = order['symbol']
        exchange = order['exchange']
        direction = order['direction']
        symbol_key = f"{symbol}_{exchange}"
        
        # Check if we have market data
        symbol_tuple = (symbol, exchange)
        if symbol_tuple not in all_data:
            self.logger.warning(f"No market data for {symbol} {exchange}")
            return False
        
        # Check if we already have a position in this symbol
        if symbol_key in self.positions:
            existing_position = self.positions[symbol_key]
            
            # Allow exit orders
            if (direction == 'sell' and existing_position['direction'] == 'long') or \
               (direction == 'buy' and existing_position['direction'] == 'short'):
                return True
            
            # Reject opening another position in the same direction
            if (direction == 'buy' and existing_position['direction'] == 'long') or \
               (direction == 'sell' and existing_position['direction'] == 'short'):
                self.logger.warning(f"Already have a {existing_position['direction']} position in {symbol}")
                return False
        
        # Check if we have enough capital and not too many positions
        if len(self.positions) >= self.config['max_positions'] and self.config['max_positions'] > 0:
            self.logger.warning(f"Maximum positions ({self.config['max_positions']}) reached")
            return False
        
        # Calculate position size if not specified
        if not order.get('quantity'):
            market_data = all_data[symbol_tuple]
            if date in market_data.index:
                bar = market_data.loc[date]
                price = bar['open']
                
                position_value = self.capital * self.config['position_size']
                order['quantity'] = position_value / price
                
                # Check minimum capital requirement
                if position_value < 1000:  # $1000 minimum position size
                    self.logger.warning(f"Position size too small: ${position_value:.2f}")
                    return False
        
        return True
    
    def _execute_order(self, order, date, execution_price):
        """
        Execute an order and update portfolio.
        
        Args:
            order (dict): Order details
            date (datetime): Execution date
            execution_price (float): Execution price
        """
        symbol = order['symbol']
        exchange = order['exchange']
        direction = order['direction']
        quantity = order['quantity']
        symbol_key = f"{symbol}_{exchange}"
        
        # Apply slippage
        if direction == 'buy':
            execution_price *= (1 + self.config['slippage'])
        else:
            execution_price *= (1 - self.config['slippage'])
        
        # Calculate commission
        commission = execution_price * quantity * self.config['commission_rate']
        
        # Execute buy order
        if direction == 'buy':
            # Close existing short position if any
            if symbol_key in self.positions and self.positions[symbol_key]['direction'] == 'short':
                self._close_position(symbol_key, date, 'strategy_exit', None, execution_price)
            
            # Open long position
            cost = execution_price * quantity + commission
            
            # Check if we have enough capital
            if cost > self.capital:
                self.logger.warning(f"Insufficient capital for {symbol} buy order")
                # Adjust quantity to available capital
                if self.config['include_partial_positions']:
                    quantity = (self.capital - commission) / execution_price
                    cost = execution_price * quantity + commission
                    
                    if quantity <= 0:
                        # Remove from open orders
                        if order in self.open_orders:
                            self.open_orders.remove(order)
                        return
                else:
                    # Remove from open orders
                    if order in self.open_orders:
                        self.open_orders.remove(order)
                    return
            
            # Update capital
            self.capital -= cost
            
            # Set stop loss and take profit
            stop_loss = None
            if order.get('stop_loss'):
                stop_loss = order['stop_loss']
            elif self.config['stop_loss_pct'] > 0:
                stop_loss = execution_price * (1 - self.config['stop_loss_pct'])
                
            take_profit = None
            if order.get('take_profit'):
                take_profit = order['take_profit']
            elif self.config['take_profit_pct'] > 0:
                take_profit = execution_price * (1 + self.config['take_profit_pct'])
            
            # Create position
            self.positions[symbol_key] = {
                'symbol': symbol,
                'exchange': exchange,
                'direction': 'long',
                'quantity': quantity,
                'entry_price': execution_price,
                'entry_date': date,
                'cost_basis': cost,
                'current_price': execution_price,
                'current_value': execution_price * quantity,
                'profit_loss': 0,
                'profit_loss_pct': 0,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'commission': commission
            }
            
            self.logger.info(f"Opened long position in {symbol} at {execution_price:.2f}")
        
        # Execute sell order
        elif direction == 'sell':
            # Close existing long position if any
            if symbol_key in self.positions and self.positions[symbol_key]['direction'] == 'long':
                self._close_position(symbol_key, date, 'strategy_exit', None, execution_price)
            else:
                # Open short position
                cost = execution_price * quantity
                commission_cost = cost * self.config['commission_rate']
                
                # Check if we have enough capital for margin
                margin_requirement = cost * 0.5  # 50% margin requirement
                if margin_requirement > self.capital:
                    self.logger.warning(f"Insufficient capital for {symbol} short margin")
                    # Adjust quantity to available capital
                    if self.config['include_partial_positions']:
                        quantity = (self.capital / 0.5) / execution_price
                        cost = execution_price * quantity
                        commission_cost = cost * self.config['commission_rate']
                        margin_requirement = cost * 0.5
                        
                        if quantity <= 0:
                            # Remove from open orders
                            if order in self.open_orders:
                                self.open_orders.remove(order)
                            return
                    else:
                        # Remove from open orders
                        if order in self.open_orders:
                            self.open_orders.remove(order)
                        return
                
                # Update capital (only commission is deducted, margin is reserved)
                self.capital -= commission_cost
                
                # Set stop loss and take profit
                stop_loss = None
                if order.get('stop_loss'):
                    stop_loss = order['stop_loss']
                elif self.config['stop_loss_pct'] > 0:
                    stop_loss = execution_price * (1 + self.config['stop_loss_pct'])
                    
                take_profit = None
                if order.get('take_profit'):
                    take_profit = order['take_profit']
                elif self.config['take_profit_pct'] > 0:
                    take_profit = execution_price * (1 - self.config['take_profit_pct'])
                
                # Create position
                self.positions[symbol_key] = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'direction': 'short',
                    'quantity': quantity,
                    'entry_price': execution_price,
                    'entry_date': date,
                    'cost_basis': cost,
                    'current_price': execution_price,
                    'current_value': cost,
                    'profit_loss': 0,
                    'profit_loss_pct': 0,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'commission': commission_cost,
                    'margin': margin_requirement
                }
                
                self.logger.info(f"Opened short position in {symbol} at {execution_price:.2f}")
        
        # Remove from open orders
        if order in self.open_orders:
            self.open_orders.remove(order)
    
    def _close_position(self, symbol_key, date, reason, all_data=None, exit_price=None):
        """
        Close a position.
        
        Args:
            symbol_key (str): Symbol key in positions dictionary
            date (datetime): Exit date
            reason (str): Reason for closing
            all_data (dict): Market data for all symbols
            exit_price (float): Optional exit price override
        """
        if symbol_key not in self.positions:
            return
        
        position = self.positions[symbol_key]
        symbol = position['symbol']
        exchange = position['exchange']
        
        # Determine exit price
        if exit_price is None and all_data is not None:
            symbol_tuple = (symbol, exchange)
            if symbol_tuple in all_data:
                data = all_data[symbol_tuple]
                if date in data.index:
                    bar = data.loc[date]
                    exit_price = bar['close']
        
        if exit_price is None:
            exit_price = position['current_price']
        
        # Apply slippage
        if position['direction'] == 'long':
            exit_price *= (1 - self.config['slippage'])
        else:
            exit_price *= (1 + self.config['slippage'])
        
        # Calculate commission
        exit_commission = exit_price * position['quantity'] * self.config['commission_rate']
        
        # Calculate profit/loss
        if position['direction'] == 'long':
            profit_loss = (exit_price - position['entry_price']) * position['quantity'] - position['commission'] - exit_commission
            profit_loss_pct = (exit_price / position['entry_price']) - 1
        else:  # short
            profit_loss = (position['entry_price'] - exit_price) * position['quantity'] - position['commission'] - exit_commission
            profit_loss_pct = 1 - (exit_price / position['entry_price'])
        
        # Update capital
        self.capital += position['quantity'] * exit_price - exit_commission
        
        # For short positions, return margin
        if position['direction'] == 'short' and 'margin' in position:
            self.capital += position['margin']
        
        # Add to closed positions
        closed_position = position.copy()
        closed_position.update({
            'exit_price': exit_price,
            'exit_date': date,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'exit_reason': reason,
            'holding_days': (date - position['entry_date']).days,
            'exit_commission': exit_commission
        })
        
        self.closed_positions.append(closed_position)
        
        # Update statistics
        self.stats['total_trades'] += 1
        
        if profit_loss > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        # Remove from positions
        del self.positions[symbol_key]
        
        self.logger.info(f"Closed {position['direction']} position in {symbol} at {exit_price:.2f} for {profit_loss:.2f} ({profit_loss_pct:.2%})")
    
    def _close_all_positions(self, date, all_data):
        """
        Close all open positions.
        
        Args:
            date (datetime): Exit date
            all_data (dict): Market data for all symbols
        """
        for symbol_key in list(self.positions.keys()):
            self._close_position(symbol_key, date, 'end_of_backtest', all_data)
    
    def _calculate_positions_value(self, date, all_data):
        """
        Calculate the total value of all open positions.
        
        Args:
            date (datetime): Current date
            all_data (dict): Market data for all symbols
            
        Returns:
            float: Total positions value
        """
        total_value = 0
        
        for symbol_key, position in self.positions.items():
            symbol = position['symbol']
            exchange = position['exchange']
            symbol_tuple = (symbol, exchange)
            
            if symbol_tuple in all_data:
                data = all_data[symbol_tuple]
                
                if date in data.index:
                    bar = data.loc[date]
                    price = bar['close']
                    
                    if position['direction'] == 'long':
                        position_value = position['quantity'] * price
                    else:  # short
                        # For shorts, profit is negative when price increases
                        initial_value = position['quantity'] * position['entry_price']
                        change_value = position['quantity'] * (position['entry_price'] - price)
                        position_value = position['margin'] + change_value
                    
                    total_value += position_value
            
        return total_value
    
    def _update_drawdowns(self):
        """Update drawdown calculations."""
        if len(self.equity_curve) < 2:
            return
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(self.equity_curve)
        
        # Calculate drawdown percentage
        drawdown_pct = (running_max[-1] - self.equity_curve[-1]) / running_max[-1]
        
        self.drawdowns.append(drawdown_pct)
    
    def _calculate_performance_metrics(self, start_date, end_date):
        """
        Calculate performance metrics.
        
        Args:
            start_date (datetime): Backtest start date
            end_date (datetime): Backtest end date
        """
        if not self.closed_positions:
            return
        
        # Basic metrics
        total_trades = len(self.closed_positions)
        winning_trades = sum(1 for p in self.closed_positions if p['profit_loss'] > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        profits = [p['profit_loss'] for p in self.closed_positions if p['profit_loss'] > 0]
        losses = [p['profit_loss'] for p in self.closed_positions if p['profit_loss'] <= 0]
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        total_profit = sum(profits)
        total_loss = sum(losses)
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Drawdown
        max_drawdown = max(self.drawdowns) if self.drawdowns else 0
        
        # Returns
        returns_array = np.array(self.returns)
        
        if len(returns_array) > 1:
            # Sharpe ratio
            excess_returns = returns_array - (self.config['risk_free_rate'] / 252)  # Daily risk-free rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
            
            # Sortino ratio (using negative returns only)
            negative_returns = returns_array[returns_array < 0]
            sortino_ratio = np.mean(excess_returns) / np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 and np.std(negative_returns) > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # CAGR
       # CAGR
        years = (end_date - start_date).days / 365.25
        cagr = (self.equity / self.config['initial_capital']) ** (1 / years) - 1 if years > 0 else 0
        
        # Average holding time
        holding_times = [p['holding_days'] for p in self.closed_positions]
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        # Update statistics
        self.stats.update({
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'cagr': cagr,
            'avg_holding_time': avg_holding_time
        })
    
    def _save_backtest_results(self, results, strategy_name):
        """
        Save backtest results to database.
        
        Args:
            results (dict): Backtest results
            strategy_name (str): Name of the strategy
            
        Returns:
            str: Backtest ID
        """
        try:
            # Create results document
            backtest_doc = {
                'strategy': strategy_name,
                'timestamp': datetime.now(),
                'start_date': results['start_date'],
                'end_date': results['end_date'],
                'timeframe': results['timeframe'],
                'initial_capital': results['initial_capital'],
                'final_capital': results['final_capital'],
                'return_pct': results['return_pct'],
                'trades': results['trades'],
                'statistics': results['statistics'],
                'config': results['config']
            }
            
            # Add equity curve dates and values separately
            # (convert to strings to avoid MongoDB date issues)
            backtest_doc['equity_timestamps'] = [
                d.strftime('%Y-%m-%d %H:%M:%S') if isinstance(d, datetime) else str(d)
                for d in results['equity_timestamps']
            ]
            backtest_doc['equity_curve'] = results['equity_curve']
            
            # Add trade details
            backtest_doc['positions'] = [
                {
                    'symbol': p['symbol'],
                    'exchange': p['exchange'],
                    'direction': p['direction'],
                    'entry_date': p['entry_date'],
                    'exit_date': p['exit_date'],
                    'entry_price': p['entry_price'],
                    'exit_price': p['exit_price'],
                    'quantity': p['quantity'],
                    'profit_loss': p['profit_loss'],
                    'profit_loss_pct': p['profit_loss_pct'],
                    'exit_reason': p['exit_reason'],
                    'holding_days': p['holding_days']
                }
                for p in results['positions']
            ]
            
            # Insert into database
            result = self.db.backtest_results_collection.insert_one(backtest_doc)
            backtest_id = str(result.inserted_id)
            
            self.logger.info(f"Saved backtest results with ID: {backtest_id}")
            
            return backtest_id
            
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}")
            return None
    
    def get_trades_by_symbol(self):
        """
        Group trades by symbol and calculate performance metrics per symbol.
        
        Returns:
            dict: Performance metrics by symbol
        """
        if not self.closed_positions:
            return {}
            
        trades_by_symbol = defaultdict(list)
        
        for position in self.closed_positions:
            symbol = position['symbol']
            exchange = position['exchange']
            symbol_key = f"{symbol}_{exchange}"
            
            trades_by_symbol[symbol_key].append(position)
        
        metrics_by_symbol = {}
        
        for symbol_key, trades in trades_by_symbol.items():
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['profit_loss'] > 0)
            losing_trades = total_trades - winning_trades
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            profits = [t['profit_loss'] for t in trades if t['profit_loss'] > 0]
            losses = [t['profit_loss'] for t in trades if t['profit_loss'] <= 0]
            
            avg_profit = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            
            total_profit = sum(profits)
            total_loss = sum(losses)
            net_profit = total_profit + total_loss
            
            profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
            
            metrics_by_symbol[symbol_key] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net_profit': net_profit,
                'profit_factor': profit_factor
            }
        
        return metrics_by_symbol
    
    def get_monthly_returns(self):
        """
        Calculate monthly returns.
        
        Returns:
            dict: Monthly returns
        """
        if len(self.equity_curve) < 2 or len(self.equity_timestamps) < 2:
            return {}
        
        # Create DataFrame with equity curve
        equity_df = pd.DataFrame({
            'date': self.equity_timestamps,
            'equity': self.equity_curve
        })
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(equity_df['date']):
            equity_df['date'] = pd.to_datetime(equity_df['date'])
        
        # Set date as index
        equity_df.set_index('date', inplace=True)
        
        # Resample to month-end
        monthly_equity = equity_df.resample('M').last()
        
        # Calculate returns
        monthly_returns = monthly_equity['equity'].pct_change().fillna(
            (monthly_equity['equity'].iloc[0] / self.config['initial_capital']) - 1
        )
        
        # Convert to dictionary
        return {
            date.strftime('%Y-%m'): float(ret)
            for date, ret in monthly_returns.items()
        }
    
    def calculate_max_consecutive_losses(self):
        """
        Calculate maximum consecutive losses.
        
        Returns:
            int: Maximum consecutive losses
        """
        if not self.closed_positions:
            return 0
        
        # Sort positions by exit date
        sorted_positions = sorted(self.closed_positions, key=lambda x: x['exit_date'])
        
        # Track profit/loss streaks
        current_streak = 0
        max_loss_streak = 0
        
        for position in sorted_positions:
            if position['profit_loss'] > 0:
                # Reset loss streak on profit
                current_streak = 0
            else:
                # Increment loss streak
                current_streak += 1
                max_loss_streak = max(max_loss_streak, current_streak)
        
        return max_loss_streak
    
    def calculate_max_consecutive_wins(self):
        """
        Calculate maximum consecutive wins.
        
        Returns:
            int: Maximum consecutive wins
        """
        if not self.closed_positions:
            return 0
        
        # Sort positions by exit date
        sorted_positions = sorted(self.closed_positions, key=lambda x: x['exit_date'])
        
        # Track profit/loss streaks
        current_streak = 0
        max_win_streak = 0
        
        for position in sorted_positions:
            if position['profit_loss'] <= 0:
                # Reset win streak on loss
                current_streak = 0
            else:
                # Increment win streak
                current_streak += 1
                max_win_streak = max(max_win_streak, current_streak)
        
        return max_win_streak
    
    def get_return_distribution(self):
        """
        Calculate return distribution statistics.
        
        Returns:
            dict: Return distribution statistics
        """
        if not self.closed_positions:
            return {}
        
        # Extract returns
        returns = [p['profit_loss_pct'] for p in self.closed_positions]
        
        if not returns:
            return {}
        
        # Calculate distribution metrics
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)
        skew = pd.Series(returns).skew()
        kurtosis = pd.Series(returns).kurtosis()
        
        # Calculate percentiles
        p10 = np.percentile(returns, 10)
        p25 = np.percentile(returns, 25)
        p75 = np.percentile(returns, 75)
        p90 = np.percentile(returns, 90)
        
        # Create distribution
        return {
            'mean': float(mean_return),
            'median': float(median_return),
            'std': float(std_return),
            'skew': float(skew),
            'kurtosis': float(kurtosis),
            'percentile_10': float(p10),
            'percentile_25': float(p25),
            'percentile_75': float(p75),
            'percentile_90': float(p90),
            'min': float(min(returns)),
            'max': float(max(returns))
        }