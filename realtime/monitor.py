# realtime/monitor.py
import logging
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

class TradingMonitor:
    """
    Real-time trading performance monitoring and risk assessment.
    """
    
    def __init__(self, db_connector, trading_engine=None, logger=None):
        """
        Initialize the trading monitor.
        
        Args:
            db_connector: MongoDB connector
            trading_engine: Trading engine instance
            logger: Logger instance
        """
        self.db = db_connector
        self.engine = trading_engine
        self.logger = logger or logging.getLogger(__name__)
        
        # State
        self.is_running = False
        self.monitoring_thread = None
        
        # Tracking
        self.risk_metrics = {}
        self.performance_metrics = {}
        self.alerts = []
        self.snapshots = []
        
        # Alert handlers
        self.alert_handlers = []
        
        # Configuration
        self.config = {
            'monitoring_interval': 60,  # Seconds between monitoring checks
            'performance_snapshot_interval': 3600,  # Seconds between performance snapshots
            'max_drawdown_alert': 5.0,  # Maximum drawdown percentage before alerting
            'max_position_size_pct': 10.0,  # Maximum position size as percentage of equity
            'max_sector_exposure_pct': 25.0,  # Maximum sector exposure as percentage of equity
            'max_daily_loss_pct': 3.0,  # Maximum daily loss percentage
            'max_risk_per_trade_pct': 1.0,  # Maximum risk per trade percentage
            'vwap_deviation_alert': 2.0,  # VWAP deviation percentage for price anomaly alerts
            'stale_price_minutes': 5,  # Minutes without price update to consider price stale
            'equity_update_interval': 300,  # Seconds between equity curve updates
            'track_trade_pnl': True,  # Track individual trade P&L
            'monitor_greeks': True,  # Monitor option Greeks
            'track_sector_exposure': True,  # Track sector exposure
            'correlation_threshold': 0.7,  # Correlation threshold for portfolio diversification alerts
            'max_concentration_pct': 20.0,  # Maximum concentration in a single position
            'volatility_window': 20,  # Window for volatility calculation
            'max_position_staleness_minutes': 30,  # Maximum minutes for position without update
            'track_strategy_performance': True,  # Track performance by strategy
            'track_instrument_type_performance': True  # Track performance by instrument type
        }
        
        # Initialize
        self._initialize()
        
        self.logger.info("Trading monitor initialized")
    
    def set_config(self, config):
        """
        Set monitor configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated trading monitor configuration")
    
    def _initialize(self):
        """
        Initialize the trading monitor.
        """
        # Load risk metrics from database
        self._load_risk_metrics()
        
        # Load performance metrics from database
        self._load_performance_metrics()
    
    def _load_risk_metrics(self):
        """
        Load risk metrics from database.
        """
        try:
            # Get latest risk metrics
            result = self.db.risk_metrics_collection.find_one(
                sort=[('timestamp', -1)]
            )
            
            if result:
                # Remove MongoDB ID
                if '_id' in result:
                    del result['_id']
                    
                self.risk_metrics = result
                self.logger.info(f"Loaded risk metrics from {result.get('timestamp')}")
                
        except Exception as e:
            self.logger.error(f"Error loading risk metrics: {e}")
    
    def _load_performance_metrics(self):
        """
        Load performance metrics from database.
        """
        try:
            # Get latest performance metrics
            result = self.db.performance_metrics_collection.find_one(
                sort=[('timestamp', -1)]
            )
            
            if result:
                # Remove MongoDB ID
                if '_id' in result:
                    del result['_id']
                    
                self.performance_metrics = result
                self.logger.info(f"Loaded performance metrics from {result.get('timestamp')}")
                
        except Exception as e:
            self.logger.error(f"Error loading performance metrics: {e}")
    
    def add_alert_handler(self, handler):
        """
        Add alert handler function.
        
        Args:
            handler: Alert handler function that accepts alert dict
        """
        if callable(handler):
            self.alert_handlers.append(handler)
            self.logger.info(f"Added alert handler {handler.__name__}")
        else:
            self.logger.error("Alert handler must be callable")
    
    def start(self):
        """
        Start the trading monitor.
        
        Returns:
            bool: Success status
        """
        if self.is_running:
            self.logger.warning("Trading monitor is already running")
            return False
            
        # Start monitoring thread
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Trading monitor started")
        
        return True
    
    def stop(self, wait=True):
        """
        Stop the trading monitor.
        
        Args:
            wait (bool): Wait for monitoring thread to complete
            
        Returns:
            bool: Success status
        """
        if not self.is_running:
            self.logger.warning("Trading monitor is not running")
            return False
            
        self.is_running = False
        
        if wait and self.monitoring_thread:
            self.monitoring_thread.join(timeout=30)
            
        self.logger.info("Trading monitor stopped")
        
        return True
    
    def _monitoring_loop(self):
        """
        Main monitoring loop.
        """
        last_snapshot_time = time.time()
        last_equity_update_time = time.time()
        
        while self.is_running:
            try:
                now = time.time()
                
                # Check risk metrics
                self._check_risk_metrics()
                
                # Take performance snapshot if interval elapsed
                if now - last_snapshot_time >= self.config['performance_snapshot_interval']:
                    self._take_performance_snapshot()
                    last_snapshot_time = now
                    
                # Update equity curve if interval elapsed
                if now - last_equity_update_time >= self.config['equity_update_interval']:
                    self._update_equity_curve()
                    last_equity_update_time = now
                    
                # Sleep until next check
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Sleep longer on error
    
    def _check_risk_metrics(self):
        """
        Check risk metrics and generate alerts if needed.
        """
        try:
            # Skip if no trading engine
            if not self.engine:
                return
                
            # Get position data
            positions = self.engine.get_open_positions()
            
            if not positions:
                return
                
            # Get account info
            account_info = None
            
            if hasattr(self.engine, 'market_data_connector') and self.engine.market_data_connector:
                account_info = self.engine.market_data_connector.get_account_info()
                
            current_equity = account_info.get('balance', 100000) if account_info else 100000
            
            # Initialize metrics
            metrics = {
                'timestamp': datetime.now(),
                'total_equity': current_equity,
                'open_positions': len(positions),
                'position_details': positions,
                'position_exposure': {},
                'sector_exposure': {},
                'instrument_exposure': {},
                'daily_pnl': 0,
                'daily_pnl_pct': 0,
                'unrealized_pnl': 0,
                'unrealized_pnl_pct': 0,
                'max_position_size': 0,
                'max_position_name': '',
                'max_sector_exposure': 0,
                'max_sector_name': '',
                'correlation_matrix': {},
                'risk_concentration': 0,
                'avg_position_staleness': 0
            }
            
            # Calculate exposures and metrics
            unrealized_pnl = 0
            sector_exposure = {}
            instrument_exposure = {}
            position_exposure = {}
            stale_positions = []
            max_position_value = 0
            max_position_name = ''
            
            for position in positions:
                symbol = position.get('symbol', '')
                exchange = position.get('exchange', '')
                position_id = position.get('position_id', '')
                
                if not symbol or not exchange or not position_id:
                    continue
                    
                # Get position value
                quantity = position.get('quantity', 0)
                entry_price = position.get('entry_price', 0)
                current_price = position.get('current_price', entry_price)
                
                position_value = quantity * current_price
                
                # Calculate position exposure
                exposure_pct = position_value / current_equity * 100
                position_exposure[f"{symbol}:{exchange}"] = exposure_pct
                
                # Check for maximum position
                if position_value > max_position_value:
                    max_position_value = position_value
                    max_position_name = symbol
                
                # Calculate unrealized P&L
                position_pnl = position.get('unrealized_pnl', 0)
                unrealized_pnl += position_pnl
                
                # Check position staleness
                if 'last_update' in position:
                    last_update = position['last_update']
                    
                    if isinstance(last_update, str):
                        last_update = datetime.strptime(last_update, '%Y-%m-%d %H:%M:%S')
                        
                    staleness = (datetime.now() - last_update).total_seconds() / 60
                    
                    if staleness > self.config['max_position_staleness_minutes']:
                        stale_positions.append({
                            'symbol': symbol,
                            'exchange': exchange,
                            'staleness_minutes': staleness
                        })
                
                # Get sector info
                if self.config['track_sector_exposure']:
                    sector = self._get_sector(symbol, exchange)
                    
                    if sector:
                        if sector not in sector_exposure:
                            sector_exposure[sector] = 0
                            
                        sector_exposure[sector] += exposure_pct
                
                # Get instrument type
                if self.config['track_instrument_type_performance']:
                    instrument_type = self._get_instrument_type(symbol, exchange)
                    
                    if instrument_type:
                        if instrument_type not in instrument_exposure:
                            instrument_exposure[instrument_type] = 0
                            
                        instrument_exposure[instrument_type] += exposure_pct
            
            # Calculate metrics
            metrics['position_exposure'] = position_exposure
            metrics['sector_exposure'] = sector_exposure
            metrics['instrument_exposure'] = instrument_exposure
            metrics['unrealized_pnl'] = unrealized_pnl
            metrics['unrealized_pnl_pct'] = unrealized_pnl / current_equity * 100 if current_equity > 0 else 0
            metrics['max_position_size'] = max_position_value / current_equity * 100 if current_equity > 0 else 0
            metrics['max_position_name'] = max_position_name
            
            # Find max sector exposure
            if sector_exposure:
                max_sector = max(sector_exposure.items(), key=lambda x: x[1])
                metrics['max_sector_exposure'] = max_sector[1]
                metrics['max_sector_name'] = max_sector[0]
                
            # Calculate daily P&L
            start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            daily_pnl_query = {
                'timestamp': {'$gte': start_of_day}
            }
            
            daily_trades = list(self.db.trades_collection.find(daily_pnl_query))
            
            daily_pnl = sum(trade.get('profit_loss', 0) for trade in daily_trades)
            metrics['daily_pnl'] = daily_pnl
            metrics['daily_pnl_pct'] = daily_pnl / current_equity * 100 if current_equity > 0 else 0
            
            # Calculate position staleness
            if stale_positions:
                avg_staleness = sum(p['staleness_minutes'] for p in stale_positions) / len(stale_positions)
                metrics['avg_position_staleness'] = avg_staleness
                metrics['stale_positions'] = stale_positions
                
            # Store risk metrics
            self.risk_metrics = metrics
            
            # Save to database
            self._save_risk_metrics(metrics)
            
            # Check for alerts
            self._check_risk_alerts(metrics)
            
        except Exception as e:
            self.logger.error(f"Error checking risk metrics: {e}")
    
    def _get_sector(self, symbol, exchange):
        """
        Get sector for a symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            
        Returns:
            str: Sector
        """
        try:
            # Try to get from database
            result = self.db.symbol_info_collection.find_one({
                'symbol': symbol,
                'exchange': exchange
            })
            
            if result and 'sector' in result:
                return result['sector']
                
            return 'Unknown'
            
        except Exception as e:
            self.logger.error(f"Error getting sector for {symbol}: {e}")
            return 'Unknown'
    
    def _get_instrument_type(self, symbol, exchange):
        """
        Get instrument type for a symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            
        Returns:
            str: Instrument type
        """
        try:
            # Try to get from database
            result = self.db.symbol_info_collection.find_one({
                'symbol': symbol,
                'exchange': exchange
            })
            
            if result and 'instrument_type' in result:
                return result['instrument_type']
                
            # Try to infer from exchange
            if exchange == 'NSE':
                return 'EQUITY'
            elif exchange == 'NFO':
                if 'CE' in symbol or 'PE' in symbol:
                    return 'OPTION'
                else:
                    return 'FUTURE'
                    
            return 'EQUITY'  # Default
            
        except Exception as e:
            self.logger.error(f"Error getting instrument type for {symbol}: {e}")
            return 'EQUITY'  # Default
    
    def _save_risk_metrics(self, metrics):
        """
        Save risk metrics to database.
        
        Args:
            metrics (dict): Risk metrics
        """
        try:
            # Create copy for storage
            metrics_copy = metrics.copy()
            
            # Convert non-serializable objects
            metrics_copy['timestamp'] = datetime.now()
            
            # Insert into database
            self.db.risk_metrics_collection.insert_one(metrics_copy)
            
        except Exception as e:
            self.logger.error(f"Error saving risk metrics: {e}")
    
    def _check_risk_alerts(self, metrics):
        """
        Check risk metrics and generate alerts if needed.
        
        Args:
            metrics (dict): Risk metrics
        """
        try:
            alerts = []
            
            # Check max drawdown
            unrealized_pnl_pct = metrics.get('unrealized_pnl_pct', 0)
            
            if unrealized_pnl_pct < -self.config['max_drawdown_alert']:
                alerts.append({
                    'type': 'MAX_DRAWDOWN',
                    'severity': 'HIGH',
                    'message': f"Current drawdown of {abs(unrealized_pnl_pct):.2f}% exceeds threshold of {self.config['max_drawdown_alert']}%",
                    'timestamp': datetime.now(),
                    'details': {
                        'current_drawdown': unrealized_pnl_pct,
                        'threshold': self.config['max_drawdown_alert']
                    }
                })
                
            # Check max position size
            max_position_size = metrics.get('max_position_size', 0)
            
            if max_position_size > self.config['max_position_size_pct']:
                alerts.append({
                    'type': 'MAX_POSITION_SIZE',
                    'severity': 'MEDIUM',
                    'message': f"Position {metrics.get('max_position_name', '')} size of {max_position_size:.2f}% exceeds threshold of {self.config['max_position_size_pct']}%",
                    'timestamp': datetime.now(),
                    'details': {
                        'symbol': metrics.get('max_position_name', ''),
                        'current_size': max_position_size,
                        'threshold': self.config['max_position_size_pct']
                    }
                })
                
            # Check sector exposure
            max_sector_exposure = metrics.get('max_sector_exposure', 0)
            
            if max_sector_exposure > self.config['max_sector_exposure_pct']:
                alerts.append({
                    'type': 'MAX_SECTOR_EXPOSURE',
                    'severity': 'MEDIUM',
                    'message': f"Sector {metrics.get('max_sector_name', '')} exposure of {max_sector_exposure:.2f}% exceeds threshold of {self.config['max_sector_exposure_pct']}%",
                    'timestamp': datetime.now(),
                    'details': {
                        'sector': metrics.get('max_sector_name', ''),
                        'current_exposure': max_sector_exposure,
                        'threshold': self.config['max_sector_exposure_pct']
                    }
                })
                
            # Check daily loss
            daily_pnl_pct = metrics.get('daily_pnl_pct', 0)
            
            if daily_pnl_pct < -self.config['max_daily_loss_pct']:
                alerts.append({
                    'type': 'MAX_DAILY_LOSS',
                    'severity': 'HIGH',
                    'message': f"Daily loss of {abs(daily_pnl_pct):.2f}% exceeds threshold of {self.config['max_daily_loss_pct']}%",
                    'timestamp': datetime.now(),
                    'details': {
                        'current_loss': daily_pnl_pct,
                        'threshold': self.config['max_daily_loss_pct']
                    }
                })
                
            # Check position staleness
            if 'stale_positions' in metrics and metrics['stale_positions']:
                position_count = len(metrics['stale_positions'])
                position_list = ', '.join([p['symbol'] for p in metrics['stale_positions'][:3]])
                
                if position_count > 3:
                    position_list += f" and {position_count - 3} more"
                
                alerts.append({
                    'type': 'STALE_POSITIONS',
                    'severity': 'MEDIUM',
                    'message': f"{position_count} positions have not been updated for over {self.config['max_position_staleness_minutes']} minutes: {position_list}",
                    'timestamp': datetime.now(),
                    'details': {
                        'positions': metrics['stale_positions'],
                        'threshold_minutes': self.config['max_position_staleness_minutes']
                    }
                })
            
            # Process alerts
            for alert in alerts:
                self._process_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Error checking risk alerts: {e}")
    
    def _process_alert(self, alert):
        """
        Process and dispatch an alert.
        
        Args:
            alert (dict): Alert data
        """
        # Store alert
        self.alerts.append(alert)
        
        # Log alert
        severity = alert.get('severity', 'MEDIUM')
        message = alert.get('message', '')
        
        if severity == 'HIGH':
            self.logger.error(f"ALERT: {message}")
        elif severity == 'MEDIUM':
            self.logger.warning(f"ALERT: {message}")
        else:
            self.logger.info(f"ALERT: {message}")
            
        # Save to database
        try:
            self.db.alerts_collection.insert_one(alert)
        except Exception as e:
            self.logger.error(f"Error saving alert to database: {e}")
            
        # Dispatch to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    def _take_performance_snapshot(self):
        """
        Take a performance snapshot.
        """
        try:
            # Skip if no trading engine
            if not self.engine:
                return
                
            # Get performance metrics
            if hasattr(self.engine, 'get_performance_metrics'):
                metrics = self.engine.get_performance_metrics()
            else:
                metrics = {}
                
            # Add timestamp
            metrics['timestamp'] = datetime.now()
            
            # Get position data
            positions = self.engine.get_open_positions()
            metrics['open_positions'] = len(positions)
            
            # Get equity data
            if hasattr(self.engine, 'market_data_connector') and self.engine.market_data_connector:
                account_info = self.engine.market_data_connector.get_account_info()
                
                if account_info:
                    metrics['current_equity'] = account_info.get('balance', 0)
            
            # Add strategy performance if enabled
            if self.config['track_strategy_performance'] and hasattr(self.engine, 'trade_history'):
                metrics['strategy_performance'] = self._calculate_strategy_performance(
                    self.engine.trade_history
                )
                
            # Add instrument type performance if enabled
            if self.config['track_instrument_type_performance'] and hasattr(self.engine, 'trade_history'):
                metrics['instrument_performance'] = self._calculate_instrument_performance(
                    self.engine.trade_history
                )
                
            # Store metrics
            self.performance_metrics = metrics
            
            # Save to database
            self._save_performance_metrics(metrics)
            
            # Add to snapshots
            self.snapshots.append(metrics)
            
            # Keep only recent snapshots in memory
            if len(self.snapshots) > 24:  # Keep last 24 snapshots (typically one day)
                self.snapshots.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error taking performance snapshot: {e}")
    
    def _calculate_strategy_performance(self, trade_history):
        """
        Calculate performance metrics by strategy.
        
        Args:
            trade_history (list): Trade history
            
        Returns:
            dict: Strategy performance metrics
        """
        try:
            if not trade_history:
                return {}
                
            # Group trades by strategy
            strategy_trades = {}
            
            for trade in trade_history:
                strategy_id = trade.get('strategy_id', 'unknown')
                
                if strategy_id not in strategy_trades:
                    strategy_trades[strategy_id] = []
                    
                strategy_trades[strategy_id].append(trade)
                
            # Calculate metrics for each strategy
            strategy_metrics = {}
            
            for strategy_id, trades in strategy_trades.items():
                # Skip strategies with no trades
                if not trades:
                    continue
                    
                # Calculate metrics
                total_trades = len(trades)
                profitable_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
                losing_trades = [t for t in trades if t.get('profit_loss', 0) <= 0]
                
                win_count = len(profitable_trades)
                loss_count = len(losing_trades)
                
                win_rate = win_count / total_trades * 100 if total_trades > 0 else 0
                
                total_profit = sum(t.get('profit_loss', 0) for t in profitable_trades)
                total_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
                
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                avg_profit = total_profit / win_count if win_count > 0 else 0
                avg_loss = total_loss / loss_count if loss_count > 0 else 0
                
                # Store metrics
                strategy_metrics[strategy_id] = {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'total_profit': total_profit,
                    'total_loss': total_loss,
                    'avg_profit': avg_profit,
                    'avg_loss': avg_loss
                }
                
            return strategy_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy performance: {e}")
            return {}
    
    def _calculate_instrument_performance(self, trade_history):
        """
        Calculate performance metrics by instrument type.
        
        Args:
            trade_history (list): Trade history
            
        Returns:
            dict: Instrument performance metrics
        """
        try:
            if not trade_history:
                return {}
                
            # Group trades by instrument type
            instrument_trades = {}
            
            for trade in trade_history:
                symbol = trade.get('symbol', '')
                exchange = trade.get('exchange', '')
                
                if not symbol or not exchange:
                    continue
                    
                instrument_type = self._get_instrument_type(symbol, exchange)
                
                if instrument_type not in instrument_trades:
                    instrument_trades[instrument_type] = []
                    
                instrument_trades[instrument_type].append(trade)
                
            # Calculate metrics for each instrument type
            instrument_metrics = {}
            
            for instrument_type, trades in instrument_trades.items():
                # Skip instrument types with no trades
                if not trades:
                    continue
                    
                # Calculate metrics
                total_trades = len(trades)
                profitable_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
                losing_trades = [t for t in trades if t.get('profit_loss', 0) <= 0]
                
                win_count = len(profitable_trades)
                loss_count = len(losing_trades)
                
                win_rate = win_count / total_trades * 100 if total_trades > 0 else 0
                
                total_profit = sum(t.get('profit_loss', 0) for t in profitable_trades)
                total_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
                
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                avg_profit = total_profit / win_count if win_count > 0 else 0
                avg_loss = total_loss / loss_count if loss_count > 0 else 0
                
                # Store metrics
                instrument_metrics[instrument_type] = {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'total_profit': total_profit,
                    'total_loss': total_loss,
                    'avg_profit': avg_profit,
                    'avg_loss': avg_loss
                }
                
            return instrument_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating instrument performance: {e}")
            return {}
    
    def _save_performance_metrics(self, metrics):
        """
        Save performance metrics to database.
        
        Args:
            metrics (dict): Performance metrics
        """
        try:
            # Create copy for storage
            metrics_copy = metrics.copy()
            
            # Convert non-serializable objects
            metrics_copy['timestamp'] = datetime.now()
            
            # Insert into database
            self.db.performance_metrics_collection.insert_one(metrics_copy)
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")
    
    def _update_equity_curve(self):
        """
        Update equity curve.
        """
        try:
            # Skip if no trading engine
            if not self.engine:
                return
                
            # Get current equity
            current_equity = 0
            
            if hasattr(self.engine, 'market_data_connector') and self.engine.market_data_connector:
                account_info = self.engine.market_data_connector.get_account_info()
                
                if account_info:
                    current_equity = account_info.get('balance', 0)
            elif hasattr(self.engine, '_get_current_equity'):
                current_equity = self.engine._get_current_equity()
                
            if current_equity <= 0:
                return
                
            # Create equity curve point
            equity_point = {
                'timestamp': datetime.now(),
                'equity': current_equity
            }
            
            # Save to database
            self.db.equity_curve_collection.insert_one(equity_point)
            
        except Exception as e:
            self.logger.error(f"Error updating equity curve: {e}")
    
    def get_risk_metrics(self):
        """
        Get current risk metrics.
        
        Returns:
            dict: Risk metrics
        """
        return self.risk_metrics
    
    def get_performance_metrics(self):
        """
        Get current performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        return self.performance_metrics
    
    def get_equity_curve(self, days=7):
        """
        Get equity curve data.
        
        Args:
            days (int): Number of days of data to retrieve
            
        Returns:
            list: Equity curve data
        """
        try:
            # Calculate start date
            start_date = datetime.now() - timedelta(days=days)
            
            # Query database
            cursor = self.db.equity_curve_collection.find({
                'timestamp': {'$gte': start_date}
            }).sort('timestamp', 1)
            
            # Convert to list
            equity_curve = list(cursor)
            
            # Remove MongoDB IDs
            for point in equity_curve:
                if '_id' in point:
                    del point['_id']
                    
            return equity_curve
            
        except Exception as e:
            self.logger.error(f"Error getting equity curve: {e}")
            return []
    
    def get_drawdown_analysis(self):
        """
        Get drawdown analysis.
        
        Returns:
            dict: Drawdown analysis
        """
        try:
            # Get equity curve for analysis
            equity_curve = self.get_equity_curve(days=90)  # Get 90 days of data
            
            if not equity_curve:
                return {
                    'max_drawdown': 0,
                    'max_drawdown_date': None,
                    'current_drawdown': 0,
                    'drawdown_periods': []
                }
                
            # Create DataFrame
            df = pd.DataFrame(equity_curve)
            
            # Calculate drawdowns
            df['peak'] = df['equity'].cummax()
            df['drawdown'] = (df['peak'] - df['equity']) / df['peak'] * 100
            
            # Calculate max drawdown
            max_drawdown = df['drawdown'].max()
            max_drawdown_idx = df['drawdown'].idxmax()
            max_drawdown_date = df.loc[max_drawdown_idx, 'timestamp'] if max_drawdown_idx is not None else None
            
            # Calculate current drawdown
            current_drawdown = df['drawdown'].iloc[-1]
            
            # Calculate drawdown periods
            drawdown_periods = []
            in_drawdown = False
            current_period = {}
            recovery_threshold = 0.5  # 50% recovery is considered end of period
            
            for i, row in df.iterrows():
                # Start of drawdown
                if not in_drawdown and row['drawdown'] > 2.0:  # 2% threshold to consider a drawdown
                    in_drawdown = True
                    current_period = {
                        'start_date': row['timestamp'],
                        'start_equity': row['equity'],
                        'max_drawdown': row['drawdown'],
                        'max_drawdown_date': row['timestamp'],
                        'end_date': None,
                        'end_equity': None,
                        'recovery_days': None
                    }
                
                # Update max drawdown in period
                if in_drawdown and row['drawdown'] > current_period['max_drawdown']:
                    current_period['max_drawdown'] = row['drawdown']
                    current_period['max_drawdown_date'] = row['timestamp']
                
                # End of drawdown (recovery)
                if in_drawdown and row['drawdown'] < current_period['max_drawdown'] * recovery_threshold:
                    in_drawdown = False
                    current_period['end_date'] = row['timestamp']
                    current_period['end_equity'] = row['equity']
                    
                    # Calculate recovery days
                    recovery_days = (current_period['end_date'] - current_period['max_drawdown_date']).days
                    current_period['recovery_days'] = recovery_days
                    
                    # Add to periods
                    drawdown_periods.append(current_period)
                    current_period = {}
            
            # If we're still in a drawdown, add it as ongoing
            if in_drawdown:
                current_period['is_ongoing'] = True
                drawdown_periods.append(current_period)
                
            # Sort by max drawdown
            drawdown_periods.sort(key=lambda x: x.get('max_drawdown', 0), reverse=True)
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_date': max_drawdown_date,
                'current_drawdown': current_drawdown,
                'drawdown_periods': drawdown_periods
            }
            
        except Exception as e:
            self.logger.error(f"Error getting drawdown analysis: {e}")
            return {
                'max_drawdown': 0,
                'max_drawdown_date': None,
                'current_drawdown': 0,
                'drawdown_periods': []
            }
    
    def get_alerts(self, count=10, severity=None, alert_type=None):
        """
        Get recent alerts.
        
        Args:
            count (int): Number of alerts to retrieve
            severity (str): Filter by severity (HIGH, MEDIUM, LOW)
            alert_type (str): Filter by alert type
            
        Returns:
            list: Recent alerts
        """
        try:
            # Build query
            query = {}
            
            if severity:
                query['severity'] = severity
                
            if alert_type:
                query['type'] = alert_type
                
            # Query database
            cursor = self.db.alerts_collection.find(query).sort('timestamp', -1).limit(count)
            
            # Convert to list
            alerts = list(cursor)
            
            # Remove MongoDB IDs
            for alert in alerts:
                if '_id' in alert:
                    del alert['_id']
                    
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting alerts: {e}")
            return []
    
    def get_strategy_performance(self):
        """
        Get performance by strategy.
        
        Returns:
            dict: Strategy performance metrics
        """
        try:
            if not self.performance_metrics or 'strategy_performance' not in self.performance_metrics:
                return {}
                
            return self.performance_metrics['strategy_performance']
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {e}")
            return {}
    
    def get_instrument_performance(self):
        """
        Get performance by instrument type.
        
        Returns:
            dict: Instrument performance metrics
        """
        try:
            if not self.performance_metrics or 'instrument_performance' not in self.performance_metrics:
                return {}
                
            return self.performance_metrics['instrument_performance']
            
        except Exception as e:
            self.logger.error(f"Error getting instrument performance: {e}")
            return {}
    
    def get_correlation_matrix(self, days=30):
        """
        Get correlation matrix for positions.
        
        Args:
            days (int): Number of days of data to use
            
        Returns:
            dict: Correlation matrix
        """
        try:
            # Skip if no trading engine
            if not self.engine or not hasattr(self.engine, 'get_open_positions'):
                return {}
                
            # Get open positions
            positions = self.engine.get_open_positions()
            
            if not positions:
                return {}
                
            # Get symbols
            symbols = []
            exchanges = []
            
            for position in positions:
                symbol = position.get('symbol')
                exchange = position.get('exchange')
                
                if symbol and exchange:
                    symbols.append(symbol)
                    exchanges.append(exchange)
                    
            if not symbols:
                return {}
                
            # Get historical data
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
            
            # Get price data
            price_data = {}
            
            if hasattr(self.engine, 'market_data_connector') and self.engine.market_data_connector:
                for i, symbol in enumerate(symbols):
                    exchange = exchanges[i]
                    
                    data = self.engine.market_data_connector.get_historical_data(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe='day',
                        from_date=start_date,
                        to_date=end_date
                    )
                    
                    if data:
                        # Extract close prices
                        closes = [bar.get('close') for bar in data if 'close' in bar]
                        
                        if closes:
                            price_data[f"{symbol}:{exchange}"] = closes
            
            if len(price_data) < 2:
                return {}
                
            # Create DataFrame
            df = pd.DataFrame(price_data)
            
            # Calculate returns
            returns_df = df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr().to_dict()
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return {}
    
    def get_performance_summary(self):
        """
        Get performance summary.
        
        Returns:
            dict: Performance summary
        """
        try:
            # Get latest metrics
            metrics = self.performance_metrics.copy() if self.performance_metrics else {}
            
            # Add drawdown analysis
            drawdown_analysis = self.get_drawdown_analysis()
            metrics['drawdown_analysis'] = drawdown_analysis
            
            # Add correlation matrix
            correlation_matrix = self.get_correlation_matrix()
            metrics['correlation_matrix'] = correlation_matrix
            
            # Add risk metrics
            risk_metrics = self.risk_metrics.copy() if self.risk_metrics else {}
            
            # Remove position details to keep size manageable
            if 'position_details' in risk_metrics:
                del risk_metrics['position_details']
                
            metrics['risk_metrics'] = risk_metrics
            
            # Add recent alerts
            alerts = self.get_alerts(count=5)
            metrics['recent_alerts'] = alerts
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def generate_daily_report(self):
        """
        Generate daily performance report.
        
        Returns:
            dict: Daily report
        """
        try:
            # Get current date
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Get performance summary
            summary = self.get_performance_summary()
            
            # Get today's trades
            start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            daily_trades_query = {
                'timestamp': {'$gte': start_of_day}
            }
            
            daily_trades = list(self.db.trades_collection.find(daily_trades_query))
            
            # Remove MongoDB IDs
            for trade in daily_trades:
                if '_id' in trade:
                    del trade['_id']
            
            # Calculate daily metrics
            daily_pnl = sum(trade.get('profit_loss', 0) for trade in daily_trades)
            win_trades = [t for t in daily_trades if t.get('profit_loss', 0) > 0]
            loss_trades = [t for t in daily_trades if t.get('profit_loss', 0) <= 0]
            
            win_count = len(win_trades)
            loss_count = len(loss_trades)
            total_trades = win_count + loss_count
            
            win_rate = win_count / total_trades * 100 if total_trades > 0 else 0
            
            # Create report
            report = {
                'date': today,
                'summary': summary,
                'daily_metrics': {
                    'pnl': daily_pnl,
                    'trade_count': total_trades,
                    'win_count': win_count,
                    'loss_count': loss_count,
                    'win_rate': win_rate
                },
                'trades': daily_trades
            }
            
            # Save to database
            self.db.daily_reports_collection.insert_one(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
            return {}