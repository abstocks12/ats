# automation/intraday_monitor.py
import logging
import threading
import time
from datetime import datetime, timedelta

class IntradayMonitor:
    """
    Monitors intraday market conditions and trading performance.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the intraday monitor.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # State
        self.is_running = False
        self.monitor_thread = None
        
        # Configuration
        self.config = {
            'check_interval': 60,  # Seconds between checks
            'market_open_time': '09:15',  # Market opening time (IST)
            'market_close_time': '15:30',  # Market closing time (IST)
            'alert_drawdown_threshold': 2.0,  # Alert if drawdown exceeds this percentage
            'alert_profit_threshold': 5.0,  # Alert if profit exceeds this percentage
            'position_update_interval': 5 * 60,  # Update positions every 5 minutes
            'stop_loss_check_interval': 30,  # Check stop losses every 30 seconds
            'market_data_update_interval': 5 * 60,  # Update market data every 5 minutes
            'volatility_check_interval': 15 * 60,  # Check volatility every 15 minutes
            'opportunity_scan_interval': 30 * 60,  # Scan for opportunities every 30 minutes
            'prediction_update_interval': 60 * 60,  # Update predictions every hour
            'log_interval': 15 * 60,  # Log status every 15 minutes
        }
        
        # Runtime data
        self.market_status = "closed"
        self.last_checks = {
            'position_update': datetime.now(),
            'stop_loss_check': datetime.now(),
            'market_data_update': datetime.now(),
            'volatility_check': datetime.now(),
            'opportunity_scan': datetime.now(),
            'prediction_update': datetime.now(),
            'log_status': datetime.now(),
        }
        
        # Tracking state
        self.portfolio_value = None
        self.max_portfolio_value = None
        self.drawdown = 0.0
        self.active_positions = []
        
        self.logger.info("Intraday monitor initialized")

    def start(self):
        """
        Start the intraday monitor.
        
        Returns:
            bool: Success status
        """
        if self.is_running:
            self.logger.warning("Intraday monitor already running")
            return False
            
        self.is_running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Intraday monitor started")
        
        return True
        
    def stop(self):
        """
        Stop the intraday monitor.
        
        Returns:
            bool: Success status
        """
        if not self.is_running:
            self.logger.warning("Intraday monitor not running")
            return False
            
        self.is_running = False
        
        # Wait for monitoring thread to terminate
        if self.monitor_thread:
            self.monitor_thread.join(timeout=30)
            self.monitor_thread = None
            
        self.logger.info("Intraday monitor stopped")
        
        return True
        
    def _monitoring_loop(self):
        """
        Main monitoring loop.
        """
        while self.is_running:
            try:
                # Check market status
                self._update_market_status()
                
                # Only perform monitoring during market hours
                if self.market_status == "open":
                    # Perform necessary checks based on intervals
                    now = datetime.now()
                    
                    # Update positions
                    if (now - self.last_checks['position_update']).total_seconds() >= self.config['position_update_interval']:
                        self._update_positions()
                        self.last_checks['position_update'] = now
                    
                    # Check stop losses
                    if (now - self.last_checks['stop_loss_check']).total_seconds() >= self.config['stop_loss_check_interval']:
                        self._check_stop_losses()
                        self.last_checks['stop_loss_check'] = now
                    
                    # Update market data
                    if (now - self.last_checks['market_data_update']).total_seconds() >= self.config['market_data_update_interval']:
                        self._update_market_data()
                        self.last_checks['market_data_update'] = now
                    
                    # Check volatility
                    if (now - self.last_checks['volatility_check']).total_seconds() >= self.config['volatility_check_interval']:
                        self._check_volatility()
                        self.last_checks['volatility_check'] = now
                    
                    # Scan for opportunities
                    if (now - self.last_checks['opportunity_scan']).total_seconds() >= self.config['opportunity_scan_interval']:
                        self._scan_opportunities()
                        self.last_checks['opportunity_scan'] = now
                    
                    # Update predictions
                    if (now - self.last_checks['prediction_update']).total_seconds() >= self.config['prediction_update_interval']:
                        self._update_predictions()
                        self.last_checks['prediction_update'] = now
                    
                    # Log status
                    if (now - self.last_checks['log_status']).total_seconds() >= self.config['log_interval']:
                        self._log_status()
                        self.last_checks['log_status'] = now
                
                # Sleep for check interval
                time.sleep(self.config['check_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config['check_interval'])
                
    def _update_market_status(self):
        """
        Update market status (open/closed).
        """
        try:
            # Import market hours module
            from trading.market_hours import MarketHours
            
            # Initialize market hours
            market_hours = MarketHours()
            
            # Check if market is open
            is_open = market_hours.is_market_open()
            
            # Update status
            prev_status = self.market_status
            self.market_status = "open" if is_open else "closed"
            
            # Log status change
            if prev_status != self.market_status:
                self.logger.info(f"Market status changed: {prev_status} -> {self.market_status}")
                
                # Perform status change actions
                if self.market_status == "open":
                    self._market_open_actions()
                else:
                    self._market_close_actions()
                    
        except Exception as e:
            self.logger.error(f"Error updating market status: {e}")
            
    def _market_open_actions(self):
        """
        Actions to perform when market opens.
        """
        try:
            self.logger.info("Market opened, initializing monitoring")
            
            # Initialize portfolio value
            self._update_portfolio_value()
            self.max_portfolio_value = self.portfolio_value
            
            # Start real-time data collection
            self._start_real_time_data()
            
            # Initialize active positions
            self._update_positions()
            
            # Log initial status
            self._log_status(force=True)
            
        except Exception as e:
            self.logger.error(f"Error in market open actions: {e}")
            
    def _market_close_actions(self):
        """
        Actions to perform when market closes.
        """
        try:
            self.logger.info("Market closed, finalizing monitoring")
            
            # Stop real-time data collection
            self._stop_real_time_data()
            
            # Update final positions
            self._update_positions()
            
            # Log final status
            self._log_status(force=True)
            
            # Generate end-of-day summary
            self._generate_eod_summary()
            
        except Exception as e:
            self.logger.error(f"Error in market close actions: {e}")
            
    def _update_portfolio_value(self):
        """
        Update current portfolio value.
        """
        try:
            if not self.db:
                return
                
            # Get position manager
            from trading.position_manager import PositionManager
            position_manager = PositionManager(self.db)
            
            # Get portfolio value
            portfolio_value = position_manager.get_portfolio_value()
            
            # Update tracking
            self.portfolio_value = portfolio_value
            
            # Update max portfolio value
            if self.max_portfolio_value is None or portfolio_value > self.max_portfolio_value:
                self.max_portfolio_value = portfolio_value
                
            # Calculate drawdown
            if self.max_portfolio_value > 0:
                self.drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value * 100
                
                # Check drawdown threshold
                if self.drawdown > self.config['alert_drawdown_threshold']:
                    self._alert_drawdown()
                    
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
            
    def _update_positions(self):
        """
        Update active positions.
        """
        try:
            if not self.db:
                return
                
            # Get position manager
            from trading.position_manager import PositionManager
            position_manager = PositionManager(self.db)
            
            # Get active positions
            positions = position_manager.get_active_positions()
            
            # Update tracking
            self.active_positions = positions
            
            # Update portfolio value
            self._update_portfolio_value()
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            
    def _check_stop_losses(self):
        """
        Check and adjust stop losses.
        """
        try:
            if not self.db or not self.active_positions:
                return
                
            # Get stop management
            from core.risk.stop_management import update_stop_loss
            
            # Get position manager
            from trading.position_manager import PositionManager
            position_manager = PositionManager(self.db)
            
            # Check each position
            for position in self.active_positions:
                try:
                    # Update stop loss
                    new_stop = update_stop_loss(position, None, self.db)
                    
                    # If stop loss changed, update position
                    if new_stop and new_stop != position.get('stop_loss'):
                        position_manager.update_stop_loss(position['_id'], new_stop)
                        self.logger.info(f"Updated stop loss for {position['symbol']}: {new_stop}")
                        
                except Exception as e:
                    self.logger.error(f"Error updating stop loss for {position.get('symbol', 'unknown')}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error checking stop losses: {e}")
            
    def _update_market_data(self):
        """
        Update market data for analysis.
        """
        try:
            if not self.db:
                return
                
            # Get market data collector
            from data.market.real_time import RealTimeDataCollector
            data_collector = RealTimeDataCollector(self.db)
            
            # Update indicators for active positions
            for position in self.active_positions:
                try:
                    symbol = position.get('symbol')
                    exchange = position.get('exchange')
                    
                    if symbol and exchange:
                        # Update indicators
                        data_collector.update_indicators(symbol, exchange)
                        
                except Exception as e:
                    self.logger.error(f"Error updating data for {position.get('symbol', 'unknown')}: {e}")
                    
            # Update market indices
            index_symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
            exchange = "NSE"
            
            for symbol in index_symbols:
                try:
                    data_collector.update_indicators(symbol, exchange)
                except Exception as e:
                    self.logger.error(f"Error updating index data for {symbol}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            
    def _check_volatility(self):
        """
        Check market volatility and adjust risk accordingly.
        """
        try:
            if not self.db:
                return
                
            # Get volatility analyzer
            from research.volatility_analyzer import VolatilityAnalyzer
            volatility_analyzer = VolatilityAnalyzer(self.db)
            
            # Get current volatility
            market_volatility = volatility_analyzer.get_current_volatility("NIFTY", "NSE")
            
            # Check volatility threshold
            if market_volatility.get('is_high_volatility', False):
                self._alert_high_volatility(market_volatility)
                
                # Adjust risk parameters
                self._adjust_risk_for_volatility(market_volatility)
                
        except Exception as e:
            self.logger.error(f"Error checking volatility: {e}")
            
    def _scan_opportunities(self):
        """
        Scan for intraday trading opportunities.
        """
        try:
            if not self.db:
                return
                
            # Get opportunity scanner
            from research.opportunity_scanner import OpportunityScanner
            opportunity_scanner = OpportunityScanner(self.db)
            
            # Scan for opportunities
            opportunities = opportunity_scanner.scan_intraday_opportunities()
            
            # Process opportunities
            if opportunities:
                self.logger.info(f"Found {len(opportunities)} intraday opportunities")
                
                # Alert about best opportunities
                if len(opportunities) > 0:
                    self._alert_opportunities(opportunities[:3])  # Top 3 opportunities
                    
        except Exception as e:
            self.logger.error(f"Error scanning opportunities: {e}")
            
    def _update_predictions(self):
        """
        Update intraday predictions.
        """
        try:
            if not self.db:
                return
                
            # Get predictor
            from ml.prediction.daily_predictor import DailyPredictor
            predictor = DailyPredictor(self.db)
            
            # Update predictions for active positions
            for position in self.active_positions:
                try:
                    symbol = position.get('symbol')
                    exchange = position.get('exchange')
                    
                    if symbol and exchange:
                        # Update prediction
                        predictor.update_intraday_prediction(symbol, exchange)
                        
                except Exception as e:
                    self.logger.error(f"Error updating prediction for {position.get('symbol', 'unknown')}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error updating predictions: {e}")
            
    def _log_status(self, force=False):
        """
        Log current status.
        
        Args:
            force (bool): Force logging regardless of interval
        """
        try:
            if not self.db:
                return
                
            # Get status information
            num_positions = len(self.active_positions)
            portfolio_value = self.portfolio_value
            drawdown = self.drawdown
            
            # Log status
            self.logger.info(f"Status: {num_positions} positions, Portfolio: ₹{portfolio_value:.2f}, Drawdown: {drawdown:.2f}%")
            
            # Log position summary
            if num_positions > 0:
                for position in self.active_positions:
                    symbol = position.get('symbol', 'Unknown')
                    entry_price = position.get('entry_price', 0)
                    current_price = position.get('current_price', 0)
                    
                    if current_price > 0 and entry_price > 0:
                        pnl_percent = (current_price - entry_price) / entry_price * 100
                        self.logger.info(f"  {symbol}: Entry: ₹{entry_price:.2f}, Current: ₹{current_price:.2f}, PnL: {pnl_percent:.2f}%")
                        
                        # Alert if profit threshold exceeded
                        if pnl_percent > self.config['alert_profit_threshold']:
                            self._alert_profit_target(position, pnl_percent)
                            
            # Store status in database
            self._store_status()
            
        except Exception as e:
            self.logger.error(f"Error logging status: {e}")
            
    def _store_status(self):
        """
        Store status in database.
        """
        try:
            if not self.db:
                return
                
            # Create status document
            status = {
                "timestamp": datetime.now(),
                "market_status": self.market_status,
                "portfolio_value": self.portfolio_value,
                "max_portfolio_value": self.max_portfolio_value,
                "drawdown": self.drawdown,
                "num_positions": len(self.active_positions),
                "position_symbols": [p.get('symbol') for p in self.active_positions if 'symbol' in p]
            }
            
            # Store in database
            self.db.intraday_status.insert_one(status)
            
        except Exception as e:
            self.logger.error(f"Error storing status: {e}")
            
    def _generate_eod_summary(self):
        """
        Generate end-of-day summary.
        """
        try:
            if not self.db:
                return
                
            # Get position manager
            from trading.position_manager import PositionManager
            position_manager = PositionManager(self.db)
            
            # Get daily performance
            daily_performance = position_manager.get_daily_performance()
            
            # Get closed positions
            closed_positions = position_manager.get_closed_positions(
                start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            )
            
            # Create EOD summary
            summary = {
                "date": datetime.now().date(),
                "portfolio_value": self.portfolio_value,
                "daily_pnl": daily_performance.get('daily_pnl', 0),
                "daily_pnl_percent": daily_performance.get('daily_pnl_percent', 0),
                "max_drawdown": self.drawdown,
                "trades_executed": len(closed_positions),
                "winning_trades": sum(1 for p in closed_positions if p.get('profit_loss', 0) > 0),
                "losing_trades": sum(1 for p in closed_positions if p.get('profit_loss', 0) <= 0),
                "largest_win": max((p.get('profit_loss_percent', 0) for p in closed_positions), default=0),
                "largest_loss": min((p.get('profit_loss_percent', 0) for p in closed_positions), default=0),
                "open_positions": len(self.active_positions)
            }
            
            # Calculate win rate
            if summary['trades_executed'] > 0:
                summary['win_rate'] = summary['winning_trades'] / summary['trades_executed'] * 100
            else:
                summary['win_rate'] = 0
                
            # Store summary in database
            self.db.eod_summaries.insert_one(summary)
            
            # Log summary
            self.logger.info(f"EOD Summary: PnL: ₹{summary['daily_pnl']:.2f} ({summary['daily_pnl_percent']:.2f}%), Trades: {summary['trades_executed']}, Win Rate: {summary['win_rate']:.2f}%")
            
            # Send EOD alert
            self._alert_eod_summary(summary)
            
        except Exception as e:
            self.logger.error(f"Error generating EOD summary: {e}")
            
    def _start_real_time_data(self):
        """
        Start real-time data collection.
        """
        try:
            # Get real-time data collector
            from data.market.real_time import RealTimeDataCollector
            data_collector = RealTimeDataCollector(self.db)
            
            # Start data collection
            data_collector.start()
            
            # Get active symbols
            symbols = []
            exchanges = []
            
            # Get all active instruments from portfolio
            if self.db:
                instruments = list(self.db.portfolio_collection.find({"status": "active"}))
                
                for instrument in instruments:
                    symbols.append(instrument.get('symbol'))
                    exchanges.append(instrument.get('exchange'))
                    
            # Add index symbols
            index_symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
            exchange = "NSE"
            
            for symbol in index_symbols:
                symbols.append(symbol)
                exchanges.append(exchange)
                
            # Subscribe to symbols
            if symbols and exchanges:
                data_collector.subscribe(symbols, exchanges)
                self.logger.info(f"Subscribed to {len(symbols)} symbols for real-time data")
                
        except Exception as e:
            self.logger.error(f"Error starting real-time data: {e}")
            
    def _stop_real_time_data(self):
        """
        Stop real-time data collection.
        """
        try:
            # Get real-time data collector
            from data.market.real_time import RealTimeDataCollector
            data_collector = RealTimeDataCollector(self.db)
            
            # Stop data collection
            data_collector.stop()
            
            self.logger.info("Real-time data collection stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping real-time data: {e}")
            
    def _adjust_risk_for_volatility(self, volatility_data):
        """
        Adjust risk parameters based on volatility.
        
        Args:
            volatility_data (dict): Volatility analysis data
        """
        try:
            if not self.db:
                return
                
            # Get volatility level
            volatility_level = volatility_data.get('volatility_level', 'normal')
            
            # Adjust position sizing
            from core.risk.position_sizing import adjust_position_sizing
            adjust_position_sizing(volatility_level, self.db)
            
            self.logger.info(f"Adjusted risk parameters for {volatility_level} volatility")
            
        except Exception as e:
            self.logger.error(f"Error adjusting risk: {e}")
            
    def _alert_drawdown(self):
        """
        Send alert for excessive drawdown.
        """
        try:
            # Import notification manager
            from communication.notification_manager import NotificationManager
            notification_manager = NotificationManager(self.db)
            
            # Build alert message
            message = f"⚠️ DRAWDOWN ALERT: {self.drawdown:.2f}% drawdown detected"
            
            # Add portfolio details
            if self.portfolio_value and self.max_portfolio_value:
                message += f"\nPortfolio value: ₹{self.portfolio_value:.2f} (from max ₹{self.max_portfolio_value:.2f})"
                
            # Add position count
            message += f"\nActive positions: {len(self.active_positions)}"
            
            # Send notification
            notification_manager.send_alert(message, level="warning")
            
        except Exception as e:
            self.logger.error(f"Error sending drawdown alert: {e}")
            
    def _alert_high_volatility(self, volatility_data):
        """
        Send alert for high volatility.
        
        Args:
            volatility_data (dict): Volatility analysis data
        """
        try:
            # Import notification manager
            from communication.notification_manager import NotificationManager
            notification_manager = NotificationManager(self.db)
            
            # Get volatility details
            volatility_level = volatility_data.get('volatility_level', 'high')
            current_value = volatility_data.get('current_value', 0)
            average_value = volatility_data.get('average_value', 0)
            
            # Build alert message
            message = f"⚠️ VOLATILITY ALERT: {volatility_level.upper()} market volatility detected"
            
            # Add volatility details
            message += f"\nCurrent volatility: {current_value:.2f}% (avg: {average_value:.2f}%)"
            
            # Add risk adjustment info
            message += f"\nRisk parameters have been adjusted for {volatility_level} volatility"
            
            # Send notification
            notification_manager.send_alert(message, level="warning")
            
        except Exception as e:
            self.logger.error(f"Error sending volatility alert: {e}")
            
    def _alert_opportunities(self, opportunities):
        """
        Send alert for detected trading opportunities.
        
        Args:
            opportunities (list): List of detected opportunities
        """
        try:
            # Import notification manager
            from communication.notification_manager import NotificationManager
            notification_manager = NotificationManager(self.db)
            
            # Build alert message
            message = f"🔍 OPPORTUNITY ALERT: {len(opportunities)} top opportunities detected"
            
            # Add opportunity details
            for i, opp in enumerate(opportunities, 1):
                symbol = opp.get('symbol', 'Unknown')
                direction = opp.get('direction', 'unknown').upper()
                confidence = opp.get('confidence', 0) * 100
                reason = opp.get('reason', 'No reason provided')
                
                message += f"\n{i}. {symbol} ({direction}) - {confidence:.1f}% confidence"
                message += f"\n   Reason: {reason}"
                
            # Send notification
            notification_manager.send_alert(message, level="info")
            
        except Exception as e:
            self.logger.error(f"Error sending opportunity alert: {e}")
            
    def _alert_profit_target(self, position, pnl_percent):
        """
        Send alert for position reaching profit target.
        
        Args:
            position (dict): Position details
            pnl_percent (float): PnL percentage
        """
        try:
            # Import notification manager
            from communication.notification_manager import NotificationManager
            notification_manager = NotificationManager(self.db)
            
            # Get position details
            symbol = position.get('symbol', 'Unknown')
            entry_price = position.get('entry_price', 0)
            current_price = position.get('current_price', 0)
            position_type = position.get('position_type', 'long').upper()
            
            # Build alert message
            message = f"🎯 PROFIT TARGET ALERT: {symbol} reached {pnl_percent:.2f}% profit"
            
            # Add position details
            message += f"\nPosition: {position_type}"
            message += f"\nEntry price: ₹{entry_price:.2f}"
            message += f"\nCurrent price: ₹{current_price:.2f}"
            
            # Add suggestion
            message += f"\nConsider taking partial profits or trailing stop"
            
            # Send notification
            notification_manager.send_alert(message, level="success")
            
        except Exception as e:
            self.logger.error(f"Error sending profit target alert: {e}")
            
    def _alert_eod_summary(self, summary):
        """
        Send end-of-day summary alert.
        
        Args:
            summary (dict): EOD summary data
        """
        try:
            # Import notification manager
            from communication.notification_manager import NotificationManager
            notification_manager = NotificationManager(self.db)
            
            # Build alert message
            message = f"📊 END OF DAY SUMMARY: {summary.get('date').strftime('%d-%b-%Y')}"
            
            # Add performance details
            message += f"\nDaily P&L: ₹{summary.get('daily_pnl', 0):.2f} ({summary.get('daily_pnl_percent', 0):.2f}%)"
            message += f"\nTrades: {summary.get('trades_executed', 0)} (Win rate: {summary.get('win_rate', 0):.1f}%)"
            message += f"\nMax drawdown: {summary.get('max_drawdown', 0):.2f}%"
            
            # Add open positions
            message += f"\nOpen positions: {summary.get('open_positions', 0)}"
            
            # Send notification
            notification_manager.send_alert(message, level="info")
            
        except Exception as e:
            self.logger.error(f"Error sending EOD summary alert: {e}")