"""
Command Processor for Slack Integration.
This module handles processing of commands received from Slack.
"""

import re
import logging
from datetime import datetime, timedelta

class CommandProcessor:
    def __init__(self, db_connector, slack_connector, trading_controller=None, portfolio_manager=None):
        """
        Initialize the command processor.
        
        Args:
            db_connector: Database connector
            slack_connector: SlackConnector instance
            trading_controller: TradingController instance
            portfolio_manager: PortfolioManager instance
        """
        self.db = db_connector
        self.slack = slack_connector
        self.trading_controller = trading_controller
        self.portfolio_manager = portfolio_manager
        self.logger = logging.getLogger(__name__)
        
        # Register available commands
        self.register_commands()
        
    def register_commands(self):
        """Register all available commands with the Slack connector"""
        commands = {
            '/report': self.handle_report_command,
            '/watchlist': self.handle_watchlist_command,
            '/performance': self.handle_performance_command,
            '/analyze': self.handle_analyze_command,
            '/add': self.handle_add_command,
            '/remove': self.handle_remove_command,
            '/start': self.handle_start_command,
            '/stop': self.handle_stop_command,
            '/alert': self.handle_alert_command,
            '/help': self.handle_help_command,
        }
        
        for command, handler in commands.items():
            self.slack.register_command(command, handler)
            
    def handle_report_command(self, args, user_id):
        """
        Handle /report command - Get prediction reports.
        
        Args:
            args (list): Command arguments
            user_id (str): User ID of the requester
            
        Returns:
            str: Response message
        """
        if not args:
            return "Please specify a report type: `/report [daily|weekly|monthly]`"
            
        report_type = args[0].lower()
        valid_types = ['daily', 'weekly', 'monthly']
        
        if report_type not in valid_types:
            return f"Invalid report type. Please use one of: {', '.join(valid_types)}"
            
        try:
            # Import report generators
            if report_type == 'daily':
                from reports.daily_prediction import DailyPredictionReport
                report_generator = DailyPredictionReport(self.db)
            elif report_type == 'weekly':
                from reports.templates.weekly_report import WeeklyReport
                report_generator = WeeklyReport(self.db)
            elif report_type == 'monthly':
                from reports.templates.monthly_report import MonthlyReport
                report_generator = MonthlyReport(self.db)
                
            # Generate report
            from reports.formatters.slack_formatter import SlackFormatter
            formatter = SlackFormatter()
            report = report_generator.generate_report()
            formatted_report = formatter.format_report(report)
            
            # Send report
            self.slack.send_message(
                text=f"{report_type.capitalize()} Report",
                blocks=formatted_report['blocks'],
                attachments=formatted_report.get('attachments')
            )
            
            return None  # No additional message needed
        except Exception as e:
            self.logger.error(f"Error generating {report_type} report: {e}")
            return f"Error generating {report_type} report: {str(e)}"
            
    def handle_watchlist_command(self, args, user_id):
        """
        Handle /watchlist command - View current watchlist.
        
        Args:
            args (list): Command arguments
            user_id (str): User ID of the requester
            
        Returns:
            str: Response message
        """
        try:
            # Get active instruments from portfolio
            instruments = list(self.db.portfolio_collection.find({"status": "active"}))
            
            if not instruments:
                return "No instruments in watchlist."
                
            # Format response
            response = "*Current Watchlist:*\n\n"
            categories = {}
            
            # Group by instrument type
            for instr in instruments:
                instr_type = instr.get("instrument_type", "equity")
                if instr_type not in categories:
                    categories[instr_type] = []
                categories[instr_type].append(instr)
                
            # Format each category
            for category, items in categories.items():
                response += f"*{category.upper()}*\n"
                for item in items:
                    symbol = item["symbol"]
                    exchange = item["exchange"]
                    enabled = "✅" if item.get("trading_config", {}).get("enabled", False) else "❌"
                    response += f"• {symbol} ({exchange}) {enabled}\n"
                response += "\n"
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error retrieving watchlist: {e}")
            return f"Error retrieving watchlist: {str(e)}"
            
    def handle_performance_command(self, args, user_id):
        """
        Handle /performance command - Check trading performance.
        
        Args:
            args (list): Command arguments
            user_id (str): User ID of the requester
            
        Returns:
            str: Response message
        """
        try:
            # Determine time period
            period = 'daily'
            if args:
                period = args[0].lower()
                valid_periods = ['daily', 'weekly', 'monthly', 'all']
                if period not in valid_periods:
                    return f"Invalid period. Please use one of: {', '.join(valid_periods)}"
            
            # Get trades from database
            query = {}
            if period != 'all':
                # Calculate date range
                today = datetime.now()
                if period == 'daily':
                    start_date = today - timedelta(days=1)
                elif period == 'weekly':
                    start_date = today - timedelta(days=7)
                elif period == 'monthly':
                    start_date = today - timedelta(days=30)
                
                query["exit_time"] = {"$gte": start_date}
                
            trades = list(self.db.trade_collection.find(query))
            
            if not trades:
                return f"No completed trades found for the {period} period."
                
            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.get("profit_loss", 0) > 0)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            total_profit = sum(t.get("profit_loss", 0) for t in trades)
            avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
            
            max_profit = max((t.get("profit_loss", 0) for t in trades), default=0)
            max_loss = min((t.get("profit_loss", 0) for t in trades), default=0)
            
            # Calculate drawdown
            equity_curve = []
            running_total = 0
            max_equity = 0
            max_drawdown = 0
            
            for trade in sorted(trades, key=lambda x: x.get("exit_time", datetime.min)):
                running_total += trade.get("profit_loss", 0)
                equity_curve.append(running_total)
                
                max_equity = max(max_equity, running_total)
                drawdown = max_equity - running_total
                max_drawdown = max(max_drawdown, drawdown)
            
            # Format response
            response = f"*Performance Summary ({period.capitalize()})*\n\n"
            response += f"*Total Trades:* {total_trades}\n"
            response += f"*Win Rate:* {win_rate:.2f}%\n"
            response += f"*Total Profit/Loss:* {total_profit:.2f}\n"
            response += f"*Average Profit/Trade:* {avg_profit_per_trade:.2f}\n"
            response += f"*Max Profit:* {max_profit:.2f}\n"
            response += f"*Max Loss:* {max_loss:.2f}\n"
            response += f"*Max Drawdown:* {max_drawdown:.2f}\n"
            
            # Add more details like strategy performance if requested
            if args and len(args) > 1 and args[1].lower() == 'detailed':
                # Group by strategy
                strategy_performance = {}
                for trade in trades:
                    strategy = trade.get("strategy", "unknown")
                    if strategy not in strategy_performance:
                        strategy_performance[strategy] = {
                            "trades": 0,
                            "wins": 0,
                            "losses": 0,
                            "profit": 0
                        }
                    
                    perf = strategy_performance[strategy]
                    perf["trades"] += 1
                    if trade.get("profit_loss", 0) > 0:
                        perf["wins"] += 1
                    else:
                        perf["losses"] += 1
                    perf["profit"] += trade.get("profit_loss", 0)
                
                response += "\n*Strategy Performance:*\n"
                for strategy, perf in strategy_performance.items():
                    win_rate = (perf["wins"] / perf["trades"]) * 100 if perf["trades"] > 0 else 0
                    response += f"• *{strategy}*: {perf['trades']} trades, {win_rate:.2f}% win rate, {perf['profit']:.2f} profit\n"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error retrieving performance: {e}")
            return f"Error retrieving performance: {str(e)}"
            
    def handle_analyze_command(self, args, user_id):
        """
        Handle /analyze command - Get detailed analysis for a stock.
        
        Args:
            args (list): Command arguments
            user_id (str): User ID of the requester
            
        Returns:
            str: Response message
        """
        if not args:
            return "Please specify a symbol to analyze: `/analyze SYMBOL`"
            
        symbol = args[0].upper()
        exchange = args[1].upper() if len(args) > 1 else "NSE"  # Default to NSE
        
        try:
            # Check if symbol exists in portfolio
            instrument = self.db.portfolio_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "status": "active"
            })
            
            if not instrument:
                return f"Symbol {symbol} not found in portfolio. Add it first with `/add {symbol} {exchange}`"
                
            # Import analysis modules
            from research.technical_analyzer import TechnicalAnalyzer
            from research.fundamental_analyzer import FundamentalAnalyzer
            from research.opportunity_scanner import OpportunityScanner
            
            # Run analysis
            tech_analyzer = TechnicalAnalyzer(self.db)
            fund_analyzer = FundamentalAnalyzer(self.db)
            opportunity_scanner = OpportunityScanner(self.db)
            
            tech_analysis = tech_analyzer.analyze(symbol, exchange)
            fund_analysis = fund_analyzer.analyze(symbol, exchange)
            opportunities = opportunity_scanner.scan_single(symbol, exchange)
            
            # Format response
            response = f"*Analysis for {symbol} ({exchange})*\n\n"
            
            # Technical Analysis
            response += "*Technical Analysis:*\n"
            if tech_analysis:
                trend = tech_analysis.get("trend", "Neutral")
                strength = tech_analysis.get("strength", "Medium")
                support = tech_analysis.get("support", "N/A")
                resistance = tech_analysis.get("resistance", "N/A")
                
                response += f"• *Trend:* {trend}\n"
                response += f"• *Strength:* {strength}\n"
                response += f"• *Support:* {support}\n"
                response += f"• *Resistance:* {resistance}\n"
                
                # Add key indicators
                indicators = tech_analysis.get("indicators", {})
                if indicators:
                    response += "• *Key Indicators:*\n"
                    for name, value in indicators.items():
                        response += f"  - {name}: {value}\n"
            else:
                response += "Technical analysis not available\n"
                
            # Fundamental Analysis
            response += "\n*Fundamental Analysis:*\n"
            if fund_analysis:
                score = fund_analysis.get("fundamental_score", "N/A")
                pe = fund_analysis.get("details", {}).get("ratios", {}).get("pe", "N/A")
                growth = fund_analysis.get("details", {}).get("growth", {}).get("eps_growth", "N/A")
                
                response += f"• *Fundamental Score:* {score}\n"
                response += f"• *P/E Ratio:* {pe}\n"
                response += f"• *EPS Growth:* {growth}\n"
                
                # Add key metrics
                metrics = fund_analysis.get("details", {}).get("ratios", {})
                if metrics:
                    response += "• *Key Metrics:*\n"
                    for name, value in list(metrics.items())[:5]:  # Show top 5 metrics
                        response += f"  - {name}: {value}\n"
            else:
                response += "Fundamental analysis not available\n"
                
            # Trading Opportunities
            response += "\n*Trading Opportunities:*\n"
            if opportunities:
                for opp in opportunities:
                    opp_type = opp.get("type", "Unknown")
                    confidence = opp.get("confidence", 0) * 100
                    
                    response += f"• *{opp_type}* (Confidence: {confidence:.1f}%)\n"
                    response += f"  - Entry: {opp.get('entry', 'N/A')}\n"
                    response += f"  - Target: {opp.get('target', 'N/A')}\n"
                    response += f"  - Stop: {opp.get('stop_loss', 'N/A')}\n"
                    response += f"  - R/R Ratio: {opp.get('risk_reward', 'N/A')}\n"
            else:
                response += "No trading opportunities found\n"
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return f"Error analyzing {symbol}: {str(e)}"
            
    def handle_add_command(self, args, user_id):
        """
        Handle /add command - Add instrument to portfolio.
        
        Args:
            args (list): Command arguments
            user_id (str): User ID of the requester
            
        Returns:
            str: Response message
        """
        if not args:
            return "Please specify a symbol to add: `/add SYMBOL [EXCHANGE] [TYPE]`"
            
        symbol = args[0].upper()
        exchange = args[1].upper() if len(args) > 1 else "NSE"  # Default to NSE
        instr_type = args[2].lower() if len(args) > 2 else "equity"  # Default to equity
        
        try:
            # Check if portfolio manager is available
            if not self.portfolio_manager:
                return "Portfolio manager not available. Please initialize it first."
                
            # Check if instrument already exists
            existing = self.db.portfolio_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "status": "active"
            })
            
            if existing:
                return f"{symbol} already in portfolio."
                
            # Add instrument
            instrument_id = self.portfolio_manager.add_instrument(
                symbol=symbol,
                exchange=exchange,
                instrument_type=instr_type
            )
            
            return f"Added {symbol} ({exchange}) to portfolio. Data collection started."
            
        except Exception as e:
            self.logger.error(f"Error adding {symbol}: {e}")
            return f"Error adding {symbol}: {str(e)}"
            
    def handle_remove_command(self, args, user_id):
        """
        Handle /remove command - Remove instrument from portfolio.
        
        Args:
            args (list): Command arguments
            user_id (str): User ID of the requester
            
        Returns:
            str: Response message
        """
        if not args:
            return "Please specify a symbol to remove: `/remove SYMBOL [EXCHANGE]`"
            
        symbol = args[0].upper()
        exchange = args[1].upper() if len(args) > 1 else "NSE"  # Default to NSE
        
        try:
            # Check if portfolio manager is available
            if not self.portfolio_manager:
                return "Portfolio manager not available. Please initialize it first."
                
            # Check if instrument exists
            existing = self.db.portfolio_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "status": "active"
            })
            
            if not existing:
                return f"{symbol} not found in portfolio."
                
            # Remove instrument
            result = self.portfolio_manager.remove_instrument(
                symbol=symbol,
                exchange=exchange
            )
            
            return f"Removed {symbol} ({exchange}) from portfolio."
            
        except Exception as e:
            self.logger.error(f"Error removing {symbol}: {e}")
            return f"Error removing {symbol}: {str(e)}"
            
    def handle_start_command(self, args, user_id):
        """
        Handle /start command - Start trading.
        
        Args:
            args (list): Command arguments
            user_id (str): User ID of the requester
            
        Returns:
            str: Response message
        """
        try:
            # Check if trading controller is available
            if not self.trading_controller:
                return "Trading controller not available. Please initialize it first."
                
            # Start trading
            result = self.trading_controller.start_trading()
            
            if result:
                return "Trading started successfully."
            else:
                return "Failed to start trading. Check logs for details."
                
        except Exception as e:
            self.logger.error(f"Error starting trading: {e}")
            return f"Error starting trading: {str(e)}"
            
    def handle_stop_command(self, args, user_id):
        """
        Handle /stop command - Stop trading.
        
        Args:
            args (list): Command arguments
            user_id (str): User ID of the requester
            
        Returns:
            str: Response message
        """
        try:
            # Check if trading controller is available
            if not self.trading_controller:
                return "Trading controller not available. Please initialize it first."
                
            # Check if we should close positions
            close_positions = False
            if args and args[0] == "--close-positions":
                close_positions = True
                
            # Stop trading
            result = self.trading_controller.stop_trading(close_positions=close_positions)
            
            if result:
                msg = "Trading stopped successfully."
                if close_positions:
                    msg += " All positions closed."
                return msg
            else:
                return "Failed to stop trading. Check logs for details."
                
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")
            return f"Error stopping trading: {str(e)}"
            
    def handle_alert_command(self, args, user_id):
        """
        Handle /alert command - Set price alert.
        
        Args:
            args (list): Command arguments
            user_id (str): User ID of the requester
            
        Returns:
            str: Response message
        """
        if len(args) < 2:
            return "Please specify a symbol and price: `/alert SYMBOL PRICE`"
            
        symbol = args[0].upper()
        
        try:
            price = float(args[1])
        except ValueError:
            return f"Invalid price: {args[1]}. Please provide a numeric value."
            
        exchange = args[2].upper() if len(args) > 2 else "NSE"  # Default to NSE
        
        try:
            # Check if symbol exists in portfolio
            instrument = self.db.portfolio_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "status": "active"
            })
            
            if not instrument:
                return f"Symbol {symbol} not found in portfolio. Add it first with `/add {symbol} {exchange}`"
                
            # Add alert to database
            alert_id = self.db.alerts_collection.insert_one({
                "symbol": symbol,
                "exchange": exchange,
                "price": price,
                "user_id": user_id,
                "created_at": datetime.now(),
                "status": "active"
            }).inserted_id
            
            return f"Price alert set for {symbol} at {price}."
            
        except Exception as e:
            self.logger.error(f"Error setting alert for {symbol}: {e}")
            return f"Error setting alert for {symbol}: {str(e)}"
            
    def handle_help_command(self, args, user_id):
        """
        Handle /help command - Display available commands.
        
        Args:
            args (list): Command arguments
            user_id (str): User ID of the requester
            
        Returns:
            str: Response message
        """
        help_text = """*Available Commands:*

- `/report [daily|weekly|monthly]` - Get prediction reports
- `/watchlist` - View current watchlist
- `/performance [daily|weekly|monthly|all]` - Check trading performance
- `/analyze SYMBOL [EXCHANGE]` - Get detailed analysis for a stock
- `/add SYMBOL [EXCHANGE] [TYPE]` - Add instrument to portfolio
- `/remove SYMBOL [EXCHANGE]` - Remove instrument from portfolio
- `/start` - Start trading
- `/stop [--close-positions]` - Stop trading
- `/alert SYMBOL PRICE [EXCHANGE]` - Set price alert
- `/help` - Display this help message

*Examples:*
- `/report daily` - Get daily prediction report
- `/analyze TATASTEEL NSE` - Analyze TATASTEEL stock
- `/add RELIANCE NSE equity` - Add RELIANCE stock to portfolio
- `/performance weekly detailed` - Get detailed weekly performance
"""
        return help_text