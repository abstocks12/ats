# automation/daily_workflow.py
import logging
import time
from datetime import datetime, timedelta
import os
import sys

class DailyWorkflow:
    """
    Implements the daily automated workflow for the trading system.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the daily workflow.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'market_open_time': '09:15',  # Market opening time (IST)
            'market_close_time': '15:30',  # Market closing time (IST)
            'data_collection_time': '04:00',  # Start data collection time
            'analysis_time': '05:00',  # Start analysis time
            'prediction_time': '06:00',  # Start prediction time
            'morning_report_time': '07:00',  # Generate morning report time
            'pre_market_time': '08:45',  # Pre-market preparation time
            'post_market_time': '15:45',  # Post-market analysis time
            'eod_report_time': '16:30',  # End of day report time
            'system_maintenance_time': '20:00',  # System maintenance time
        }
        
        self.logger.info("Daily workflow initialized")
    
    def register_tasks(self, scheduler):
        """
        Register all daily tasks with the scheduler.
        
        Args:
            scheduler: Scheduler instance
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info("Registering daily tasks")
            
            # Global market data collection (4:00 AM)
            scheduler.schedule_daily(
                func=self.collect_global_data,
                time_str=self.config['data_collection_time'],
                name="Global Market Data Collection"
            )
            
            # Data processing and analysis (5:00 AM)
            scheduler.schedule_daily(
                func=self.process_and_analyze_data,
                time_str=self.config['analysis_time'],
                name="Data Processing and Analysis"
            )
            
            # Model predictions and morning report (6:00 AM)
            scheduler.schedule_daily(
                func=self.generate_predictions,
                time_str=self.config['prediction_time'],
                name="Model Predictions"
            )
            
            # Morning report delivery (7:00 AM)
            scheduler.schedule_daily(
                func=self.generate_morning_report,
                time_str=self.config['morning_report_time'],
                name="Morning Report Generation"
            )
            
            # Pre-market preparation (8:45 AM)
            scheduler.schedule_daily(
                func=self.pre_market_preparation,
                time_str=self.config['pre_market_time'],
                name="Pre-Market Preparation"
            )
            
            # Intraday monitoring (starts at market open)
            scheduler.schedule_daily(
                func=self.start_intraday_monitor,
                time_str=self.config['market_open_time'],
                name="Start Intraday Monitoring"
            )
            
            # Post-market analysis (15:45 PM)
            scheduler.schedule_daily(
                func=self.post_market_analysis,
                time_str=self.config['post_market_time'],
                name="Post-Market Analysis"
            )
            
            # End of day report (16:30 PM)
            scheduler.schedule_daily(
                func=self.generate_eod_report,
                time_str=self.config['eod_report_time'],
                name="End of Day Report"
            )
            
            # System maintenance (20:00 PM)
            scheduler.schedule_daily(
                func=self.system_maintenance,
                time_str=self.config['system_maintenance_time'],
                name="System Maintenance"
            )
            
            self.logger.info("Daily tasks registered successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering daily tasks: {e}")
            return False
    
    def collect_global_data(self):
        """
        Collect global market data (4:00 AM).
        """
        try:
            self.logger.info("Starting global market data collection")
            
            # Import required modules
            from data.global_markets.indices_collector import IndicesCollector
            from data.global_markets.forex_collector import ForexCollector
            from data.global_markets.economic_calendar import EconomicCalendar
            
            # Initialize collectors
            indices_collector = IndicesCollector(self.db)
            forex_collector = ForexCollector(self.db)
            economic_calendar = EconomicCalendar(self.db)
            
            # Collect global index data
            indices_collector.collect_data()
            
            # Collect forex data
            forex_collector.collect_data()
            
            # Update economic calendar
            economic_calendar.update_calendar()
            
            # Collect alternative data
            from data.alternative.social_sentiment import SocialSentimentCollector
            from data.alternative.google_trends import GoogleTrendsCollector
            
            # Initialize alternative data collectors
            sentiment_collector = SocialSentimentCollector(self.db)
            trends_collector = GoogleTrendsCollector(self.db)
            
            # Collect social sentiment data
            sentiment_collector.collect_data()
            
            # Collect Google Trends data
            trends_collector.collect_data()
            
            self.logger.info("Global market data collection completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting global market data: {e}")
            return False
    
    def process_and_analyze_data(self):
        """
        Process and analyze collected data (5:00 AM).
        """
        try:
            self.logger.info("Starting data processing and analysis")
            
            # Import required modules
            from data.market.historical_data import HistoricalDataCollector
            from research.technical_analyzer import TechnicalAnalyzer
            from research.fundamental_analyzer import FundamentalAnalyzer
            from research.market_analysis import MarketAnalyzer
            from research.volatility_analyzer import VolatilityAnalyzer
            from research.correlation_analyzer import CorrelationAnalyzer
            
            # Get active instruments
            instruments = list(self.db.portfolio_collection.find({"status": "active"}))
            
            self.logger.info(f"Processing data for {len(instruments)} instruments")
            
            # Process each instrument
            for instrument in instruments:
                symbol = instrument.get('symbol')
                exchange = instrument.get('exchange')
                
                # Update historical data
                historical_collector = HistoricalDataCollector(self.db)
                historical_collector.update_data(symbol, exchange)
                
                # Run technical analysis
                tech_analyzer = TechnicalAnalyzer(self.db)
                tech_analyzer.analyze(symbol, exchange)
                
                # Run fundamental analysis if data is available
                if instrument.get('data_collection_status', {}).get('financial', False):
                    fund_analyzer = FundamentalAnalyzer(self.db)
                    fund_analyzer.analyze(symbol, exchange)
            
            # Run market analysis
            market_analyzer = MarketAnalyzer(self.db)
            market_analyzer.analyze_market_regime()
            
            # Run volatility analysis
            volatility_analyzer = VolatilityAnalyzer(self.db)
            volatility_analyzer.analyze_all()
            
            # Run correlation analysis
            correlation_analyzer = CorrelationAnalyzer(self.db)
            correlation_analyzer.analyze_all()
            
            self.logger.info("Data processing and analysis completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing and analyzing data: {e}")
            return False
    
    def generate_predictions(self):
        """
        Generate model predictions (6:00 AM).
        """
        try:
            self.logger.info("Starting model predictions")
            
            # Import required modules
            from ml.prediction.daily_predictor import DailyPredictor
            from ml.prediction.overnight_gap_predictor import OvernightGapPredictor
            from ml.prediction.sector_rotation_analyzer import SectorRotationAnalyzer
            
            # Initialize predictors
            daily_predictor = DailyPredictor(self.db)
            gap_predictor = OvernightGapPredictor(self.db)
            sector_analyzer = SectorRotationAnalyzer(self.db)
            
            # Generate daily predictions for all active instruments
            daily_predictor.predict_all()
            
            # Generate overnight gap predictions
            gap_predictor.predict_all()
            
            # Analyze sector rotation
            sector_analyzer.analyze()
            
            self.logger.info("Model predictions completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return False
    
    def generate_morning_report(self):
        """
        Generate morning report (7:00 AM).
        """
        try:
            self.logger.info("Generating morning report")
            
            # Import required modules
            from reports.daily_prediction import DailyPredictionReport
            from reports.report_formatter import ReportFormatter
            from communication.report_distributor import ReportDistributor
            
            # Initialize report generators
            prediction_report = DailyPredictionReport(self.db)
            report_formatter = ReportFormatter()
            report_distributor = ReportDistributor(self.db)
            
            # Generate daily prediction report
            report_data = prediction_report.generate_report()
            
            # Format report for WhatsApp
            from reports.formatters.whatsapp_formatter import WhatsAppFormatter
            whatsapp_formatter = WhatsAppFormatter()
            whatsapp_report = whatsapp_formatter.format(report_data)
            
            # Distribute report via WhatsApp
            report_distributor.distribute_via_whatsapp(whatsapp_report)
            
            self.logger.info("Morning report generated and distributed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating morning report: {e}")
            return False
    
    def pre_market_preparation(self):
        """
        Pre-market preparation (8:45 AM).
        """
        try:
            self.logger.info("Starting pre-market preparation")
            
            # Import required modules
            from trading.trading_controller import TradingController
            from trading.position_manager import PositionManager
            
            # Initialize trading controller
            trading_controller = TradingController(self.db)
            
            # Check market status
            from trading.market_hours import MarketHours
            market_hours = MarketHours()
            
            if not market_hours.is_trading_day():
                self.logger.info("Not a trading day, skipping pre-market preparation")
                return True
            
            # Prepare strategies
            trading_controller.prepare_strategies()
            
            # Check for any overnight positions
            position_manager = PositionManager(self.db)
            overnight_positions = position_manager.get_overnight_positions()
            
            if overnight_positions:
                self.logger.info(f"Found {len(overnight_positions)} overnight positions")
                
                # Adjust stop-loss levels for overnight positions
                position_manager.adjust_overnight_positions()
            
            # Prepare order batches for market open
            from realtime.execution.batch_processor import BatchProcessor
            batch_processor = BatchProcessor(self.db)
            batch_processor.prepare_market_open_orders()
            
            self.logger.info("Pre-market preparation completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in pre-market preparation: {e}")
            return False
    
    def start_intraday_monitor(self):
        """
        Start intraday monitoring (9:15 AM).
        """
        try:
            self.logger.info("Starting intraday monitoring")
            
            # Import required modules
            from automation.intraday_monitor import IntradayMonitor
            
            # Start intraday monitor
            intraday_monitor = IntradayMonitor(self.db)
            intraday_monitor.start()
            
            # Start trading system if configured
            from trading.trading_controller import TradingController
            
            # Read trading configuration
            trading_enabled = self._get_config_value('trading_enabled', False)
            trading_mode = self._get_config_value('trading_mode', 'paper')
            
            if trading_enabled:
                # Start trading controller
                trading_controller = TradingController(self.db, mode=trading_mode)
                trading_controller.start_trading()
                
                self.logger.info(f"Trading system started in {trading_mode} mode")
            else:
                self.logger.info("Trading system not enabled, running in monitoring mode only")
            
            self.logger.info("Intraday monitoring started")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting intraday monitoring: {e}")
            return False
    
    def post_market_analysis(self):
        """
        Post-market analysis (15:45 PM).
        """
        try:
            self.logger.info("Starting post-market analysis")
            
            # Stop trading system
            from trading.trading_controller import TradingController
            trading_controller = TradingController(self.db)
            trading_controller.stop_trading()
            
            # Import required modules
            from automation.eod_analyzer import EODAnalyzer
            
            # Run end-of-day analysis
            eod_analyzer = EODAnalyzer(self.db)
            eod_analyzer.analyze()
            
            # Validate daily predictions
            from automation.prediction_validator import PredictionValidator
            validator = PredictionValidator(self.db)
            validator.validate_daily_predictions()
            
            self.logger.info("Post-market analysis completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in post-market analysis: {e}")
            return False
    
    def generate_eod_report(self):
        """
        Generate end-of-day report (16:30 PM).
        """
        try:
            self.logger.info("Generating end-of-day report")
            
            # Import required modules
            from reports.templates.eod_report import EODReport
            from reports.report_formatter import ReportFormatter
            from communication.report_distributor import ReportDistributor
            
            # Initialize report generators
            eod_report = EODReport(self.db)
            report_formatter = ReportFormatter()
            report_distributor = ReportDistributor(self.db)
            
            # Generate EOD report
            report_data = eod_report.generate_report()
            
            # Format report for WhatsApp
            from reports.formatters.whatsapp_formatter import WhatsAppFormatter
            whatsapp_formatter = WhatsAppFormatter()
            whatsapp_report = whatsapp_formatter.format(report_data)
            
            # Format report for PDF
            from reports.formatters.pdf_formatter import PDFFormatter
            pdf_formatter = PDFFormatter()
            pdf_report = pdf_formatter.format(report_data)
            pdf_path = pdf_formatter.save(pdf_report, f"eod_report_{datetime.now().strftime('%Y%m%d')}.pdf")
            
            # Distribute reports
            report_distributor.distribute_via_whatsapp(whatsapp_report)
            report_distributor.distribute_via_email(pdf_path, "End of Day Trading Report")
            
            self.logger.info("End-of-day report generated and distributed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating end-of-day report: {e}")
            return False
    
    def system_maintenance(self):
        """
        System maintenance (20:00 PM).
        """
        try:
            self.logger.info("Starting system maintenance")
            
            # Database optimization
            self._optimize_database()
            
            # Backup database
            self._backup_database()
            
            # Clean up old data
            self._cleanup_old_data()
            
            # Check for system updates
            self._check_for_updates()
            
            # Schedule next day's model retraining
            self._schedule_model_retraining()
            
            self.logger.info("System maintenance completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in system maintenance: {e}")
            return False
    
    def _optimize_database(self):
        """
        Optimize database performance.
        """
        try:
            self.logger.info("Optimizing database")
            
            if not self.db:
                self.logger.warning("No database connection, skipping optimization")
                return
                
            # Run MongoDB maintenance commands
            # Note: This requires admin privileges
            try:
                # Get database stats
                stats = self.db.command("dbStats")
                self.logger.info(f"Database size: {stats.get('dataSize') / (1024 * 1024):.2f} MB")
                
                # Run compact on collections with significant data
                collections = ["market_data", "news_data", "financial_data", "predictions", "trades"]
                
                for collection in collections:
                    try:
                        self.db.command("compact", collection)
                        self.logger.info(f"Compacted collection: {collection}")
                    except Exception as e:
                        self.logger.warning(f"Could not compact collection {collection}: {e}")
                        
            except Exception as e:
                self.logger.warning(f"Could not run database maintenance: {e}")
                
            # Refresh indexes
            collections = self.db.list_collection_names()
            
            for collection in collections:
                indexes = list(self.db[collection].list_indexes())
                self.logger.info(f"Collection {collection} has {len(indexes)} indexes")
                
            self.logger.info("Database optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error optimizing database: {e}")
    
    def _backup_database(self):
        """
        Backup database.
        """
        try:
            self.logger.info("Backing up database")
            
            # Get backup directory from config
            backup_dir = self._get_config_value("backup_directory", "backups")
            
            # Create backup directory if it doesn't exist
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
                
            # Generate backup filename
            backup_filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.archive"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Execute mongodump command
            import subprocess
            
            # Get database connection info
            db_host = self._get_config_value("mongodb_host", "localhost")
            db_port = self._get_config_value("mongodb_port", "27017")
            db_name = self._get_config_value("mongodb_db", "trading")
            
            # Build mongodump command
            cmd = [
                "mongodump",
                "--host", db_host,
                "--port", str(db_port),
                "--db", db_name,
                "--archive=" + backup_path,
                "--gzip"
            ]
            
            # Add authentication if configured
            db_user = self._get_config_value("mongodb_user", "")
            db_pass = self._get_config_value("mongodb_pass", "")
            
            if db_user and db_pass:
                cmd.extend(["--username", db_user, "--password", db_pass])
                
            # Run command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Database backed up to {backup_path}")
                
                # Remove old backups
                max_backups = self._get_config_value("max_backups", 7)
                self._cleanup_old_backups(backup_dir, max_backups)
            else:
                self.logger.error(f"Database backup failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error backing up database: {e}")
    
    def _cleanup_old_backups(self, backup_dir, max_backups):
        """
        Remove old backups exceeding the maximum number.
        
        Args:
            backup_dir (str): Backup directory
            max_backups (int): Maximum number of backups to keep
        """
        try:
            # Get list of backup files
            backup_files = [f for f in os.listdir(backup_dir) if f.startswith("backup_") and f.endswith(".archive")]
            
            # Sort by creation time (oldest first)
            backup_files.sort(key=lambda f: os.path.getctime(os.path.join(backup_dir, f)))
            
            # Remove old backups
            while len(backup_files) > max_backups:
                old_backup = backup_files.pop(0)
                old_path = os.path.join(backup_dir, old_backup)
                
                try:
                    os.remove(old_path)
                    self.logger.info(f"Removed old backup: {old_backup}")
                except Exception as e:
                    self.logger.error(f"Could not remove old backup {old_backup}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {e}")
    
    def _cleanup_old_data(self):
        """
        Clean up old data from the database.
        """
        try:
            self.logger.info("Cleaning up old data")
            
            if not self.db:
                self.logger.warning("No database connection, skipping cleanup")
                return
                
            # Get retention periods from config
            market_data_days = self._get_config_value("market_data_retention_days", 365)
            news_data_days = self._get_config_value("news_data_retention_days", 90)
            predictions_days = self._get_config_value("predictions_retention_days", 30)
            
            # Calculate cutoff dates
            market_cutoff = datetime.now() - timedelta(days=market_data_days)
            news_cutoff = datetime.now() - timedelta(days=news_data_days)
            predictions_cutoff = datetime.now() - timedelta(days=predictions_days)
            
            # Clean up market data
            market_result = self.db.market_data.delete_many({
                "timestamp": {"$lt": market_cutoff}
            })
            
            self.logger.info(f"Deleted {market_result.deleted_count} old market data records")
            
            # Clean up news data
            news_result = self.db.news_data.delete_many({
                "published_date": {"$lt": news_cutoff}
            })
            
            self.logger.info(f"Deleted {news_result.deleted_count} old news data records")
            
            # Clean up predictions
            predictions_result = self.db.predictions.delete_many({
                "date": {"$lt": predictions_cutoff}
            })
            
            self.logger.info(f"Deleted {predictions_result.deleted_count} old predictions")
            
            # Clean up completed tasks
            one_week_ago = datetime.now() - timedelta(days=7)
            tasks_result = self.db.tasks_collection.delete_many({
                "status": "completed",
                "end_time": {"$lt": one_week_ago}
            })
            
            self.logger.info(f"Deleted {tasks_result.deleted_count} old completed tasks")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def _check_for_updates(self):
        """
        Check for system updates.
        """
        try:
            self.logger.info("Checking for system updates")
            
            # This would typically check a repository or update server
            # For now, we'll just log the current version
            version = self._get_config_value("system_version", "1.0.0")
            self.logger.info(f"Current system version: {version}")
            
            # In a real implementation, this would check for updates and apply them
            
        except Exception as e:
            self.logger.error(f"Error checking for updates: {e}")
    
    def _schedule_model_retraining(self):
        """
        Schedule model retraining for off-hours.
        """
        try:
            self.logger.info("Scheduling model retraining")
            
            # Import scheduler module
            from automation.scheduler import Scheduler
            from automation.model_retraining import ModelRetraining
            
            # Get scheduler instance
            scheduler = Scheduler(self.db)
            
            # Get retraining time from config (default 01:00 AM)
            retraining_time = self._get_config_value("model_retraining_time", "01:00")
            
            # Create model retraining instance
            model_retraining = ModelRetraining(self.db)
            
            # Schedule retraining
            scheduler.schedule_daily(
                func=model_retraining.retrain_all_models,
                time_str=retraining_time,
                name="Daily Model Retraining"
            )
            
            self.logger.info(f"Model retraining scheduled for {retraining_time}")
            
        except Exception as e:
            self.logger.error(f"Error scheduling model retraining: {e}")
    
    def _get_config_value(self, key, default=None):
        """
        Get configuration value from database or default.
        
        Args:
            key (str): Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        try:
            if not self.db:
                return default
                
            # Check if config collection exists
            if "config" not in self.db.list_collection_names():
                return default
                
            # Get config value
            config = self.db.config.find_one({"key": key})
            
            if config:
                return config.get("value", default)
                
            return default
            
        except Exception as e:
            self.logger.error(f"Error getting config value {key}: {e}")
            return default