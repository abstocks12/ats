# automation/weekly_workflow.py
import logging
from datetime import datetime, timedelta
import os

class WeeklyWorkflow:
    """
    Implements the weekly automated workflow for the trading system.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the weekly workflow.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'weekend_analysis_day': 5,  # 0=Monday, 5=Saturday
            'weekend_analysis_time': '10:00',  # Weekend analysis time
            'portfolio_review_day': 6,  # Sunday
            'portfolio_review_time': '12:00',  # Portfolio review time
            'strategy_optimization_day': 6,  # Sunday
            'strategy_optimization_time': '14:00',  # Strategy optimization time
            'weekly_report_day': 6,  # Sunday
            'weekly_report_time': '18:00',  # Weekly report time
            'database_maintenance_day': 6,  # Sunday
            'database_maintenance_time': '22:00',  # Database maintenance time
        }
        
        self.logger.info("Weekly workflow initialized")
    
    def register_tasks(self, scheduler):
        """
        Register all weekly tasks with the scheduler.
        
        Args:
            scheduler: Scheduler instance
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info("Registering weekly tasks")
            
            # Weekend market analysis (Saturday 10:00 AM)
            scheduler.schedule_weekly(
                func=self.weekend_market_analysis,
                day_of_week=self.config['weekend_analysis_day'],
                time_str=self.config['weekend_analysis_time'],
                name="Weekend Market Analysis"
            )
            
            # Portfolio review (Sunday 12:00 PM)
            scheduler.schedule_weekly(
                func=self.portfolio_review,
                day_of_week=self.config['portfolio_review_day'],
                time_str=self.config['portfolio_review_time'],
                name="Weekly Portfolio Review"
            )
            
            # Strategy optimization (Sunday 14:00 PM)
            scheduler.schedule_weekly(
                func=self.strategy_optimization,
                day_of_week=self.config['strategy_optimization_day'],
                time_str=self.config['strategy_optimization_time'],
                name="Weekly Strategy Optimization"
            )
            
            # Weekly report (Sunday 18:00 PM)
            scheduler.schedule_weekly(
                func=self.generate_weekly_report,
                day_of_week=self.config['weekly_report_day'],
                time_str=self.config['weekly_report_time'],
                name="Weekly Report Generation"
            )
            
            # Database maintenance (Sunday 22:00 PM)
            scheduler.schedule_weekly(
                func=self.database_maintenance,
                day_of_week=self.config['database_maintenance_day'],
                time_str=self.config['database_maintenance_time'],
                name="Weekly Database Maintenance"
            )
            
            self.logger.info("Weekly tasks registered successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering weekly tasks: {e}")
            return False
    
    def weekend_market_analysis(self):
        """
        Perform weekend market analysis (Saturday 10:00 AM).
        """
        try:
            self.logger.info("Starting weekend market analysis")
            
            # Import required modules
            from research.market_analysis import MarketAnalyzer
            from research.correlation_analyzer import CorrelationAnalyzer
            from research.volatility_analyzer import VolatilityAnalyzer
            from ml.prediction.sector_rotation_analyzer import SectorRotationAnalyzer
            
            # Run weekly market analysis
            market_analyzer = MarketAnalyzer(self.db)
            market_analyzer.analyze_weekly_trends()
            
            # Run weekly correlation analysis
            correlation_analyzer = CorrelationAnalyzer(self.db)
            correlation_analyzer.analyze_weekly_correlations()
            
            # Run volatility regime analysis
            volatility_analyzer = VolatilityAnalyzer(self.db)
            volatility_analyzer.analyze_volatility_regimes()
            
            # Run sector rotation analysis
            sector_analyzer = SectorRotationAnalyzer(self.db)
            sector_analyzer.analyze_weekly_rotation()
            
            self.logger.info("Weekend market analysis completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in weekend market analysis: {e}")
            return False
    
    def portfolio_review(self):
        """
        Perform weekly portfolio review (Sunday 12:00 PM).
        """
        try:
            self.logger.info("Starting weekly portfolio review")
            
            # Import required modules
            from portfolio.portfolio_manager import PortfolioManager
            from research.opportunity_scanner import OpportunityScanner
            
            # Initialize portfolio manager
            portfolio_manager = PortfolioManager(self.db)
            
            # Analyze current portfolio performance
            portfolio_analysis = portfolio_manager.analyze_portfolio_performance(
                timeframe="weekly"
            )
            
            # Scan for new opportunities
            opportunity_scanner = OpportunityScanner(self.db)
            new_opportunities = opportunity_scanner.scan_for_new_symbols()
            
            # Review existing symbols
            portfolio_review = portfolio_manager.review_portfolio_symbols()
            
            # Store review results
            self.db.portfolio_reviews.insert_one({
                "date": datetime.now(),
                "type": "weekly",
                "portfolio_analysis": portfolio_analysis,
                "new_opportunities": new_opportunities,
                "symbol_review": portfolio_review
            })
            
            self.logger.info("Weekly portfolio review completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in weekly portfolio review: {e}")
            return False
    
    def strategy_optimization(self):
        """
        Perform weekly strategy optimization (Sunday 14:00 PM).
        """
        try:
            self.logger.info("Starting weekly strategy optimization")
            
            # Import required modules
            from backtesting.optimizer import StrategyOptimizer
            
            # Initialize optimizer
            optimizer = StrategyOptimizer(self.db)
            
            # Get active strategies
            active_strategies = self._get_active_strategies()
            
            # Optimize each strategy
            for strategy in active_strategies:
                try:
                    self.logger.info(f"Optimizing strategy: {strategy['name']}")
                    
                    # Run optimization
                    results = optimizer.optimize_strategy(
                        strategy_name=strategy['name'],
                        strategy_class=strategy['class'],
                        param_grid=strategy['param_grid']
                    )
                    
                    # Update strategy parameters
                    self._update_strategy_parameters(strategy['name'], results['best_params'])
                    
                    self.logger.info(f"Strategy optimization completed for {strategy['name']}")
                    
                except Exception as e:
                    self.logger.error(f"Error optimizing strategy {strategy['name']}: {e}")
            
            # Validate optimized strategies
            from backtesting.validator import StrategyValidator
            validator = StrategyValidator(self.db)
            validation_results = validator.validate_all_strategies()
            
            # Store validation results
            self.db.strategy_validations.insert_one({
                "date": datetime.now(),
                "validation_results": validation_results
            })
            
            self.logger.info("Weekly strategy optimization completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in weekly strategy optimization: {e}")
            return False
    
    def generate_weekly_report(self):
        """
        Generate weekly report (Sunday 18:00 PM).
        """
        try:
            self.logger.info("Generating weekly report")
            
            # Import required modules
            from reports.templates.weekly_report import WeeklyReport
            from reports.formatters.pdf_formatter import PDFFormatter
            from communication.report_distributor import ReportDistributor
            
            # Initialize report generators
            weekly_report = WeeklyReport(self.db)
            pdf_formatter = PDFFormatter()
            report_distributor = ReportDistributor(self.db)
            
            # Generate weekly report
            report_data = weekly_report.generate_report()
            
            # Format report for PDF
            pdf_report = pdf_formatter.format(report_data)
            pdf_path = pdf_formatter.save(pdf_report, f"weekly_report_{datetime.now().strftime('%Y%m%d')}.pdf")
            
            # Distribute report via email
            report_distributor.distribute_via_email(
                pdf_path, 
                "Weekly Trading System Report",
                recipients=self._get_report_recipients()
            )
            
            # Format report for WhatsApp
            from reports.formatters.whatsapp_formatter import WhatsAppFormatter
            whatsapp_formatter = WhatsAppFormatter()
            whatsapp_report = whatsapp_formatter.format(report_data, summary_only=True)
            
            # Distribute summary via WhatsApp
            report_distributor.distribute_via_whatsapp(whatsapp_report)
            
            self.logger.info("Weekly report generated and distributed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating weekly report: {e}")
            return False
    
    def database_maintenance(self):
        """
        Perform weekly database maintenance (Sunday 22:00 PM).
        """
        try:
            self.logger.info("Starting weekly database maintenance")
            
            if not self.db:
                self.logger.warning("No database connection, skipping maintenance")
                return True
                
            # Perform database maintenance tasks
            
            # 1. Reindex collections
            collections = self.db.list_collection_names()
            
            for collection in collections:
                try:
                    self.logger.info(f"Reindexing collection: {collection}")
                    self.db.command("reIndex", collection)
                except Exception as e:
                    self.logger.warning(f"Could not reindex collection {collection}: {e}")
            
            # 2. Run database stats
            stats = self.db.command("dbStats")
            self.logger.info(f"Database size: {stats.get('dataSize') / (1024 * 1024):.2f} MB")
            
            # 3. Clean up old log data
            log_cutoff = datetime.now() - timedelta(days=30)
            
            log_result = self.db.system_logs.delete_many({
                "timestamp": {"$lt": log_cutoff}
            })
            
            self.logger.info(f"Deleted {log_result.deleted_count} old log entries")
            
            # 4. Create full database backup
            self._create_full_backup()
            
            self.logger.info("Weekly database maintenance completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in weekly database maintenance: {e}")
            return False
    
    def _create_full_backup(self):
        """
        Create a full database backup.
        """
        try:
            self.logger.info("Creating full database backup")
            
            # Get backup directory from config
            backup_dir = self._get_config_value("backup_directory", "backups")
            
            # Create backup directory if it doesn't exist
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
                
            # Generate backup filename
            backup_filename = f"weekly_backup_{datetime.now().strftime('%Y%m%d')}.archive"
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
                self.logger.info(f"Full database backed up to {backup_path}")
            else:
                self.logger.error(f"Database backup failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error creating full backup: {e}")
    
    def _get_active_strategies(self):
        """
        Get list of active strategies for optimization.
        
        Returns:
            list: List of active strategies
        """
        try:
            # In a real implementation, this would come from the database
            # For now, return a sample list
            
            from core.strategies.technical import TechnicalStrategy
            from core.strategies.statistical_arbitrage import StatisticalArbitrageStrategy
            
            strategies = [
                {
                    "name": "SMA_Crossover",
                    "class": TechnicalStrategy,
                    "param_grid": {
                        "short_period": [5, 10, 15, 20],
                        "long_period": [30, 40, 50, 60],
                        "exit_after": [3, 5, 7]
                    }
                },
                {
                    "name": "MeanReversion",
                    "class": StatisticalArbitrageStrategy,
                    "param_grid": {
                        "lookback_period": [10, 15, 20, 25],
                        "entry_z": [1.5, 2.0, 2.5],
                        "exit_z": [0.5, 0.25, 0]
                    }
                }
            ]
            
            return strategies
            
        except Exception as e:
            self.logger.error(f"Error getting active strategies: {e}")
            return []
    
    def _update_strategy_parameters(self, strategy_name, parameters):
        """
        Update strategy parameters in the database.
        
        Args:
            strategy_name (str): Strategy name
            parameters (dict): Strategy parameters
        """
        try:
            if not self.db:
                return
                
            # Update strategy parameters
            self.db.strategy_parameters.update_one(
                {"name": strategy_name},
                {"$set": {
                    "parameters": parameters,
                    "updated_at": datetime.now()
                }},
                upsert=True
            )
            
            self.logger.info(f"Updated parameters for strategy {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Error updating strategy parameters: {e}")
    
    def _get_report_recipients(self):
        """
        Get list of email recipients for reports.
        
        Returns:
            list: List of email addresses
        """
        try:
            # In a real implementation, this would come from the database
            # For now, return a sample list
            return ["admin@example.com"]
            
        except Exception as e:
            self.logger.error(f"Error getting report recipients: {e}")
            return []
    
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