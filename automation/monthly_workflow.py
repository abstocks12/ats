# automation/monthly_workflow.py
import logging
from datetime import datetime, timedelta
import os
import calendar
import time

class MonthlyWorkflow:
    """
    Implements the monthly automated workflow for the trading system.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the monthly workflow.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'performance_review_day': 1,  # First day of month
            'performance_review_time': '10:00',  # Performance review time
            'model_evaluation_day': 2,  # Second day of month
            'model_evaluation_time': '10:00',  # Model evaluation time
            'strategy_review_day': 3,  # Third day of month
            'strategy_review_time': '10:00',  # Strategy review time
            'monthly_report_day': 5,  # Fifth day of month
            'monthly_report_time': '12:00',  # Monthly report time
            'system_audit_day': 28,  # Fourth last day of month (approx)
            'system_audit_time': '14:00',  # System audit time
        }
        
        self.logger.info("Monthly workflow initialized")
    
    def register_tasks(self, scheduler):
        """
        Register all monthly tasks with the scheduler.
        
        Args:
            scheduler: Scheduler instance
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info("Registering monthly tasks")
            
            # Monthly performance review (1st day of month, 10:00 AM)
            scheduler.schedule_monthly(
                func=self.monthly_performance_review,
                day_of_month=self.config['performance_review_day'],
                time_str=self.config['performance_review_time'],
                name="Monthly Performance Review"
            )
            
            # Model evaluation (2nd day of month, 10:00 AM)
            scheduler.schedule_monthly(
                func=self.model_evaluation,
                day_of_month=self.config['model_evaluation_day'],
                time_str=self.config['model_evaluation_time'],
                name="Monthly Model Evaluation"
            )
            
            # Strategy review (3rd day of month, 10:00 AM)
            scheduler.schedule_monthly(
                func=self.strategy_review,
                day_of_month=self.config['strategy_review_day'],
                time_str=self.config['strategy_review_time'],
                name="Monthly Strategy Review"
            )
            
            # Monthly report (5th day of month, 12:00 PM)
            scheduler.schedule_monthly(
                func=self.generate_monthly_report,
                day_of_month=self.config['monthly_report_day'],
                time_str=self.config['monthly_report_time'],
                name="Monthly Report Generation"
            )
            
            # System audit (28th day of month, 14:00 PM)
            scheduler.schedule_monthly(
                func=self.system_audit,
                day_of_month=self.config['system_audit_day'],
                time_str=self.config['system_audit_time'],
                name="Monthly System Audit"
            )
            
            self.logger.info("Monthly tasks registered successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering monthly tasks: {e}")
            return False
    
    def monthly_performance_review(self):
        """
        Perform monthly performance review (1st day of month, 10:00 AM).
        """
        try:
            self.logger.info("Starting monthly performance review")
            
            # Import required modules
            from portfolio.portfolio_manager import PortfolioManager
            from trading.position_manager import PositionManager
            
            # Initialize managers
            portfolio_manager = PortfolioManager(self.db)
            position_manager = PositionManager(self.db)
            
            # Get previous month's date range
            today = datetime.now()
            first_day_prev_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
            last_day_prev_month = today.replace(day=1) - timedelta(days=1)
            
            # Analyze portfolio performance
            portfolio_performance = portfolio_manager.analyze_portfolio_performance(
                start_date=first_day_prev_month,
                end_date=last_day_prev_month,
                timeframe="monthly"
            )
            
            # Get trading statistics
            trading_stats = position_manager.get_trading_statistics(
                start_date=first_day_prev_month,
                end_date=last_day_prev_month
            )
            
            # Analyze by instrument type
            equity_performance = portfolio_manager.analyze_by_instrument_type(
                instrument_type="equity",
                start_date=first_day_prev_month,
                end_date=last_day_prev_month
            )
            
            futures_performance = portfolio_manager.analyze_by_instrument_type(
                instrument_type="futures",
                start_date=first_day_prev_month,
                end_date=last_day_prev_month
            )
            
            options_performance = portfolio_manager.analyze_by_instrument_type(
                instrument_type="options",
                start_date=first_day_prev_month,
                end_date=last_day_prev_month
            )
            
            # Analyze by strategy
            strategy_performance = portfolio_manager.analyze_by_strategy(
                start_date=first_day_prev_month,
                end_date=last_day_prev_month
            )
            
            # Analyze by sector
            sector_performance = portfolio_manager.analyze_by_sector(
                start_date=first_day_prev_month,
                end_date=last_day_prev_month
            )
            
            # Store performance review
            review_data = {
                "date": datetime.now(),
                "period": {
                    "start_date": first_day_prev_month,
                    "end_date": last_day_prev_month
                },
                "portfolio_performance": portfolio_performance,
                "trading_statistics": trading_stats,
                "instrument_performance": {
                    "equity": equity_performance,
                    "futures": futures_performance,
                    "options": options_performance
                },
                "strategy_performance": strategy_performance,
                "sector_performance": sector_performance
            }
            
            # Save to database
            self.db.performance_reviews.insert_one(review_data)
            
            self.logger.info("Monthly performance review completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in monthly performance review: {e}")
            return False
    
    def model_evaluation(self):
        """
        Perform monthly model evaluation (2nd day of month, 10:00 AM).
        """
        try:
            self.logger.info("Starting monthly model evaluation")
            
            # Import required modules
            from ml.training.evaluator import ModelEvaluator
            from ml.prediction.daily_predictor import DailyPredictor
            from ml.prediction.overnight_gap_predictor import OvernightGapPredictor
            from automation.prediction_validator import PredictionValidator
            
            # Initialize evaluators
            model_evaluator = ModelEvaluator(self.db)
            prediction_validator = PredictionValidator(self.db)
            
            # Get previous month's date range
            today = datetime.now()
            first_day_prev_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
            last_day_prev_month = today.replace(day=1) - timedelta(days=1)
            
            # Evaluate all models
            model_evaluation = model_evaluator.evaluate_all_models(
                start_date=first_day_prev_month,
                end_date=last_day_prev_month
            )
            
            # Validate daily predictions
            daily_validation = prediction_validator.validate_monthly_predictions(
                prediction_type="daily",
                start_date=first_day_prev_month,
                end_date=last_day_prev_month
            )
            
            # Validate overnight gap predictions
            gap_validation = prediction_validator.validate_monthly_predictions(
                prediction_type="overnight_gap",
                start_date=first_day_prev_month,
                end_date=last_day_prev_month
            )
            
            # Check for model drift
            model_drift = model_evaluator.check_model_drift(
                period_months=3
            )
            
            # Identify models for retraining
            retraining_candidates = model_evaluator.identify_retraining_candidates(
                accuracy_threshold=0.55,
                drift_threshold=0.1
            )
            
            # Store evaluation results
            evaluation_data = {
                "date": datetime.now(),
                "period": {
                    "start_date": first_day_prev_month,
                    "end_date": last_day_prev_month
                },
                "model_evaluation": model_evaluation,
                "prediction_validation": {
                    "daily": daily_validation,
                    "overnight_gap": gap_validation
                },
                "model_drift": model_drift,
                "retraining_candidates": retraining_candidates
            }
            
            # Save to database
            self.db.model_evaluations.insert_one(evaluation_data)
            
            # Schedule retraining for identified models
            if retraining_candidates:
                self._schedule_model_retraining(retraining_candidates)
            
            self.logger.info("Monthly model evaluation completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in monthly model evaluation: {e}")
            return False
    
    def strategy_review(self):
        """
        Perform monthly strategy review (3rd day of month, 10:00 AM).
        """
        try:
            self.logger.info("Starting monthly strategy review")
            
            # Import required modules
            from backtesting.performance import PerformanceAnalyzer
            from backtesting.validator import StrategyValidator
            
            # Initialize analyzers
            performance_analyzer = PerformanceAnalyzer(self.db)
            strategy_validator = StrategyValidator(self.db)
            
            # Get previous month's date range
            today = datetime.now()
            first_day_prev_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
            last_day_prev_month = today.replace(day=1) - timedelta(days=1)
            
            # Get active strategies
            active_strategies = self._get_active_strategies()
            
            # Analyze each strategy
            strategy_results = []
            
            for strategy in active_strategies:
                try:
                    # Analyze strategy performance
                    performance = performance_analyzer.analyze_strategy_performance(
                        strategy_name=strategy['name'],
                        start_date=first_day_prev_month,
                        end_date=last_day_prev_month
                    )
                    
                    # Validate strategy
                    validation = strategy_validator.validate_strategy(
                        strategy_name=strategy['name'],
                        strategy_class=strategy['class']
                    )
                    
                    # Compare to benchmark
                    benchmark_comparison = performance_analyzer.compare_to_benchmark(
                        strategy_name=strategy['name'],
                        benchmark_symbol="NIFTY",
                        start_date=first_day_prev_month,
                        end_date=last_day_prev_month
                    )
                    
                    # Add to results
                    strategy_results.append({
                        "strategy_name": strategy['name'],
                        "performance": performance,
                        "validation": validation,
                        "benchmark_comparison": benchmark_comparison
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing strategy {strategy['name']}: {e}")
            
            # Identify strategies for optimization
            optimization_candidates = self._identify_optimization_candidates(strategy_results)
            
            # Identify strategies for retirement
            retirement_candidates = self._identify_retirement_candidates(strategy_results)
            
            # Store review results
            review_data = {
                "date": datetime.now(),
                "period": {
                    "start_date": first_day_prev_month,
                    "end_date": last_day_prev_month
                },
                "strategy_results": strategy_results,
                "optimization_candidates": optimization_candidates,
                "retirement_candidates": retirement_candidates
            }
            
            # Save to database
            self.db.strategy_reviews.insert_one(review_data)
            
            # Schedule optimization for identified strategies
            if optimization_candidates:
                self._schedule_strategy_optimization(optimization_candidates)
                
            # Disable retired strategies
            if retirement_candidates:
                self._retire_strategies(retirement_candidates)
            
            self.logger.info("Monthly strategy review completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in monthly strategy review: {e}")
            return False
    
    def generate_monthly_report(self):
        """
        Generate monthly report (5th day of month, 12:00 PM).
        """
        try:
            self.logger.info("Generating monthly report")
            
            # Import required modules
            from reports.templates.monthly_report import MonthlyReport
            from reports.formatters.pdf_formatter import PDFFormatter
            from communication.report_distributor import ReportDistributor
            
            # Initialize report generators
            monthly_report = MonthlyReport(self.db)
            pdf_formatter = PDFFormatter()
            report_distributor = ReportDistributor(self.db)
            
            # Get previous month's date range
            today = datetime.now()
            first_day_prev_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
            last_day_prev_month = today.replace(day=1) - timedelta(days=1)
            month_name = first_day_prev_month.strftime("%B %Y")
            
            # Generate monthly report
            report_data = monthly_report.generate_report(
                start_date=first_day_prev_month,
                end_date=last_day_prev_month
            )
            
            # Format report for PDF
            pdf_report = pdf_formatter.format(report_data)
            pdf_path = pdf_formatter.save(pdf_report, f"monthly_report_{first_day_prev_month.strftime('%Y%m')}.pdf")
            
            # Distribute report via email
            report_distributor.distribute_via_email(
                pdf_path, 
                f"Monthly Trading System Report - {month_name}",
                recipients=self._get_report_recipients()
            )
            
            # Format report for WhatsApp
            from reports.formatters.whatsapp_formatter import WhatsAppFormatter
            whatsapp_formatter = WhatsAppFormatter()
            whatsapp_report = whatsapp_formatter.format(report_data, summary_only=True)
            
            # Distribute summary via WhatsApp
            report_distributor.distribute_via_whatsapp(whatsapp_report)
            
            self.logger.info("Monthly report generated and distributed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating monthly report: {e}")
            return False
    
    def system_audit(self):
        """
        Perform monthly system audit (28th day of month, 14:00 PM).
        """
        try:
            self.logger.info("Starting monthly system audit")
            
            # Perform database audit
            db_audit = self._audit_database()
            
            # Perform filesystem audit
            fs_audit = self._audit_filesystem()
            
            # Perform configuration audit
            config_audit = self._audit_configuration()
            
            # Perform security audit
            security_audit = self._audit_security()
            
            # Perform resource usage audit
            resource_audit = self._audit_resource_usage()
            
            # Compile audit report
            audit_report = {
                "date": datetime.now(),
                "database_audit": db_audit,
                "filesystem_audit": fs_audit,
                "configuration_audit": config_audit,
                "security_audit": security_audit,
                "resource_audit": resource_audit,
                "issues": []
            }
            
            # Identify issues
            audit_report["issues"] = self._identify_audit_issues(audit_report)
            
            # Save audit report
            self.db.system_audits.insert_one(audit_report)
            
            # Address critical issues
            critical_issues = [i for i in audit_report["issues"] if i.get("severity") == "critical"]
            
            if critical_issues:
                self.logger.warning(f"Found {len(critical_issues)} critical issues that require attention")
                self._notify_admin_of_issues(critical_issues)
            
            self.logger.info("Monthly system audit completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in monthly system audit: {e}")
            return False
    
    def _audit_database(self):
        """
        Perform database audit.
        
        Returns:
            dict: Audit results
        """
        try:
            self.logger.info("Performing database audit")
            
            if not self.db:
                return {"status": "error", "message": "No database connection"}
                
            # Get database stats
            stats = self.db.command("dbStats")
            
            # Compute collection sizes
            collection_sizes = {}
            collections = self.db.list_collection_names()
            
            for collection in collections:
                coll_stats = self.db.command("collStats", collection)
                collection_sizes[collection] = {
                    "size_mb": coll_stats.get("size", 0) / (1024 * 1024),
                    "count": coll_stats.get("count", 0),
                    "avg_obj_size": coll_stats.get("avgObjSize", 0),
                    "storage_size_mb": coll_stats.get("storageSize", 0) / (1024 * 1024),
                    "indexes": len(coll_stats.get("indexSizes", {}))
                }
            
            # Check for database issues
            issues = []
            
            # Check for large collections
            for collection, stats in collection_sizes.items():
                if stats["size_mb"] > 1000:  # 1 GB
                    issues.append(f"Collection {collection} is large: {stats['size_mb']:.2f} MB")
            
            # Check index usage if available
            try:
                for collection in collections:
                    index_stats = list(self.db[collection].aggregate([
                        {"$indexStats": {}}
                    ]))
                    
                    for idx_stat in index_stats:
                        idx_name = idx_stat.get("name")
                        idx_usage = idx_stat.get("accesses", {}).get("ops", 0)
                        
                        if idx_usage == 0 and idx_name != "_id_":
                            issues.append(f"Unused index {idx_name} on collection {collection}")
            except Exception as idx_err:
                issues.append(f"Could not check index usage: {idx_err}")
            
            # Return audit results
            return {
                "status": "completed",
                "database_size_mb": stats.get("dataSize", 0) / (1024 * 1024),
                "storage_size_mb": stats.get("storageSize", 0) / (1024 * 1024),
                "collections": len(collections),
                "collection_sizes": collection_sizes,
                "issues": issues
            }
            
        except Exception as e:
            self.logger.error(f"Error in database audit: {e}")
            return {"status": "error", "message": str(e)}
    
    def _audit_filesystem(self):
        """
        Perform filesystem audit.
        
        Returns:
            dict: Audit results
        """
        try:
            self.logger.info("Performing filesystem audit")
            
            # Check disk usage
            import shutil
            
            # Get current directory
            current_dir = os.path.abspath(os.getcwd())
            
            # Get disk usage
            total, used, free = shutil.disk_usage(current_dir)
            
            # Convert to MB
            total_mb = total / (1024 * 1024)
            used_mb = used / (1024 * 1024)
            free_mb = free / (1024 * 1024)
            
            # Calculate usage percentage
            usage_percent = (used / total) * 100
            
            # Check for files by type
            file_types = {
                "py": 0,
                "json": 0,
                "log": 0,
                "csv": 0,
                "pdf": 0,
                "backup": 0,
                "other": 0
            }
            
            # Get total file size by type
            file_sizes = {
                "py": 0,
                "json": 0,
                "log": 0,
                "csv": 0,
                "pdf": 0,
                "backup": 0,
                "other": 0
            }
            
            # Check log directory size
            log_dir = os.path.join(current_dir, "logs")
            log_dir_size = 0
            
            if os.path.exists(log_dir):
                for root, dirs, files in os.walk(log_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        log_dir_size += os.path.getsize(file_path)
            
            # Check backup directory size
            backup_dir = os.path.join(current_dir, "backups")
            backup_dir_size = 0
            
            if os.path.exists(backup_dir):
                for root, dirs, files in os.walk(backup_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        backup_dir_size += os.path.getsize(file_path)
            
            # Check data directory size
            data_dir = os.path.join(current_dir, "data")
            data_dir_size = 0
            
            if os.path.exists(data_dir):
                for root, dirs, files in os.walk(data_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        data_dir_size += os.path.getsize(file_path)
            
            # Identify issues
            issues = []
            
            if usage_percent > 90:
                issues.append(f"Disk usage is high: {usage_percent:.2f}%")
                
            if log_dir_size > 1024 * 1024 * 1024:  # 1 GB
                issues.append(f"Log directory is large: {log_dir_size / (1024 * 1024):.2f} MB")
                
            # Return audit results
            return {
                "status": "completed",
                "disk_usage": {
                    "total_mb": total_mb,
                    "used_mb": used_mb,
                    "free_mb": free_mb,
                    "usage_percent": usage_percent
                },
                "directory_sizes": {
                    "log_dir_mb": log_dir_size / (1024 * 1024),
                    "backup_dir_mb": backup_dir_size / (1024 * 1024),
                    "data_dir_mb": data_dir_size / (1024 * 1024)
                },
                "issues": issues
            }
            
        except Exception as e:
            self.logger.error(f"Error in filesystem audit: {e}")
            return {"status": "error", "message": str(e)}
    
    def _audit_configuration(self):
        """
        Perform configuration audit.
        
        Returns:
            dict: Audit results
        """
        try:
            self.logger.info("Performing configuration audit")
            
            # Check configuration values
            config_values = {}
            issues = []
            
            # Required configurations
            required_configs = [
                "mongodb_host",
                "mongodb_port",
                "mongodb_db",
                "zerodha_api_key",
                "zerodha_api_secret",
                "whatsapp_api_key",
                "email_smtp_server",
                "email_smtp_port",
                "email_username"
            ]
            
            # Check each required config
            for config_key in required_configs:
                value = self._get_config_value(config_key)
                
                if value is None:
                    issues.append(f"Missing required configuration: {config_key}")
                else:
                    # Store masked value for sensitive configs
                    if "api_key" in config_key or "secret" in config_key or "password" in config_key:
                        config_values[config_key] = "****"
                    else:
                        config_values[config_key] = value
            
            # Check for credentials expiration
            api_key_date = self._get_config_value("zerodha_api_key_date")
            
            if api_key_date:
                try:
                    key_date = datetime.strptime(api_key_date, "%Y-%m-%d")
                    days_active = (datetime.now() - key_date).days
                    
                    if days_active > 180:  # 6 months
                        issues.append(f"Zerodha API key is old ({days_active} days), consider rotating")
                except Exception:
                    issues.append("Invalid Zerodha API key date format")
            
            # Return audit results
            return {
                "status": "completed",
                "config_values": config_values,
                "issues": issues
            }
            
        except Exception as e:
            self.logger.error(f"Error in configuration audit: {e}")
            return {"status": "error", "message": str(e)}
    
    def _audit_security(self):
        """
        Perform security audit.
        
        Returns:
            dict: Audit results
        """
        try:
            self.logger.info("Performing security audit")
            
            issues = []
            
            # Check for environment variables with credentials
            env_vars = os.environ
            for var_name, var_value in env_vars.items():
                if ("KEY" in var_name or "SECRET" in var_name or "PASS" in var_name or "TOKEN" in var_name) and var_value:
                    issues.append(f"Sensitive information in environment variable: {var_name}")
            
            # Check .env file permissions if it exists
            env_file = os.path.join(os.getcwd(), ".env")
            if os.path.exists(env_file):
                try:
                    import stat
                    file_stat = os.stat(env_file)
                    file_mode = file_stat.st_mode
                    
                    # Check if world-readable
                    if file_mode & stat.S_IROTH:
                        issues.append(".env file is world-readable")
                except Exception as e:
                    issues.append(f"Could not check .env file permissions: {e}")
            
            # Check backup directory permissions
            backup_dir = os.path.join(os.getcwd(), "backups")
            if os.path.exists(backup_dir):
                try:
                    import stat
                    dir_stat = os.stat(backup_dir)
                    dir_mode = dir_stat.st_mode
                    
                    # Check if world-readable or world-writable
                    if dir_mode & stat.S_IROTH or dir_mode & stat.S_IWOTH:
                        issues.append("Backup directory has too permissive permissions")
                except Exception as e:
                    issues.append(f"Could not check backup directory permissions: {e}")
            
            # Return audit results
            return {
                "status": "completed",
                "issues": issues
            }
            
        except Exception as e:
            self.logger.error(f"Error in security audit: {e}")
            return {"status": "error", "message": str(e)}
    
    def _audit_resource_usage(self):
        """
        Perform resource usage audit.
        
        Returns:
            dict: Audit results
        """
        try:
            self.logger.info("Performing resource usage audit")
            
            # Check memory usage
            import psutil
            
            # Get memory usage
            memory = psutil.virtual_memory()
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get process info
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            process_cpu = process.cpu_percent(interval=1)
            
            # Identify issues
            issues = []
            
            if memory.percent > 90:
                issues.append(f"System memory usage is high: {memory.percent}%")
                
            if cpu_percent > 90:
                issues.append(f"System CPU usage is high: {cpu_percent}%")
                
            if process_memory > 1000:  # 1 GB
                issues.append(f"Process memory usage is high: {process_memory:.2f} MB")
            
            # Return audit results
            return {
                "status": "completed",
                "system_memory": {
                    "total_mb": memory.total / (1024 * 1024),
                    "available_mb": memory.available / (1024 * 1024),
                    "used_mb": memory.used / (1024 * 1024),
                    "percent": memory.percent
                },
                "system_cpu": {
                    "percent": cpu_percent
                },
                "process": {
                    "memory_mb": process_memory,
                    "cpu_percent": process_cpu
                },
                "issues": issues
            }
            
        except Exception as e:
            self.logger.error(f"Error in resource usage audit: {e}")
            return {"status": "error", "message": str(e)}
    
    def _identify_audit_issues(self, audit_report):
        """
        Identify issues from audit report.
        
        Args:
            audit_report (dict): Audit report
            
        Returns:
            list: List of issues
        """
        try:
            issues = []
            
            # Process database issues
            if "database_audit" in audit_report and "issues" in audit_report["database_audit"]:
                for issue in audit_report["database_audit"]["issues"]:
                    issues.append({
                        "component": "database",
                        "issue": issue,
                        "severity": "warning"
                    })
            
            # Process filesystem issues
            if "filesystem_audit" in audit_report and "issues" in audit_report["filesystem_audit"]:
                for issue in audit_report["filesystem_audit"]["issues"]:
                    severity = "critical" if "high" in issue else "warning"
                    issues.append({
                        "component": "filesystem",
                        "issue": issue,
                        "severity": severity
                    })
            
            # Process configuration issues
            if "configuration_audit" in audit_report and "issues" in audit_report["configuration_audit"]:
                for issue in audit_report["configuration_audit"]["issues"]:
                    severity = "critical" if "Missing required" in issue else "warning"
                    issues.append({
                        "component": "configuration",
                        "issue": issue,
                        "severity": severity
                    })
            
            # Process security issues
            if "security_audit" in audit_report and "issues" in audit_report["security_audit"]:
                for issue in audit_report["security_audit"]["issues"]:
                    issues.append({
                        "component": "security",
                        "issue": issue,
                        "severity": "critical"
                    })
            
            # Process resource issues
            if "resource_audit" in audit_report and "issues" in audit_report["resource_audit"]:
                for issue in audit_report["resource_audit"]["issues"]:
                    severity = "critical" if "high" in issue else "warning"
                    issues.append({
                        "component": "resources",
                        "issue": issue,
                        "severity": severity
                    })
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Error identifying audit issues: {e}")
            return []
    
    def _notify_admin_of_issues(self, issues):
        """
        Notify administrators of critical issues.
        
        Args:
            issues (list): List of critical issues
        """
        try:
            self.logger.info(f"Notifying admin of {len(issues)} critical issues")
            
            # Import required modules
            from communication.report_distributor import ReportDistributor
            
            # Initialize distributor
            distributor = ReportDistributor(self.db)
            
            # Format message
            message = "CRITICAL SYSTEM ISSUES DETECTED:\n\n"
            
            for i, issue in enumerate(issues, 1):
                message += f"{i}. [{issue.get('component', 'unknown')}] {issue.get('issue', 'Unknown issue')}\n"
                
            message += "\nPlease address these issues immediately to ensure system stability."
            
            # Send email notification
            distributor.distribute_via_email(
                content=message,
                subject="CRITICAL: Trading System Issues Detected",
                recipients=self._get_admin_emails(),
                attachment_path=None,
                is_html=False
            )
            
            # Send WhatsApp notification
            distributor.distribute_via_whatsapp(message)
            
            self.logger.info("Admin notification sent")
            
        except Exception as e:
            self.logger.error(f"Error notifying admin of issues: {e}")
    
    def _schedule_model_retraining(self, retraining_candidates):
        """
        Schedule retraining for identified models.
        
        Args:
            retraining_candidates (list): List of models for retraining
        """
        try:
            self.logger.info(f"Scheduling retraining for {len(retraining_candidates)} models")
            
            # Import scheduler module
            from automation.scheduler import Scheduler
            from automation.model_retraining import ModelRetraining
            
            # Get scheduler instance
            scheduler = Scheduler(self.db)
            
            # Create model retraining instance
            model_retraining = ModelRetraining(self.db)
            
            # Schedule retraining tasks with 1-hour intervals to avoid resource contention
            delay_hours = 1
            
            for model in retraining_candidates:
                # Schedule specific model retraining
                scheduler.schedule_in(
                    func=model_retraining.retrain_model,
                    delay=delay_hours * 3600,  # Convert to seconds
                    args=[model['symbol'], model['exchange'], model['model_type']],
                    name=f"Retrain {model['model_type']} model for {model['symbol']}"
                )
                
                delay_hours += 1
                
            self.logger.info("Model retraining scheduled")
            
        except Exception as e:
            self.logger.error(f"Error scheduling model retraining: {e}")
    
    def _schedule_strategy_optimization(self, optimization_candidates):
        """
        Schedule optimization for identified strategies.
        
        Args:
            optimization_candidates (list): List of strategies for optimization
        """
        try:
            self.logger.info(f"Scheduling optimization for {len(optimization_candidates)} strategies")
            
            # Import scheduler module
            from automation.scheduler import Scheduler
            from backtesting.optimizer import StrategyOptimizer
            
            # Get scheduler instance
            scheduler = Scheduler(self.db)
            
            # Create strategy optimizer instance
            strategy_optimizer = StrategyOptimizer(self.db)
            
            # Schedule optimization tasks with 2-hour intervals to avoid resource contention
            delay_hours = 1
            
            for strategy in optimization_candidates:
                # Schedule specific strategy optimization
                scheduler.schedule_in(
                    func=strategy_optimizer.optimize_strategy,
                    delay=delay_hours * 3600,  # Convert to seconds
                    args=[strategy['name'], strategy['class'], strategy['param_grid']],
                    name=f"Optimize {strategy['name']} strategy"
                )
                
                delay_hours += 2
                
            self.logger.info("Strategy optimization scheduled")
            
        except Exception as e:
            self.logger.error(f"Error scheduling strategy optimization: {e}")
    
    def _retire_strategies(self, retirement_candidates):
        """
        Disable retired strategies.
        
        Args:
            retirement_candidates (list): List of strategies to retire
        """
        try:
            self.logger.info(f"Retiring {len(retirement_candidates)} strategies")
            
            if not self.db:
                self.logger.warning("No database connection, skipping strategy retirement")
                return
                
            # Update each strategy
            for strategy in retirement_candidates:
                # Update strategy status
                self.db.strategy_parameters.update_one(
                    {"name": strategy['name']},
                    {"$set": {
                        "status": "retired",
                        "retired_at": datetime.now(),
                        "retirement_reason": strategy.get('reason', 'Poor performance')
                    }}
                )
                
                self.logger.info(f"Retired strategy: {strategy['name']}")
                
            # Notify about retired strategies
            self._notify_strategy_retirement(retirement_candidates)
            
        except Exception as e:
            self.logger.error(f"Error retiring strategies: {e}")
    
    def _notify_strategy_retirement(self, retirement_candidates):
        """
        Notify about retired strategies.
        
        Args:
            retirement_candidates (list): List of retired strategies
        """
        try:
            self.logger.info("Sending strategy retirement notification")
            
            # Import required modules
            from communication.report_distributor import ReportDistributor
            
            # Initialize distributor
            distributor = ReportDistributor(self.db)
            
            # Format message
            message = "STRATEGY RETIREMENT NOTIFICATION:\n\n"
            
            for i, strategy in enumerate(retirement_candidates, 1):
                message += f"{i}. {strategy['name']}: {strategy.get('reason', 'Poor performance')}\n"
                
            message += "\nThese strategies have been disabled and will not be used for trading until reviewed."
            
            # Send email notification
            distributor.distribute_via_email(
                content=message,
                subject="Trading Strategy Retirement Notice",
                recipients=self._get_strategy_notification_emails(),
                attachment_path=None,
                is_html=False
            )
            
            self.logger.info("Strategy retirement notification sent")
            
        except Exception as e:
            self.logger.error(f"Error sending strategy retirement notification: {e}")
    
    def _identify_optimization_candidates(self, strategy_results):
        """
        Identify strategies for optimization.
        
        Args:
            strategy_results (list): List of strategy performance results
            
        Returns:
            list: List of strategies for optimization
        """
        try:
            optimization_candidates = []
            
            for result in strategy_results:
                # Check for strategies that underperform but are still viable
                if (result.get('performance', {}).get('sharpe_ratio', 0) < 1.0 and 
                    result.get('performance', {}).get('profit_factor', 0) > 1.2 and
                    result.get('benchmark_comparison', {}).get('relative_performance', 0) < 0):
                    
                    # Get strategy details
                    strategy_name = result.get('strategy_name')
                    strategy_class = self._get_strategy_class(strategy_name)
                    param_grid = self._get_strategy_param_grid(strategy_name)
                    
                    if strategy_class and param_grid:
                        optimization_candidates.append({
                            'name': strategy_name,
                            'class': strategy_class,
                            'param_grid': param_grid,
                            'reason': 'Underperforming but viable'
                        })
                
                # Check for strategies with declining performance
                elif (result.get('performance', {}).get('win_rate_change', 0) < -0.05 or
                      result.get('performance', {}).get('profit_factor_change', 0) < -0.2):
                    
                    # Get strategy details
                    strategy_name = result.get('strategy_name')
                    strategy_class = self._get_strategy_class(strategy_name)
                    param_grid = self._get_strategy_param_grid(strategy_name)
                    
                    if strategy_class and param_grid:
                        optimization_candidates.append({
                            'name': strategy_name,
                            'class': strategy_class,
                            'param_grid': param_grid,
                            'reason': 'Declining performance'
                        })
            
            return optimization_candidates
            
        except Exception as e:
            self.logger.error(f"Error identifying optimization candidates: {e}")
            return []
    
    def _identify_retirement_candidates(self, strategy_results):
        """
        Identify strategies for retirement.
        
        Args:
            strategy_results (list): List of strategy performance results
            
        Returns:
            list: List of strategies for retirement
        """
        try:
            retirement_candidates = []
            
            for result in strategy_results:
                # Check for strategies with negative returns
                if (result.get('performance', {}).get('total_return', 0) < -0.05 and
                    result.get('performance', {}).get('consecutive_losing_months', 0) >= 2):
                    
                    retirement_candidates.append({
                        'name': result.get('strategy_name'),
                        'reason': 'Sustained negative returns'
                    })
                
                # Check for strategies with poor validation results
                elif (result.get('validation', {}).get('is_valid', True) == False and
                      result.get('validation', {}).get('statistical_significance', 0) < 0.9):
                    
                    retirement_candidates.append({
                        'name': result.get('strategy_name'),
                        'reason': 'Failed statistical validation'
                    })
                
                # Check for strategies with very poor benchmark comparison
                elif (result.get('benchmark_comparison', {}).get('relative_performance', 0) < -0.2 and
                      result.get('benchmark_comparison', {}).get('underperformance_months', 0) >= 2):
                    
                    retirement_candidates.append({
                        'name': result.get('strategy_name'),
                        'reason': 'Significant underperformance vs benchmark'
                    })
            
            return retirement_candidates
            
        except Exception as e:
            self.logger.error(f"Error identifying retirement candidates: {e}")
            return []
    
    def _get_strategy_class(self, strategy_name):
        """
        Get strategy class by name.
        
        Args:
            strategy_name (str): Strategy name
            
        Returns:
            class: Strategy class or None if not found
        """
        try:
            # In a real implementation, this would use dynamic imports
            # For now, return predefined mappings
            
            from core.strategies.technical import TechnicalStrategy
            from core.strategies.statistical_arbitrage import StatisticalArbitrageStrategy
            from core.strategies.event_driven import EventDrivenStrategy
            
            # Strategy mapping
            strategy_classes = {
                "SMA_Crossover": TechnicalStrategy,
                "RSI_Reversal": TechnicalStrategy,
                "MeanReversion": StatisticalArbitrageStrategy,
                "PairTrading": StatisticalArbitrageStrategy,
                "EarningsAnnouncement": EventDrivenStrategy,
                "NewsMomentum": EventDrivenStrategy
            }
            
            return strategy_classes.get(strategy_name)
            
        except Exception as e:
            self.logger.error(f"Error getting strategy class for {strategy_name}: {e}")
            return None
    
    def _get_strategy_param_grid(self, strategy_name):
        """
        Get parameter grid for strategy optimization.
        
        Args:
            strategy_name (str): Strategy name
            
        Returns:
            dict: Parameter grid or None if not found
        """
        try:
            # In a real implementation, this would come from the database
            # For now, return predefined mappings
            
            # Parameter grids
            param_grids = {
                "SMA_Crossover": {
                    "short_period": [5, 10, 15, 20],
                    "long_period": [30, 40, 50, 60],
                    "exit_after": [3, 5, 7]
                },
                "RSI_Reversal": {
                    "rsi_period": [7, 14, 21],
                    "overbought": [70, 75, 80],
                    "oversold": [20, 25, 30],
                    "exit_after": [3, 5, 7]
                },
                "MeanReversion": {
                    "lookback_period": [10, 15, 20, 25],
                    "entry_z": [1.5, 2.0, 2.5],
                    "exit_z": [0.5, 0.25, 0]
                },
                "PairTrading": {
                    "lookback_period": [30, 60, 90],
                    "entry_z": [2.0, 2.5, 3.0],
                    "exit_z": [0.5, 0.25, 0]
                },
                "EarningsAnnouncement": {
                    "entry_days_before": [1, 2, 3],
                    "hold_days_after": [1, 2, 3],
                    "min_volatility": [0.02, 0.03, 0.04]
                },
                "NewsMomentum": {
                    "sentiment_threshold": [0.6, 0.7, 0.8],
                    "volume_increase": [1.5, 2.0, 2.5],
                    "hold_period": [1, 2, 3]
                }
            }
            
            return param_grids.get(strategy_name)
            
        except Exception as e:
            self.logger.error(f"Error getting parameter grid for {strategy_name}: {e}")
            return None
    
    def _get_active_strategies(self):
        """
        Get list of active strategies.
        
        Returns:
            list: List of active strategies
        """
        try:
            # In a real implementation, this would come from the database
            # For now, return a sample list
            
            from core.strategies.technical import TechnicalStrategy
            from core.strategies.statistical_arbitrage import StatisticalArbitrageStrategy
            from core.strategies.event_driven import EventDrivenStrategy
            
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
                },
                {
                    "name": "EarningsAnnouncement",
                    "class": EventDrivenStrategy,
                    "param_grid": {
                        "entry_days_before": [1, 2, 3],
                        "hold_days_after": [1, 2, 3],
                        "min_volatility": [0.02, 0.03, 0.04]
                    }
                }
            ]
            
            return strategies
            
        except Exception as e:
            self.logger.error(f"Error getting active strategies: {e}")
            return []
    
    def _get_report_recipients(self):
        """
        Get list of email recipients for reports.
        
        Returns:
            list: List of email addresses
        """
        try:
            # In a real implementation, this would come from the database
            # For now, return a sample list
            return ["admin@example.com", "trader@example.com"]
            
        except Exception as e:
            self.logger.error(f"Error getting report recipients: {e}")
            return ["admin@example.com"]
    
    def _get_admin_emails(self):
        """
        Get list of admin email addresses.
        
        Returns:
            list: List of email addresses
        """
        try:
            # In a real implementation, this would come from the database
            # For now, return a sample list
            return ["admin@example.com"]
            
        except Exception as e:
            self.logger.error(f"Error getting admin emails: {e}")
            return ["admin@example.com"]
    
    def _get_strategy_notification_emails(self):
        """
        Get list of email addresses for strategy notifications.
        
        Returns:
            list: List of email addresses
        """
        try:
            # In a real implementation, this would come from the database
            # For now, return a sample list
            return ["admin@example.com", "trader@example.com"]
            
        except Exception as e:
            self.logger.error(f"Error getting strategy notification emails: {e}")
            return ["admin@example.com"]
    
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