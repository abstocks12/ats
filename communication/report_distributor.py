"""
Report Distributor for the Automated Trading System.
This module handles the generation and distribution of reports.
"""

import logging
from datetime import datetime, timedelta
import threading
import time
import os
import tempfile

class ReportDistributor:
    def __init__(self, db_connector, notification_manager, report_config=None):
        """
        Initialize the report distributor.
        
        Args:
            db_connector: Database connector
            notification_manager: NotificationManager instance
            report_config (dict): Report configuration
        """
        self.db = db_connector
        self.notification_manager = notification_manager
        self.config = report_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize report generators
        self.report_generators = {}
        self._initialize_report_generators()
        
        # Initialize report formatters
        self.report_formatters = {}
        self._initialize_report_formatters()
        
    def _initialize_report_generators(self):
        """Initialize report generators"""
        try:
            # Import daily prediction report generator
            from reports.daily_prediction import DailyPredictionReport
            self.report_generators["daily"] = DailyPredictionReport(self.db)
            
            # Add more report generators as needed
            try:
                from reports.templates.morning_report import MorningReport
                self.report_generators["morning"] = MorningReport(self.db)
            except ImportError:
                self.logger.warning("Morning report generator not available")
                
            try:
                from reports.templates.eod_report import EODReport
                self.report_generators["eod"] = EODReport(self.db)
            except ImportError:
                self.logger.warning("EOD report generator not available")
                
            try:
                from reports.templates.weekly_report import WeeklyReport
                self.report_generators["weekly"] = WeeklyReport(self.db)
            except ImportError:
                self.logger.warning("Weekly report generator not available")
                
            try:
                from reports.templates.monthly_report import MonthlyReport
                self.report_generators["monthly"] = MonthlyReport(self.db)
            except ImportError:
                self.logger.warning("Monthly report generator not available")
                
        except ImportError as e:
            self.logger.error(f"Error initializing report generators: {e}")
            
    def _initialize_report_formatters(self):
        """Initialize report formatters"""
        try:
            # Initialize Slack formatter
            from communication.slack.formatter import SlackFormatter
            self.report_formatters["slack"] = SlackFormatter()
            
            # Add more formatters as needed
            try:
                from reports.formatters.pdf_formatter import PDFFormatter
                self.report_formatters["pdf"] = PDFFormatter()
            except ImportError:
                self.logger.warning("PDF formatter not available")
                
        except ImportError as e:
            self.logger.error(f"Error initializing report formatters: {e}")
            
    def generate_report(self, report_type, params=None):
        """
        Generate a report.
        
        Args:
            report_type (str): Type of report to generate
            params (dict): Additional parameters for report generation
            
        Returns:
            dict: Generated report data or None if failed
        """
        if report_type not in self.report_generators:
            self.logger.error(f"Unknown report type: {report_type}")
            return None
            
        try:
            generator = self.report_generators[report_type]
            report = generator.generate_report(params)
            
            # Store report in database
            if report:
                try:
                    self.db.reports_collection.insert_one({
                        "type": report_type,
                        "data": report,
                        "generated_at": datetime.now(),
                        "params": params or {}
                    })
                except Exception as e:
                    self.logger.error(f"Error storing report in database: {e}")
                    
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating {report_type} report: {e}")
            return None
            
    def format_report(self, report_data, format_type):
        """
        Format a report.
        
        Args:
            report_data (dict): Report data to format
            format_type (str): Type of formatting to apply
            
        Returns:
            dict: Formatted report data or None if failed
        """
        if format_type not in self.report_formatters:
            self.logger.error(f"Unknown format type: {format_type}")
            return None
            
        try:
            formatter = self.report_formatters[format_type]
            
            if format_type == "pdf":
                # For PDF, we need to create a file
                return formatter.format_report(report_data)
            else:
                # For other formats, just return the formatted data
                return formatter.format_report(report_data)
                
        except Exception as e:
            self.logger.error(f"Error formatting report as {format_type}: {e}")
            return None
            
    def distribute_report(self, report_type, channels=None, params=None):
        """
        Generate and distribute a report.
        
        Args:
            report_type (str): Type of report to generate
            channels (list): List of channels to distribute to (default: configured channels)
            params (dict): Additional parameters for report generation
            
        Returns:
            bool: Success status
        """
        try:
            # Generate report
            report_data = self.generate_report(report_type, params)
            if not report_data:
                return False
                
            # Use configured channels if none provided
            if channels is None:
                channels = self.config.get("default_channels", ["slack"])
                
            # Distribute to each channel
            success = True
            
            for channel in channels:
                if channel == "slack":
                    success = success and self._send_to_slack(report_type, report_data)
                elif channel == "pdf":
                    success = success and self._generate_pdf(report_type, report_data)
                # Add more channels as needed
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error distributing {report_type} report: {e}")
            return False
            
    def _send_to_slack(self, report_type, report_data):
        """Send report to Slack"""
        try:
            # Format report for Slack
            formatted = self.format_report(report_data, "slack")
            if not formatted:
                return False
                
            # Send notification
            self.notification_manager.slack.send_message(
                text=f"{report_type.capitalize()} Report",
                blocks=formatted.get("blocks"),
                attachments=formatted.get("attachments")
            )
            
            self.logger.info(f"Sent {report_type} report to Slack")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending report to Slack: {e}")
            return False
            
    def _generate_pdf(self, report_type, report_data):
        """Generate PDF report"""
        try:
            # Format report as PDF
            pdf_data = self.format_report(report_data, "pdf")
            if not pdf_data:
                return False
                
            # Save PDF to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_path = temp_file.name
                with open(temp_path, "wb") as f:
                    f.write(pdf_data)
                    
            # Upload PDF to Slack
            self.notification_manager.slack.upload_file(
                file_path=temp_path,
                title=f"{report_type.capitalize()} Report - {datetime.now().strftime('%Y-%m-%d')}",
                initial_comment=f"Here's the {report_type} report for {datetime.now().strftime('%Y-%m-%d')}"
            )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            self.logger.info(f"Generated and shared PDF for {report_type} report")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {e}")
            return False
            
    def schedule_distribution(self, schedule_config=None):
        """
        Schedule automatic report distribution.
        
        Args:
            schedule_config (dict): Configuration for scheduled reports
            
        Returns:
            bool: Success status
        """
        config = schedule_config or self.config.get("schedule", {})
        if not config:
            self.logger.warning("No schedule configuration provided")
            return False
            
        try:
            # TODO: Implement scheduling logic using the automation framework
            # For now, just return True
            self.logger.info("Report distribution scheduling not yet implemented")
            return True
            
        except Exception as e:
            self.logger.error(f"Error scheduling report distribution: {e}")
            return False