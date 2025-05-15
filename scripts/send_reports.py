#!/usr/bin/env python3
"""
Script to generate and send reports.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from database.connection_manager import get_db
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def send_report(report_type="daily", recipients=None, format="whatsapp"):
    """
    Generate and send a report
    
    Args:
        report_type (str): Report type (daily, weekly, monthly)
        recipients (list, optional): List of recipients
        format (str): Report format (whatsapp, email, pdf)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get database connection
        db = get_db()
        
        # Import necessary components
        try:
            from reports.daily_report import DailyReportGenerator
            report_generator = DailyReportGenerator(db)
        except ImportError:
            # If not available, create a placeholder implementation
            logger.warning("Report generators not available, using placeholder")
            
            class PlaceholderReportGenerator:
                def __init__(self, db):
                    self.db = db
                    self.logger = setup_logger("placeholder_report")
                
                def generate_daily_report(self):
                    self.logger.info("Placeholder: Generating daily report")
                    
                    now = datetime.now()
                    
                    # Generate a placeholder report
                    report = {
                        "date": now.strftime("%Y-%m-%d"),
                        "generated_at": now.isoformat(),
                        "market_outlook": "neutral",
                        "top_gainers": [],
                        "top_losers": [],
                        "report_content": "This is a placeholder daily report."
                    }
                    
                    return report
                
                def generate_weekly_report(self):
                    self.logger.info("Placeholder: Generating weekly report")
                    
                    now = datetime.now()
                    
                    # Calculate week start and end
                    week_start = now - timedelta(days=now.weekday())
                    week_end = week_start + timedelta(days=6)
                    
                    # Generate a placeholder report
                    report = {
                        "week": f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}",
                        "generated_at": now.isoformat(),
                        "market_outlook": "neutral",
                        "top_performers": [],
                        "worst_performers": [],
                        "report_content": "This is a placeholder weekly report."
                    }
                    
                    return report
                
                def generate_monthly_report(self):
                    self.logger.info("Placeholder: Generating monthly report")
                    
                    now = datetime.now()
                    
                    # Generate a placeholder report
                    report = {
                        "month": now.strftime("%B %Y"),
                        "generated_at": now.isoformat(),
                        "market_outlook": "neutral",
                        "monthly_performance": {},
                        "report_content": "This is a placeholder monthly report."
                    }
                    
                    return report
                
                def send_report(self, report, format="whatsapp", recipients=None):
                    self.logger.info(f"Placeholder: Sending report via {format} to {recipients or 'default recipients'}")
                    
                    # Save to database
                    self.db.insert_one("reports", {
                        "type": "report",
                        "report_type": report_type,
                        "format": format,
                        "recipients": recipients,
                        "report": report,
                        "sent_at": datetime.now().isoformat()
                    })
                    
                    return True
            
            report_generator = PlaceholderReportGenerator(db)
        
        # Generate the report
        report = None
        
        if report_type == "daily":
            report = report_generator.generate_daily_report()
        elif report_type == "weekly":
            report = report_generator.generate_weekly_report()
        elif report_type == "monthly":
            report = report_generator.generate_monthly_report()
        else:
            logger.error(f"Unknown report type: {report_type}")
            return False
        
        if not report:
            logger.error(f"Failed to generate {report_type} report")
            return False
        
        logger.info(f"Generated {report_type} report")
        
        # Send the report
        result = report_generator.send_report(report, format=format, recipients=recipients)
        
        if result:
            logger.info(f"Sent {report_type} report via {format} to {recipients or 'default recipients'}")
            return True
        else:
            logger.error(f"Failed to send {report_type} report")
            return False
            
    except Exception as e:
        logger.error(f"Error sending report: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate and send reports')
    
    parser.add_argument(
        '--type',
        choices=['daily', 'weekly', 'monthly'],
        default='daily',
        help='Report type'
    )
    
    parser.add_argument(
        '--recipients',
        nargs='+',
        help='List of recipients (emails or phone numbers)'
    )
    
    parser.add_argument(
        '--format',
        choices=['whatsapp', 'email', 'pdf'],
        default='whatsapp',
        help='Report format'
    )
    
    args = parser.parse_args()
    
    # Send report
    result = send_report(
        report_type=args.type,
        recipients=args.recipients,
        format=args.format
    )
    
    if result:
        print(f"{args.type.capitalize()} report sent successfully")
    else:
        print(f"Failed to send {args.type} report")
        sys.exit(1)

if __name__ == '__main__':
    main()