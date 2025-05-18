# reports/formatters/whatsapp_formatter.py (Session 47: Report Templates & Formatters)

import logging
from datetime import datetime

class WhatsAppFormatter:
    """
    Formats reports for WhatsApp delivery.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the WhatsApp formatter.
        
        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("WhatsApp formatter initialized")
    
    def format(self, report_data, summary_only=False):
        """
        Format report for WhatsApp.
        
        Args:
            report_data (dict): Report data
            summary_only (bool): Whether to include only summary
            
        Returns:
            str: Formatted report
        """
        try:
            self.logger.info("Formatting report for WhatsApp")
            
            # Determine report type
            if 'market_summary' in report_data and 'prediction_summary' in report_data:
                return self.format_daily_prediction(report_data, summary_only)
            elif 'market_summary' in report_data and 'trading_performance' in report_data:
                return self.format_eod_report(report_data, summary_only)
            elif 'market_outlook' in report_data and 'global_markets' in report_data:
                return self.format_morning_report(report_data, summary_only)
            else:
                return self.format_generic_report(report_data, summary_only)
                
        except Exception as e:
            self.logger.error(f"Error formatting report for WhatsApp: {e}")
            return f"Error formatting report: {e}"
    
    def format_daily_prediction(self, report_data, summary_only=False):
        """
        Format daily prediction report for WhatsApp.
        
        Args:
            report_data (dict): Report data
            summary_only (bool): Whether to include only summary
            
        Returns:
            str: Formatted report
        """
        try:
            date_str = report_data.get('date', datetime.now()).strftime("%d-%b-%Y")
            
            # Start with header
            message = f"*Daily Prediction Report - {date_str}*\n\n"
            
            # Market summary
            market_summary = report_data.get('market_summary', {})
            
            direction = market_summary.get('direction', 'unknown').title()
            nifty_change = market_summary.get('nifty', {}).get('change', 0)
            
            message += f"*Market Summary:* {direction} (NIFTY {nifty_change:+.2f}%)\n"
            
            # Add advances/declines
            advances = market_summary.get('advances', 0)
            declines = market_summary.get('declines', 0)
            
            message += f"Advances: {advances}, Declines: {declines}\n\n"
            
            # Prediction summary
            prediction_summary = report_data.get('prediction_summary', {})
            
            count = prediction_summary.get('count', 0)
            up_count = prediction_summary.get('up_count', 0)
            down_count = prediction_summary.get('down_count', 0)
            
            message += f"*Prediction Summary:* {count} predictions\n"
            message += f"Bullish: {up_count}, Bearish: {down_count}\n"
            
            yesterday_accuracy = prediction_summary.get('yesterday_accuracy', 0)
            message += f"Yesterday's accuracy: {yesterday_accuracy:.2f}%\n\n"
            
            # If summary only, return here
            if summary_only:
                message += "*Top Opportunities* (check detailed report)\n"
                return message
            
            # Top opportunities
            top_opportunities = report_data.get('top_opportunities', {})
            
            # Bullish opportunities
            bullish = top_opportunities.get('bullish', [])
            
            if bullish:
                message += "*Bullish Opportunities:*\n"
                
                for opp in bullish:
                    symbol = opp.get('symbol', 'Unknown')
                    confidence = opp.get('confidence', 0) * 100
                    target = opp.get('target_price', 0)
                    
                    message += f"• {symbol} ({confidence:.1f}%)"
                    
                    if target > 0:
                        message += f", Target: ₹{target:.2f}"
                    
                    message += "\n"
                
                message += "\n"
            
            # Bearish opportunities
            bearish = top_opportunities.get('bearish', [])
            
            if bearish:
                message += "*Bearish Opportunities:*\n"
                
                for opp in bearish:
                    symbol = opp.get('symbol', 'Unknown')
                    confidence = opp.get('confidence', 0) * 100
                    target = opp.get('target_price', 0)
                    
                    message += f"• {symbol} ({confidence:.1f}%)"
                    
                    if target > 0:
                        message += f", Target: ₹{target:.2f}"
                    
                    message += "\n"
                
                message += "\n"
            
            # Sector summary
            sector_summary = report_data.get('sector_summary', {})
            
            # Top performing sectors
            top_sectors = sector_summary.get('top_sectors', [])
            
            if top_sectors:
                message += "*Top Sectors:*\n"
                
                for sector in top_sectors:
                    name = sector.get('name', 'Unknown')
                    change = sector.get('change', 0)
                    
                    message += f"• {name}: +{change:.2f}%\n"
                
                message += "\n"
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error formatting daily prediction for WhatsApp: {e}")
            return f"Error formatting report: {e}"
    
    def format_eod_report(self, report_data, summary_only=False):
        """
        Format EOD report for WhatsApp.
        
        Args:
            report_data (dict): Report data
            summary_only (bool): Whether to include only summary
            
        Returns:
            str: Formatted report
        """
        try:
            date_str = report_data.get('date', datetime.now()).strftime("%d-%b-%Y")
            
            # Start with header
            message = f"*End-of-Day Report - {date_str}*\n\n"
            
            # Market summary
            market_summary = report_data.get('market_summary', {})
            
            direction = market_summary.get('direction', 'unknown').title()
            nifty_change = market_summary.get('nifty', {}).get('change', 0)
            volume_change = market_summary.get('volume_change', 0)
            
            message += f"*Market Summary:* {direction}\n"
            message += f"NIFTY: {nifty_change:+.2f}%, Volume: {volume_change:+.2f}%\n"
            
            # Add breadth
            breadth = market_summary.get('breadth', {})
            advances = breadth.get('advances', 0)
            declines = breadth.get('declines', 0)
            
            message += f"Advances: {advances}, Declines: {declines}\n\n"
            
            # Trading performance
            trading_performance = report_data.get('trading_performance', {})
            
            total_trades = trading_performance.get('total_trades', 0)
            win_rate = trading_performance.get('win_rate', 0)
            net_pnl = trading_performance.get('net_pnl', 0)
            
            message += f"*Trading Performance:*\n"
            message += f"Trades: {total_trades}, Win Rate: {win_rate:.2f}%\n"
            message += f"Net P&L: ₹{net_pnl:.2f}\n\n"
            
            # If summary only, return here
            if summary_only:
                message += "*See full report for details*\n"
                return message
            
            # Prediction performance
            prediction_performance = report_data.get('prediction_performance', {})
            
            if prediction_performance.get('status') != 'no_data':
                accuracy = prediction_performance.get('accuracy', 0)
                total_predictions = prediction_performance.get('total_predictions', 0)
                
                message += f"*Prediction Performance:*\n"
                message += f"Accuracy: {accuracy:.2f}% ({prediction_performance.get('correct_predictions', 0)}/{total_predictions})\n"
                
                # Compare to historical
                historical = prediction_performance.get('historical_comparison', {})
                difference = historical.get('difference', 0)
                
                if difference > 0:
                    message += f"↑ {difference:.2f}% better than average\n\n"
                elif difference < 0:
                    message += f"↓ {-difference:.2f}% worse than average\n\n"
                else:
                    message += "Same as average\n\n"
            
            # Sector performance
            sector_performance = report_data.get('sector_performance', {})
            
            # Top performing sectors
            top_sectors = sector_performance.get('top_sectors', [])
            
            if top_sectors:
                message += "*Top Sectors:*\n"
                
                for sector in top_sectors:
                    name = sector.get('name', 'Unknown')
                    change = sector.get('change', 0)
                    
                    message += f"• {name}: +{change:.2f}%\n"
                
                message += "\n"
            
            # Market analysis
            market_analysis = report_data.get('market_analysis', {})
            
            # Key observations
            observations = market_analysis.get('observations', [])
            
            if observations:
                message += "*Key Observations:*\n"
                
                for i, observation in enumerate(observations[:3], 1):
                    message += f"{i}. {observation}\n"
                
                message += "\n"
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error formatting EOD report for WhatsApp: {e}")
            return f"Error formatting report: {e}"
    
    def format_morning_report(self, report_data, summary_only=False):
        """
        Format morning report for WhatsApp.
        
        Args:
            report_data (dict): Report data
            summary_only (bool): Whether to include only summary
            
        Returns:
            str: Formatted report
        """
        try:
            date_str = report_data.get('date', datetime.now()).strftime("%d-%b-%Y")
            
            # Check if trading day
            if not report_data.get('is_trading_day', True):
                return f"*Morning Report - {date_str}*\n\n{report_data.get('message', 'Markets are closed today.')}"
            
            # Start with header
            message = f"*Morning Report - {date_str}*\n\n"
            
            # Market outlook
            market_outlook = report_data.get('market_outlook', {})
            
            outlook = market_outlook.get('outlook', 'neutral').title()
            description = market_outlook.get('description', '')
            
            message += f"*Market Outlook:* {outlook}\n"
            message += f"{description}\n\n"
            
            # Global markets
            global_markets = report_data.get('global_markets', {})
            
            us_sentiment = global_markets.get('us_sentiment', 'neutral').title()
            asian_sentiment = global_markets.get('asian_sentiment', 'neutral').title()
            
            message += f"*Global Markets:*\n"
            message += f"US: {us_sentiment}, Asia: {asian_sentiment}\n\n"
            
            # If summary only, return here
            if summary_only:
                message += "*See full report for details*\n"
                return message
            
            # Economic events
            economic_events = report_data.get('economic_events', {})
            
            high_importance = economic_events.get('high_importance', [])
            
            if high_importance:
                message += f"*Economic Events:* {len(high_importance)} high-importance\n\n"
            
            # Opportunities
            opportunities = report_data.get('opportunities', {})
            
            # Bullish opportunities
            bullish = opportunities.get('bullish', [])
            
            if bullish:
                message += "*Top Bullish Ideas:*\n"
                
                for opp in bullish:
                    symbol = opp.get('symbol', 'Unknown')
                    confidence = opp.get('confidence', 0) * 100
                    
                    message += f"• {symbol} ({confidence:.1f}%)\n"
                
                message += "\n"
            
            # Bearish opportunities
            bearish = opportunities.get('bearish', [])
            
            if bearish:
                message += "*Top Bearish Ideas:*\n"
                
                for opp in bearish:
                    symbol = opp.get('symbol', 'Unknown')
                    confidence = opp.get('confidence', 0) * 100
                    
                    message += f"• {symbol} ({confidence:.1f}%)\n"
                
                message += "\n"
            
            # Important news
            news = report_data.get('news', [])
            
            if news:
                message += "*Important News:*\n"
                
                for item in news[:3]:
                    title = item.get('title', 'Unknown')
                    source = item.get('source', '')
                    
                    message += f"• {title} ({source})\n"
                
                message += "\n"
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error formatting morning report for WhatsApp: {e}")
            return f"Error formatting report: {e}"
    
    def format_generic_report(self, report_data, summary_only=False):
        """
        Format generic report for WhatsApp.
        
        Args:
            report_data (dict): Report data
            summary_only (bool): Whether to include only summary
            
        Returns:
            str: Formatted report
        """
        try:
            # Basic formatting for generic report
            message = "*Report Summary*\n\n"
            
            # Format date if available
            if 'date' in report_data:
                date_str = report_data['date'].strftime("%d-%b-%Y")
                message += f"Date: {date_str}\n\n"
            
            # Extract key information
            key_sections = []
            
            for key, value in report_data.items():
                if key in ['date', 'error']:
                    continue
                
                # Convert section to title case for header
                section = key.replace('_', ' ').title()
                
                if isinstance(value, dict) and len(value) > 0:
                    key_sections.append(section)
                elif isinstance(value, list) and len(value) > 0:
                    key_sections.append(section)
            
            # Add key sections to summary
            if key_sections:
                message += "*Report Contents:*\n"
                
                for section in key_sections:
                    message += f"• {section}\n"
                
                message += "\n"
            
            # If error occurred
            if 'error' in report_data:
                message += f"*Error:* {report_data['error']}\n\n"
            
            # If summary only, return here
            if summary_only:
                message += "*See full report for details*\n"
                return message
            
            # Format each section
            for key, value in report_data.items():
                if key in ['date', 'error']:
                    continue
                
                # Convert section to title case for header
                section = key.replace('_', ' ').title()
                message += f"*{section}*\n"
                
                # Format section content
                if isinstance(value, dict):
                    for k, v in list(value.items())[:5]:  # Limit to first 5 items
                        item = k.replace('_', ' ').title()
                        
                        if isinstance(v, (dict, list)):
                            message += f"• {item}: (see details)\n"
                        else:
                            message += f"• {item}: {v}\n"
                    
                    if len(value) > 5:
                        message += f"• ...{len(value) - 5} more items\n"
                        
                elif isinstance(value, list):
                    for i, item in enumerate(value[:5], 1):  # Limit to first 5 items
                        if isinstance(item, dict):
                            # Try to find a name or title
                            name = item.get('name', item.get('title', f"Item {i}"))
                            message += f"• {name}\n"
                        else:
                            message += f"• {item}\n"
                    
                    if len(value) > 5:
                        message += f"• ...{len(value) - 5} more items\n"
                        
                else:
                    message += f"{value}\n"
                
                message += "\n"
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error formatting generic report for WhatsApp: {e}")
            return f"Error formatting report: {e}"