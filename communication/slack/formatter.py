"""
Formatter for Slack Messages.
This module handles formatting of reports and notifications for Slack.
"""

import logging
import json
from datetime import datetime

class SlackFormatter:
    def __init__(self):
        """Initialize the Slack formatter"""
        self.logger = logging.getLogger(__name__)

    def format_report(self, report_data):
        """
        Format a report for Slack.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            dict: Formatted report with blocks and attachments
        """
        if not report_data:
            return {"blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "No report data available."}}]}
            
        report_type = report_data.get("type", "unknown")
        
        # Call appropriate formatter based on report type
        if report_type == "daily_prediction":
            return self._format_daily_prediction(report_data)
        elif report_type == "performance":
            return self._format_performance_report(report_data)
        elif report_type == "weekly":
            return self._format_weekly_report(report_data)
        elif report_type == "monthly":
            return self._format_monthly_report(report_data)
        else:
            # Generic formatter for unknown report types
            return self._format_generic_report(report_data)
            
    def _format_daily_prediction(self, report_data):
        """Format daily prediction report"""
        blocks = []
        
        # Header
        date_str = report_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Daily Prediction Report - {date_str}",
                "emoji": True
            }
        })
        
        # Market summary
        market_summary = report_data.get("market_summary", "No market summary available.")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Market Summary:*\n{market_summary}"
            }
        })
        
        # Divider
        blocks.append({"type": "divider"})
        
        # Top predictions
        predictions = report_data.get("predictions", [])
        if predictions:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Top Predictions:*"
                }
            })
            
            for pred in predictions[:5]:  # Top 5 predictions
                symbol = pred.get("symbol", "Unknown")
                direction = pred.get("prediction", "neutral")
                confidence = pred.get("confidence", 0) * 100
                
                # Emoji based on direction
                emoji = "🔺" if direction == "up" else "🔻" if direction == "down" else "➡️"
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{symbol}*: {emoji} {direction.upper()} (Confidence: {confidence:.1f}%)\n"
                               f"Target: {pred.get('target_price', 'N/A')} | Stop Loss: {pred.get('stop_loss', 'N/A')}\n"
                               f"Supporting Factors: {', '.join(pred.get('supporting_factors', ['N/A']))}"
                    }
                })
        else:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "No predictions available for today."
                }
            })
            
        # Sector performance
        sector_performance = report_data.get("sector_performance", {})
        if sector_performance:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Sector Performance:*"
                }
            })
            
            sector_text = ""
            for sector, perf in sector_performance.items():
                change = perf.get("change", 0)
                arrow = "🔺" if change > 0 else "🔻" if change < 0 else "➡️"
                sector_text += f"• {sector}: {arrow} {change:.2f}%\n"
                
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": sector_text
                }
            })
            
        return {"blocks": blocks}
        
    def _format_performance_report(self, report_data):
        """Format performance report"""
        blocks = []
        
        # Header
        period = report_data.get("period", "all")
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Trading Performance Report - {period.capitalize()}",
                "emoji": True
            }
        })
        
        # Summary
        summary = report_data.get("summary", {})
        if summary:
            total_trades = summary.get("total_trades", 0)
            win_rate = summary.get("win_rate", 0)
            total_profit = summary.get("total_profit", 0)
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Summary:*\n"
                           f"• Total Trades: {total_trades}\n"
                           f"• Win Rate: {win_rate:.2f}%\n"
                           f"• Total Profit/Loss: {total_profit:.2f}\n"
                           f"• Max Drawdown: {summary.get('max_drawdown', 'N/A')}\n"
                }
            })
            
        # Strategy performance
        strategies = report_data.get("strategies", {})
        if strategies:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Strategy Performance:*"
                }
            })
            
            for strategy, perf in strategies.items():
                profit = perf.get("profit", 0)
                trades = perf.get("trades", 0)
                win_rate = perf.get("win_rate", 0)
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{strategy}*:\n"
                               f"• Trades: {trades}\n"
                               f"• Win Rate: {win_rate:.2f}%\n"
                               f"• Profit/Loss: {profit:.2f}\n"
                    }
                })
                
        # Recent trades
        recent_trades = report_data.get("recent_trades", [])
        if recent_trades:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Recent Trades:*"
                }
            })
            
            for trade in recent_trades[:5]:  # Show last 5 trades
                symbol = trade.get("symbol", "Unknown")
                profit = trade.get("profit_loss", 0)
                emoji = "✅" if profit > 0 else "❌"
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{symbol}* ({trade.get('strategy', 'Unknown')}): {emoji} {profit:.2f}\n"
                               f"Entry: {trade.get('entry_price', 'N/A')} | Exit: {trade.get('exit_price', 'N/A')}\n"
                               f"Time: {trade.get('exit_time', 'N/A')}"
                    }
                })
                
        return {"blocks": blocks}
        
    def _format_weekly_report(self, report_data):
        """Format weekly report"""
        # Similar structure to daily report but with weekly focus
        blocks = []
        
        # Header
        week_str = report_data.get("week", "Current Week")
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Weekly Market Report - {week_str}",
                "emoji": True
            }
        })
        
        # Market summary
        market_summary = report_data.get("market_summary", "No market summary available.")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Market Summary:*\n{market_summary}"
            }
        })
        
        # Key events
        events = report_data.get("key_events", [])
        if events:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Key Events:*"
                }
            })
            
            events_text = ""
            for event in events:
                events_text += f"• {event.get('date', 'N/A')}: {event.get('description', 'N/A')}\n"
                
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": events_text
                }
            })
            
        # Top performers
        performers = report_data.get("top_performers", [])
        if performers:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Top Performers:*"
                }
            })
            
            perf_text = ""
            for perf in performers:
                symbol = perf.get("symbol", "Unknown")
                change = perf.get("change", 0)
                perf_text += f"• {symbol}: {change:.2f}%\n"
                
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": perf_text
                }
            })
            
        # Week ahead outlook
        outlook = report_data.get("outlook", "No outlook available.")
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Week Ahead Outlook:*\n{outlook}"
            }
        })
        
        return {"blocks": blocks}
        
    def _format_monthly_report(self, report_data):
        """Format monthly report"""
        # Similar to weekly report but with monthly focus
        blocks = []
        
        # Header
        month_str = report_data.get("month", "Current Month")
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Monthly Market Report - {month_str}",
                "emoji": True
            }
        })
        
        # Market summary
        market_summary = report_data.get("market_summary", "No market summary available.")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Market Summary:*\n{market_summary}"
            }
        })
        
        # Sector analysis
        sectors = report_data.get("sector_analysis", {})
        if sectors:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Sector Analysis:*"
                }
            })
            
            sector_text = ""
            for sector, analysis in sectors.items():
                change = analysis.get("change", 0)
                outlook = analysis.get("outlook", "Neutral")
                sector_text += f"• *{sector}*: {change:.2f}% | Outlook: {outlook}\n"
                
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": sector_text
                }
            })
            
        # Economic indicators
        indicators = report_data.get("economic_indicators", [])
        if indicators:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Economic Indicators:*"
                }
            })
            
            ind_text = ""
            for ind in indicators:
                name = ind.get("name", "Unknown")
                value = ind.get("value", "N/A")
                change = ind.get("change", "N/A")
                ind_text += f"• *{name}*: {value} ({change})\n"
                
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ind_text
                }
            })
            
        # Monthly outlook
        outlook = report_data.get("outlook", "No outlook available.")
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Month Ahead Outlook:*\n{outlook}"
            }
        })
        
        return {"blocks": blocks}
        
    def _format_generic_report(self, report_data):
        """Format generic report for unknown report types"""
        blocks = []
        
        # Header
        report_type = report_data.get("type", "Report")
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{report_type.capitalize()} Report",
                "emoji": True
            }
        })
        
        # Process each section in the report
        for key, value in report_data.items():
            if key in ["type", "date"]:  # Skip metadata fields
                continue
                
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{key.replace('_', ' ').capitalize()}:*"
                }
            })
            
            # Format based on value type
            if isinstance(value, str):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": value
                    }
                })
            elif isinstance(value, (list, tuple)):
                # List of items
                items_text = ""
                for item in value:
                    if isinstance(item, dict):
                        # Try to format dict items nicely
                        item_text = ""
                        for k, v in item.items():
                            item_text += f"*{k.replace('_', ' ').capitalize()}*: {v}\n"
                        items_text += f"• {item_text}\n"
                    else:
                        items_text += f"• {item}\n"
                        
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": items_text
                    }
                })
            elif isinstance(value, dict):
                # Dictionary of values
                dict_text = ""
                for k, v in value.items():
                    dict_text += f"• *{k.replace('_', ' ').capitalize()}*: {v}\n"
                    
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": dict_text
                    }
                })
            else:
                # Other types
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": str(value)
                    }
                })
                
        return {"blocks": blocks}
    
    def format_notification(self, notification_type, data):
        """
        Format a notification for Slack.
        
        Args:
            notification_type (str): Type of notification
            data (dict): Notification data
            
        Returns:
            dict: Formatted notification with blocks
        """
        # Call appropriate formatter based on notification type
        if notification_type == "trade_executed":
            return self._format_trade_notification(data)
        elif notification_type == "system_alert":
            return self._format_system_alert(data)
        elif notification_type == "price_alert":
            return self._format_price_alert(data)
        elif notification_type == "prediction_alert":
            return self._format_prediction_alert(data)
        else:
            # Generic formatter for unknown notification types
            return self._format_generic_notification(notification_type, data)
            
    def _format_trade_notification(self, data):
        """Format trade execution notification"""
        blocks = []
        
        # Determine trade direction and emoji
        trade_type = data.get("trade_type", "unknown")
        if trade_type.lower() == "buy":
            emoji = "🟢"
            action = "BUY"
        elif trade_type.lower() == "sell":
            emoji = "🔴"
            action = "SELL"
        else:
            emoji = "⚪"
            action = trade_type.upper()
            
        symbol = data.get("symbol", "Unknown")
        
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} Trade Executed: {action} {symbol}",
                "emoji": True
            }
        })
        
        # Trade details
        blocks.append({
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Symbol:*\n{symbol}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Exchange:*\n{data.get('exchange', 'N/A')}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Price:*\n{data.get('price', 'N/A')}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Quantity:*\n{data.get('quantity', 'N/A')}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Strategy:*\n{data.get('strategy', 'N/A')}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Time:*\n{data.get('time', 'N/A')}"
                }
            ]
        })
        
        # Trade parameters
        blocks.append({
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Stop Loss:*\n{data.get('stop_loss', 'N/A')}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Target:*\n{data.get('target', 'N/A')}"
                }
            ]
        })
        
        # Trade signals
        signals = data.get("signals", [])
        if signals:
            signal_text = "*Signals:*\n"
            for signal in signals:
                signal_text += f"• {signal}\n"
                
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": signal_text
                }
            })
            
        # Notes
        notes = data.get("notes", "")
        if notes:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Notes:*\n{notes}"
                }
            })
            
        return {"blocks": blocks}
        
    def _format_system_alert(self, data):
        """Format system alert notification"""
        blocks = []
        
        # Determine alert level and emoji
        level = data.get("level", "info")
        if level.lower() == "critical":
            emoji = "🚨"
        elif level.lower() == "warning":
            emoji = "⚠️"
        elif level.lower() == "error":
            emoji = "❌"
        else:
            emoji = "ℹ️"
            
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} System Alert: {level.upper()}",
                "emoji": True
            }
        })
        
        # Alert message
        message = data.get("message", "No message provided.")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": message
            }
        })
        
        # Additional details if available
        details = data.get("details", {})
        if details:
            details_text = "*Details:*\n"
            for key, value in details.items():
                details_text += f"• *{key.replace('_', ' ').capitalize()}*: {value}\n"
                
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": details_text
                }
            })
            
        # Timestamp
        time_str = data.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"*Time:* {time_str}"
                }
            ]
        })
        
        return {"blocks": blocks}
        
    def _format_price_alert(self, data):
        """Format price alert notification"""
        blocks = []
        
        symbol = data.get("symbol", "Unknown")
        price = data.get("price", "N/A")
        condition = data.get("condition", "reached")
        
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🔔 Price Alert: {symbol}",
                "emoji": True
            }
        })
        
        # Alert details
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{symbol} has {condition} price level {price}."
            }
        })
        
        # Current price and change
        current_price = data.get("current_price", "N/A")
        change = data.get("change", "N/A")
        if current_price != "N/A":
            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Current Price:*\n{current_price}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Change:*\n{change}"
                    }
                ]
            })
            
        # Timestamp
        time_str = data.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"*Time:* {time_str}"
                }
            ]
        })
        
        return {"blocks": blocks}
        
    def _format_prediction_alert(self, data):
        """Format prediction alert notification"""
        blocks = []
        
        symbol = data.get("symbol", "Unknown")
        prediction = data.get("prediction", "neutral")
        
        # Emoji based on prediction
        if prediction.lower() == "up":
            emoji = "🔺"
        elif prediction.lower() == "down":
            emoji = "🔻"
        else:
            emoji = "➡️"
            
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} Prediction Alert: {symbol}",
                "emoji": True
            }
        })
        
        # Prediction details
        confidence = data.get("confidence", 0) * 100
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"New prediction for {symbol}: *{prediction.upper()}* with {confidence:.1f}% confidence."
            }
        })
        
        # Price targets
        blocks.append({
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Current Price:*\n{data.get('current_price', 'N/A')}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Target Price:*\n{data.get('target_price', 'N/A')}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Stop Loss:*\n{data.get('stop_loss', 'N/A')}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Timeframe:*\n{data.get('timeframe', 'N/A')}"
                }
            ]
        })
        
        # Supporting factors
        factors = data.get("supporting_factors", [])
        if factors:
            factors_text = "*Supporting Factors:*\n"
            for factor in factors:
                factors_text += f"• {factor}\n"
                
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": factors_text
                }
            })
            
        return {"blocks": blocks}
        
    def _format_generic_notification(self, notification_type, data):
        """Format generic notification"""
        blocks = []
        
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Notification: {notification_type.replace('_', ' ').capitalize()}",
                "emoji": True
            }
        })
        
        # Process each field in the data
        message = data.get("message", "")
        if message:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            })
            
        # Format remaining fields
        fields = []
        for key, value in data.items():
            if key in ["message", "time"]:  # Skip already processed fields
                continue
                
            fields.append({
                "type": "mrkdwn",
                "text": f"*{key.replace('_', ' ').capitalize()}:*\n{value}"
            })
            
        # Add fields in pairs
        while fields:
            field_pair = fields[:2]
            fields = fields[2:]
            
            blocks.append({
                "type": "section",
                "fields": field_pair
            })
            
        # Timestamp
        time_str = data.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"*Time:* {time_str}"
                }
            ]
        })
        
        return {"blocks": blocks}