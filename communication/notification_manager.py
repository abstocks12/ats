"""
Notification Manager for the Automated Trading System.
This module handles sending notifications to various channels.
"""

import logging
from datetime import datetime
import threading
import time
import queue

class NotificationManager:
    def __init__(self, db_connector, slack_connector=None):
        """
        Initialize the notification manager.
        
        Args:
            db_connector: Database connector
            slack_connector: SlackConnector instance
        """
        self.db = db_connector
        self.slack = slack_connector
        self.logger = logging.getLogger(__name__)
        
        # Queue for asynchronous notifications
        self.notification_queue = queue.Queue()
        self.running = False
        self.notification_thread = None
        
        # Formatter for Slack notifications
        if self.slack:
            from communication.slack.formatter import SlackFormatter
            self.slack_formatter = SlackFormatter()
            
    def start(self):
        """Start the notification manager"""
        if self.running:
            self.logger.warning("Notification manager is already running")
            return
            
        self.running = True
        self.notification_thread = threading.Thread(target=self._process_notification_queue)
        self.notification_thread.daemon = True
        self.notification_thread.start()
        self.logger.info("Notification manager started")
        
    def stop(self):
        """Stop the notification manager"""
        if not self.running:
            self.logger.warning("Notification manager is not running")
            return
            
        self.running = False
        # If thread is alive, wait for it to finish
        if self.notification_thread and self.notification_thread.is_alive():
            self.notification_thread.join(timeout=5)
        self.logger.info("Notification manager stopped")
        
    def _process_notification_queue(self):
        """Process notifications from the queue"""
        while self.running:
            try:
                # Get notification from queue with 1 second timeout
                try:
                    notification = self.notification_queue.get(timeout=1)
                except queue.Empty:
                    continue
                    
                # Process notification
                notification_type = notification.get("type")
                channels = notification.get("channels", [])
                data = notification.get("data", {})
                
                # Send to each channel
                for channel in channels:
                    if channel == "slack" and self.slack:
                        self._send_to_slack(notification_type, data)
                    elif channel == "database":
                        self._store_in_database(notification_type, data)
                        
                # Mark as done
                self.notification_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing notification: {e}")
                time.sleep(1)  # Avoid spinning on error
                
    def _send_to_slack(self, notification_type, data):
        """Send notification to Slack"""
        try:
            # Format notification for Slack
            formatted = self.slack_formatter.format_notification(notification_type, data)
            
            # Send to Slack
            channel_id = data.get("slack_channel_id", None)
            self.slack.send_message(
                text=f"Notification: {notification_type.replace('_', ' ').capitalize()}",
                channel_id=channel_id,
                blocks=formatted.get("blocks"),
                attachments=formatted.get("attachments")
            )
            
            self.logger.debug(f"Sent {notification_type} notification to Slack")
            
        except Exception as e:
            self.logger.error(f"Error sending notification to Slack: {e}")
            
    def _store_in_database(self, notification_type, data):
        """Store notification in database"""
        try:
            # Create notification record
            notification = {
                "type": notification_type,
                "data": data,
                "created_at": datetime.now(),
                "status": "sent"
            }
            
            # Store in database
            result = self.db.notifications_collection.insert_one(notification)
            self.logger.debug(f"Stored {notification_type} notification in database: {result.inserted_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing notification in database: {e}")
            
    def send_notification(self, notification_type, data, channels=None, async_send=True):
        """
        Send a notification.
        
        Args:
            notification_type (str): Type of notification
            data (dict): Notification data
            channels (list): List of channels to send notification to (default: ["slack", "database"])
            async_send (bool): Whether to send notification asynchronously
            
        Returns:
            bool: Success status
        """
        if channels is None:
            channels = ["slack", "database"]
            
        try:
            # Create notification object
            notification = {
                "type": notification_type,
                "channels": channels,
                "data": data,
                "timestamp": datetime.now()
            }
            
            # Send asynchronously or synchronously
            if async_send:
                self.notification_queue.put(notification)
                return True
            else:
                # Send immediately
                for channel in channels:
                    if channel == "slack" and self.slack:
                        self._send_to_slack(notification_type, data)
                    elif channel == "database":
                        self._store_in_database(notification_type, data)
                return True
                
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            return False
            
    def send_trade_notification(self, trade_data, async_send=True):
        """
        Send a trade execution notification.
        
        Args:
            trade_data (dict): Trade data
            async_send (bool): Whether to send notification asynchronously
            
        Returns:
            bool: Success status
        """
        return self.send_notification("trade_executed", trade_data, async_send=async_send)
        
    def send_system_alert(self, message, level="info", details=None, async_send=True):
        """
        Send a system alert notification.
        
        Args:
            message (str): Alert message
            level (str): Alert level (info, warning, error, critical)
            details (dict): Additional details
            async_send (bool): Whether to send notification asynchronously
            
        Returns:
            bool: Success status
        """
        data = {
            "message": message,
            "level": level,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if details:
            data["details"] = details
            
        return self.send_notification("system_alert", data, async_send=async_send)
        
    def send_price_alert(self, symbol, price, condition="reached", current_price=None, change=None, async_send=True):
        """
        Send a price alert notification.
        
        Args:
            symbol (str): Symbol
            price (float): Alert price
            condition (str): Alert condition (reached, above, below)
            current_price (float): Current price
            change (str): Price change description
            async_send (bool): Whether to send notification asynchronously
            
        Returns:
            bool: Success status
        """
        data = {
            "symbol": symbol,
            "price": price,
            "condition": condition,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if current_price is not None:
            data["current_price"] = current_price
        
        if change is not None:
            data["change"] = change
            
        return self.send_notification("price_alert", data, async_send=async_send)
        
    def send_prediction_alert(self, prediction_data, async_send=True):
        """
        Send a prediction alert notification.
        
        Args:
            prediction_data (dict): Prediction data
            async_send (bool): Whether to send notification asynchronously
            
        Returns:
            bool: Success status
        """
        data = dict(prediction_data)
        data["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return self.send_notification("prediction_alert", data, async_send=async_send)