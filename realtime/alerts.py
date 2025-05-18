# realtime/alerts.py
import logging
import threading
import time
from datetime import datetime, timedelta
import json
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class AlertManager:
    """
    Manages and dispatches trading alerts to various channels.
    """
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the alert manager.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # State
        self.is_running = False
        self.processing_thread = None
        
        # Alert handlers
        self.handlers = {}
        
        # Alert buffer
        self.alert_buffer = []
        
        # Configuration
        self.config = {
            'buffer_flush_interval': 60,  # Seconds between buffer flushes
            'max_buffer_size': 100,  # Maximum size of alert buffer
            'min_alert_interval': 300,  # Minimum interval between similar alerts (seconds)
            'notification_methods': ['database', 'log'],  # Default notification methods
            'email_config': {
                'enabled': False,
                'server': '',
                'port': 587,
                'username': '',
                'password': '',
                'from_address': '',
                'to_addresses': []
            },
            'sms_config': {
                'enabled': False,
                'provider': '',
                'api_key': '',
                'from_number': '',
                'to_numbers': []
            },
            'webhook_config': {
                'enabled': False,
                'url': '',
                'headers': {},
                'authentication': {}
            },
            'whatsapp_config': {
                'enabled': False,
                'api_key': '',
                'from_number': '',
                'to_numbers': []
            },
            'alert_throttling': True,  # Enable alert throttling
            'severity_thresholds': {
                'LOW': 0,  # All low severity alerts
                'MEDIUM': 3,  # Max 3 medium alerts per hour
                'HIGH': 10  # Max 10 high alerts per hour
            },
            'alert_grouping': True,  # Group similar alerts
            'alert_retention_days': 30  # Days to retain alerts in database
        }
        
        # Register default handlers
        self._register_default_handlers()
        
        self.logger.info("Alert manager initialized")
    
    def set_config(self, config):
        """
        Set alert manager configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        # Update top-level config
        for key, value in config.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                # Merge nested dict
                self.config[key].update(value)
            else:
                # Replace value
                self.config[key] = value
                
        self.logger.info(f"Updated alert manager configuration")
    
    def _register_default_handlers(self):
        """
        Register default alert handlers.
        """
        # Database handler
        self.register_handler('database', self._database_handler)
        
        # Log handler
        self.register_handler('log', self._log_handler)
    
    def register_handler(self, name, handler):
        """
        Register an alert handler.
        
        Args:
            name (str): Handler name
            handler: Handler function that accepts alert dict
            
        Returns:
            bool: Success status
        """
        if name in self.handlers:
            self.logger.warning(f"Handler {name} already registered, overwriting")
            
        if not callable(handler):
            self.logger.error(f"Handler {name} is not callable")
            return False
            
        self.handlers[name] = handler
        self.logger.info(f"Registered alert handler: {name}")
        
        return True
    
    def unregister_handler(self, name):
        """
        Unregister an alert handler.
        
        Args:
            name (str): Handler name
            
        Returns:
            bool: Success status
        """
        if name not in self.handlers:
            self.logger.warning(f"Handler {name} not found")
            return False
            
        del self.handlers[name]
        self.logger.info(f"Unregistered alert handler: {name}")
        
        return True
    
    def start(self):
        """
        Start the alert manager.
        
        Returns:
            bool: Success status
        """
        if self.is_running:
            self.logger.warning("Alert manager is already running")
            return False
            
        # Register additional handlers based on config
        self._register_configured_handlers()
        
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Alert manager started")
        
        return True
    
    def stop(self, wait=True):
        """
        Stop the alert manager.
        
        Args:
            wait (bool): Wait for processing thread to complete
            
        Returns:
            bool: Success status
        """
        if not self.is_running:
            self.logger.warning("Alert manager is not running")
            return False
            
        self.is_running = False
        
        if wait and self.processing_thread:
            self.processing_thread.join(timeout=30)
            
        self.logger.info("Alert manager stopped")
        
        return True
    
    def _register_configured_handlers(self):
        """
        Register handlers based on configuration.
        """
        # Email handler
        if self.config['email_config']['enabled']:
            self.register_handler('email', self._email_handler)
            
            # Add to notification methods if not already included
            if 'email' not in self.config['notification_methods']:
                self.config['notification_methods'].append('email')
        
        # SMS handler
        if self.config['sms_config']['enabled']:
            self.register_handler('sms', self._sms_handler)
            
            # Add to notification methods if not already included
            if 'sms' not in self.config['notification_methods']:
                self.config['notification_methods'].append('sms')
        
        # Webhook handler
        if self.config['webhook_config']['enabled']:
            self.register_handler('webhook', self._webhook_handler)
            
            # Add to notification methods if not already included
            if 'webhook' not in self.config['notification_methods']:
                self.config['notification_methods'].append('webhook')
        
        # WhatsApp handler
        if self.config['whatsapp_config']['enabled']:
            self.register_handler('whatsapp', self._whatsapp_handler)
            
            # Add to notification methods if not already included
            if 'whatsapp' not in self.config['notification_methods']:
                self.config['notification_methods'].append('whatsapp')
    
    def _processing_loop(self):
        """
        Main processing loop for alerts.
        """
        last_flush_time = time.time()
        
        while self.is_running:
            try:
                now = time.time()
                
                # Flush buffer if interval elapsed or buffer is full
                if (now - last_flush_time >= self.config['buffer_flush_interval'] or 
                    len(self.alert_buffer) >= self.config['max_buffer_size']):
                    self._flush_alert_buffer()
                    last_flush_time = now
                
                # Clean up old alerts periodically
                if now - last_flush_time >= 86400:  # Once per day
                    self._cleanup_old_alerts()
                    
                # Sleep briefly
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in alert processing loop: {e}")
                time.sleep(60)  # Sleep longer on error
    
    def _flush_alert_buffer(self):
        """
        Flush the alert buffer.
        """
        if not self.alert_buffer:
            return
            
        try:
            # Process alerts
            for alert in self.alert_buffer:
                # Send to each configured notification method
                for method in self.config['notification_methods']:
                    if method in self.handlers:
                        try:
                            self.handlers[method](alert)
                        except Exception as e:
                            self.logger.error(f"Error in alert handler {method}: {e}")
            
            # Clear buffer
            self.alert_buffer = []
            
        except Exception as e:
            self.logger.error(f"Error flushing alert buffer: {e}")
    
    def _cleanup_old_alerts(self):
        """
        Clean up old alerts from database.
        """
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.config['alert_retention_days'])
            
            # Delete old alerts
            result = self.db.alerts_collection.delete_many({
                'timestamp': {'$lt': cutoff_date}
            })
            
            if result.deleted_count > 0:
                self.logger.info(f"Cleaned up {result.deleted_count} old alerts")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old alerts: {e}")
    
    def add_alert(self, alert):
        """
        Add an alert to the buffer.
        
        Args:
            alert (dict): Alert data
            
        Returns:
            bool: Success status
        """
        try:
            # Validate alert
            if not isinstance(alert, dict):
                self.logger.error("Alert must be a dictionary")
                return False
                
            # Required fields
            required_fields = ['type', 'message', 'severity']
            
            for field in required_fields:
                if field not in alert:
                    self.logger.error(f"Alert missing required field: {field}")
                    return False
            
            # Ensure timestamp
            if 'timestamp' not in alert:
                alert['timestamp'] = datetime.now()
                
            # Check for throttling
            if self.config['alert_throttling'] and self._should_throttle_alert(alert):
                self.logger.info(f"Throttling alert of type {alert['type']}")
                return False
                
            # Check for grouping
            if self.config['alert_grouping']:
                grouped_alert = self._group_similar_alerts(alert)
                
                if grouped_alert is None:
                    # Alert was grouped, no need to add it
                    return True
                    
                # Use grouped alert
                alert = grouped_alert
            
            # Add to buffer
            self.alert_buffer.append(alert)
            
            # Flush immediately for high severity
            if alert.get('severity') == 'HIGH':
                self._flush_alert_buffer()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding alert: {e}")
            return False
    
    def _should_throttle_alert(self, alert):
        """
        Check if an alert should be throttled.
        
        Args:
            alert (dict): Alert data
            
        Returns:
            bool: True if alert should be throttled
        """
        try:
            # Get alert type and severity
            alert_type = alert.get('type')
            severity = alert.get('severity', 'MEDIUM')
            
            # Check if we have a threshold for this severity
            if severity not in self.config['severity_thresholds']:
                return False
                
            threshold = self.config['severity_thresholds'][severity]
            
            # 0 means no throttling
            if threshold == 0:
                return False
                
            # Check recent alerts of same type
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            count = self.db.alerts_collection.count_documents({
                'type': alert_type,
                'severity': severity,
                'timestamp': {'$gte': one_hour_ago}
            })
            
            return count >= threshold
            
        except Exception as e:
            self.logger.error(f"Error checking alert throttling: {e}")
            return False
    
    def _group_similar_alerts(self, alert):
        """
        Group similar alerts.
        
        Args:
            alert (dict): Alert data
            
        Returns:
            dict: Grouped alert or None if alert was added to an existing group
        """
        try:
            # Get alert type
            alert_type = alert.get('type')
            
            # Check for recent similar alerts
            min_interval = self.config['min_alert_interval']
            cutoff_time = datetime.now() - timedelta(seconds=min_interval)
            
            # Query for similar alerts
            query = {
                'type': alert_type,
                'timestamp': {'$gte': cutoff_time}
            }
            
            # Add additional grouping criteria based on alert type
            if alert_type == 'MAX_DRAWDOWN' and 'details' in alert:
                # Group drawdown alerts by threshold
                threshold = alert.get('details', {}).get('threshold')
                
                if threshold:
                    query['details.threshold'] = threshold
            elif alert_type == 'MAX_POSITION_SIZE' and 'details' in alert:
                # Group position size alerts by symbol
                symbol = alert.get('details', {}).get('symbol')
                
                if symbol:
                    query['details.symbol'] = symbol
            elif alert_type == 'MAX_SECTOR_EXPOSURE' and 'details' in alert:
                # Group sector exposure alerts by sector
                sector = alert.get('details', {}).get('sector')
                
                if sector:
                    query['details.sector'] = sector
            
            # Look for existing group
            existing = self.db.alert_groups_collection.find_one(query)
            
            if existing:
                # Update existing group
                group_id = existing.get('_id')
                
                # Increment count
                count = existing.get('count', 1) + 1
                
                # Update group
                self.db.alert_groups_collection.update_one(
                    {'_id': group_id},
                    {
                        '$set': {
                            'last_timestamp': datetime.now(),
                            'count': count,
                            'last_message': alert.get('message')
                        }
                    }
                )
                
                # Original alert was grouped, no need to add it
                return None
            else:
                # Create new group
                group = {
                    'type': alert_type,
                    'severity': alert.get('severity'),
                    'first_timestamp': datetime.now(),
                    'last_timestamp': datetime.now(),
                    'count': 1,
                    'first_message': alert.get('message'),
                    'last_message': alert.get('message')
                }
                
                # Add details if present
                if 'details' in alert:
                    group['details'] = alert.get('details')
                    
                # Insert group
                self.db.alert_groups_collection.insert_one(group)
                
                # Return original alert
                return alert
                
        except Exception as e:
            self.logger.error(f"Error grouping alerts: {e}")
            return alert  # Return original alert on error
    
    def _database_handler(self, alert):
        """
        Handle alert by storing in database.
        
        Args:
            alert (dict): Alert data
        """
        try:
            # Insert into database
            self.db.alerts_collection.insert_one(alert)
            
        except Exception as e:
            self.logger.error(f"Error storing alert in database: {e}")
    
    def _log_handler(self, alert):
        """
        Handle alert by logging.
        
        Args:
            alert (dict): Alert data
        """
        try:
            # Get severity and message
            severity = alert.get('severity', 'MEDIUM')
            message = alert.get('message', '')
            
            # Log with appropriate level
            if severity == 'HIGH':
                self.logger.error(f"ALERT: {message}")
            elif severity == 'MEDIUM':
                self.logger.warning(f"ALERT: {message}")
            else:
                self.logger.info(f"ALERT: {message}")
                
        except Exception as e:
            self.logger.error(f"Error logging alert: {e}")
    
    def _email_handler(self, alert):
        """
        Handle alert by sending email.
        
        Args:
            alert (dict): Alert data
        """
        try:
            # Check if enabled
            if not self.config['email_config']['enabled']:
                return
                
            # Get email config
            config = self.config['email_config']
            
            # Only send HIGH and MEDIUM severity alerts via email
            severity = alert.get('severity', 'MEDIUM')
            
            if severity == 'LOW':
                return
                
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = config['from_address']
            msg['To'] = ', '.join(config['to_addresses'])
            msg['Subject'] = f"Trading Alert: {alert.get('type', 'Alert')} - {severity}"
            
            # Create email body
            body = f"""
            <html>
            <body>
                <h2>Trading Alert</h2>
                <p><strong>Type:</strong> {alert.get('type', 'Alert')}</p>
                <p><strong>Severity:</strong> {severity}</p>
                <p><strong>Message:</strong> {alert.get('message', '')}</p>
                <p><strong>Time:</strong> {alert.get('timestamp', datetime.now())}</p>
            """
            
            # Add details if available
            if 'details' in alert:
                body += "<h3>Details:</h3><ul>"
                
                for key, value in alert['details'].items():
                    body += f"<li><strong>{key}:</strong> {value}</li>"
                    
                body += "</ul>"
                
            body += """
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(config['server'], config['port'])
            server.starttls()
            server.login(config['username'], config['password'])
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Sent email alert: {alert.get('type')}")
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
    
    def _sms_handler(self, alert):
        """
        Handle alert by sending SMS.
        
        Args:
            alert (dict): Alert data
        """
        try:
            # Check if enabled
            if not self.config['sms_config']['enabled']:
                return
                
            # Get SMS config
            config = self.config['sms_config']
            
            # Only send HIGH severity alerts via SMS
            severity = alert.get('severity', 'MEDIUM')
            
            if severity != 'HIGH':
                return
                
            # Create SMS message
            message = f"ALERT: {alert.get('type', 'Alert')} - {alert.get('message', '')}"
            
            # Truncate if too long
            if len(message) > 160:
                message = message[:157] + "..."
                
            # Send SMS based on provider
            provider = config['provider'].lower()
            
            if provider == 'twilio':
                self._send_twilio_sms(message, config)
            elif provider == 'aws_sns':
                self._send_aws_sns(message, config)
            else:
                self.logger.error(f"Unsupported SMS provider: {provider}")
                
        except Exception as e:
            self.logger.error(f"Error sending SMS alert: {e}")
    
    def _send_twilio_sms(self, message, config):
        """
        Send SMS via Twilio.
        
        Args:
            message (str): Message to send
            config (dict): SMS configuration
        """
        try:
            # Import twilio client
            from twilio.rest import Client
            
            # Create client
            # Create client
            client = Client(config.get('account_sid'), config.get('api_key'))
            
            # Send message to each recipient
            for to_number in config['to_numbers']:
                client.messages.create(
                    body=message,
                    from_=config['from_number'],
                    to=to_number
                )
                
            self.logger.info(f"Sent Twilio SMS alert to {len(config['to_numbers'])} recipients")
            
        except ImportError:
            self.logger.error("Twilio library not installed. Install with 'pip install twilio'")
        except Exception as e:
            self.logger.error(f"Error sending Twilio SMS: {e}")
    
    def _send_aws_sns(self, message, config):
        """
        Send SMS via AWS SNS.
        
        Args:
            message (str): Message to send
            config (dict): SMS configuration
        """
        try:
            # Import boto3
            import boto3
            
            # Create SNS client
            sns = boto3.client(
                'sns',
                aws_access_key_id=config.get('aws_access_key_id'),
                aws_secret_access_key=config.get('aws_secret_access_key'),
                region_name=config.get('region_name', 'us-east-1')
            )
            
            # Send message to each recipient
            for to_number in config['to_numbers']:
                sns.publish(
                    PhoneNumber=to_number,
                    Message=message,
                    MessageAttributes={
                        'AWS.SNS.SMS.SenderID': {
                            'DataType': 'String',
                            'StringValue': 'TRADING'
                        },
                        'AWS.SNS.SMS.SMSType': {
                            'DataType': 'String',
                            'StringValue': 'Transactional'
                        }
                    }
                )
                
            self.logger.info(f"Sent AWS SNS SMS alert to {len(config['to_numbers'])} recipients")
            
        except ImportError:
            self.logger.error("Boto3 library not installed. Install with 'pip install boto3'")
        except Exception as e:
            self.logger.error(f"Error sending AWS SNS SMS: {e}")
    
    def _webhook_handler(self, alert):
        """
        Handle alert by sending to webhook.
        
        Args:
            alert (dict): Alert data
        """
        try:
            # Check if enabled
            if not self.config['webhook_config']['enabled']:
                return
                
            # Get webhook config
            config = self.config['webhook_config']
            
            # Create payload
            payload = {
                'alert_type': alert.get('type', 'Alert'),
                'severity': alert.get('severity', 'MEDIUM'),
                'message': alert.get('message', ''),
                'timestamp': str(alert.get('timestamp', datetime.now())),
                'details': alert.get('details', {})
            }
            
            # Get headers
            headers = config.get('headers', {})
            
            # Add content type if not specified
            if 'Content-Type' not in headers:
                headers['Content-Type'] = 'application/json'
                
            # Add authentication if specified
            auth = None
            auth_config = config.get('authentication', {})
            
            auth_type = auth_config.get('type', '').lower()
            
            if auth_type == 'basic':
                auth = (auth_config.get('username', ''), auth_config.get('password', ''))
            elif auth_type == 'bearer':
                if 'Authorization' not in headers:
                    headers['Authorization'] = f"Bearer {auth_config.get('token', '')}"
            
            # Send request
            response = requests.post(
                config['url'],
                json=payload,
                headers=headers,
                auth=auth,
                timeout=10
            )
            
            # Check response
            if response.status_code >= 200 and response.status_code < 300:
                self.logger.info(f"Sent webhook alert: {response.status_code}")
            else:
                self.logger.error(f"Webhook error: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}")
    
    def _whatsapp_handler(self, alert):
        """
        Handle alert by sending WhatsApp message.
        
        Args:
            alert (dict): Alert data
        """
        try:
            # Check if enabled
            if not self.config['whatsapp_config']['enabled']:
                return
                
            # Get WhatsApp config
            config = self.config['whatsapp_config']
            
            # Only send HIGH and MEDIUM severity alerts via WhatsApp
            severity = alert.get('severity', 'MEDIUM')
            
            if severity == 'LOW':
                return
                
            # Create message
            message = f"*TRADING ALERT*\n\n"
            message += f"*Type:* {alert.get('type', 'Alert')}\n"
            message += f"*Severity:* {severity}\n"
            message += f"*Message:* {alert.get('message', '')}\n"
            message += f"*Time:* {alert.get('timestamp', datetime.now())}\n"
            
            # Add details if available
            if 'details' in alert:
                message += "\n*Details:*\n"
                
                for key, value in alert['details'].items():
                    message += f"- *{key}:* {value}\n"
            
            # Use Twilio for WhatsApp if configured
            if config.get('provider', '').lower() == 'twilio':
                self._send_twilio_whatsapp(message, config)
            else:
                self.logger.error(f"Unsupported WhatsApp provider: {config.get('provider')}")
                
        except Exception as e:
            self.logger.error(f"Error sending WhatsApp alert: {e}")
    
    def _send_twilio_whatsapp(self, message, config):
        """
        Send WhatsApp message via Twilio.
        
        Args:
            message (str): Message to send
            config (dict): WhatsApp configuration
        """
        try:
            # Import twilio client
            from twilio.rest import Client
            
            # Create client
            client = Client(config.get('account_sid'), config.get('api_key'))
            
            # Format numbers for WhatsApp
            from_number = f"whatsapp:{config['from_number']}"
            
            # Send message to each recipient
            for to_number in config['to_numbers']:
                to_whatsapp = f"whatsapp:{to_number}"
                
                client.messages.create(
                    body=message,
                    from_=from_number,
                    to=to_whatsapp
                )
                
            self.logger.info(f"Sent WhatsApp alert to {len(config['to_numbers'])} recipients")
            
        except ImportError:
            self.logger.error("Twilio library not installed. Install with 'pip install twilio'")
        except Exception as e:
            self.logger.error(f"Error sending WhatsApp message: {e}")
    
    def create_alert(self, alert_type, message, severity='MEDIUM', details=None):
        """
        Create and add an alert.
        
        Args:
            alert_type (str): Alert type
            message (str): Alert message
            severity (str): Alert severity (HIGH, MEDIUM, LOW)
            details (dict): Additional details
            
        Returns:
            bool: Success status
        """
        # Create alert dict
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now()
        }
        
        # Add details if provided
        if details:
            alert['details'] = details
            
        # Add alert
        return self.add_alert(alert)
    
    def create_price_alert(self, symbol, exchange, condition, target_price, current_price=None):
        """
        Create a price alert.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            condition (str): Condition (ABOVE, BELOW)
            target_price (float): Target price
            current_price (float): Current price (optional)
            
        Returns:
            bool: Success status
        """
        # Format condition
        condition = condition.upper()
        
        if condition not in ['ABOVE', 'BELOW']:
            self.logger.error(f"Invalid price alert condition: {condition}")
            return False
            
        # Create message
        message = f"{symbol} price is {condition.lower()} {target_price}"
        
        if current_price:
            message += f" (current: {current_price})"
            
        # Create details
        details = {
            'symbol': symbol,
            'exchange': exchange,
            'condition': condition,
            'target_price': target_price,
            'current_price': current_price
        }
        
        # Create and add alert
        return self.create_alert(
            alert_type='PRICE_ALERT',
            message=message,
            severity='MEDIUM',
            details=details
        )
    
    def create_volume_alert(self, symbol, exchange, condition, target_volume, current_volume=None):
        """
        Create a volume alert.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            condition (str): Condition (ABOVE, BELOW)
            target_volume (float): Target volume
            current_volume (float): Current volume (optional)
            
        Returns:
            bool: Success status
        """
        # Format condition
        condition = condition.upper()
        
        if condition not in ['ABOVE', 'BELOW']:
            self.logger.error(f"Invalid volume alert condition: {condition}")
            return False
            
        # Create message
        message = f"{symbol} volume is {condition.lower()} {target_volume}"
        
        if current_volume:
            message += f" (current: {current_volume})"
            
        # Create details
        details = {
            'symbol': symbol,
            'exchange': exchange,
            'condition': condition,
            'target_volume': target_volume,
            'current_volume': current_volume
        }
        
        # Create and add alert
        return self.create_alert(
            alert_type='VOLUME_ALERT',
            message=message,
            severity='MEDIUM',
            details=details
        )
    
    def create_technical_alert(self, symbol, exchange, indicator, value, condition, threshold):
        """
        Create a technical indicator alert.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            indicator (str): Technical indicator
            value (float): Current indicator value
            condition (str): Condition (ABOVE, BELOW, CROSSES_ABOVE, CROSSES_BELOW)
            threshold (float): Threshold value
            
        Returns:
            bool: Success status
        """
        # Format condition
        condition = condition.upper()
        
        if condition not in ['ABOVE', 'BELOW', 'CROSSES_ABOVE', 'CROSSES_BELOW']:
            self.logger.error(f"Invalid technical alert condition: {condition}")
            return False
            
        # Format condition for message
        if condition == 'CROSSES_ABOVE':
            condition_text = 'crossed above'
        elif condition == 'CROSSES_BELOW':
            condition_text = 'crossed below'
        else:
            condition_text = condition.lower()
            
        # Create message
        message = f"{symbol} {indicator} ({value:.2f}) {condition_text} {threshold:.2f}"
            
        # Create details
        details = {
            'symbol': symbol,
            'exchange': exchange,
            'indicator': indicator,
            'value': value,
            'condition': condition,
            'threshold': threshold
        }
        
        # Create and add alert
        return self.create_alert(
            alert_type='TECHNICAL_ALERT',
            message=message,
            severity='MEDIUM',
            details=details
        )
    
    def create_position_alert(self, symbol, exchange, position_type, condition, current_pnl=None):
        """
        Create a position alert.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            position_type (str): Position type (LONG, SHORT)
            condition (str): Condition (PROFIT_TARGET, STOP_LOSS, TRAILING_STOP)
            current_pnl (float): Current P&L (optional)
            
        Returns:
            bool: Success status
        """
        # Format condition
        condition = condition.upper()
        
        if condition not in ['PROFIT_TARGET', 'STOP_LOSS', 'TRAILING_STOP']:
            self.logger.error(f"Invalid position alert condition: {condition}")
            return False
            
        # Format condition for message
        if condition == 'PROFIT_TARGET':
            condition_text = 'reached profit target'
            severity = 'MEDIUM'
        elif condition == 'STOP_LOSS':
            condition_text = 'hit stop loss'
            severity = 'HIGH'
        else:
            condition_text = 'hit trailing stop'
            severity = 'MEDIUM'
            
        # Create message
        message = f"{symbol} {position_type.lower()} position {condition_text}"
        
        if current_pnl is not None:
            message += f" (P&L: {current_pnl:.2f})"
            
        # Create details
        details = {
            'symbol': symbol,
            'exchange': exchange,
            'position_type': position_type,
            'condition': condition,
            'current_pnl': current_pnl
        }
        
        # Create and add alert
        return self.create_alert(
            alert_type='POSITION_ALERT',
            message=message,
            severity=severity,
            details=details
        )
    
    def create_system_alert(self, component, status, message, severity='MEDIUM'):
        """
        Create a system alert.
        
        Args:
            component (str): System component
            status (str): Component status
            message (str): Alert message
            severity (str): Alert severity
            
        Returns:
            bool: Success status
        """
        # Create full message
        full_message = f"System {component}: {status} - {message}"
            
        # Create details
        details = {
            'component': component,
            'status': status
        }
        
        # Create and add alert
        return self.create_alert(
            alert_type='SYSTEM_ALERT',
            message=full_message,
            severity=severity,
            details=details
        )
    
    def get_alerts(self, count=10, alert_type=None, severity=None):
        """
        Get recent alerts.
        
        Args:
            count (int): Number of alerts to retrieve
            alert_type (str): Filter by alert type
            severity (str): Filter by severity
            
        Returns:
            list: Recent alerts
        """
        try:
            # Build query
            query = {}
            
            if alert_type:
                query['type'] = alert_type
                
            if severity:
                query['severity'] = severity
                
            # Query database
            cursor = self.db.alerts_collection.find(query).sort('timestamp', -1).limit(count)
            
            # Convert to list
            alerts = list(cursor)
            
            # Remove MongoDB IDs
            for alert in alerts:
                if '_id' in alert:
                    del alert['_id']
                    
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting alerts: {e}")
            return []
    
    def get_alert_groups(self, count=10, alert_type=None, severity=None):
        """
        Get alert groups.
        
        Args:
            count (int): Number of groups to retrieve
            alert_type (str): Filter by alert type
            severity (str): Filter by severity
            
        Returns:
            list: Alert groups
        """
        try:
            # Build query
            query = {}
            
            if alert_type:
                query['type'] = alert_type
                
            if severity:
                query['severity'] = severity
                
            # Query database
            cursor = self.db.alert_groups_collection.find(query).sort('last_timestamp', -1).limit(count)
            
            # Convert to list
            groups = list(cursor)
            
            # Remove MongoDB IDs
            for group in groups:
                if '_id' in group:
                    del group['_id']
                    
            return groups
            
        except Exception as e:
            self.logger.error(f"Error getting alert groups: {e}")
            return []
    
    def get_alert_statistics(self, days=7):
        """
        Get alert statistics.
        
        Args:
            days (int): Number of days to include
            
        Returns:
            dict: Alert statistics
        """
        try:
            # Calculate start date
            start_date = datetime.now() - timedelta(days=days)
            
            # Query database
            pipeline = [
                {
                    '$match': {
                        'timestamp': {'$gte': start_date}
                    }
                },
                {
                    '$group': {
                        '_id': {
                            'type': '$type',
                            'severity': '$severity',
                            'day': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$timestamp'}}
                        },
                        'count': {'$sum': 1}
                    }
                },
                {
                    '$sort': {
                        '_id.day': 1,
                        '_id.type': 1
                    }
                }
            ]
            
            cursor = self.db.alerts_collection.aggregate(pipeline)
            
            # Process results
            results = list(cursor)
            
            # Create statistics
            statistics = {
                'by_type': {},
                'by_severity': {},
                'by_day': {},
                'total': 0
            }
            
            for result in results:
                # Extract fields
                alert_type = result['_id']['type']
                severity = result['_id']['severity']
                day = result['_id']['day']
                count = result['count']
                
                # Update total
                statistics['total'] += count
                
                # Update by type
                if alert_type not in statistics['by_type']:
                    statistics['by_type'][alert_type] = 0
                    
                statistics['by_type'][alert_type] += count
                
                # Update by severity
                if severity not in statistics['by_severity']:
                    statistics['by_severity'][severity] = 0
                    
                statistics['by_severity'][severity] += count
                
                # Update by day
                if day not in statistics['by_day']:
                    statistics['by_day'][day] = {
                        'total': 0,
                        'by_type': {},
                        'by_severity': {}
                    }
                    
                statistics['by_day'][day]['total'] += count
                
                # Update day by type
                if alert_type not in statistics['by_day'][day]['by_type']:
                    statistics['by_day'][day]['by_type'][alert_type] = 0
                    
                statistics['by_day'][day]['by_type'][alert_type] += count
                
                # Update day by severity
                if severity not in statistics['by_day'][day]['by_severity']:
                    statistics['by_day'][day]['by_severity'][severity] = 0
                    
                statistics['by_day'][day]['by_severity'][severity] += count
                
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error getting alert statistics: {e}")
            return {
                'by_type': {},
                'by_severity': {},
                'by_day': {},
                'total': 0
            }