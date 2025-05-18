"""
Slack Connector for the Automated Trading System.
This module handles the connection to Slack API and provides methods for sending messages
and receiving commands from Slack.
"""

import os
import time
import logging
import threading
import json
from datetime import datetime

import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from flask import Flask, request, jsonify

class SlackConnector:
    def __init__(self, token=None, channel_id=None, db_connector=None, port=3000):
        """
        Initialize the Slack connector with the provided token and channel ID.
        
        Args:
            token (str): Slack bot token (starts with xoxb-)
            channel_id (str): Default channel ID to send messages to
            db_connector: Database connector for storing messages
            port (int): Port for the Flask server to listen on for events
        """
        self.token = token or os.environ.get("SLACK_BOT_TOKEN")
        if not self.token:
            raise ValueError("Slack token is required. Set SLACK_BOT_TOKEN environment variable or pass token parameter.")
            
        self.channel_id = channel_id or os.environ.get("SLACK_CHANNEL_ID")
        self.client = WebClient(token=self.token)
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        self.port = port
        self.command_callbacks = {}
        
        # Set up Flask server for receiving events
        self.app = Flask(__name__)
        self.setup_routes()
        self.server_thread = None
        
        # Test connection
        try:
            response = self.client.auth_test()
            self.bot_id = response["user_id"]
            self.logger.info(f"Connected to Slack as {response['user']} in team {response['team']}")
        except SlackApiError as e:
            self.logger.error(f"Failed to connect to Slack: {e}")
            raise

    def setup_routes(self):
        """Set up Flask routes for Slack events"""
        @self.app.route('/slack/events', methods=['POST'])
        def slack_events():
            data = request.json
            
            # Verify Slack challenge during setup
            if "challenge" in data:
                return jsonify({"challenge": data["challenge"]})
                
            # Process event
            if "event" in data:
                self._process_event(data["event"])
                
            return jsonify({"status": "ok"})

    def _process_event(self, event):
        """Process Slack events"""
        # Process message events
        if event.get("type") == "message" and "subtype" not in event:
            user_id = event.get("user")
            channel_id = event.get("channel")
            text = event.get("text", "").strip()
            
            # Store message in database if available
            if self.db:
                try:
                    self.db.slack_messages.insert_one({
                        "user_id": user_id,
                        "channel_id": channel_id,
                        "text": text,
                        "timestamp": datetime.now()
                    })
                except Exception as e:
                    self.logger.error(f"Failed to store message in database: {e}")
            
            # Check for commands
            if text.startswith('/'):
                self._process_command(text, user_id, channel_id)
                
    def _process_command(self, text, user_id, channel_id):
        """Process commands from Slack messages"""
        parts = text.split()
        if not parts:
            return
            
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if command in self.command_callbacks:
            try:
                response = self.command_callbacks[command](args, user_id)
                if response:
                    self.send_message(response, channel_id)
            except Exception as e:
                error_msg = f"Error processing command {command}: {e}"
                self.logger.error(error_msg)
                self.send_message(f"Error: {error_msg}", channel_id)
    
    def register_command(self, command, callback):
        """
        Register a callback function for a specific command.
        
        Args:
            command (str): Command to register (without the leading /)
            callback (callable): Function to call when the command is received
        """
        if not command.startswith('/'):
            command = f'/{command}'
        self.command_callbacks[command] = callback
        self.logger.info(f"Registered command: {command}")
    
    def start_server(self):
        """Start the Flask server in a separate thread"""
        if self.server_thread is not None and self.server_thread.is_alive():
            self.logger.warning("Server is already running")
            return
            
        def run_server():
            self.app.run(host='0.0.0.0', port=self.port)
            
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        self.logger.info(f"Started Slack event server on port {self.port}")
    
    def stop_server(self):
        """Stop the Flask server"""
        if self.server_thread and self.server_thread.is_alive():
            # Terminate Flask server
            self.logger.info("Stopping Slack event server")
            requests.post(f"http://localhost:{self.port}/shutdown")
            self.server_thread.join(timeout=5)
            self.server_thread = None
    
    def send_message(self, text, channel_id=None, blocks=None, attachments=None):
        """
        Send a message to a Slack channel.
        
        Args:
            text (str): Message text
            channel_id (str): Channel ID to send message to (defaults to self.channel_id)
            blocks (list): Slack blocks for rich formatting
            attachments (list): Slack attachments
            
        Returns:
            dict: Response from Slack API
        """
        try:
            channel = channel_id or self.channel_id
            if not channel:
                raise ValueError("Channel ID is required for sending messages")
                
            response = self.client.chat_postMessage(
                channel=channel,
                text=text,
                blocks=blocks,
                attachments=attachments
            )
            
            # Store message in database if available
            if self.db:
                try:
                    self.db.sent_messages.insert_one({
                        "channel_id": channel,
                        "text": text,
                        "timestamp": datetime.now(),
                        "message_ts": response["ts"]
                    })
                except Exception as e:
                    self.logger.error(f"Failed to store sent message in database: {e}")
                    
            return response
        except SlackApiError as e:
            self.logger.error(f"Failed to send message: {e}")
            return None
    
    def upload_file(self, file_path, title=None, channels=None, initial_comment=None):
        """
        Upload a file to Slack.
        
        Args:
            file_path (str): Path to the file to upload
            title (str): Title of the file
            channels (str or list): Channel ID(s) to share the file with
            initial_comment (str): Initial comment to add to the file
            
        Returns:
            dict: Response from Slack API
        """
        try:
            channels_str = channels or self.channel_id
            if isinstance(channels, list):
                channels_str = ",".join(channels)
                
            response = self.client.files_upload(
                file=file_path,
                title=title,
                channels=channels_str,
                initial_comment=initial_comment
            )
            return response
        except SlackApiError as e:
            self.logger.error(f"Failed to upload file: {e}")
            return None
            
    def update_status(self, status_text, status_emoji=None):
        """
        Update the bot's status.
        
        Args:
            status_text (str): Status text
            status_emoji (str): Status emoji (e.g., ":chart_with_upwards_trend:")
            
        Returns:
            dict: Response from Slack API
        """
        try:
            response = self.client.users_profile_set(
                profile={
                    "status_text": status_text,
                    "status_emoji": status_emoji or "",
                }
            )
            return response
        except SlackApiError as e:
            self.logger.error(f"Failed to update status: {e}")
            return None
            
    def get_user_info(self, user_id):
        """
        Get information about a Slack user.
        
        Args:
            user_id (str): User ID
            
        Returns:
            dict: User information
        """
        try:
            response = self.client.users_info(user=user_id)
            return response["user"]
        except SlackApiError as e:
            self.logger.error(f"Failed to get user info: {e}")
            return None