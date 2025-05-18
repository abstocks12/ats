"""
Conversation Manager for the Automated Trading System.
This module handles interactive conversations with users.
"""

import logging
import re
from datetime import datetime
import threading
import time
import queue

class ConversationManager:
    def __init__(self, db_connector, slack_connector, trading_controller=None, portfolio_manager=None):
        """
        Initialize the conversation manager.
        
        Args:
            db_connector: Database connector
            slack_connector: SlackConnector instance
            trading_controller: TradingController instance
            portfolio_manager: PortfolioManager instance
        """
        self.db = db_connector
        self.slack = slack_connector
        self.trading_controller = trading_controller
        self.portfolio_manager = portfolio_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize command processor
        from communication.slack.command_processor import CommandProcessor
        self.command_processor = CommandProcessor(
            db_connector=self.db,
            slack_connector=self.slack,
            trading_controller=self.trading_controller,
            portfolio_manager=self.portfolio_manager
        )
        
        # Active conversations
        self.active_conversations = {}
        
        # Register message handler
        self._register_message_handler()
        
    def _register_message_handler(self):
        """Register a message handler with Slack connector"""
        if not self.slack:
            self.logger.warning("Slack connector not available")
            return
            
        # TODO: Implement more comprehensive message handling
        
    def start_conversation(self, user_id, conversation_type, initial_data=None):
        """
        Start a new conversation with a user.
        
        Args:
            user_id (str): User ID
            conversation_type (str): Type of conversation
            initial_data (dict): Initial data for the conversation
            
        Returns:
            str: Conversation ID or None if failed
        """
        try:
            # Create conversation object
            conversation_id = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            conversation = {
                "id": conversation_id,
                "user_id": user_id,
                "type": conversation_type,
                "data": initial_data or {},
                "state": "started",
                "start_time": datetime.now(),
                "last_update": datetime.now(),
                "messages": []
            }
            
            # Store in active conversations
            self.active_conversations[conversation_id] = conversation
            
            # Store in database
            try:
                self.db.conversations_collection.insert_one(conversation)
            except Exception as e:
                self.logger.error(f"Error storing conversation in database: {e}")
                
            return conversation_id
            
        except Exception as e:
            self.logger.error(f"Error starting conversation: {e}")
            return None
            
    def end_conversation(self, conversation_id):
        """
        End a conversation.
        
        Args:
            conversation_id (str): Conversation ID
            
        Returns:
            bool: Success status
        """
        try:
            # Check if conversation exists
            if conversation_id not in self.active_conversations:
                self.logger.warning(f"Conversation {conversation_id} not found")
                return False
                
            # Update conversation state
            conversation = self.active_conversations[conversation_id]
            conversation["state"] = "ended"
            conversation["end_time"] = datetime.now()
            
            # Update in database
            try:
                self.db.conversations_collection.update_one(
                    {"id": conversation_id},
                    {"$set": {
                        "state": "ended",
                        "end_time": conversation["end_time"]
                    }}
                )
            except Exception as e:
                self.logger.error(f"Error updating conversation in database: {e}")
                
            # Remove from active conversations
            del self.active_conversations[conversation_id]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error ending conversation: {e}")
            return False
            
    def add_message(self, conversation_id, sender, message, metadata=None):
        """
        Add a message to a conversation.
        
        Args:
            conversation_id (str): Conversation ID
            sender (str): Sender ID
            message (str): Message text
            metadata (dict): Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Check if conversation exists
            if conversation_id not in self.active_conversations:
                self.logger.warning(f"Conversation {conversation_id} not found")
                return False
                
            # Create message object
            msg = {
                "sender": sender,
                "text": message,
                "timestamp": datetime.now(),
                "metadata": metadata or {}
            }
            
            # Add to conversation
            conversation = self.active_conversations[conversation_id]
            conversation["messages"].append(msg)
            conversation["last_update"] = datetime.now()
            
            # Update in database
            try:
                self.db.conversations_collection.update_one(
                    {"id": conversation_id},
                    {"$push": {"messages": msg},
                     "$set": {"last_update": conversation["last_update"]}}
                )
            except Exception as e:
                self.logger.error(f"Error updating conversation in database: {e}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding message to conversation: {e}")
            return False
            
    def process_message(self, user_id, message, channel_id=None):
        """
        Process a message from a user.
        
        Args:
            user_id (str): User ID
            message (str): Message text
            channel_id (str): Channel ID
            
        Returns:
            str: Response message or None if no response
        """
        try:
            # Check for command
            if message.startswith('/'):
                # Extract command and args
                parts = message.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                # Handle command
                if hasattr(self.command_processor, f"handle_{command[1:]}_command"):
                    handler = getattr(self.command_processor, f"handle_{command[1:]}_command")
                    response = handler(args, user_id)
                    return response
                    
            # Check for active conversation
            user_conversations = [cid for cid, conv in self.active_conversations.items() 
                                 if conv["user_id"] == user_id and conv["state"] == "started"]
                                 
            if user_conversations:
                # Use the most recent conversation
                conversation_id = max(user_conversations, 
                                      key=lambda cid: self.active_conversations[cid]["last_update"])
                
                # Add message to conversation
                self.add_message(conversation_id, user_id, message)
                
                # Process message based on conversation type
                conversation = self.active_conversations[conversation_id]
                
                # TODO: Implement conversation state processing
                return None
                
            else:
                # No active conversation, check if we need to start one
                conversation_type = self._detect_conversation_type(message)
                if conversation_type:
                    # Start new conversation
                    conversation_id = self.start_conversation(user_id, conversation_type)
                    if conversation_id:
                        # Add message to conversation
                        self.add_message(conversation_id, user_id, message)
                        
                        # Process message based on conversation type
                        # TODO: Implement conversation processing
                        return None
                
            # No command, no active conversation, and no new conversation needed
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return f"Error processing message: {str(e)}"
            
    def _detect_conversation_type(self, message):
        """
        Detect the type of conversation based on the message.
        
        Args:
            message (str): Message text
            
        Returns:
            str: Conversation type or None if not detected
        """
        message = message.lower()
        
        # TODO: Implement more sophisticated conversation detection
        if "trade" in message or "position" in message:
            return "trading"
        elif "analyze" in message or "report" in message:
            return "analysis"
        elif "add" in message or "remove" in message:
            return "portfolio"
            
        return None