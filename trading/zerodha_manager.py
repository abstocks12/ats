# trading/zerodha_manager.py
"""
Zerodha Connection Manager.
Manages Zerodha connections and ensures valid sessions.
"""

import os
import sys
import logging
from datetime import datetime

from config import settings
from realtime.zerodha_integration import ZerodhaConnector
from utils.logging_utils import setup_logger

# Global zerodha connector instance
_zerodha_connector = None

# Logger setup
logger = setup_logger(__name__)

def get_zerodha_connector(force_new=False):
    """
    Get a valid Zerodha connector with active session
    
    Args:
        force_new (bool): Force creation of a new connector
        
    Returns:
        ZerodhaConnector: Zerodha connector instance
    """
    global _zerodha_connector
    
    if _zerodha_connector is None or force_new:
        try:
            # Check for saved access token
            token_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'config', 'zerodha_token.txt')
            
            access_token = None
            if os.path.exists(token_file):
                try:
                    with open(token_file, 'r') as f:
                        access_token = f.read().strip()
                    logger.info("Found saved Zerodha access token")
                except Exception as e:
                    logger.error(f"Error reading access token from file: {e}")
            
            # Create new connector with the token if available
            _zerodha_connector = ZerodhaConnector(
                api_key=settings.ZERODHA_API_KEY,
                api_secret=settings.ZERODHA_API_SECRET,
                access_token=access_token
            )
            
            # Check connection
            if not _zerodha_connector.is_connected():
                logger.warning("Zerodha not connected, access token may be missing or expired")
                logger.info("Run scripts/zerodha_login.py to generate a new access token")
                
                # Make sure simulated_mode attribute exists
                if not hasattr(_zerodha_connector, 'simulated_mode'):
                    _zerodha_connector.simulated_mode = True
                
        except Exception as e:
            logger.error(f"Error creating Zerodha connector: {e}")
            _zerodha_connector = None
    
    return _zerodha_connector

def ensure_zerodha_connection():
    """
    Ensure Zerodha connection is established and valid
    
    Returns:
        bool: True if connected, False otherwise
    """
    connector = get_zerodha_connector()
    
    if connector and connector.is_connected():
        logger.info("Zerodha connection is valid")
        return True
    
    if connector and connector.simulated_mode:
        logger.warning("Zerodha running in simulated mode")
        return True
    
    logger.error("Zerodha connection failed. Please run scripts/zerodha_login.py to authenticate")
    return False