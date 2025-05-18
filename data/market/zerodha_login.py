#!/usr/bin/env python3
"""
Script for Zerodha login and token generation.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market.zerodha_connector import ZerodhaConnector
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Zerodha Login')
    parser.add_argument('--request-token', required=True, help='Request token from Zerodha login redirect')
    
    args = parser.parse_args()
    
    try:
        # Initialize ZerodhaConnector
        connector = ZerodhaConnector()
        
        # Generate session using the request token
        success = connector.generate_session(args.request_token)
        
        if success:
            print("Authentication successful! Access token saved.")
            return 0
        else:
            print("Authentication failed. Please try again.")
            return 1
            
    except Exception as e:
        print(f"Error during authentication: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())