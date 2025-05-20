# scripts/zerodha_login.py
#!/usr/bin/env python3
"""
Script to login to Zerodha.
"""

import json
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta

from realtime.zerodha_integration import ZerodhaConnector
from config import settings
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def zerodha_login(request_token=None):
    """
    Login to Zerodha
    
    Args:
        request_token (str, optional): Request token from login URL
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create Zerodha connector
        connector = ZerodhaConnector(
            api_key=settings.ZERODHA_API_KEY,
            api_secret=settings.ZERODHA_API_SECRET
        )
        
        if request_token:
            # Generate session from request token
            result = connector.generate_session(request_token)
            if result:
                # Get the access token
                access_token = connector.access_token
                
                # Create the auth_tokens directory if it doesn't exist
                token_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    'auth_tokens'
                )
                os.makedirs(token_dir, exist_ok=True)
                
                # Save the access token to auth_tokens/zerodha_token.json in JSON format
                token_file = os.path.join(token_dir, 'zerodha_token.json')
                
                # Create token data with expiry set to 1 day from now
                token_data = {
                    'access_token': access_token,
                    'api_key': settings.ZERODHA_API_KEY,
                    'expiry': (datetime.now() + timedelta(days=1)).isoformat()
                }
                
                with open(token_file, 'w') as f:
                    json.dump(token_data, f)
                
                logger.info(f"Successfully logged in to Zerodha. Token saved to {token_file}")
                
                # For backward compatibility, also save to the old location
                old_token_file = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'config', 
                    'zerodha_token.txt'
                )
                os.makedirs(os.path.dirname(old_token_file), exist_ok=True)
                
                with open(old_token_file, 'w') as f:
                    f.write(access_token)
                
                logger.info(f"Token also saved to old location: {old_token_file} for backward compatibility")
                
                return True
            else:
                logger.error("Failed to login to Zerodha with provided request token")
                return False
        else:
            # Generate login URL
            login_url = connector.generate_login_url()
            if login_url:
                print("\n" + "="*80)
                print(f"Please login using this URL: {login_url}")
                print("After login, you will be redirected to a URL containing request_token parameter")
                print("Copy the request_token parameter and run this script again with --request-token parameter")
                print("="*80 + "\n")
                return True
            else:
                logger.error("Failed to generate Zerodha login URL")
                return False
    except Exception as e:
        logger.error(f"Error logging in to Zerodha: {e}")
        import traceback
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        return False
    
def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Login to Zerodha')
    
    parser.add_argument(
        '--request-token',
        help='Request token from Zerodha login URL'
    )
    
    args = parser.parse_args()
    
    # Login to Zerodha
    result = zerodha_login(request_token=args.request_token)
    
    if not result:
        sys.exit(1)

if __name__ == '__main__':
    main()