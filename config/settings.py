"""
Main configuration file for the Automated Trading System.
Contains default settings and environment variable handling.
"""

import os
import json
from datetime import datetime
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data_storage')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Environment-based configuration
# Load from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Using environment variables directly.")

# MongoDB settings
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'automated_trading')
MONGO_USERNAME = os.getenv('MONGO_USERNAME', '')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD', '')

# Zerodha API settings
ZERODHA_API_KEY = os.getenv('ZERODHA_API_KEY', '')
ZERODHA_API_SECRET = os.getenv('ZERODHA_API_SECRET', '')
ZERODHA_USER_ID = os.getenv('ZERODHA_USER_ID', '')
ZERODHA_USER_PASSWORD = os.getenv('ZERODHA_USER_PASSWORD', '')
ZERODHA_TOTP_KEY = os.getenv('ZERODHA_TOTP_KEY', '')

# WhatsApp settings
WHATSAPP_ENABLED = os.getenv('WHATSAPP_ENABLED', 'False').lower() == 'true'
WHATSAPP_API_KEY = os.getenv('WHATSAPP_API_KEY', '')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID', '')
WHATSAPP_RECIPIENT = os.getenv('WHATSAPP_RECIPIENT', '')


# System settings
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
TRADING_MODE = os.getenv('TRADING_MODE', 'paper')  # 'paper' or 'live'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Application settings
MAX_INSTRUMENTS = int(os.getenv('MAX_INSTRUMENTS', '10'))
DEFAULT_POSITION_SIZE_PERCENT = float(os.getenv('DEFAULT_POSITION_SIZE_PERCENT', '5.0'))
DEFAULT_MAX_RISK_PERCENT = float(os.getenv('DEFAULT_MAX_RISK_PERCENT', '1.0'))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))

# Trading hours (IST)
MARKET_OPEN_HOUR = int(os.getenv('MARKET_OPEN_HOUR', '9'))  # 9 AM
MARKET_OPEN_MINUTE = int(os.getenv('MARKET_OPEN_MINUTE', '15'))  # 9:15 AM
MARKET_CLOSE_HOUR = int(os.getenv('MARKET_CLOSE_HOUR', '15'))  # 3 PM
MARKET_CLOSE_MINUTE = int(os.getenv('MARKET_CLOSE_MINUTE', '30'))  # 3:30 PM

# Scheduling
DAILY_DATA_UPDATE_HOUR = int(os.getenv('DAILY_DATA_UPDATE_HOUR', '6'))  # 6 AM
PRE_MARKET_PREP_HOUR = int(os.getenv('PRE_MARKET_PREP_HOUR', '8'))  # 8 AM
POST_MARKET_ANALYSIS_HOUR = int(os.getenv('POST_MARKET_ANALYSIS_HOUR', '16'))  # 4 PM

# Time periods for data collection
HISTORICAL_DAYS_DEFAULT = int(os.getenv('HISTORICAL_DAYS_DEFAULT', '365'))  # 1 year
NEWS_DAYS_DEFAULT = int(os.getenv('NEWS_DAYS_DEFAULT', '30'))  # 30 days

# User agent for web requests
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"

# Web scraping settings
SCRAPING_DELAY = float(os.getenv('SCRAPING_DELAY', '2.0'))  # seconds between requests
SCRAPING_TIMEOUT = int(os.getenv('SCRAPING_TIMEOUT', '30'))  # seconds
SCRAPING_RETRY_COUNT = int(os.getenv('SCRAPING_RETRY_COUNT', '3'))

# News sources
NEWS_SOURCES = {
    'zerodha_pulse': {
        'enabled': True,
        'base_url': 'https://pulse.zerodha.com/',
        'search_url': 'https://pulse.zerodha.com/?q={}'
    },
    'economic_times': {
        'enabled': True,
        'markets_url': 'https://economictimes.indiatimes.com/markets',
        'banking_url': 'https://economictimes.indiatimes.com/industry/banking/finance'
    },
    'bloomberg': {
        'enabled': True,
        'url': 'https://www.bloombergquint.com/markets'
    },
    'hindu_business': {
        'enabled': True,
        'url': 'https://www.thehindu.com/business/markets'
    }
}

# Financial data sources
FINANCIAL_SOURCES = {
    'screener': {
        'enabled': True,
        'url_template': 'https://www.screener.in/company/{}/'
    }
}
# Slack settings
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN', '')
SLACK_CHANNEL_ID = os.getenv('SLACK_CHANNEL_ID', '')
SLACK_API_SIGNING_SECRET = os.getenv('SLACK_API_SIGNING_SECRET', '')

# Set to True if Slack integration should be enabled
SLACK_ENABLED = os.getenv('SLACK_ENABLED', 'False').lower() == 'true'

# Channel settings for different notification types
SLACK_CHANNELS = {
    'trades': os.getenv('SLACK_TRADES_CHANNEL', SLACK_CHANNEL_ID),
    'alerts': os.getenv('SLACK_ALERTS_CHANNEL', SLACK_CHANNEL_ID),
    'reports': os.getenv('SLACK_REPORTS_CHANNEL', SLACK_CHANNEL_ID),
    'system': os.getenv('SLACK_SYSTEM_CHANNEL', SLACK_CHANNEL_ID)
}

# System integration and component settings
SYSTEM_COMPONENTS = {
    'notification_manager': True,
    'report_distributor': True,
    'conversation_manager': SLACK_ENABLED,
    'scheduler': True,
    'market_data_collector': True,
    'market_analyzer': True,
    'portfolio_manager': True,
    'trading_controller': True,
    'time_series_partitioner': True
}


# Notification settings
NOTIFICATION_CONFIG = {
    'channels': {
        'slack': SLACK_ENABLED,
        'database': True,
        'whatsapp': WHATSAPP_ENABLED
    },
    'levels': {
        'debug': ['database'],
        'info': ['database', 'slack'],
        'warning': ['database', 'slack'],
        'error': ['database', 'slack'],
        'critical': ['database', 'slack', 'whatsapp'] if WHATSAPP_ENABLED else ['database', 'slack']
    },
    'batch_notifications': True,
    'batch_interval': 300,  # seconds
    'max_slack_notifications_per_minute': 10
}
# Scheduler settings
SCHEDULER_CONFIG = {
    'min_interval': 1,  # Minimum time between task checks (seconds)
    'max_concurrent_tasks': 10,  # Maximum number of concurrent tasks
    'task_timeout': 3600,  # Default task timeout (seconds)
    'retry_failed_tasks': True,  # Retry failed tasks
    'max_retries': 3,  # Maximum number of retries for failed tasks
    'retry_delay': 300,  # Delay between retries (seconds)
    'persistent_storage': True,  # Store tasks in database
    'log_task_output': True,  # Log task output
    'shutdown_timeout': 60,  # Maximum time to wait for running tasks on shutdown (seconds)
    'auto_recover': True  # Automatically recover tasks from database on startup
}
# Email settings
EMAIL_SENDER = os.getenv('EMAIL_SENDER', '')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', 587))
EMAIL_RECIPIENTS = os.getenv('EMAIL_RECIPIENTS', '').split(',')

# Set to True if email integration should be enabled
EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'False').lower() == 'true'
# Report distribution settings
REPORT_CONFIG = {
    'default_channels': ['slack', 'database'],
    'formats': ['text', 'pdf', 'html'],
    'schedule': {
        'daily_morning': {
            'time': '09:00',
            'report_type': 'daily',
            'channels': ['slack', 'database']
        },
        'daily_evening': {
            'time': '16:30',
            'report_type': 'daily',
            'channels': ['slack', 'database']
        },
        'weekly': {
            'day': 'Friday',
            'time': '16:00',
            'report_type': 'weekly',
            'channels': ['slack', 'database', 'email'] if EMAIL_ENABLED else ['slack', 'database']
        },
        'monthly': {
            'day': 1,
            'time': '09:00',
            'report_type': 'monthly',
            'channels': ['slack', 'database', 'email'] if EMAIL_ENABLED else ['slack', 'database']
        }
    }
}

# Sector to keywords mapping
# This helps in filtering news that might be relevant to a stock's sector
SECTOR_KEYWORDS = {
    'banking': [
        'banking', 'bank', 'RBI', 'Reserve Bank', 'NBFC', 'NPA', 'credit growth',
        'deposit', 'lending', 'monetary policy', 'interest rate', 'repo rate',
        'central bank', 'financial services', 'loan', 'credit'
    ],
    'it': [
        'IT', 'software', 'technology', 'digital', 'cloud', 'SaaS', 'artificial intelligence',
        'AI', 'machine learning', 'ML', 'automation', 'tech'
    ],
    'pharma': [
        'pharma', 'pharmaceutical', 'drug', 'medicine', 'healthcare', 'FDA', 'USFDA',
        'clinical trial', 'vaccine', 'biotech', 'research'
    ],
    'auto': [
        'automobile', 'auto', 'car', 'vehicle', 'EV', 'electric vehicle', 'automotive',
        'two-wheeler', 'four-wheeler', 'commercial vehicle'
    ],
    'energy': [
        'energy', 'power', 'electricity', 'renewable', 'solar', 'wind', 'hydro',
        'oil', 'gas', 'petroleum', 'coal', 'thermal'
    ],
    'fmcg': [
        'FMCG', 'consumer goods', 'retail', 'food', 'beverage', 'household',
        'personal care', 'packaged goods'
    ],
    'metal': [
        'metal', 'steel', 'aluminum', 'copper', 'zinc', 'iron ore', 'mining',
        'metallurgy', 'commodity'
    ]
}

# Default database collections
MONGODB_COLLECTIONS = {
    'portfolio': 'portfolio',
    'market_data': 'market_data',
    'news': 'news',
    'financial': 'financial',
    'predictions': 'predictions',
    'trades': 'trades',
    'performance': 'performance',
    'system_logs': 'system_logs',
    'tasks': 'tasks',  # Add this line
    'conversations': 'conversations',  # For conversation management
    'notifications': 'notifications',  # For storing notification history
    'reports': 'reports'  
    # For report storage
    # 'portfolio': 'portfolio',
    # 'market_data': 'market_data',
    # 'news': 'news',
    # 'financial': 'financial',
    # 'predictions': 'predictions',
    # 'trades': 'trades',
    # 'performance': 'performance',
    # 'system_logs': 'system_logs',
    # 'tasks': 'tasks'
}

# System version
VERSION = "1.0.0"

def get_version_info():
    """Return system version information"""
    return {
        'version': VERSION,
        'environment': 'Production' if not DEBUG else 'Development',
        'trading_mode': TRADING_MODE.capitalize(),
        'timestamp': datetime.now().isoformat()
    }

def load_custom_config(config_path):
    """Load custom configuration from a JSON file"""
    try:
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
            
        # Update global variables
        for key, value in custom_config.items():
            if key in globals():
                globals()[key] = value
                
        return True
    except Exception as e:
        print(f"Error loading custom config: {e}")
        return False