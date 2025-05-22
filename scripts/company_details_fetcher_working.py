#!/usr/bin/env python3
"""
NSE Stock Details Scraper

This script scrapes stock details from NSE India and saves them to the portfolio collection.
It generates rich company details including name, sector, industry, and keywords.

Usage:
    python nse_stock_details.py --symbol SBIN
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import requests
from bs4 import BeautifulSoup

# Try multiple approaches to find the config module
try:
    # First try direct import
    from config import settings
except ImportError:
    try:
        # Try adding current directory's parent to path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import settings
    except ImportError:
        try:
            # Try adding script directory's parent to path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            sys.path.append(parent_dir)
            from config import settings
        except ImportError:
            # Define minimal settings as fallback
            class Settings:
                LOG_DIR = 'logs'
                USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            
            settings = Settings()
            print("Warning: Could not import config.settings, using default settings")

# Try to import the database connector
try:
    from database.connection_manager import get_db
except ImportError:
    try:
        # Try adding script directory's parent to path for db import
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        sys.path.append(parent_dir)
        from database.connection_manager import get_db
    except ImportError:
        # Mock database for testing
        print("Warning: Could not import database connection, using mock database")
        
        class MockCollection:
            def find_one(self, query):
                print(f"Mock find_one: {query}")
                return None
                
            def update_one(self, query, update, upsert=False):
                print(f"Mock update_one: {query}, {update}")
                return type('obj', (object,), {'modified_count': 1})
                
            def insert_one(self, document):
                print(f"Mock insert_one: {document.get('symbol')}")
                return type('obj', (object,), {'inserted_id': 'mock_id'})
        
        class MockDB:
            def __init__(self):
                self.portfolio_collection = MockCollection()
        
        def get_db():
            return MockDB()

# Configure logging
log_dir = getattr(settings, 'LOG_DIR', 'logs')
os.makedirs(log_dir, exist_ok=True)

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Check if handlers already exist to avoid duplicate handlers
if not logger.handlers:
    # Create handlers
    file_handler = logging.FileHandler(os.path.join(log_dir, 'nse_scraper.log'))
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

class NSEStockDetailsScraper:
    """Scrapes stock details from NSE India and saves to database"""
    
    def __init__(self):
        """Initialize the scraper"""
        self.logger = logger
        self.db = get_db()
        
        # Base URLs and endpoints
        self.base_url = "https://www.nseindia.com"
        self.quote_url = "https://www.nseindia.com/get-quotes/equity?symbol={}"
        self.api_quote_url = "https://www.nseindia.com/api/quote-equity?symbol={}"
        self.company_info_url = "https://www.nseindia.com/api/quote-equity?symbol={}&section=trade_info"
        self.peers_url = "https://www.nseindia.com/api/quote-equity?symbol={}&section=peer_comparison"
        
        # Headers to mimic browser request
        self.headers = {
            'User-Agent': getattr(settings, 'USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Session for maintaining cookies
        self.session = requests.Session()
        
        # Setup rate limiting
        self.last_api_call_time = 0
        self.min_time_between_calls = 1  # seconds
    
    # ... rest of the class remains unchanged ...
        
        # Setup rate limiting
        self.last_api_call_time = 0
        self.min_time_between_calls = 1  # seconds
    
    def _rate_limit(self):
        """Apply rate limiting for API calls"""
        current_time = time.time()
        elapsed = current_time - self.last_api_call_time
        
        if elapsed < self.min_time_between_calls:
            sleep_time = self.min_time_between_calls - elapsed
            time.sleep(sleep_time)
        
        self.last_api_call_time = time.time()
    
    def _initialize_session(self):
        """Initialize session to get cookies required for NSE API calls"""
        try:
            self.logger.info("Initializing session with NSE India")
            response = self.session.get(self.base_url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to initialize session: {response.status_code}")
                return False
            
            self.logger.info(f"Session initialized successfully with cookies: {len(self.session.cookies)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing session: {e}")
            return False
    
    def fetch_stock_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch stock details from NSE India
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock details or None if failed
        """
        # Initialize session if needed
        if not self.session.cookies:
            if not self._initialize_session():
                return None
        
        try:
            # First, visit the quote page to ensure cookies are set correctly
            self._rate_limit()
            quote_page_url = self.quote_url.format(symbol)
            self.logger.info(f"Visiting quote page for {symbol}: {quote_page_url}")
            
            response = self.session.get(quote_page_url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to access quote page: {response.status_code}")
                return None
            
            # Now fetch the JSON API data
            self._rate_limit()
            api_url = self.api_quote_url.format(symbol)
            self.logger.info(f"Fetching API data for {symbol}: {api_url}")
            
            headers = self.headers.copy()
            headers['Accept'] = 'application/json, text/plain, */*'
            headers['Referer'] = quote_page_url
            
            response = self.session.get(api_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch API data: {response.status_code}")
                return None
            
            data = response.json()
            
            # Check if we have valid data
            if not data or 'info' not in data:
                self.logger.error(f"Incomplete or missing data for {symbol}")
                return None
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching stock details for {symbol}: {e}")
            return None
    
    def fetch_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed company information from NSE India
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company information or None if failed
        """
        try:
            self._rate_limit()
            api_url = self.company_info_url.format(symbol)
            self.logger.info(f"Fetching company info for {symbol}: {api_url}")
            
            headers = self.headers.copy()
            headers['Accept'] = 'application/json, text/plain, */*'
            headers['Referer'] = self.quote_url.format(symbol)
            
            response = self.session.get(api_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch company info: {response.status_code}")
                return None
            
            data = response.json()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching company info for {symbol}: {e}")
            return None
    
    def fetch_peers(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch peer comparison data from NSE India
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with peer comparison data or None if failed
        """
        try:
            self._rate_limit()
            api_url = self.peers_url.format(symbol)
            self.logger.info(f"Fetching peer comparison for {symbol}: {api_url}")
            
            headers = self.headers.copy()
            headers['Accept'] = 'application/json, text/plain, */*'
            headers['Referer'] = self.quote_url.format(symbol)
            
            response = self.session.get(api_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch peer comparison: {response.status_code}")
                return None
            
            data = response.json()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching peer comparison for {symbol}: {e}")
            return None
    
    def extract_industry_peers(self, peers_data: Dict[str, Any]) -> List[str]:
        """
        Extract industry peers from peer comparison data
        
        Args:
            peers_data: Peer comparison data from NSE
            
        Returns:
            List of peer company symbols
        """
        peers = []
        
        try:
            if peers_data and 'peerComparisonData' in peers_data:
                peer_list = peers_data['peerComparisonData']
                for peer in peer_list:
                    if 'symbol' in peer and peer['symbol']:
                        peers.append(peer['symbol'])
        except Exception as e:
            self.logger.error(f"Error extracting peers: {e}")
        
        return peers
    
    def generate_keywords(self, stock_data: Dict[str, Any], peers: List[str] = None) -> List[str]:
        """
        Generate keywords based on stock data
        
        Args:
            stock_data: Stock data from NSE
            peers: List of peer companies
            
        Returns:
            List of keywords
        """
        keywords = []
        
        try:
            info = stock_data.get('info', {})
            
            # Add symbol
            symbol = info.get('symbol')
            if symbol:
                keywords.append(symbol)
            
            # Add company name
            company_name = info.get('companyName')
            if company_name:
                keywords.append(company_name)
                
                # Add variations of company name
                if " Limited" in company_name:
                    keywords.append(company_name.replace(" Limited", " Ltd"))
                if " Ltd" in company_name:
                    keywords.append(company_name.replace(" Ltd", " Limited"))
                
                # Remove common suffixes for another variation
                for suffix in [" Limited", " Ltd", " Corporation", " Corp", " Inc", " Company", " Co"]:
                    if company_name.endswith(suffix):
                        keywords.append(company_name[:-len(suffix)])
                
                # Add name parts
                name_parts = company_name.split()
                for part in name_parts:
                    if len(part) > 3 and part.lower() not in ['ltd', 'limited', 'inc', 'corp', 'corporation', 'company', 'and', 'the']:
                        keywords.append(part)
            
            # Add industry and sector
            industry = info.get('industry')
            if industry:
                keywords.append(industry)
            
            sector = info.get('sector')
            if sector:
                keywords.append(sector)
            
            # Add special keywords for specific companies
            if symbol == "SBIN" or "State Bank of India" in (company_name or ""):
                extra_keywords = ["SBI", "State Bank", "India's largest bank", "PSU Bank", "Public Sector Bank"]
                keywords.extend(extra_keywords)
            elif symbol == "HDFCBANK" or "HDFC Bank" in (company_name or ""):
                extra_keywords = ["HDFC Bank", "HDFC", "Private sector bank"]
                keywords.extend(extra_keywords)
            elif symbol == "ICICIBANK" or "ICICI Bank" in (company_name or ""):
                extra_keywords = ["ICICI Bank", "ICICI", "Private sector bank"]
                keywords.extend(extra_keywords)
            elif symbol == "BANKBARODA" or "Bank of Baroda" in (company_name or ""):
                extra_keywords = ["BOB", "Bank of Baroda", "PSU Bank"]
                keywords.extend(extra_keywords)
            elif symbol == "PNB" or "Punjab National Bank" in (company_name or ""):
                extra_keywords = ["Punjab National Bank", "PNB", "PSU Bank"]
                keywords.extend(extra_keywords)
            
            # Add peer companies (top 3) as relevant keywords
            if peers and len(peers) > 0:
                top_peers = peers[:3]
                keywords.extend(top_peers)
            
            # Add sector specific keywords
            if sector and sector.lower() in ["banking", "financial services", "finance"]:
                banking_keywords = ["interest rates", "RBI", "Reserve Bank", "NPA", "credit growth", 
                                   "deposit rates", "lending", "loans", "banking sector"]
                keywords.extend(banking_keywords)
            elif sector and sector.lower() in ["technology", "it services", "software"]:
                tech_keywords = ["software", "digital", "cloud", "AI", "technology sector", "IT services"]
                keywords.extend(tech_keywords)
            elif sector and sector.lower() in ["pharma", "healthcare", "pharmaceutical"]:
                pharma_keywords = ["drug", "medicine", "healthcare", "pharma sector", "FDA", "clinical trials"]
                keywords.extend(pharma_keywords)
            
        except Exception as e:
            self.logger.error(f"Error generating keywords: {e}")
        
        # Clean and remove duplicates while preserving order
        clean_keywords = []
        for kw in keywords:
            if isinstance(kw, str) and kw.strip() and kw.strip().lower() not in [k.lower() for k in clean_keywords]:
                clean_keywords.append(kw.strip())
        
        return clean_keywords
    
    def prepare_company_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Prepare complete company details by collecting and processing data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with complete company details or None if failed
        """
        try:
            # Step 1: Fetch basic stock data
            stock_data = self.fetch_stock_details(symbol)
            if not stock_data:
                self.logger.error(f"Could not fetch stock data for {symbol}")
                return None
            
            # Step 2: Fetch additional company info
            company_info = self.fetch_company_info(symbol)
            
            # Step 3: Fetch peers data
            peers_data = self.fetch_peers(symbol)
            peers = self.extract_industry_peers(peers_data) if peers_data else []
            
            # Step 4: Generate keywords
            keywords = self.generate_keywords(stock_data, peers)
            
            # Extract metadata
            info = stock_data.get('info', {})
            metadata = stock_data.get('metadata', {})
            securityInfo = stock_data.get('securityInfo', {})
            
            # Step 5: Prepare company details document
            company_details = {
                "symbol": symbol,
                "exchange": "NSE",  # This is NSE specific
                "company_name": info.get('companyName'),
                "industry": info.get('industry'),
                "sector": info.get('sector'),
                "series": metadata.get('series'),
                "isin": info.get('isin'),
                "status": info.get('status'),
                "listing_date": info.get('listingDate'),
                "industry_info": info.get('industryInfo'),
                "keywords": keywords,
                "peers": peers,
                "face_value": securityInfo.get('faceValue'),
                "updated_at": datetime.now()
            }
            
            # Add price data
            if 'priceInfo' in stock_data:
                price_info = stock_data['priceInfo']
                company_details["price_data"] = {
                    "last_price": price_info.get('lastPrice'),
                    "change": price_info.get('change'),
                    "pct_change": price_info.get('pChange'),
                    "prev_close": price_info.get('previousClose'),
                    "open": price_info.get('open'),
                    "close": price_info.get('close'),
                    "high": price_info.get('intraDayHighLow', {}).get('max'),
                    "low": price_info.get('intraDayHighLow', {}).get('min'),
                    "volume": price_info.get('totalTradedVolume'),
                    "value": price_info.get('totalTradedValue'),
                    "week_high": price_info.get('weekHighLow', {}).get('max'),
                    "week_low": price_info.get('weekHighLow', {}).get('min')
                }
            
            # Add financial ratios if available
            if 'securityInfo' in stock_data:
                security_info = stock_data['securityInfo']
                company_details["financial_ratios"] = {
                    "eps": security_info.get('eps'),
                    "pe_ratio": security_info.get('pe'),
                    "pb_ratio": security_info.get('pb'),
                    "dividend_yield": security_info.get('yield')
                }
            
            # Add trade info if available
            if company_info and 'marketDeptOrderBook' in company_info:
                market_data = company_info['marketDeptOrderBook']
                company_details["market_data"] = {
                    "market_cap": market_data.get('marketCapital'),
                    "delivery_quantity": market_data.get('deliveryQuantity'),
                    "delivery_percentage": market_data.get('deliveryToTradedQuantity'),
                    "total_traded_value": market_data.get('totalTradedValue')
                }
            
            # Add detailed company information if available
            if company_info and 'tradeInfo' in company_info:
                trade_info = company_info['tradeInfo']
                if 'companyInfo' in trade_info:
                    comp_info = trade_info['companyInfo']
                    company_details["company_info"] = {
                        "address": comp_info.get('address'),
                        "phone": comp_info.get('phone'),
                        "email": comp_info.get('email'),
                        "website": comp_info.get('website')
                    }
            
            return company_details
            
        except Exception as e:
            self.logger.error(f"Error preparing company details for {symbol}: {e}")
            return None
    
    def save_to_database(self, company_details: Dict[str, Any]) -> bool:
        """
        Save company details to database
        
        Args:
            company_details: Company details to save
            
        Returns:
            True if successful, False otherwise
        """
        if not company_details:
            self.logger.error("No company details provided")
            return False
        
        symbol = company_details.get('symbol')
        exchange = company_details.get('exchange')
        
        if not symbol:
            self.logger.error("No symbol provided in company details")
            return False
        
        try:
            # Check if instrument exists
            query = {"symbol": symbol}
            if exchange:
                query["exchange"] = exchange
            
            # Find existing instrument
            existing = self.db.portfolio_collection.find_one(query)
            
            if existing:
                # Update existing instrument
                self.logger.info(f"Updating company details for {symbol}:{exchange}")
                result = self.db.portfolio_collection.update_one(
                    query,
                    {"$set": {
                        "company_details": company_details,
                        "last_updated": datetime.now()
                    }}
                )
                success = result.modified_count > 0
            else:
                # Create new instrument entry
                self.logger.info(f"Creating new instrument entry for {symbol}:{exchange}")
                new_instrument = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "company_name": company_details.get('company_name'),
                    "sector": company_details.get('sector'),
                    "industry": company_details.get('industry'),
                    "company_details": company_details,
                    "added_at": datetime.now(),
                    "last_updated": datetime.now()
                }
                result = self.db.portfolio_collection.insert_one(new_instrument)
                success = result.inserted_id is not None
            
            if success:
                self.logger.info(f"Successfully saved company details for {symbol}:{exchange}")
                return True
            else:
                self.logger.warning(f"No changes made when saving company details for {symbol}:{exchange}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving company details to database: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fetch and save NSE stock details')
    parser.add_argument('--symbol', '-s', required=True, help='Stock symbol')
    parser.add_argument('--output', '-o', help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    try:
        # Initialize scraper
        scraper = NSEStockDetailsScraper()
        
        # Prepare company details
        company_details = scraper.prepare_company_details(args.symbol)
        
        if not company_details:
            logger.error(f"Failed to prepare company details for {args.symbol}")
            return 1
        
        # Save to database
        success = scraper.save_to_database(company_details)
        
        if not success:
            logger.error(f"Failed to save company details for {args.symbol}")
            return 1
        
        # Save to output file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(company_details, f, indent=2, default=str)
            print(f"Saved company details to {args.output}")
        
        # Print summary
        print(f"\nCompany Details for {args.symbol}:NSE")
        print(f"Name: {company_details.get('company_name')}")
        print(f"Sector: {company_details.get('sector')}")
        print(f"Industry: {company_details.get('industry')}")
        
        if company_details.get('financial_ratios'):
            print("\nFinancial Ratios:")
            for key, value in company_details['financial_ratios'].items():
                if value is not None:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\nKeywords ({len(company_details.get('keywords', []))}):")
        print(f"  {', '.join(company_details.get('keywords', []))}")
        
        if company_details.get('peers'):
            print(f"\nPeers: {', '.join(company_details.get('peers', []))}")
        
        print(f"\nCompany details successfully saved to database")
        return 0
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())