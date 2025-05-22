#!/usr/bin/env python3
"""
NSE Stock Details Scraper

This script scrapes stock details from NSE India and saves them to the portfolio collection.

Usage:
    python nse_stock_details.py --symbol SBIN
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime

import requests

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'nse_scraper.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for database import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from database.connection_manager import get_db

class NSEStockDetailsScraper:
    """Scrapes stock details from NSE India and saves to database"""
    
    def __init__(self):
        """Initialize the scraper"""
        self.logger = logger
        
        # Get database connection
        self.db = get_db()
        
        # Test database connection
        try:
            self.db.portfolio_collection.find_one({})
            logger.info("Database connection verified")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        
        # Base URLs and endpoints
        self.base_url = "https://www.nseindia.com"
        self.quote_url = "https://www.nseindia.com/get-quotes/equity?symbol={}"
        self.api_quote_url = "https://www.nseindia.com/api/quote-equity?symbol={}"
        self.company_info_url = "https://www.nseindia.com/api/quote-equity?symbol={}&section=trade_info"
        self.peers_url = "https://www.nseindia.com/api/quote-equity?symbol={}&section=peer_comparison"
        
        # Headers to mimic browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Session for maintaining cookies
        self.session = requests.Session()
        
        # Rate limiting
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
            logger.info("Initializing session with NSE India")
            response = self.session.get(self.base_url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Failed to initialize session: {response.status_code}")
                return False
            
            logger.info(f"Session initialized successfully with cookies: {len(self.session.cookies)}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing session: {e}")
            return False
    
    def fetch_stock_data(self, symbol):
        """Fetch all stock data from NSE for the given symbol"""
        # Initialize session
        if not self._initialize_session():
            return None
        
        try:
            # Step 1: Visit the quote page to ensure cookies are set correctly
            self._rate_limit()
            quote_page_url = self.quote_url.format(symbol)
            logger.info(f"Visiting quote page for {symbol}: {quote_page_url}")
            
            response = self.session.get(quote_page_url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Failed to access quote page: {response.status_code}")
                return None
            
            # Step 2: Fetch basic stock data
            self._rate_limit()
            api_url = self.api_quote_url.format(symbol)
            
            headers = self.headers.copy()
            headers['Accept'] = 'application/json, text/plain, */*'
            headers['Referer'] = quote_page_url
            
            logger.info(f"Fetching stock data from: {api_url}")
            response = self.session.get(api_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch stock data: {response.status_code}")
                return None
            
            stock_data = response.json()
            
            # Step 3: Fetch company info
            self._rate_limit()
            trade_info_url = self.company_info_url.format(symbol)
            
            logger.info(f"Fetching company info from: {trade_info_url}")
            response = self.session.get(trade_info_url, headers=headers, timeout=10)
            
            company_info = None
            if response.status_code == 200:
                company_info = response.json()
            else:
                logger.warning(f"Failed to fetch company info: {response.status_code}")
            
            # Step 4: Fetch peer comparison
            self._rate_limit()
            peers_url = self.peers_url.format(symbol)
            
            logger.info(f"Fetching peer comparison from: {peers_url}")
            response = self.session.get(peers_url, headers=headers, timeout=10)
            
            peers_data = None
            if response.status_code == 200:
                peers_data = response.json()
            else:
                logger.warning(f"Failed to fetch peer comparison: {response.status_code}")
            
            return {
                'stock_data': stock_data,
                'company_info': company_info,
                'peers_data': peers_data
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def extract_keywords(self, stock_data, peers_data):
        """Extract keywords from stock data"""
        keywords = []
        
        try:
            if not stock_data or 'info' not in stock_data:
                return keywords
                
            info = stock_data['info']
            
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
            
            # Add peer companies
            if peers_data and 'peerComparisonData' in peers_data:
                peer_list = peers_data['peerComparisonData']
                for i, peer in enumerate(peer_list):
                    if i >= 3:  # Limit to top 3 peers
                        break
                    if 'symbol' in peer and peer['symbol']:
                        keywords.append(peer['symbol'])
            
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
            logger.error(f"Error extracting keywords: {e}")
        
        # Clean and remove duplicates while preserving order
        clean_keywords = []
        for kw in keywords:
            if isinstance(kw, str) and kw.strip() and kw.strip().lower() not in [k.lower() for k in clean_keywords]:
                clean_keywords.append(kw.strip())
        
        return clean_keywords
    
    def extract_peers(self, peers_data):
        """Extract peer symbols from peers data"""
        peers = []
        
        try:
            if peers_data and 'peerComparisonData' in peers_data:
                peer_list = peers_data['peerComparisonData']
                for peer in peer_list:
                    if 'symbol' in peer and peer['symbol']:
                        peers.append(peer['symbol'])
        except Exception as e:
            logger.error(f"Error extracting peers: {e}")
        
        return peers
    
    def save_to_database(self, symbol, data):
        """Save stock details to database"""
        if not data or 'stock_data' not in data or not data['stock_data']:
            logger.error("No stock data to save")
            return False
        
        stock_data = data['stock_data']
        company_info = data.get('company_info')
        peers_data = data.get('peers_data')
        
        # Extract basic info from stock data
        info = stock_data.get('info', {})
        metadata = stock_data.get('metadata', {})
        
        # Extract financial ratios
        security_info = stock_data.get('securityInfo', {})
        
        # Extract company address from company_info
        company_address = {}
        if company_info and 'tradeInfo' in company_info and 'companyInfo' in company_info['tradeInfo']:
            comp_info = company_info['tradeInfo']['companyInfo']
            company_address = {
                'address': comp_info.get('address'),
                'phone': comp_info.get('phone'),
                'email': comp_info.get('email'),
                'website': comp_info.get('website')
            }
        
        # Extract market data
        market_data = {}
        if company_info and 'marketDeptOrderBook' in company_info:
            market_book = company_info['marketDeptOrderBook']
            market_data = {
                'market_cap': market_book.get('marketCapital'),
                'delivery_quantity': market_book.get('deliveryQuantity'),
                'delivery_percentage': market_book.get('deliveryToTradedQuantity')
            }
        
        # Extract price data
        price_data = {}
        if 'priceInfo' in stock_data:
            price_info = stock_data['priceInfo']
            price_data = {
                'last_price': price_info.get('lastPrice'),
                'prev_close': price_info.get('previousClose'),
                'change': price_info.get('change'),
                'pct_change': price_info.get('pChange'),
                'open': price_info.get('open'),
                'high': price_info.get('intraDayHighLow', {}).get('max'),
                'low': price_info.get('intraDayHighLow', {}).get('min'),
                'close': price_info.get('close'),
                '52_week_high': price_info.get('weekHighLow', {}).get('max'),
                '52_week_low': price_info.get('weekHighLow', {}).get('min')
            }
        
        # Generate keywords
        keywords = self.extract_keywords(stock_data, peers_data)
        
        # Extract peers
        peers = self.extract_peers(peers_data)
        
        # Prepare document for insertion/update
        company_data = {
            "symbol": symbol,
            "exchange": "NSE",
            "company_name": info.get('companyName'),
            "industry": info.get('industry'),
            "sector": info.get('sector'),
            "series": metadata.get('series'),
            "keywords": keywords,
            "peers": peers,
            "isin": info.get('isin'),
            "face_value": security_info.get('faceValue'),
            "status": info.get('status'),
            "listing_date": info.get('listingDate'),
            "eps": security_info.get('eps'),
            "pe_ratio": security_info.get('pe'),
            "pb_ratio": security_info.get('pb'),
            "dividend_yield": security_info.get('yield'),
            # Market data
            "market_cap": market_data.get('market_cap'),
            "delivery_quantity": market_data.get('delivery_quantity'),
            "delivery_percentage": market_data.get('delivery_percentage'),
            # Price data
            "last_price": price_data.get('last_price'),
            "prev_close": price_data.get('prev_close'),
            "change": price_data.get('change'),
            "pct_change": price_data.get('pct_change'),
            "open_price": price_data.get('open'),
            "high_price": price_data.get('high'),
            "low_price": price_data.get('low'),
            "close_price": price_data.get('close'),
            "week_high": price_data.get('52_week_high'),
            "week_low": price_data.get('52_week_low'),
            # Contact info
            "address": company_address.get('address'),
            "phone": company_address.get('phone'),
            "email": company_address.get('email'),
            "website": company_address.get('website'),
            # Update timestamps
            "last_updated": datetime.now()
        }
        
        try:
            # Check if instrument exists
            query = {"symbol": symbol, "exchange": "NSE"}
            
            # Find existing instrument
            existing = self.db.portfolio_collection.find_one(query)
            
            if existing:
                # Update existing instrument
                logger.info(f"Updating stock details for {symbol}")
                
                # Extract the current value of fields we don't want to overwrite if they're None
                for field in ['keywords', 'peers', 'address', 'phone', 'email', 'website']:
                    if company_data.get(field) is None and existing.get(field) is not None:
                        company_data[field] = existing[field]
                
                # Add added_at if not present
                if 'added_at' not in company_data and 'added_at' in existing:
                    company_data['added_at'] = existing['added_at']
                
                result = self.db.portfolio_collection.update_one(
                    query,
                    {"$set": company_data}
                )
                
                if result.modified_count > 0:
                    logger.info(f"Successfully updated stock details for {symbol}")
                    return True
                else:
                    logger.warning(f"No changes made when updating stock details for {symbol}")
                    return True  # Return true anyway as it's not a failure
            else:
                # Create new instrument entry
                logger.info(f"Creating new instrument entry for {symbol}")
                
                # Add added_at timestamp for new entries
                company_data["added_at"] = datetime.now()
                
                result = self.db.portfolio_collection.insert_one(company_data)
                
                if result.inserted_id:
                    logger.info(f"Successfully created new instrument entry for {symbol}")
                    return True
                else:
                    logger.error(f"Failed to create new instrument entry for {symbol}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error saving stock details to database: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
        
        # Fetch stock data
        logger.info(f"Fetching stock data for {args.symbol}")
        stock_data = scraper.fetch_stock_data(args.symbol)
        
        if not stock_data:
            logger.error(f"Failed to fetch stock data for {args.symbol}")
            return 1
        
        # Save to database
        logger.info(f"Saving stock data for {args.symbol} to database")
        success = scraper.save_to_database(args.symbol, stock_data)
        
        if not success:
            logger.error(f"Failed to save stock data for {args.symbol}")
            return 1
        
        # Save to output file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(stock_data, f, indent=2, default=str)
            print(f"Saved stock data to {args.output}")
        
        print(f"\nStock details for {args.symbol} successfully saved to database")
        
        # Print summary of data saved
        info = stock_data['stock_data'].get('info', {})
        price_info = stock_data['stock_data'].get('priceInfo', {})
        security_info = stock_data['stock_data'].get('securityInfo', {})
        
        print(f"Company: {info.get('companyName')}")
        print(f"Sector: {info.get('sector')}")
        print(f"Industry: {info.get('industry')}")
        print(f"Price: ₹{price_info.get('lastPrice')} ({price_info.get('pChange')}%)")
        print(f"PE Ratio: {security_info.get('pe')}")
        print(f"Market Cap: ₹{stock_data.get('company_info', {}).get('marketDeptOrderBook', {}).get('marketCapital')}")
        print(f"52 Week Range: ₹{price_info.get('weekHighLow', {}).get('min')} - ₹{price_info.get('weekHighLow', {}).get('max')}")
        
        keywords = scraper.extract_keywords(stock_data['stock_data'], stock_data['peers_data'])
        print(f"\nKeywords ({len(keywords)}):")
        print(', '.join(keywords))
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())