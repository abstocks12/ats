"""News Aggregator Module for the Automated Trading System.
Collects and aggregates news from specified sources with improved debugging and error handling.
"""

import os
import sys
import time
from datetime import datetime, timedelta
import hashlib
import re
import traceback
from typing import List, Dict, Any, Optional, Union, Set
from bs4 import BeautifulSoup
import requests
import ssl
import nltk
from urllib.parse import urljoin, urlparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from config import settings
    from utils.logging_utils import setup_logger, log_error, log_data_collection
    from utils.helper_functions import retry_function, get_date_range, normalize_symbol, extract_stock_symbol
    from data.news.sentiment_analyzer import SentimentAnalyzer
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit("Required modules not found. Cannot run in standalone mode as this is for production.")

class NewsAggregator:
    """
    Aggregates news from multiple sources, processes them, and stores in database.
    """
    
    def __init__(self, db_connector=None, debug_mode=True):
        """
        Initialize news aggregator
        
        Args:
            db_connector: Database connector (optional, will use global connection if not provided)
            debug_mode: Whether to enable additional debugging
        """
        self.logger = setup_logger(__name__)
        self.db = db_connector or self._get_db_connection()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.debug_mode = debug_mode
        
        # Set user agent for requests
        self.headers = {
            'User-Agent': settings.USER_AGENT
        }
        
        # Define scraping delay to avoid rate limiting
        self.scraping_delay = settings.SCRAPING_DELAY
        
        # Add debugging header
        if self.debug_mode:
            self.logger.info("=" * 50)
            self.logger.info("INITIALIZING NEWS AGGREGATOR IN DEBUG MODE")
            self.logger.info("=" * 50)
            
        # Initialize source configs
        self.sources = self._initialize_sources()
        
        # Log configured sources
        self.logger.info(f"Configured news sources: {', '.join(self.sources.keys())}")

        try:
            # Try to create an unverified SSL context for NLTK
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # Download NLTK data
            nltk.download('punkt', quiet=True)
            self.logger.info("NLTK data downloaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to download NLTK data: {e}")
            
        # Create debug directory if in debug mode
        if self.debug_mode:
            os.makedirs("debug", exist_ok=True)
            self.logger.info("Created debug directory")
    
    def _get_db_connection(self):
        """
        Get database connection
        
        Returns:
            Database connector
        """
        try:
            from database.connection_manager import get_db
            return get_db()
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _initialize_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize news source configurations
        
        Returns:
            dict: Source configurations
        """
        # Initialize with only the specified sources
        sources = {}
        
        # Zerodha Pulse
        sources['zerodha_pulse'] = {
            'enabled': True,
            'base_url': 'https://pulse.zerodha.com',
            'format': 'https://pulse.zerodha.com/#{}'  # Format for direct tag search
        }
        
        # Zerodha Markets
        sources['zerodha_markets'] = {
            'enabled': True,
            'base_url': 'https://zerodha.com/markets/stocks/',
            'url_format': 'https://zerodha.com/markets/stocks/{}'  # Format for company URL
        }
        
        # The Hindu Business
        sources['hindu_business'] = {
            'enabled': True,
            'url': 'https://www.thehindu.com/business/markets'
        }
        
        # Times of India
        sources['times_of_india'] = {
            'enabled': True,
            'base_url': 'https://timesofindia.indiatimes.com',
            'topic_url': 'https://timesofindia.indiatimes.com/topic/{}'
        }
        
        return sources
    
    def retry_request(self, url, max_retries=3, delay=1):
        """
        Make HTTP request with retry logic
        
        Args:
            url: URL to request
            max_retries: Maximum number of retries
            delay: Delay between retries in seconds
            
        Returns:
            Response object or None
        """
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Making request to {url} (attempt {attempt+1}/{max_retries})")
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    self.logger.debug(f"Request to {url} successful")
                    return response
                else:
                    self.logger.warning(f"Request to {url} failed with status {response.status_code}")
                    
            except requests.RequestException as e:
                self.logger.warning(f"Request to {url} failed: {e}")
            
            # Wait before retrying
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
        
        self.logger.error(f"All {max_retries} request attempts to {url} failed")
        return None
    
    def save_debug_content(self, source_name, content, suffix="html"):
        """
        Save content to debug file
        
        Args:
            source_name: Name of the source
            content: Content to save
            suffix: File suffix
        """
        if not self.debug_mode:
            return
            
        debug_dir = "debug"
        filename = f"{debug_dir}/{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{suffix}"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"Saved debug content to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save debug content: {e}")
    
    def collect_news(self, symbol: str, exchange: str = None, days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Collect news for a specific instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            days (int): Number of days to look back
            limit (int): Maximum number of news items to return
            
        Returns:
            list: List of news items
        """
        start_time = time.time()
        self.logger.info(f"Collecting news for {symbol}{':'+exchange if exchange else ''} (last {days} days)")
        
        # Normalize symbol
        normalized_symbol = normalize_symbol(symbol, exchange) if exchange else symbol
        
        # Get date range
        start_date, end_date = get_date_range(days)
        
        # Get keywords for symbol - with better error handling
        try:
            keywords = self._get_symbol_keywords(symbol, exchange)
            self.logger.info(f"Using keywords for {symbol}: {keywords}")
        except Exception as e:
            self.logger.error(f"Error getting keywords for {symbol}: {e}")
            # Default to symbol as the only keyword
            keywords = [symbol]
        
        # Get sector keywords if available
        try:
            sector_keywords = self._get_sector_keywords(symbol, exchange)
            if sector_keywords:
                self.logger.info(f"Found sector keywords: {sector_keywords}")
            else:
                self.logger.info(f"No sector keywords found for {symbol}")
        except Exception as e:
            self.logger.warning(f"Error getting sector keywords: {e}")
            sector_keywords = []
        
        # Initialize results
        all_news = []
        
        # Collect news from each source
        for source_name, source_config in self.sources.items():
            if not source_config.get('enabled', False):
                self.logger.debug(f"Skipping disabled source: {source_name}")
                continue
            
            try:
                source_start_time = time.time()
                self.logger.info(f"Collecting news from {source_name} for {symbol}")
                
                # Call appropriate collector method based on source name
                collector_method = getattr(self, f"_collect_{source_name}", None)
                
                if collector_method:
                    news_items = collector_method(
                        symbol=symbol,
                        exchange=exchange,
                        keywords=keywords,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Verify we got valid results
                    if news_items is None:
                        self.logger.warning(f"Got None result from {source_name}")
                        news_items = []
                    
                    source_duration = time.time() - source_start_time
                    
                    if news_items:
                        # Log the details of what we found
                        self.logger.info(f"Found {len(news_items)} items from {source_name} - processing them")
                        
                        # Add source information
                        for item in news_items:
                            item['source'] = source_name
                            
                            # Add normalized symbol
                            if 'symbols' not in item:
                                item['symbols'] = []
                            
                            if normalized_symbol and normalized_symbol not in item['symbols']:
                                item['symbols'].append(normalized_symbol)
                            
                            # Add sector keywords as categories if applicable
                            if sector_keywords and 'categories' in item:
                                for keyword in sector_keywords:
                                    if keyword.lower() in item['title'].lower() or \
                                    (item.get('description') and keyword.lower() in item['description'].lower()):
                                        if keyword not in item['categories']:
                                            item['categories'].append(keyword)
                        
                        all_news.extend(news_items)
                        self.logger.info(f"Collected {len(news_items)} news items from {source_name} in {source_duration:.2f}s")
                    else:
                        self.logger.info(f"No news found from {source_name} for {symbol} (took {source_duration:.2f}s)")
                else:
                    self.logger.warning(f"No collector method found for source: {source_name}")
            
            except Exception as e:
                error_detail = traceback.format_exc()
                self.logger.error(f"Error collecting from {source_name}: {e}\n{error_detail}")
                log_error(e, context={"action": "collect_news", "source": source_name, "symbol": symbol})
                
                # Save exception to debug file
                if self.debug_mode:
                    debug_file = f"debug/error_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(debug_file, 'w') as f:
                        f.write(f"Error collecting from {source_name} for {symbol}:\n")
                        f.write(error_detail)
                    self.logger.info(f"Saved error details to {debug_file}")
        
        # Log what we found
        self.logger.info(f"Total news items collected from all sources: {len(all_news)}")
        
        # Process and filter collected news
        processed_news = self._process_news_items(all_news, keywords)
        
        # Save to database - with error handling
        try:
            saved_count = self._save_to_database(processed_news)
            self.logger.info(f"Saved {saved_count} news items to database")
        except Exception as e:
            self.logger.error(f"Failed to save news items to database: {e}")
            saved_count = 0
        
        total_duration = time.time() - start_time
        self.logger.info(f"Completed news collection for {symbol}{':'+exchange if exchange else ''} (total time: {total_duration:.2f}s)")
        
        # Return the news items (limited by the specified limit)
        return processed_news[:limit] if limit else processed_news

    def _get_symbol_keywords(self, symbol: str, exchange: str = None) -> List[str]:
        """
        Get keywords for a symbol
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            list: List of keywords
        """
        # Get instrument details from portfolio collection
        instrument = None
        if exchange:
            # Try to get from portfolio_collection
            instrument = self.db.portfolio_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            })
            
            # If not found or no company_name, try to get from portfolio collection structure
            if not instrument or 'company_name' not in instrument:
                # Try to get from the company_details.keywords field in portfolio collection
                portfolio_entry = self.db.portfolio_collection.find_one({
                    "symbol": symbol,
                    "exchange": exchange
                })
                
                if portfolio_entry and 'company_details' in portfolio_entry:
                    company_details = portfolio_entry['company_details']
                    # Use keywords if available
                    if 'keywords' in company_details and company_details['keywords']:
                        return company_details['keywords']
                    # Use company_name if available
                    elif 'company_name' in company_details and company_details['company_name']:
                        # Create a new instrument-like object with the company name
                        instrument = {'company_name': company_details['company_name']}
        
        keywords = [symbol]
        
        if instrument and 'company_name' in instrument and instrument['company_name']:
            # Add company name if available
            keywords.append(instrument['company_name'])
            
            # Add parts of company name
            parts = instrument['company_name'].split()
            if len(parts) > 1:
                for part in parts:
                    if len(part) > 3 and part.lower() not in ['ltd', 'limited', 'inc', 'corp']:
                        keywords.append(part)
        
        # Try to get company details from financial data
        financial_data = None
        if exchange:
            financial_data = self.db.financial_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            })
        
        if financial_data and 'company_name' in financial_data and financial_data['company_name']:
            if financial_data['company_name'] not in keywords:
                keywords.append(financial_data['company_name'])
                
                # Add parts of company name
                parts = financial_data['company_name'].split()
                if len(parts) > 1:
                    for part in parts:
                        if len(part) > 3 and part.lower() not in ['ltd', 'limited', 'inc', 'corp']:
                            if part not in keywords:
                                keywords.append(part)
                                
        # If no company keywords found, fallback to symbol variations
        if len(keywords) <= 1:
            # Add symbol without spaces
            if ' ' in symbol:
                keywords.append(symbol.replace(' ', ''))
            
            # Add symbol without special chars
            clean_symbol = re.sub(r'[^a-zA-Z0-9]', '', symbol)
            if clean_symbol != symbol and clean_symbol not in keywords:
                keywords.append(clean_symbol)
        
        # Remove duplicates while preserving order
        unique_keywords = []
        for kw in keywords:
            if kw not in unique_keywords:
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _get_sector_keywords(self, symbol: str, exchange: str = None) -> List[str]:
        """
        Get sector keywords for a symbol
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            list: List of sector keywords
        """
        if not exchange:
            return []
            
        # Get instrument details
        instrument = self.db.portfolio_collection.find_one({
            "symbol": symbol,
            "exchange": exchange
        })
        
        if not instrument:
            return []
        
        # Check both portfolio structure formats
        sector = None
        
        # Check for sector in top-level structure
        if 'sector' in instrument and instrument['sector']:
            sector = instrument['sector']
        # Check for sector in company_details
        elif 'company_details' in instrument and instrument['company_details']:
            if 'sector' in instrument['company_details'] and instrument['company_details']['sector']:
                sector = instrument['company_details']['sector']
            # Check for industry as an alternative
            elif 'industry' in instrument['company_details'] and instrument['company_details']['industry']:
                sector = instrument['company_details']['industry']
        
        # If no sector found, return empty list
        if not sector:
            self.logger.warning(f"No sector found for {symbol}:{exchange}")
            return []
        
        # Get sector keywords from settings
        sector_keywords = settings.SECTOR_KEYWORDS.get(sector.lower(), [])
        
        # If no sector keywords found in settings, use the sector itself
        if not sector_keywords:
            sector_keywords = [sector]
        
        return sector_keywords
    
    def _process_news_items(self, news_items: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Process and filter news items
        
        Args:
            news_items (list): List of news items
            keywords (list): Keywords to filter by
            
        Returns:
            list: Processed news items
        """
        self.logger.info(f"Processing {len(news_items)} news items")
        processed_news = []
        seen_hashes = set()
        duplicates = 0
        
        for item in news_items:
            # Generate unique hash for deduplication
            item_hash = self._generate_news_hash(item)
            
            # Skip if duplicate
            if item_hash in seen_hashes:
                duplicates += 1
                continue
            
            seen_hashes.add(item_hash)
            
            # Ensure required fields
            if 'title' not in item:
                self.logger.debug(f"Skipping item with no title: {item}")
                continue
            
            # Set default values for missing fields
            if 'description' not in item:
                item['description'] = ""
            
            if 'url' not in item:
                item['url'] = ""
            
            if 'published_date' not in item:
                item['published_date'] = datetime.now()
            
            if 'categories' not in item:
                item['categories'] = []
            
            if 'symbols' not in item:
                item['symbols'] = []
                
                # Extract symbols from title and description
                title_symbols = extract_stock_symbol(item['title'])
                if title_symbols and title_symbols not in item['symbols']:
                    item['symbols'].append(title_symbols)
                
                desc_symbols = extract_stock_symbol(item['description'])
                if desc_symbols and desc_symbols not in item['symbols']:
                    item['symbols'].append(desc_symbols)
            
            # Add sentiment if not already present
            if 'sentiment' not in item or 'sentiment_score' not in item:
                sentiment, score = self.sentiment_analyzer.analyze_sentiment(
                    item['title'], 
                    item.get('description', '')
                )
                
                item['sentiment'] = sentiment
                item['sentiment_score'] = score
                
                if self.debug_mode:
                    self.logger.debug(f"Sentiment for '{item['title']}': {sentiment} ({score})")
            
            # Add to processed items
            processed_news.append(item)
        
        # Sort by published date (newest first)
        processed_news.sort(key=lambda x: x.get('published_date', datetime.now()), reverse=True)
        
        self.logger.info(f"Processed {len(processed_news)} unique news items (found {duplicates} duplicates)")
        
        return processed_news
    
    def _save_to_database(self, news_items: List[Dict[str, Any]]) -> int:
        """
        Save news items to database
        
        Args:
            news_items (list): List of news items
            
        Returns:
            int: Number of items saved
        """
        if not news_items:
            self.logger.info("No news items to save to database")
            return 0
        
        self.logger.info(f"Attempting to save {len(news_items)} news items to database")
        
        saved_count = 0
        already_exists = 0
        errors = 0
        
        for idx, item in enumerate(news_items):
            try:
                # Ensure we have all required fields
                if 'title' not in item or not item['title']:
                    self.logger.warning(f"Skipping item #{idx} - Missing title")
                    continue
                    
                if 'source' not in item or not item['source']:
                    item['source'] = "Unknown"
                
                # Check if already exists using consistent access pattern
                existing = self.db.news_collection.find_one({
                    "title": item['title'],
                    "source": item['source']
                })
                
                if not existing:
                    # Add timestamp if not present
                    if 'scraped_at' not in item:
                        item['scraped_at'] = datetime.now()
                    
                    # Ensure other fields are present
                    for field in ['description', 'url', 'published_date', 'categories', 'symbols']:
                        if field not in item:
                            if field in ['categories', 'symbols']:
                                item[field] = []
                            elif field == 'published_date':
                                item[field] = datetime.now()
                            else:
                                item[field] = ""
                    
                    # Insert to database - with explicit error handling
                    try:
                        self.logger.debug(f"Inserting news item {idx+1}/{len(news_items)}: '{item['title']}'")
                        result = self.db.news_collection.insert_one(item)
                        
                        if result and result.inserted_id:
                            saved_count += 1
                            self.logger.debug(f"Successfully saved item with ID: {result.inserted_id}")
                        else:
                            self.logger.warning(f"Failed to save item: '{item['title']}' - No inserted_id returned")
                            errors += 1
                            
                    except pymongo.errors.PyMongoError as mongo_err:
                        self.logger.error(f"MongoDB error saving item: '{item['title']}' - {mongo_err}")
                        errors += 1
                else:
                    already_exists += 1
                    self.logger.debug(f"Item already exists: '{item['title']}'")
                    
            except Exception as e:
                self.logger.error(f"Error processing news item for database: {e}")
                # Print the full exception traceback for debugging
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                
                log_error(e, context={"action": "save_news_to_db", "title": item.get('title', 'Unknown')})
                errors += 1
        
        self.logger.info(f"Database save results: {saved_count} new, {already_exists} existing, {errors} errors")
        return saved_count

    def _generate_news_hash(self, news_item: Dict[str, Any]) -> str:
        """
        Generate a unique hash for a news item
        
        Args:
            news_item (dict): News item
            
        Returns:
            str: Hash string
        """
        # Create a string with title, source, and date for hashing
        hash_input = f"{news_item.get('title', '')}"
        hash_input += f"|{news_item.get('source', '')}"
        
        if 'published_date' in news_item:
            if isinstance(news_item['published_date'], datetime):
                hash_input += f"|{news_item['published_date'].isoformat()}"
            else:
                hash_input += f"|{news_item['published_date']}"
        
        # Generate MD5 hash
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _collect_zerodha_pulse(self, symbol: str, exchange: str = None, keywords: List[str] = None, 
                              start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Collect news from Zerodha Pulse
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            keywords (list): Keywords to search for
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            list: List of news items
        """
        news_items = []
        
        # Get config
        config = self.sources['zerodha_pulse']
        base_url = config['base_url']
        search_format = config['format']
        
        # Try each keyword and symbol as a hashtag
        search_terms = [symbol.lower()] + [kw.lower() for kw in keywords if kw.lower() != symbol.lower()]
        
        # Also add sector-specific tags
        sector_tags = ["results", "bank", "finance", "market", "stocks"]
        search_terms.extend(sector_tags)
        
        for term in search_terms:
            try:
                # Remove spaces and special chars for hashtag
                tag = term.replace(' ', '').replace('-', '').replace('&', '')
                url = search_format.format(tag)
                self.logger.info(f"Searching Zerodha Pulse for tag: #{tag}")
                
                # Make request
                response = self.retry_request(url)
                
                if not response:
                    self.logger.warning(f"Failed to get response from {url}")
                    continue
                
                # Save debug content
                if self.debug_mode:
                    self.save_debug_content(f"zerodha_pulse_{tag}", response.text)
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract posts - adjust selectors as needed for Pulse's actual structure
                posts = soup.select('.post, article, .post-item')
                self.logger.info(f"Found {len(posts)} posts for tag #{tag}")
                
                if len(posts) == 0:
                    # Try alternative selectors
                    posts = soup.select('.feed-item, .feed-card')
                    self.logger.info(f"Found {len(posts)} posts using alternative selector")
                
                for post in posts:
                    try:
                        # Extract title
                        title_elem = post.select_one('h2, h3, .post-title, .feed-title')
                        if not title_elem:
                            continue
                            
                        title = title_elem.text.strip()
                        
                        # Check if relevant to symbol or keywords
                        is_relevant = False
                        for kw in keywords:
                            if kw.lower() in title.lower():
                                is_relevant = True
                                self.logger.debug(f"Found relevant post with keyword '{kw}': {title}")
                                break
                        
                        if not is_relevant and term not in sector_tags:
                            # For non-sector tags, require keyword relevance
                            continue
                        
                        # Extract description
                        desc_elem = post.select_one('p, .post-content, .feed-content')
                        description = desc_elem.text.strip() if desc_elem else ""
                        
                        # Double-check relevance with description if not already confirmed
                        if not is_relevant:
                            for kw in keywords:
                                if kw.lower() in description.lower():
                                    is_relevant = True
                                    self.logger.debug(f"Found relevant post with keyword '{kw}' in description")
                                    break
                            
                            if not is_relevant:
                                continue
                        
                        # Extract date
                        date_elem = post.select_one('.post-date, time, .date, .timestamp')
                        published_date = None
                        
                        if date_elem:
                            date_text = date_elem.text.strip()
                            try:
                                # Parse date - try different formats
                                for date_format in ['%d %b %Y', '%B %d, %Y', '%d %b, %Y', '%d-%m-%Y', '%Y-%m-%d']:
                                    try:
                                        published_date = datetime.strptime(date_text, date_format)
                                        break
                                    except ValueError:
                                        continue
                                
                                # Handle relative dates like "2 hours ago"
                                if not published_date and ('ago' in date_text.lower()):
                                    if 'min' in date_text.lower():
                                        mins = int(re.search(r'(\d+)', date_text).group(1))
                                        published_date = datetime.now() - timedelta(minutes=mins)
                                    elif 'hour' in date_text.lower():
                                        hours = int(re.search(r'(\d+)', date_text).group(1))
                                        published_date = datetime.now() - timedelta(hours=hours)
                                    elif 'day' in date_text.lower():
                                        days = int(re.search(r'(\d+)', date_text).group(1))
                                        published_date = datetime.now() - timedelta(days=days)
                                    else:
                                        published_date = datetime.now()
                                
                                if not published_date:
                                    published_date = datetime.now()
                            except Exception as e:
                                self.logger.debug(f"Error parsing date '{date_text}': {e}")
                                published_date = datetime.now()
                        else:
                            published_date = datetime.now()
                        
                        # Skip if outside date range
                        if start_date and end_date:
                            if published_date < start_date or published_date > end_date:
                                continue
                        
                        # Extract URL
                        url = ""
                        link_elem = post.find('a')
                        if link_elem and 'href' in link_elem.attrs:
                            url = link_elem['href']
                            if not url.startswith('http'):
                                url = urljoin(base_url, url)
                        
                        # Create news item
                        news_item = {
                            'title': title,
                            'description': description,
                            'url': url,
                            'published_date': published_date,
                            'source': 'zerodha_pulse',
                            'categories': [tag],  # Add tag as category
                            'symbols': [symbol]
                        }
                        
                        # Add to results
                        news_items.append(news_item)
                        self.logger.debug(f"Found article: {title}")
                        
                    except Exception as e:
                        self.logger.error(f"Error parsing Zerodha Pulse post: {e}")
                
                # Delay between requests
                time.sleep(self.scraping_delay)
                
            except Exception as e:
                self.logger.error(f"Error searching Zerodha Pulse for tag '{term}': {e}")
        
        self.logger.info(f"Collected {len(news_items)} news items from Zerodha Pulse")
        return news_items
    
    def _collect_zerodha_markets(self, symbol: str, exchange: str = None, keywords: List[str] = None, 
                            start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Collect news from Zerodha Markets
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            keywords (list): Keywords to search for
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            list: List of news items
        """
        news_items = []
        
        # Get config
        config = self.sources['zerodha_markets']
        
        # Special cases for known symbols - this is the most reliable approach
        special_cases = {
            'SBIN': 'state-bank-of-india',
            'INFY': 'infosys',
            'HDFCBANK': 'hdfc-bank',
            'INDUSINDBK': 'indusind-bank',
            'TCS': 'tata-consultancy-services',
            'RELIANCE': 'reliance-industries',
            'ICICIBANK': 'icici-bank',
            'TATAMOTORS': 'tata-motors',
            'TATASTEEL': 'tata-steel',
            'AXISBANK': 'axis-bank',
            'BAJAJFINSV': 'bajaj-finserv',
            'BAJFINANCE': 'bajaj-finance',
            'BHARTIARTL': 'bharti-airtel',
            'KOTAKBANK': 'kotak-mahindra-bank',
            'MARUTI': 'maruti-suzuki-india',
            'SUNPHARMA': 'sun-pharmaceutical-industries',
            'WIPRO': 'wipro'
        }
        
        # Prepare a list of possible URL formats to try
        url_formats = []
        
        # First check the special cases
        if symbol in special_cases:
            url_formats.append(config['url_format'].format(special_cases[symbol]))
        
        # Get company name from portfolio collection or keywords
        company_name = None
        if exchange:
            # First check in portfolio collection
            portfolio_entry = self.db.portfolio_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            })
            
            # Check both possible structures
            if portfolio_entry:
                if 'company_name' in portfolio_entry and portfolio_entry['company_name']:
                    company_name = portfolio_entry['company_name']
                elif 'company_details' in portfolio_entry and portfolio_entry['company_details']:
                    if 'company_name' in portfolio_entry['company_details'] and portfolio_entry['company_details']['company_name']:
                        company_name = portfolio_entry['company_details']['company_name']
        
        # If we have a company name, add URL formats based on it
        if company_name:
            # Format 1: Just company name in kebab-case without -limited suffix
            url_name = company_name.lower().replace('&', 'and').replace(' ', '-').replace('.', '').replace(',', '')
            url_name = re.sub(r'(-ltd|-limited)$', '', url_name)
            url_formats.append(config['url_format'].format(url_name))
            
            # Format 2: Company name with -limited suffix
            url_formats.append(config['url_format'].format(url_name + '-limited'))
        
        # Try with symbol directly 
        url_name = symbol.lower().replace('&', 'and').replace(' ', '-').replace('.', '').replace(',', '')
        url_formats.append(config['url_format'].format(url_name))
        
        # Try with keywords
        if keywords:
            # Get company name from keywords (excluding the symbol itself)
            company_names = [kw for kw in keywords if kw.lower() != symbol.lower()]
            
            # Try with first keyword that might be a company name
            for kw in company_names:
                if len(kw) > 3 and kw.lower() not in ['ltd', 'limited', 'inc', 'corp']:
                    url_name = kw.lower().replace('&', 'and').replace(' ', '-').replace('.', '').replace(',', '')
                    url_formats.append(config['url_format'].format(url_name))
        
        # Remove duplicates
        url_formats = list(dict.fromkeys(url_formats))
        
        # Try each URL format until we get a response
        response = None
        successful_url = None
        
        for url in url_formats:
            try:
                self.logger.info(f"Fetching from Zerodha Markets: {url}")
                response = self.retry_request(url)
                
                if response and response.status_code == 200:
                    successful_url = url
                    break
                else:
                    self.logger.warning(f"Failed to get valid response from {url}")
            except Exception as e:
                self.logger.error(f"Error requesting {url}: {e}")
        
        if not response or not successful_url:
            self.logger.warning("All URL attempts failed for Zerodha Markets")
            return news_items
        
        # Save debug content
        if self.debug_mode:
            self.save_debug_content("zerodha_markets", response.text)
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find news container - Zerodha might have different section IDs
        news_section = None
        news_section_selectors = ['#news', '.news-section', '.company-news', '#company-news', '#related-news', '#recent-news']
        
        for selector in news_section_selectors:
            section = soup.select_one(selector)
            if section:
                news_section = section
                self.logger.info(f"Found news section with selector: {selector}")
                break
        
        # Set articles to an empty list initially
        articles = []
        
        if not news_section:
            self.logger.warning("No explicit news section found, looking for articles in the main page")
            
            # Try to look for news items directly with various selectors
            selector_options = [
                '.news-item, .stock-news-item, .news-article', 
                '.news-items li, .news-list li',
                'ul li a', 
                '.news div, .news-block div',
                '.news-feed .item, .feed-item',
                'article, .article-item',
                '.content-section ul li, .section-content ul li'
            ]
            
            # Try each selector until we find articles
            for selector in selector_options:
                articles = soup.select(selector)
                if articles:
                    self.logger.info(f"Found {len(articles)} articles using selector: {selector}")
                    break
            
            if not articles:
                # Last resort - look for any UL/OL with LI elements that contain A tags
                list_elements = soup.select('ul, ol')
                for list_elem in list_elements:
                    list_items = list_elem.select('li')
                    if list_items and all(item.find('a') for item in list_items[:3]):  # Check if first few items have links
                        articles = list_items
                        self.logger.info(f"Found {len(articles)} articles in list element")
                        break
        else:
            # Extract news items from the news section
            selector_options = [
                '.news-item, li, .news-article, .news-link',
                'li a, div a',
                'p a, .content a'
            ]
            
            # Try each selector until we find articles
            for selector in selector_options:
                articles = news_section.select(selector)
                if articles:
                    self.logger.info(f"Found {len(articles)} articles in news section using selector: {selector}")
                    break
        
        if not articles:
            self.logger.warning("No news items found on Zerodha Markets page")
            return news_items
        
        self.logger.info(f"Successfully found {len(articles)} articles on Zerodha Markets page")
        
        for article in articles:
            try:
                # Extract title and URL
                title = ""
                url = ""
                
                # First, try to get from a tags directly
                if article.name == 'a':
                    title = article.text.strip()
                    if 'href' in article.attrs:
                        url = article['href']
                else:
                    # Otherwise look for a tag within the element
                    link_elem = article.find('a')
                    if link_elem:
                        title = link_elem.text.strip()
                        if 'href' in link_elem.attrs:
                            url = link_elem['href']
                    else:
                        # If still no title/link, try other elements
                        title_elem = article.select_one('h3, h4, .title, .headline, strong')
                        if title_elem:
                            title = title_elem.text.strip()
                
                # Skip if we couldn't extract a title
                if not title:
                    continue
                    
                # Skip items that are clearly not news (like "About Us", "Contact", etc.)
                skip_terms = ["about us", "contact", "privacy policy", "terms", "home"]
                if any(term in title.lower() for term in skip_terms):
                    continue
                
                # Format URL if needed
                if url and not url.startswith('http'):
                    url = urljoin(config['base_url'], url)
                
                # Extract description
                desc_elem = article.select_one('p, .description, .summary')
                description = desc_elem.text.strip() if desc_elem else ""
                
                # Extract date (with fallback to current date)
                date_elem = article.select_one('.date, time, .timestamp, .news-date')
                published_date = None
                
                if date_elem:
                    date_text = date_elem.text.strip()
                    try:
                        # Parse date - try different formats
                        for date_format in ['%d %b %Y', '%b %d, %Y', '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y']:
                            try:
                                published_date = datetime.strptime(date_text, date_format)
                                break
                            except ValueError:
                                continue
                        
                        # Handle relative dates
                        if not published_date and ('ago' in date_text.lower()):
                            if 'min' in date_text.lower():
                                mins = int(re.search(r'(\d+)', date_text).group(1))
                                published_date = datetime.now() - timedelta(minutes=mins)
                            elif 'hour' in date_text.lower():
                                hours = int(re.search(r'(\d+)', date_text).group(1))
                                published_date = datetime.now() - timedelta(hours=hours)
                            elif 'day' in date_text.lower():
                                days = int(re.search(r'(\d+)', date_text).group(1))
                                published_date = datetime.now() - timedelta(days=days)
                            else:
                                published_date = datetime.now()
                        
                        if not published_date:
                            published_date = datetime.now()
                    except Exception as e:
                        self.logger.debug(f"Error parsing date '{date_text}': {e}")
                        published_date = datetime.now()
                else:
                    published_date = datetime.now()
                
                # Skip if outside date range
                if start_date and end_date:
                    if published_date < start_date or published_date > end_date:
                        continue
                
                # Create news item
                news_item = {
                    'title': title,
                    'description': description,
                    'url': url,
                    'published_date': published_date,
                    'source': 'zerodha_markets',
                    'categories': [],
                    'symbols': [symbol]
                }
                
                # Add to results
                news_items.append(news_item)
                self.logger.debug(f"Added article: {title}")
                
            except Exception as e:
                self.logger.error(f"Error parsing Zerodha Markets article: {e}")
        
        self.logger.info(f"Collected {len(news_items)} news items from Zerodha Markets")
        return news_items

    def _collect_hindu_business(self, symbol: str, exchange: str = None, keywords: List[str] = None, 
                               start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Collect news from The Hindu Business
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            keywords (list): Keywords to search for
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            list: List of news items
        """
        news_items = []
        
        # Get config
        config = self.sources['hindu_business']
        url = config['url']
        
        try:
            self.logger.info(f"Fetching from The Hindu Business: {url}")
            
            # Make request
            response = self.retry_request(url)
            
            if not response:
                self.logger.warning(f"Failed to get response from {url}")
                return news_items
            
            # Save debug content
            if self.debug_mode:
                self.save_debug_content("hindu_business", response.text)
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract news items
            articles = soup.select('.story-card, article, .archive-list li')
            
            self.logger.info(f"Found {len(articles)} articles on The Hindu Business page")
            
            if len(articles) == 0:
                # Try alternative selectors
                articles = soup.select('.article-container, .article')
                self.logger.info(f"Found {len(articles)} articles using alternative selector")
                
                if len(articles) == 0:
                    articles = soup.select('.lead-story, .other-story')
                    self.logger.info(f"Found {len(articles)} articles using second alternative selector")
            
            for article in articles:
                try:
                    # Extract title
                    title_elem = article.select_one('h2, h3, .title, .story-card-headline')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    
                    # Check if relevant to symbol or keywords
                    is_relevant = False
                    for keyword in keywords:
                        if keyword.lower() in title.lower():
                            is_relevant = True
                            self.logger.debug(f"Found relevant article with keyword '{keyword}': {title}")
                            break
                    
                    if not is_relevant:
                        continue
                    
                    # Extract URL
                    url = ""
                    link_elem = title_elem if title_elem.name == 'a' else title_elem.find('a')
                    if link_elem and 'href' in link_elem.attrs:
                        url = link_elem['href']
                        if not url.startswith('http'):
                            url = urljoin('https://www.thehindu.com', url)
                    
                    # Extract description
                    desc_elem = article.select_one('p, .summary, .story-card-text')
                    description = desc_elem.text.strip() if desc_elem else ""
                    
                    # Extract date
                    date_elem = article.select_one('.dateline, .timestamp, .date, .update-time')
                    published_date = None
                    
                    if date_elem:
                        date_text = date_elem.text.strip()
                        try:
                            # Parse date - try different formats
                            for date_format in ['%B %d, %Y', '%b %d, %Y', '%d %b %Y', '%Y-%m-%d']:
                                try:
                                    published_date = datetime.strptime(date_text, date_format)
                                    break
                                except ValueError:
                                    continue
                            
                            if not published_date:
                                # Try to extract date using regex
                                date_match = re.search(r'(\w+ \d{1,2}, \d{4})', date_text)
                                if date_match:
                                    try:
                                        published_date = datetime.strptime(date_match.group(1), '%B %d, %Y')
                                    except ValueError:
                                        published_date = datetime.now()
                                else:
                                    published_date = datetime.now()
                        except Exception as e:
                            self.logger.debug(f"Error parsing date '{date_text}': {e}")
                            published_date = datetime.now()
                    else:
                        published_date = datetime.now()
                    
                    # Skip if outside date range
                    if start_date and end_date:
                        if published_date < start_date or published_date > end_date:
                            continue
                    
                    # Create news item
                    news_item = {
                        'title': title,
                        'description': description,
                        'url': url,
                        'published_date': published_date,
                        'source': 'hindu_business',
                        'categories': [],
                        'symbols': [symbol]
                    }
                    
                    # Add to results
                    news_items.append(news_item)
                    
                except Exception as e:
                    self.logger.error(f"Error parsing The Hindu article: {e}")
            
        except Exception as e:
            self.logger.error(f"Error fetching The Hindu Business page: {e}")
        
        self.logger.info(f"Collected {len(news_items)} news items from The Hindu Business")
        return news_items
    
    def _collect_times_of_india(self, symbol: str, exchange: str = None, keywords: List[str] = None, 
                              start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Collect news from Times of India
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            keywords (list): Keywords to search for
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            list: List of news items
        """
        news_items = []
        
        # Get config
        config = self.sources['times_of_india']
        topic_url_format = config['topic_url']
        
        # Search for each keyword
        for keyword in keywords:
            try:
                # Format keyword for URL
                url_keyword = keyword.lower().replace(' ', '-').replace('&', 'and')
                url = topic_url_format.format(url_keyword)
                
                self.logger.info(f"Searching Times of India for topic: {keyword}")
                
                # Make request
                response = self.retry_request(url)
                
                if not response:
                    self.logger.warning(f"Failed to get response from {url}")
                    continue
                
                # Save debug content
                if self.debug_mode:
                    self.save_debug_content(f"times_of_india_{url_keyword}", response.text)
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract articles
                articles = soup.select('.article, .topic-listing li, .card-container')
                
                self.logger.info(f"Found {len(articles)} articles for topic '{keyword}'")
                
                if len(articles) == 0:
                    # Try alternative selectors
                    articles = soup.select('.news-card, .media-body')
                    self.logger.info(f"Found {len(articles)} articles using alternative selector")
                
                for article in articles:
                    try:
                        # Extract title
                        title_elem = article.select_one('h2, h3, .title, .headline, a[title]')
                        if not title_elem:
                            continue
                            
                        title = title_elem.text.strip()
                        
                        # Extract URL
                        url = ""
                        link_elem = title_elem if title_elem.name == 'a' else article.find('a')
                        if link_elem and 'href' in link_elem.attrs:
                            url = link_elem['href']
                            if not url.startswith('http'):
                                url = urljoin(config['base_url'], url)
                        
                        # Extract description
                        desc_elem = article.select_one('p, .summary, .content')
                        description = desc_elem.text.strip() if desc_elem else ""
                        
                        # Extract date
                        date_elem = article.select_one('.date, time, .timestamp, .meta')
                        published_date = None
                        
                        if date_elem:
                            date_text = date_elem.text.strip()
                            try:
                                # Parse date - try different formats
                                for date_format in ['%b %d, %Y', '%d %b %Y', '%B %d, %Y', '%Y-%m-%d']:
                                    try:
                                        published_date = datetime.strptime(date_text, date_format)
                                        break
                                    except ValueError:
                                        continue
                                
                                # Handle relative dates
                                if not published_date and ('ago' in date_text.lower()):
                                    if 'min' in date_text.lower():
                                        mins = int(re.search(r'(\d+)', date_text).group(1))
                                        published_date = datetime.now() - timedelta(minutes=mins)
                                    elif 'hour' in date_text.lower():
                                        hours = int(re.search(r'(\d+)', date_text).group(1))
                                        published_date = datetime.now() - timedelta(hours=hours)
                                    elif 'day' in date_text.lower():
                                        days = int(re.search(r'(\d+)', date_text).group(1))
                                        published_date = datetime.now() - timedelta(days=days)
                                    else:
                                        published_date = datetime.now()
                                
                                if not published_date:
                                    # Try to extract date using regex
                                    date_match = re.search(r'(\w+ \d{1,2}, \d{4})', date_text)
                                    if date_match:
                                        try:
                                            published_date = datetime.strptime(date_match.group(1), '%B %d, %Y')
                                        except ValueError:
                                            published_date = datetime.now()
                                    else:
                                        published_date = datetime.now()
                            except Exception as e:
                                self.logger.debug(f"Error parsing date '{date_text}': {e}")
                                published_date = datetime.now()
                        else:
                            published_date = datetime.now()
                        
                        # Skip if outside date range
                        if start_date and end_date:
                            if published_date < start_date or published_date > end_date:
                                continue
                        
                        # Create news item
                        news_item = {
                            'title': title,
                            'description': description,
                            'url': url,
                            'published_date': published_date,
                            'source': 'times_of_india',
                            'categories': [keyword],
                            'symbols': [symbol]
                        }
                        
                        # Add to results
                        news_items.append(news_item)
                        
                    except Exception as e:
                        self.logger.error(f"Error parsing Times of India article: {e}")
                
                # Delay between requests
                time.sleep(self.scraping_delay)
                
            except Exception as e:
                self.logger.error(f"Error searching Times of India for keyword '{keyword}': {e}")
        
        self.logger.info(f"Collected {len(news_items)} news items from Times of India")
        return news_items
    
    def get_recent_news(self, symbol: str = None, exchange: str = None, days: int = 7, 
                      limit: int = 20, categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent news from database
        
        Args:
            symbol (str, optional): Filter by symbol
            exchange (str, optional): Filter by exchange
            days (int): Number of days to look back
            limit (int): Maximum number of news items to return
            categories (list, optional): Filter by categories
            
        Returns:
            list: List of news items
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Build query
        query = {
            "published_date": {
                "$gte": start_date,
                "$lte": end_date
            }
        }
        
        # Add symbol filter if provided
        if symbol:
            # Normalize symbol
            normalized_symbol = normalize_symbol(symbol, exchange) if exchange else symbol
            query["symbols"] = normalized_symbol
        
        # Add categories filter if provided
        if categories:
            categories_query = []
            for category in categories:
                categories_query.append({"categories": category})
            
            if categories_query:
                query["$or"] = categories_query
        
        self.logger.info(f"Querying database for recent news with filters: {query}")
        
        try:
            # Get news from database
            news_items = list(self.db.news_collection.find(
                query,
                sort=[("published_date", -1)],
                limit=limit
            ))
            
            # Convert ObjectId to string for JSON serialization
            for item in news_items:
                if '_id' in item:
                    item['_id'] = str(item['_id'])
            
            self.logger.info(f"Found {len(news_items)} recent news items")
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error querying database for recent news: {e}")
            return []
    
    def search_news(self, query: str, days: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search news by keyword
        
        Args:
            query (str): Search query
            days (int): Number of days to look back
            limit (int): Maximum number of news items to return
            
        Returns:
            list: List of news items
        """
        self.logger.info(f"Searching news for query: '{query}' (last {days} days)")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Build MongoDB query
        db_query = {
            "published_date": {
                "$gte": start_date,
                "$lte": end_date
            },
            "$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}}
            ]
        }
        
        try:
            # Get news from database
            news_items = list(self.db.news_collection.find(
                db_query,
                sort=[("published_date", -1)],
                limit=limit
            ))
            
            # Convert ObjectId to string for JSON serialization
            for item in news_items:
                if '_id' in item:
                    item['_id'] = str(item['_id'])
            
            self.logger.info(f"Found {len(news_items)} news items matching query '{query}'")
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error searching news database: {e}")
            return []


# Helper function for standalone usage
def collect_news_for_symbol(symbol, exchange=None, days=30, limit=50, debug_mode=True):
    """
    Standalone helper function to collect news for a given symbol
    
    Args:
        symbol (str): Stock symbol
        exchange (str, optional): Exchange code
        days (int): Number of days to look back
        limit (int): Maximum number of news items to return
        debug_mode (bool): Whether to enable debug mode
        
    Returns:
        list: Collected news items
    """
    print(f"Collecting news for {symbol}{':'+exchange if exchange else ''} (last {days} days)")
    
    # Create aggregator
    from database.connection_manager import get_db
    db = get_db()
    aggregator = NewsAggregator(db_connector=db, debug_mode=debug_mode)
    
    # Collect news
    news_items = aggregator.collect_news(symbol, exchange, days=days, limit=limit)
    
    print(f"Collected {len(news_items)} news items for {symbol}")
    
    return news_items


# Main function for standalone usage
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='News Aggregator Tool')
    parser.add_argument('--symbol', '-s', type=str, required=True, help='Stock symbol')
    parser.add_argument('--exchange', '-e', type=str, help='Exchange code')
    parser.add_argument('--days', '-d', type=int, default=30, help='Number of days to look back')
    parser.add_argument('--limit', '-l', type=int, default=50, help='Maximum number of news items')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--output', '-o', type=str, help='Output file (JSON format)')
    
    args = parser.parse_args()
    
    # Collect news
    news_items = collect_news_for_symbol(
        args.symbol, 
        args.exchange, 
        days=args.days, 
        limit=args.limit,
        debug_mode=args.debug
    )
    
    # Print summary
    print(f"\nCollected {len(news_items)} news items:")
    
    for i, item in enumerate(news_items[:10], 1):
        print(f"{i}. {item['title']} ({item['source']}, {item['published_date']})")
    
    if len(news_items) > 10:
        print(f"...and {len(news_items) - 10} more")
    
    # Save to file if requested
    if args.output:
        import json
        
        # Convert datetime objects to string
        for item in news_items:
            if isinstance(item.get('published_date'), datetime):
                item['published_date'] = item['published_date'].isoformat()
            if isinstance(item.get('scraped_at'), datetime):
                item['scraped_at'] = item['scraped_at'].isoformat()
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(news_items, f, indent=2)
        
        print(f"\nSaved {len(news_items)} news items to {args.output}")