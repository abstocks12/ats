"""
News Aggregator Module for the Automated Trading System.
Collects and aggregates news from multiple sources.
"""

import os
import sys
import time
from datetime import datetime, timedelta
import hashlib
import re
from typing import List, Dict, Any, Optional, Union, Set
from bs4 import BeautifulSoup
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings
from database.connection_manager import get_db
from utils.logging_utils import setup_logger, log_error, log_data_collection
from utils.helper_functions import retry_function, get_date_range, normalize_symbol, extract_stock_symbol
from data.news.sentiment_analyzer import SentimentAnalyzer

class NewsAggregator:
    """
    Aggregates news from multiple sources, processes them, and stores in database.
    """
    
    def __init__(self, db=None):
        """
        Initialize news aggregator
        
        Args:
            db: Database connector (optional, will use global connection if not provided)
        """
        self.logger = setup_logger(__name__)
        self.db = db or get_db()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Set user agent for requests
        self.headers = {
            'User-Agent': settings.USER_AGENT
        }
        
        # Define scraping delay to avoid rate limiting
        self.scraping_delay = settings.SCRAPING_DELAY
        
        # Initialize source configs
        self.sources = self._initialize_sources()
    
    def _initialize_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize news source configurations
        
        Returns:
            dict: Source configurations
        """
        # Start with default sources from settings
        sources = settings.NEWS_SOURCES.copy()
        
        # Add Zerodha Markets
        sources['zerodha_markets'] = {
            'enabled': True,
            'base_url': 'https://zerodha.com/markets/stocks/',
            'url_template': 'https://zerodha.com/markets/stocks/{}'
        }
        
        # Add Tijori Finance
        sources['tijori_finance'] = {
            'enabled': True,
            'base_url': 'https://www.tijorifinance.com/company/',
            'url_template': 'https://www.tijorifinance.com/company/{}'
        }
        
        return sources
    
    def collect_news(self, symbol: str, exchange: str, days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
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
        self.logger.info(f"Collecting news for {symbol}:{exchange} (last {days} days)")
        
        # Normalize symbol
        normalized_symbol = normalize_symbol(symbol, exchange)
        
        # Get date range
        start_date, end_date = get_date_range(days)
        
        # Get keywords for symbol
        keywords = self._get_symbol_keywords(symbol, exchange)
        
        # Get sector keywords if available
        sector_keywords = self._get_sector_keywords(symbol, exchange)
        
        # Initialize results
        all_news = []
        
        # Collect news from each source
        for source_name, source_config in self.sources.items():
            if not source_config.get('enabled', False):
                continue
            
            try:
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
                    
                    if news_items:
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
                        self.logger.info(f"Collected {len(news_items)} news items from {source_name}")
                    else:
                        self.logger.info(f"No news found from {source_name} for {symbol}")
            
            except Exception as e:
                log_error(e, context={"action": "collect_news", "source": source_name, "symbol": symbol})
        
        # Process and filter collected news
        processed_news = self._process_news_items(all_news, keywords)
        
        # Save to database
        saved_count = self._save_to_database(processed_news)
        
        self.logger.info(f"Saved {saved_count} news items for {symbol}:{exchange}")
        
        # Return the news items (limited by the specified limit)
        return processed_news[:limit] if limit else processed_news
    
    def _get_symbol_keywords(self, symbol: str, exchange: str) -> List[str]:
        """
        Get keywords for a symbol
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            list: List of keywords
        """
        # Get instrument details
        instrument = self.db.portfolio_collection.find_one({
            "symbol": symbol,
            "exchange": exchange
        })
        
        keywords = [symbol]
        
        if instrument:
            # Add company name if available
            if 'company_name' in instrument:
                keywords.append(instrument['company_name'])
                
                # Add parts of company name
                parts = instrument['company_name'].split()
                if len(parts) > 1:
                    for part in parts:
                        if len(part) > 3 and part.lower() not in ['ltd', 'limited', 'inc', 'corp']:
                            keywords.append(part)
        
        # Try to get company details from financial data
        financial_data = self.db.financial_collection.find_one({
            "symbol": symbol,
            "exchange": exchange
        })
        
        if financial_data and 'company_name' in financial_data:
            if financial_data['company_name'] not in keywords:
                keywords.append(financial_data['company_name'])
                
                # Add parts of company name
                parts = financial_data['company_name'].split()
                if len(parts) > 1:
                    for part in parts:
                        if len(part) > 3 and part.lower() not in ['ltd', 'limited', 'inc', 'corp']:
                            if part not in keywords:
                                keywords.append(part)
        
        return keywords
    
    def _get_sector_keywords(self, symbol: str, exchange: str) -> List[str]:
        """
        Get sector keywords for a symbol
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            list: List of sector keywords
        """
        # Get instrument details
        instrument = self.db.portfolio_collection.find_one({
            "symbol": symbol,
            "exchange": exchange
        })
        
        if not instrument or 'sector' not in instrument:
            return []
        
        sector = instrument['sector'].lower()
        
        # Get sector keywords from settings
        sector_keywords = settings.SECTOR_KEYWORDS.get(sector, [])
        
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
        processed_news = []
        seen_hashes = set()
        
        for item in news_items:
            # Generate unique hash for deduplication
            item_hash = self._generate_news_hash(item)
            
            # Skip if duplicate
            if item_hash in seen_hashes:
                continue
            
            seen_hashes.add(item_hash)
            
            # Ensure required fields
            if 'title' not in item:
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
            
            # Add to processed items
            processed_news.append(item)
        
        # Sort by published date (newest first)
        processed_news.sort(key=lambda x: x.get('published_date', datetime.now()), reverse=True)
        
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
            return 0
        
        saved_count = 0
        
        for item in news_items:
            try:
                # Check if already exists
                existing = self.db.news_collection.find_one({
                    "title": item['title'],
                    "source": item.get('source', 'Unknown')
                })
                
                if not existing:
                    # Add timestamp
                    item['scraped_at'] = datetime.now()
                    
                    # Insert to database
                    result = self.db.news_collection.insert_one(item)
                    
                    if result.inserted_id:
                        saved_count += 1
                        
            except Exception as e:
                log_error(e, context={"action": "save_news_to_db", "title": item.get('title', 'Unknown')})
        
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
    
    def _collect_zerodha_pulse(self, symbol: str, exchange: str, keywords: List[str], 
                              start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
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
        # Defer to pulse_scraper.py if available
        try:
            from data.news.pulse_scraper import PulseScraper
            
            # Initialize scraper
            scraper = PulseScraper(self.db)
            
            # Run scraper for this symbol
            return scraper.scrape_for_symbol(symbol, exchange, days=(end_date - start_date).days)
            
        except ImportError:
            self.logger.warning("PulseScraper not available, using fallback implementation")
            
            # Fallback implementation
            news_items = []
            
            # Construct URL
            base_url = self.sources['zerodha_pulse']['base_url']
            search_url = self.sources['zerodha_pulse']['search_url']
            
            # Try each keyword
            for keyword in keywords:
                try:
                    url = search_url.format(keyword)
                    
                    # Make request
                    response = requests.get(url, headers=self.headers)
                    
                    if response.status_code != 200:
                        continue
                    
                    # Parse HTML
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract posts
                    posts = soup.select('.post')
                    
                    for post in posts:
                        try:
                            # Extract title
                            title_elem = post.select_one('.post-title')
                            if not title_elem:
                                continue
                                
                            title = title_elem.text.strip()
                            
                            # Extract description
                            desc_elem = post.select_one('.post-content')
                            description = desc_elem.text.strip() if desc_elem else ""
                            
                            # Extract date
                            date_elem = post.select_one('.post-date')
                            published_date = None
                            
                            if date_elem:
                                date_text = date_elem.text.strip()
                                try:
                                    # Parse date
                                    published_date = datetime.strptime(date_text, '%d %b %Y')
                                except ValueError:
                                    published_date = datetime.now()
                            else:
                                published_date = datetime.now()
                            
                            # Skip if outside date range
                            if published_date < start_date or published_date > end_date:
                                continue
                            
                            # Extract URL
                            url = ""
                            link_elem = post.select_one('a')
                            if link_elem and 'href' in link_elem.attrs:
                                url = link_elem['href']
                                if not url.startswith('http'):
                                    url = base_url + url.lstrip('/')
                            
                            # Create news item
                            news_item = {
                                'title': title,
                                'description': description,
                                'url': url,
                                'published_date': published_date,
                                'source': 'zerodha_pulse',
                                'categories': [],
                                'symbols': [symbol]
                            }
                            
                            # Add to results
                            news_items.append(news_item)
                            
                        except Exception as e:
                            log_error(e, context={"action": "parse_pulse_post", "symbol": symbol})
                    
                    # Delay between requests
                    time.sleep(self.scraping_delay)
                    
                except Exception as e:
                    log_error(e, context={"action": "collect_zerodha_pulse", "keyword": keyword})
            
            return news_items
    
    def _collect_economic_times(self, symbol: str, exchange: str, keywords: List[str], 
                               start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Collect news from Economic Times
        
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
        config = self.sources['economic_times']
        
        # Try markets and banking URLs
        urls = [config['markets_url']]
        
        if 'banking_url' in config:
            urls.append(config['banking_url'])
        
        for url in urls:
            try:
                # Make request
                response = requests.get(url, headers=self.headers)
                
                if response.status_code != 200:
                    continue
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract news items
                articles = soup.select('.article')
                
                for article in articles:
                    try:
                        # Extract title
                        title_elem = article.select_one('.title')
                        if not title_elem:
                            continue
                            
                        title = title_elem.text.strip()
                        
                        # Check if relevant to symbol
                        is_relevant = False
                        for keyword in keywords:
                            if keyword.lower() in title.lower():
                                is_relevant = True
                                break
                        
                        if not is_relevant:
                            continue
                        
                        # Extract URL
                        url = ""
                        link_elem = title_elem.find('a')
                        if link_elem and 'href' in link_elem.attrs:
                            url = link_elem['href']
                            if not url.startswith('http'):
                                url = 'https://economictimes.indiatimes.com' + url
                        
                        # Extract description
                        desc_elem = article.select_one('.summary')
                        description = desc_elem.text.strip() if desc_elem else ""
                        
                        # Extract date
                        date_elem = article.select_one('.date')
                        published_date = None
                        
                        if date_elem:
                            date_text = date_elem.text.strip()
                            try:
                                # Parse date
                                published_date = datetime.strptime(date_text, '%d %b %Y, %H:%M %p')
                            except ValueError:
                                try:
                                    published_date = datetime.strptime(date_text, '%d %b %Y')
                                except ValueError:
                                    published_date = datetime.now()
                        else:
                            published_date = datetime.now()
                        
                        # Skip if outside date range
                        if published_date < start_date or published_date > end_date:
                            continue
                        
                        # Create news item
                        news_item = {
                            'title': title,
                            'description': description,
                            'url': url,
                            'published_date': published_date,
                            'source': 'economic_times',
                            'categories': [],
                            'symbols': [symbol]
                        }
                        
                        # Add to results
                        news_items.append(news_item)
                        
                    except Exception as e:
                        log_error(e, context={"action": "parse_et_article", "symbol": symbol})
                
                # Delay between requests
                time.sleep(self.scraping_delay)
                
            except Exception as e:
                log_error(e, context={"action": "collect_economic_times", "url": url})
        
        return news_items
    
    def _collect_bloomberg(self, symbol: str, exchange: str, keywords: List[str], 
                          start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Collect news from Bloomberg
        
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
        config = self.sources['bloomberg']
        url = config['url']
        
        try:
            # Make request
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                return news_items
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract news items
            articles = soup.select('.story-package-module__story')
            
            for article in articles:
                try:
                    # Extract title
                    title_elem = article.select_one('.story-package-module__headline')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    
                    # Check if relevant to symbol
                    is_relevant = False
                    for keyword in keywords:
                        if keyword.lower() in title.lower():
                            is_relevant = True
                            break
                    
                    if not is_relevant:
                        continue
                    
                    # Extract URL
                    url = ""
                    link_elem = article.find('a')
                    if link_elem and 'href' in link_elem.attrs:
                        url = link_elem['href']
                        if not url.startswith('http'):
                            url = 'https://www.bloombergquint.com' + url
                    
                    # Extract description
                    desc_elem = article.select_one('.story-package-module__summary')
                    description = desc_elem.text.strip() if desc_elem else ""
                    
                    # Extract date
                    date_elem = article.select_one('.story-package-module__timestamp')
                    published_date = None
                    
                    if date_elem:
                        date_text = date_elem.text.strip()
                        try:
                            # Parse date
                            if 'ago' in date_text.lower():
                                # Handle relative time (e.g. "2 hours ago")
                                published_date = datetime.now()
                            else:
                                published_date = datetime.strptime(date_text, '%b %d %Y')
                        except ValueError:
                            published_date = datetime.now()
                    else:
                        published_date = datetime.now()
                    
                    # Skip if outside date range
                    if published_date < start_date or published_date > end_date:
                        continue
                    
                    # Create news item
                    news_item = {
                        'title': title,
                        'description': description,
                        'url': url,
                        'published_date': published_date,
                        'source': 'bloomberg',
                        'categories': [],
                        'symbols': [symbol]
                    }
                    
                    # Add to results
                    news_items.append(news_item)
                    
                except Exception as e:
                    log_error(e, context={"action": "parse_bloomberg_article", "symbol": symbol})
            
        except Exception as e:
            log_error(e, context={"action": "collect_bloomberg", "symbol": symbol})
        
        return news_items
    
    def _collect_hindu_business(self, symbol: str, exchange: str, keywords: List[str], 
                               start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
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
            # Make request
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                return news_items
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract news items
            articles = soup.select('.story-card')
            
            for article in articles:
                try:
                    # Extract title
                    title_elem = article.select_one('.story-card-headline')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    
                    # Check if relevant to symbol
                    is_relevant = False
                    for keyword in keywords:
                        if keyword.lower() in title.lower():
                            is_relevant = True
                            break
                    
                    if not is_relevant:
                        continue
                    
                    # Extract URL
                    url = ""
                    link_elem = article.find('a')
                    if link_elem and 'href' in link_elem.attrs:
                        url = link_elem['href']
                        if not url.startswith('http'):
                            url = 'https://www.thehindu.com' + url
                    
                    # Extract description
                    desc_elem = article.select_one('.story-card-text')
                    description = desc_elem.text.strip() if desc_elem else ""
                    
                    # Extract date
                    date_elem = article.select_one('.dateline')
                    published_date = None
                    
                    if date_elem:
                        date_text = date_elem.text.strip()
                        try:
                            # Parse date
                            published_date = datetime.strptime(date_text, '%B %d, %Y')
                        except ValueError:
                            published_date = datetime.now()
                    else:
                        published_date = datetime.now()
                    
                    # Skip if outside date range
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
                    log_error(e, context={"action": "parse_hindu_article", "symbol": symbol})
            
        except Exception as e:
            log_error(e, context={"action": "collect_hindu_business", "symbol": symbol})
        
        return news_items
    
    def _collect_zerodha_markets(self, symbol: str, exchange: str, keywords: List[str], 
                                start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
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
        
        # Generate URL for this symbol
        # Convert to lowercase and hyphenate for URL
        url_symbol = symbol.lower().replace('&', 'and').replace(' ', '-')
        url = config['url_template'].format(url_symbol)
        
        try:
            # Make request
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                return news_items
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract news section
            news_section = soup.select_one('#news')
            
            if not news_section:
                return news_items
            
            # Extract news items
            articles = news_section.select('.news-item')
            
            for article in articles:
                try:
                    # Extract title
                    title_elem = article.select_one('h3')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    
                    # Extract URL
                    url = ""
                    link_elem = article.find('a')
                    if link_elem and 'href' in link_elem.attrs:
                        url = link_elem['href']
                    
                    # Extract description
                    desc_elem = article.select_one('p')
                    description = desc_elem.text.strip() if desc_elem else ""
                    
                    # Extract date
                    date_elem = article.select_one('.date')
                    published_date = None
                    
                    if date_elem:
                        date_text = date_elem.text.strip()
                        try:
                            # Parse date
                            published_date = datetime.strptime(date_text, '%d %b %Y')
                        except ValueError:
                            published_date = datetime.now()
                    else:
                        published_date = datetime.now()
                    
                    # Skip if outside date range
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
                    
                except Exception as e:
                    log_error(e, context={"action": "parse_zerodha_markets_article", "symbol": symbol})
            
        except Exception as e:
            log_error(e, context={"action": "collect_zerodha_markets", "symbol": symbol})
        
        return news_items
    
    def _collect_tijori_finance(self, symbol: str, exchange: str, keywords: List[str], 
                              start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
       """
       Collect news from Tijori Finance
       
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
       config = self.sources['tijori_finance']
       
       # Generate URL for this symbol
       # Convert to lowercase and hyphenate for URL
       url_symbol = symbol.lower().replace('&', 'and').replace(' ', '-')
       url = config['url_template'].format(url_symbol)
       
       try:
           # Make request
           response = requests.get(url, headers=self.headers)
           
           if response.status_code != 200:
               return news_items
           
           # Parse HTML
           soup = BeautifulSoup(response.text, 'html.parser')
           
           # Extract news section
           news_section = soup.select_one('.news-section')
           
           if not news_section:
               # Try alternative selector
               news_section = soup.select_one('#company-news')
               
               if not news_section:
                   return news_items
           
           # Extract news items
           articles = news_section.select('.news-item, .article')
           
           for article in articles:
               try:
                   # Extract title
                   title_elem = article.select_one('h3, .title')
                   if not title_elem:
                       continue
                       
                   title = title_elem.text.strip()
                   
                   # Extract URL
                   url = ""
                   link_elem = article.find('a')
                   if link_elem and 'href' in link_elem.attrs:
                       url = link_elem['href']
                       if not url.startswith('http'):
                           url = config['base_url'] + url.lstrip('/')
                   
                   # Extract description
                   desc_elem = article.select_one('p, .summary')
                   description = desc_elem.text.strip() if desc_elem else ""
                   
                   # Extract date
                   date_elem = article.select_one('.date, .timestamp')
                   published_date = None
                   
                   if date_elem:
                       date_text = date_elem.text.strip()
                       try:
                           # Try different date formats
                           for date_format in ['%d %b %Y', '%B %d, %Y', '%Y-%m-%d']:
                               try:
                                   published_date = datetime.strptime(date_text, date_format)
                                   break
                               except ValueError:
                                   continue
                           
                           if published_date is None:
                               published_date = datetime.now()
                       except Exception:
                           published_date = datetime.now()
                   else:
                       published_date = datetime.now()
                   
                   # Skip if outside date range
                   if published_date < start_date or published_date > end_date:
                       continue
                   
                   # Create news item
                   news_item = {
                       'title': title,
                       'description': description,
                       'url': url,
                       'published_date': published_date,
                       'source': 'tijori_finance',
                       'categories': [],
                       'symbols': [symbol]
                   }
                   
                   # Add to results
                   news_items.append(news_item)
                   
               except Exception as e:
                   log_error(e, context={"action": "parse_tijori_finance_article", "symbol": symbol})
           
       except Exception as e:
           log_error(e, context={"action": "collect_tijori_finance", "symbol": symbol})
       
       return news_items
   
    def get_recent_news(self, symbol: str = None, exchange: str = None, days: int = 7, 
                      limit: int = 20) -> List[Dict[str, Any]]:
       """
       Get recent news from database
       
       Args:
           symbol (str, optional): Filter by symbol
           exchange (str, optional): Filter by exchange
           days (int): Number of days to look back
           limit (int): Maximum number of news items to return
           
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
           normalized_symbol = normalize_symbol(symbol, exchange)
           query["symbols"] = normalized_symbol
       
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
       
       return news_items
   
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
       
       return news_items