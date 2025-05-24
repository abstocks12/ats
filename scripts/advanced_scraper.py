import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import time
import logging
from typing import List, Dict, Any, Optional, Set
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from urllib.parse import urlparse, urljoin
from textblob import TextBlob
import threading
from queue import Queue, PriorityQueue
import pymongo
from pymongo import MongoClient
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveNewsScraper:
    """
    A comprehensive news scraper that collects news from multiple global and Indian sources
    for intraday trading decisions.
    """
    
    def __init__(self, mongodb_uri="mongodb://localhost:27017/", db_name="trading_system"):
        # MongoDB setup
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        
        # Collections
        self.raw_news = self.db.raw_news_collection
        self.processed_news = self.db.processed_news_collection
        self.news_alerts = self.db.news_alerts_collection
        self.news_performance = self.db.news_performance_collection
        
        # Create indexes
        self._create_indexes()
        
        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Deduplication cache
        self.seen_urls = set()
        self.seen_hashes = set()
        self._load_seen_items()
        
        # Processing queues
        self.high_priority_queue = PriorityQueue()
        self.normal_queue = Queue()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Configure news sources
        self._configure_sources()
        
    def _create_indexes(self):
        """Create MongoDB indexes for efficient querying"""
        # Raw news indexes
        self.raw_news.create_index([("url", 1)], unique=True)
        self.raw_news.create_index([("hash", 1)])
        self.raw_news.create_index([("published_date", -1)])
        self.raw_news.create_index([("scraped_date", -1)])
        self.raw_news.create_index([("status", 1)])
        self.raw_news.create_index([("priority", 1)])
        
        # Processed news indexes
        self.processed_news.create_index([("raw_news_id", 1)])
        self.processed_news.create_index([("entities.companies", 1)])
        self.processed_news.create_index([("trading_relevance.affected_symbols", 1)])
        self.processed_news.create_index([("processed_date", -1)])
        
    def _load_seen_items(self):
        """Load recently seen URLs and hashes to prevent duplicates"""
        # Load URLs from last 24 hours
        cutoff_date = datetime.utcnow() - timedelta(days=1)
        recent_news = self.raw_news.find(
            {"scraped_date": {"$gte": cutoff_date}},
            {"url": 1, "hash": 1}
        )
        
        for news in recent_news:
            self.seen_urls.add(news.get("url", ""))
            self.seen_hashes.add(news.get("hash", ""))
            
        logger.info(f"Loaded {len(self.seen_urls)} seen URLs")
        
    def _configure_sources(self):
        """Configure all news sources"""
        self.sources = {
            # Indian Business News
            'economic_times': {
                'priority': 1,
                'rss_feeds': [
                    'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
                    'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
                    'https://economictimes.indiatimes.com/news/company/rssfeeds/2143429.cms',
                    'https://economictimes.indiatimes.com/markets/commodities/rssfeeds/1808152121.cms',
                    'https://economictimes.indiatimes.com/markets/forex/rssfeeds/1808152122.cms'
                ],
                'web_urls': [
                    'https://economictimes.indiatimes.com/markets/stocks/liveblog',
                    'https://economictimes.indiatimes.com/markets/stocks/recos'
                ],
                'selectors': {
                    'title': ['h1.artTitle', 'h1'],
                    'content': ['div.artText', 'div.article-content'],
                    'time': ['time', 'span.time']
                }
            },
            
            'business_standard': {
                'priority': 1,
                'rss_feeds': [
                    'https://www.business-standard.com/rss/home_page_top_stories.rss',  # Try this
                    'https://www.business-standard.com/rss/latest.rss',  # Alternative
                    'https://www.business-standard.com/rss/markets-106.rss',
                    'https://www.business-standard.com/rss/companies-101.rss'
                ],
                'web_urls': [
                    'https://www.business-standard.com/markets',
                    'https://www.business-standard.com/companies/results'
                ],
                'selectors': {
                    'title': ['h1.headline', 'h1'],
                    'content': ['div.p-content', 'div.story-content'],
                    'time': ['div.updated-on', 'time']
                }
            },
            
            'moneycontrol': {
                'priority': 1,
                'rss_feeds': [
                    'https://www.moneycontrol.com/rss/business.xml',
                    'https://www.moneycontrol.com/rss/marketreports.xml',
                    'https://www.moneycontrol.com/rss/results.xml',
                    'https://www.moneycontrol.com/rss/currency.xml'
                ],
                'web_urls': [
                    'https://www.moneycontrol.com/news/business/',
                    'https://www.moneycontrol.com/news/announcements/'
                ],
                'selectors': {
                    'title': ['h1.article_title', 'h1'],
                    'content': ['div.content_wrapper', 'div.article_content'],
                    'time': ['div.article_schedule', 'time']
                }
            },
            
            'livemint': {
                'priority': 1,
                'rss_feeds': [
                    'https://www.livemint.com/rss/markets',
                    'https://www.livemint.com/rss/companies',
                    'https://www.livemint.com/rss/economy',
                    'https://www.livemint.com/rss/money'
                ],
                'web_urls': [
                    'https://www.livemint.com/market',
                    'https://www.livemint.com/companies'
                ],
                'selectors': {
                    'title': ['h1.headline', 'h1'],
                    'content': ['div.mainContent', 'div.content'],
                    'time': ['span.articleInfo', 'time']
                }
            },
            # Add to your scraper configuration:

            # For commodity prices affecting Indian markets
            'commodity_sources': {
                'mcx_data': {
                    'url': 'https://www.mcxindia.com/',
                    'check_interval': 60
                },
                'crude_oil': {
                    'rss_feeds': ['https://oilprice.com/rss/main'],
                    'keywords': ['crude', 'WTI', 'Brent', 'OPEC']
                }
            },

            # For FII/DII data
            'market_participant_data': {
                'fii_dii': {
                    'url': 'https://www.nseindia.com/api/fiidiiTradeReact',
                    'check_interval': 300
                }
            },

            # For breaking news via Twitter
            'social_media': {
                'twitter_monitors': [
                    '@NSEIndia',
                    '@BSEIndia', 
                    '@SEBI_India',
                    '@RBI',
                    '@FinMinIndia'
                ]
            },
            'financial_express': {
                'priority': 2,
                'rss_feeds': [
                    'https://www.financialexpress.com/business/rss',  # Try this
                    'https://www.financialexpress.com/stock-market/rss',  # Alternative
                    'https://www.financialexpress.com/market/rss'
                ],
                'web_urls': [
                    'https://www.financialexpress.com/market/',
                    'https://www.financialexpress.com/economy/'
                ],
                'selectors': {
                    'title': ['h1.wp-block-post-title', 'h1'],
                    'content': ['div.wp-block-post-content', 'div.content'],
                    'time': ['div.post-date', 'time']
                }
            },
            
            'hindu_business': {
                'priority': 2,
                'rss_feeds': [
                    'https://www.thehindubusinessline.com/markets/?service=rss',
                    'https://www.thehindubusinessline.com/companies/?service=rss',
                    'https://www.thehindubusinessline.com/economy/?service=rss'
                ],
                'web_urls': [
                    'https://www.thehindubusinessline.com/markets/',
                    'https://www.thehindubusinessline.com/economy/'
                ],
                'selectors': {
                    'title': ['h1.title', 'h1'],
                    'content': ['div[itemprop="articleBody"]', 'div.article-content'],
                    'time': ['none', 'time']
                }
            },
            
            # Global News Sources
            'reuters_global': {
                'priority': 1,
                'rss_feeds': [
                    'https://news.google.com/rss/search?q=india+business+reuters&hl=en-IN',  # Google News RSS for Reuters India
                    'https://feeds.reuters.com/reuters/INtopNews',  # India Top News
                    'https://feeds.reuters.com/reuters/INbusinessNews'
                ],
                'keywords_filter': ['India', 'Sensex', 'Nifty', 'Mumbai', 'Delhi', 'RBI', 'Rupee']  # Broader keywords
            },
            
            'bloomberg': {
                'priority': 1,
                'rss_feeds': [
                    'https://feeds.bloomberg.com/markets/news.rss',
                    'https://feeds.bloomberg.com/politics/news.rss'
                ],
                'keywords_filter': ['India', 'Asia', 'emerging markets', 'Fed', 'Dollar']
            },
            
            # Commodity & Currency
            'investing_com': {
                'priority': 2,
                'web_urls': [
                    'https://in.investing.com/news/commodities-news',
                    'https://in.investing.com/news/forex-news'
                ],
                'selectors': {
                    'title': ['h1.articleHeader', 'h1'],
                    'content': ['div.articlePage', 'div.content'],
                    'time': ['span.date', 'time']
                }
            },
            
            # Official Sources
            'nse_announcements': {
                'priority': 1,
                'api_url': 'https://www.nseindia.com/api/corporates-announcements',
                'type': 'api',
                'check_interval': 60
            },
            
            'sebi_updates': {
                'priority': 1,
                'web_urls': ['https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListingAll=yes'],
                'type': 'official',
                'check_interval': 300
            }
        }
        
        # High priority keywords for immediate action
        self.high_priority_keywords = {
            'fraud': ['fraud', 'scam', 'investigation', 'scandal', 'embezzlement'],
            'regulatory': ['SEBI action', 'RBI penalty', 'banned', 'suspended', 'delisted'],
            'earnings': ['earnings beat', 'earnings miss', 'profit warning', 'guidance cut'],
            'deals': ['acquisition', 'merger', 'takeover', 'buyout', 'stake sale'],
            'major_events': ['bankruptcy', 'default', 'CEO resign', 'CFO resign'],
            'positive': ['breakthrough', 'FDA approval', 'major order', 'contract win'],
            'macro': ['rate cut', 'rate hike', 'Fed decision', 'RBI policy', 'budget']
        }
        
        # Company mappings for entity extraction
        self.company_aliases = {
            'TCS': ['Tata Consultancy', 'TCS'],
            'INFY': ['Infosys', 'INFY'],
            'RELIANCE': ['Reliance Industries', 'RIL', 'Reliance'],
            'HDFC': ['HDFC Bank', 'HDFC'],
            'ICICIBANK': ['ICICI Bank', 'ICICI'],
            'SBIN': ['State Bank', 'SBI', 'State Bank of India'],
            'WIPRO': ['Wipro', 'WIPRO'],
            'HCLTECH': ['HCL Tech', 'HCL Technologies'],
            'TATAMOTORS': ['Tata Motors', 'Tata Motor'],
            'TATASTEEL': ['Tata Steel', 'TISCO'],
            # Add more mappings as needed
        }
        
    def start_collection(self):
        """Start the news collection process"""
        logger.info("Starting comprehensive news collection...")
        
        # Start processing threads
        processing_thread = threading.Thread(target=self._process_queue)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start collection loops
        collection_threads = []
        
        # High priority sources - check every 30 seconds
        high_priority_thread = threading.Thread(
            target=self._collection_loop,
            args=(30, ['economic_times', 'business_standard', 'moneycontrol', 'reuters_global'])
        )
        high_priority_thread.daemon = True
        collection_threads.append(high_priority_thread)
        
        # Medium priority sources - check every 2 minutes
        medium_priority_thread = threading.Thread(
            target=self._collection_loop,
            args=(120, ['livemint', 'financial_express', 'hindu_business'])
        )
        medium_priority_thread.daemon = True
        collection_threads.append(medium_priority_thread)
        
        # Official sources - check every 5 minutes
        official_thread = threading.Thread(
            target=self._collection_loop,
            args=(300, ['nse_announcements', 'sebi_updates'])
        )
        official_thread.daemon = True
        collection_threads.append(official_thread)
        
        # Start all threads
        for thread in collection_threads:
            thread.start()
            
        logger.info("All collection threads started")
        
        # Keep running
        try:
            while True:
                time.sleep(60)
                self._log_statistics()
        except KeyboardInterrupt:
            logger.info("Stopping news collection...")

    def check_position_news(self, your_positions):
        """Check for negative news on your positions"""
        alerts = []
        
        for position in your_positions:
            # Check last 30 minutes of news
            cutoff = datetime.utcnow() - timedelta(minutes=30)
            
            negative_news = self.db.processed_news_collection.find({
                "processed_date": {"$gte": cutoff},
                "trading_relevance.affected_symbols": position['symbol'],
                "sentiment.overall": {"$lt": -0.3}
            })
            
            for news in negative_news:
                alerts.append({
                    'symbol': position['symbol'],
                    'alert': 'NEGATIVE_NEWS',
                    'urgency': 'HIGH' if news['trading_relevance']['score'] > 0.8 else 'MEDIUM',
                    'news': news
                })
                
        return alerts
      
    def _collection_loop(self, interval, sources):
        """Collection loop for specific sources"""
        while True:
            try:
                futures = []
                
                for source_name in sources:
                    if source_name in self.sources:
                        future = self.executor.submit(self._collect_from_source, source_name)
                        futures.append(future)
                
                # Wait for all to complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.error(f"Error in collection: {e}")
                        
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                
            time.sleep(interval)
            
    def _collect_from_source(self, source_name):
        """Collect news from a specific source"""
        source = self.sources[source_name]
        collected = 0
        
        try:
            # Collect from RSS feeds
            if 'rss_feeds' in source:
                for feed_url in source['rss_feeds']:
                    items = self._parse_rss_feed(feed_url, source_name)
                    collected += len(items)
                    
            # Collect from web pages
            if 'web_urls' in source:
                for web_url in source['web_urls']:
                    items = self._scrape_webpage(web_url, source_name, source.get('selectors', {}))
                    collected += len(items)
                    
            # Collect from API
            if 'api_url' in source:
                items = self._fetch_from_api(source['api_url'], source_name)
                collected += len(items)
                
            logger.info(f"Collected {collected} items from {source_name}")
            
        except Exception as e:
            logger.error(f"Error collecting from {source_name}: {e}")
            
        return collected
        
    def _parse_rss_feed(self, feed_url, source_name):
        """Parse RSS feed and extract news items"""
        items = []
        
        try:
            feed = feedparser.parse(feed_url)
            source = self.sources[source_name]
            
            for entry in feed.entries[:20]:  # Limit to recent 20 items
                # Check if already seen
                if entry.link in self.seen_urls:
                    continue
                    
                # Extract basic info
                item = {
                    'url': entry.link,
                    'title': entry.title,
                    'description': entry.get('summary', ''),
                    'source': source_name,
                    'source_category': 'rss',
                    'published_date': self._parse_date(entry.get('published', '')),
                    'scraped_date': datetime.utcnow(),
                    'hash': hashlib.md5(entry.link.encode()).hexdigest(),
                    'status': 'new'
                }
                
                # Check for high priority keywords
                item['priority'] = self._calculate_priority(item['title'] + ' ' + item['description'])
                
                # Apply keyword filter if specified
                if 'keywords_filter' in source:
                    if not any(keyword.lower() in (item['title'] + item['description']).lower() 
                             for keyword in source['keywords_filter']):
                        continue
                
                # Extract tags
                item['tags'] = self._extract_tags(item['title'] + ' ' + item['description'])
                
                # Save to database
                try:
                    self.raw_news.insert_one(item)
                    self.seen_urls.add(item['url'])
                    self.seen_hashes.add(item['hash'])
                    
                    # Add to processing queue
                    if item['priority'] == 'high':
                        self.high_priority_queue.put((1, item['_id']))
                    else:
                        self.normal_queue.put(item['_id'])
                        
                    items.append(item)
                    
                except pymongo.errors.DuplicateKeyError:
                    # Already exists
                    pass
                    
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url}: {e}")
            
        return items
        
    def _scrape_webpage(self, url, source_name, selectors):
        """Scrape news from webpage"""
        items = []
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles - this is generic, customize per source
            articles = soup.find_all(['article', 'div'], class_=re.compile('article|story|news-item'))
            
            for article in articles[:10]:  # Limit to 10 per page
                try:
                    # Extract title
                    title = None
                    for selector in selectors.get('title', ['h2', 'h3']):
                        title_elem = article.select_one(selector)
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            break
                    
                    if not title:
                        continue
                    
                    # Extract URL
                    link_elem = article.find('a', href=True)
                    if not link_elem:
                        continue
                        
                    article_url = urljoin(url, link_elem['href'])
                    
                    # Check if already seen
                    if article_url in self.seen_urls:
                        continue
                    
                    # Extract content preview
                    content = ""
                    for selector in selectors.get('content', ['p']):
                        content_elem = article.select_one(selector)
                        if content_elem:
                            content = content_elem.get_text(strip=True)[:500]
                            break
                    
                    # Create news item
                    item = {
                        'url': article_url,
                        'title': title,
                        'description': content,
                        'source': source_name,
                        'source_category': 'web',
                        'published_date': datetime.utcnow(),  # Will be updated if found
                        'scraped_date': datetime.utcnow(),
                        'hash': hashlib.md5(article_url.encode()).hexdigest(),
                        'status': 'new'
                    }
                    
                    # Calculate priority
                    item['priority'] = self._calculate_priority(title + ' ' + content)
                    
                    # Extract tags
                    item['tags'] = self._extract_tags(title + ' ' + content)
                    
                    # Save to database
                    try:
                        self.raw_news.insert_one(item)
                        self.seen_urls.add(item['url'])
                        self.seen_hashes.add(item['hash'])
                        
                        # Add to processing queue
                        if item['priority'] == 'high':
                            self.high_priority_queue.put((1, item['_id']))
                        else:
                            self.normal_queue.put(item['_id'])
                            
                        items.append(item)
                        
                    except pymongo.errors.DuplicateKeyError:
                        pass
                        
                except Exception as e:
                    logger.error(f"Error parsing article: {e}")
                    
        except Exception as e:
            logger.error(f"Error scraping webpage {url}: {e}")
            
        return items
        
    def _fetch_from_api(self, api_url, source_name):
        """Fetch data from API endpoints"""
        items = []
        
        try:
            # Example for NSE announcements
            if source_name == 'nse_announcements':
                # This would need proper implementation based on actual API
                headers = {
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json'
                }
                
                response = self.session.get(api_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Process announcements (structure depends on actual API)
                    # This is a placeholder - adapt to actual API response
                    announcements = data.get('data', [])
                    
                    for announcement in announcements[:20]:
                        item = {
                            'url': f"https://nseindia.com/announcement/{announcement.get('id', '')}",
                            'title': announcement.get('subject', ''),
                            'description': announcement.get('description', ''),
                            'source': source_name,
                            'source_category': 'official',
                            'published_date': self._parse_date(announcement.get('date', '')),
                            'scraped_date': datetime.utcnow(),
                            'status': 'new',
                            'priority': 'high',  # Official announcements are high priority
                            'tags': ['official', 'nse', 'announcement']
                        }
                        
                        # Extract company symbol if present
                        if 'symbol' in announcement:
                            item['tags'].append(announcement['symbol'])
                            
                        # Save to database
                        try:
                            self.raw_news.insert_one(item)
                            items.append(item)
                            self.high_priority_queue.put((1, item['_id']))
                        except pymongo.errors.DuplicateKeyError:
                            pass
                            
        except Exception as e:
            logger.error(f"Error fetching from API {api_url}: {e}")
            
        return items
        
    def _calculate_priority(self, text):
        """Calculate news priority based on keywords"""
        text_lower = text.lower()
        
        # Check high priority keywords
        for category, keywords in self.high_priority_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return 'high'
                    
        # Check for company names
        for symbol, aliases in self.company_aliases.items():
            for alias in aliases:
                if alias.lower() in text_lower:
                    return 'medium'
                    
        return 'low'
        
    def _extract_tags(self, text):
        """Extract relevant tags from text"""
        tags = []
        text_lower = text.lower()
        
        # Extract keyword categories
        for category, keywords in self.high_priority_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    tags.append(category)
                    break
                    
        # Extract company references
        for symbol, aliases in self.company_aliases.items():
            for alias in aliases:
                if alias.lower() in text_lower:
                    tags.append(symbol)
                    break
                    
        # Extract sectors
        sectors = ['banking', 'it', 'pharma', 'auto', 'fmcg', 'metal', 'energy', 'telecom']
        for sector in sectors:
            if sector in text_lower:
                tags.append(sector)
                
        return list(set(tags))  # Remove duplicates
        
    def _parse_date(self, date_str):
        """Parse date string to datetime"""
        if not date_str:
            return datetime.utcnow()
            
        # Try common date formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S',
            '%d %b %Y',
            '%d-%m-%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.replace('GMT', '+0000'), fmt)
            except:
                continue
                
        return datetime.utcnow()
        
    def _process_queue(self):
        """Process news items from the queue"""
        while True:
            try:
                # Process high priority items first
                if not self.high_priority_queue.empty():
                    _, news_id = self.high_priority_queue.get()
                    self._process_news_item(news_id)
                    
                # Process normal priority items
                elif not self.normal_queue.empty():
                    news_id = self.normal_queue.get()
                    self._process_news_item(news_id)
                    
                else:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error processing queue: {e}")
                time.sleep(5)
                
    def _process_news_item(self, news_id):
        """Process a single news item"""
        try:
            # Fetch the news item
            news_item = self.raw_news.find_one({'_id': news_id})
            if not news_item:
                return
                
            # Skip if already processed
            if news_item.get('status') == 'processed':
                return
                
            # Perform sentiment analysis
            sentiment = self._analyze_sentiment(news_item['title'] + ' ' + news_item.get('description', ''))
            
            # Extract entities
            entities = self._extract_entities(news_item)
            
            # Calculate trading relevance
            trading_relevance = self._calculate_trading_relevance(news_item, entities, sentiment)
            
            # Create processed news document
            processed = {
                'raw_news_id': news_id,
                'processed_date': datetime.utcnow(),
                'entities': entities,
                'sentiment': sentiment,
                'trading_relevance': trading_relevance,
                'classification': {
                    'news_type': self._classify_news_type(news_item),
                    'market_impact': self._assess_market_impact(news_item, entities),
                    'urgency': news_item.get('priority', 'low')
                }
            }
            
            # Save processed news
            self.processed_news.insert_one(processed)
            
            # Update raw news status
            self.raw_news.update_one(
                {'_id': news_id},
                {'$set': {'status': 'processed'}}
            )
            
            # Generate alerts if needed
            if trading_relevance['score'] > 0.7 and news_item.get('priority') == 'high':
                self._generate_alert(news_item, processed)
                
            logger.info(f"Processed news: {news_item['title'][:50]}... Score: {trading_relevance['score']}")
            
        except Exception as e:
            logger.error(f"Error processing news item {news_id}: {e}")
            
    def _analyze_sentiment(self, text):
        """Analyze sentiment of news text"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Enhanced sentiment with confidence
            sentiment = {
                'overall': polarity,
                'confidence': abs(polarity),
                'emotions': {
                    'positive': max(0, polarity),
                    'negative': abs(min(0, polarity)),
                    'neutral': 1 - abs(polarity)
                }
            }
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'overall': 0,
                'confidence': 0,
                'emotions': {'positive': 0, 'negative': 0, 'neutral': 1}
            }
            
    def _extract_entities(self, news_item):
        """Extract entities from news item"""
        entities = {
            'companies': [],
            'sectors': [],
            'people': [],
            'locations': [],
            'currencies': [],
            'numbers': {
                'percentages': [],
                'amounts': []
            }
        }
        
        text = news_item['title'] + ' ' + news_item.get('description', '')
        text_lower = text.lower()
        
        # Extract companies
        for symbol, aliases in self.company_aliases.items():
            for alias in aliases:
                if alias.lower() in text_lower:
                    entities['companies'].append(symbol)
                    break
                    
        # Extract numbers
        # Percentages
        percentages = re.findall(r'(\d+\.?\d*)\s*%', text)
        entities['numbers']['percentages'] = [float(p) for p in percentages]
        
        # Amounts (crore, lakh, million, billion)
        amounts = re.findall(r'₹?\s*(\d+\.?\d*)\s*(crore|lakh|million|billion)', text, re.IGNORECASE)
        entities['numbers']['amounts'] = amounts
        
        # Extract sectors from tags
        # Extract sectors from tags
        entities['sectors'] = [tag for tag in news_item.get('tags', []) 
                                if tag in ['banking', 'it', 'pharma', 'auto', 'fmcg', 'metal', 'energy', 'telecom']]
        
        # Extract currencies
        currencies = re.findall(r'\b(USD|INR|EUR|GBP|JPY|CNY)\b', text)
        entities['currencies'] = list(set(currencies))
        
        # Extract locations
        locations = ['India', 'US', 'USA', 'China', 'Europe', 'Asia', 'Mumbai', 'Delhi', 'Bangalore']
        for location in locations:
            if location.lower() in text_lower:
                entities['locations'].append(location)
                
        return entities
        
    def _calculate_trading_relevance(self, news_item, entities, sentiment):
        """Calculate trading relevance score"""
        relevance = {
            'score': 0,
            'affected_symbols': [],
            'expected_impact': 'neutral',
            'time_horizon': 'medium-term',
            'confidence': 0
        }
        
        # Base score from priority
        if news_item.get('priority') == 'high':
            relevance['score'] = 0.6
        elif news_item.get('priority') == 'medium':
            relevance['score'] = 0.4
        else:
            relevance['score'] = 0.2
            
        # Adjust for entities
        if entities['companies']:
            relevance['score'] += 0.2
            relevance['affected_symbols'] = entities['companies']
            
        # Adjust for sentiment
        if abs(sentiment['overall']) > 0.5:
            relevance['score'] += 0.1
            relevance['expected_impact'] = 'positive' if sentiment['overall'] > 0 else 'negative'
            
        # Adjust for specific keywords
        text_lower = (news_item['title'] + news_item.get('description', '')).lower()
        
        # Immediate impact keywords
        immediate_keywords = ['breaking', 'alert', 'urgent', 'flash', 'just in', 'live']
        if any(keyword in text_lower for keyword in immediate_keywords):
            relevance['time_horizon'] = 'immediate'
            relevance['score'] += 0.1
            
        # High impact events
        high_impact = ['earnings beat', 'earnings miss', 'acquisition', 'merger', 'bankruptcy', 
                        'fraud', 'scam', 'ceo resign', 'rbi rate', 'fed rate']
        if any(keyword in text_lower for keyword in high_impact):
            relevance['score'] = min(relevance['score'] + 0.2, 1.0)
            
        # Set confidence
        relevance['confidence'] = min(relevance['score'] * sentiment['confidence'], 1.0)
        
        # Cap score at 1.0
        relevance['score'] = min(relevance['score'], 1.0)
        
        return relevance
        
    def _classify_news_type(self, news_item):
        """Classify the type of news"""
        text_lower = (news_item['title'] + news_item.get('description', '')).lower()
        
        # Check various news types
        if any(word in text_lower for word in ['earnings', 'result', 'profit', 'revenue', 'quarter']):
            return 'earnings'
        elif any(word in text_lower for word in ['merger', 'acquisition', 'takeover', 'buyout']):
            return 'merger_acquisition'
        elif any(word in text_lower for word in ['rbi', 'sebi', 'regulation', 'policy', 'circular']):
            return 'regulatory'
        elif any(word in text_lower for word in ['dividend', 'bonus', 'split', 'buyback']):
            return 'corporate_action'
        elif any(word in text_lower for word in ['ceo', 'cfo', 'director', 'resign', 'appoint']):
            return 'management_change'
        elif any(word in text_lower for word in ['deal', 'contract', 'order', 'win', 'award']):
            return 'business_deal'
        elif any(word in text_lower for word in ['fraud', 'scam', 'investigation', 'penalty']):
            return 'negative_event'
        else:
            return 'general'
            
    def _assess_market_impact(self, news_item, entities):
        """Assess market impact scope"""
        # If specific companies mentioned
        if entities['companies']:
            return 'company_specific'
        # If sector mentioned
        elif entities['sectors']:
            return 'sector_wide'
        # If macro keywords
        elif any(tag in news_item.get('tags', []) for tag in ['macro', 'economy', 'rbi', 'fed']):
            return 'market_wide'
        else:
            return 'limited'
            
    def _generate_alert(self, news_item, processed):
        """Generate alert for high-impact news"""
        try:
            alert = {
                'processed_news_id': processed['_id'],
                'alert_type': 'high_impact',
                'symbols': processed['trading_relevance']['affected_symbols'],
                'message': self._format_alert_message(news_item, processed),
                'created_date': datetime.utcnow(),
                'sent': False,
                'channels': ['slack', 'email'],
                'priority': 1 if news_item.get('priority') == 'high' else 2,
                'metadata': {
                    'sentiment': processed['sentiment']['overall'],
                    'relevance_score': processed['trading_relevance']['score'],
                    'news_type': processed['classification']['news_type']
                }
            }
            
            self.news_alerts.insert_one(alert)
            logger.info(f"Generated alert for: {news_item['title'][:50]}...")
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
            
    def _format_alert_message(self, news_item, processed):
        """Format alert message"""
        sentiment_emoji = "🟢" if processed['sentiment']['overall'] > 0.2 else "🔴" if processed['sentiment']['overall'] < -0.2 else "🟡"
        
        message = f"{sentiment_emoji} *{news_item['title']}*\n\n"
        
        if processed['trading_relevance']['affected_symbols']:
            message += f"📊 Affected: {', '.join(processed['trading_relevance']['affected_symbols'])}\n"
            
        message += f"💹 Impact: {processed['trading_relevance']['expected_impact'].title()}\n"
        message += f"⏱️ Timeframe: {processed['trading_relevance']['time_horizon'].replace('_', ' ').title()}\n"
        message += f"📈 Relevance: {processed['trading_relevance']['score']:.0%}\n"
        
        if news_item.get('description'):
            message += f"\n{news_item['description'][:200]}...\n"
            
        message += f"\n🔗 {news_item['url']}"
        
        return message
        
    def _log_statistics(self):
        """Log collection statistics"""
        try:
            # Count news collected in last hour
            hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            stats = {
                'last_hour': self.raw_news.count_documents({'scraped_date': {'$gte': hour_ago}}),
                'processed': self.raw_news.count_documents({'status': 'processed'}),
                'pending': self.raw_news.count_documents({'status': 'new'}),
                'high_priority': self.raw_news.count_documents({'priority': 'high', 'scraped_date': {'$gte': hour_ago}}),
                'alerts_generated': self.news_alerts.count_documents({'created_date': {'$gte': hour_ago}})
            }
            
            logger.info(f"Stats - Last hour: {stats['last_hour']}, Processed: {stats['processed']}, "
                        f"Pending: {stats['pending']}, High Priority: {stats['high_priority']}, "
                        f"Alerts: {stats['alerts_generated']}")
                        
        except Exception as e:
            logger.error(f"Error logging statistics: {e}")
            
    def get_latest_news(self, symbols=None, hours=1, news_type=None):
        """Get latest processed news for specific symbols"""
        query = {
            'processed_date': {'$gte': datetime.utcnow() - timedelta(hours=hours)}
        }
        
        if symbols:
            query['trading_relevance.affected_symbols'] = {'$in': symbols}
            
        if news_type:
            query['classification.news_type'] = news_type
            
        # Join with raw news to get full details
        pipeline = [
            {'$match': query},
            {'$lookup': {
                'from': 'raw_news_collection',
                'localField': 'raw_news_id',
                'foreignField': '_id',
                'as': 'raw_news'
            }},
            {'$unwind': '$raw_news'},
            {'$sort': {'trading_relevance.score': -1}},
            {'$limit': 50}
        ]
        
        results = list(self.processed_news.aggregate(pipeline))
        
        # Format for easy consumption
        formatted_news = []
        for item in results:
            formatted_news.append({
                'title': item['raw_news']['title'],
                'url': item['raw_news']['url'],
                'source': item['raw_news']['source'],
                'published': item['raw_news']['published_date'],
                'sentiment': item['sentiment']['overall'],
                'relevance': item['trading_relevance']['score'],
                'affected_symbols': item['trading_relevance']['affected_symbols'],
                'impact': item['trading_relevance']['expected_impact'],
                'type': item['classification']['news_type']
            })
            
        return formatted_news
        
    def get_market_sentiment(self, hours=4):
        """Get overall market sentiment from recent news"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        pipeline = [
            {'$match': {'processed_date': {'$gte': cutoff}}},
            {'$group': {
                '_id': None,
                'avg_sentiment': {'$avg': '$sentiment.overall'},
                'total_news': {'$sum': 1},
                'positive_news': {
                    '$sum': {'$cond': [{'$gt': ['$sentiment.overall', 0.2]}, 1, 0]}
                },
                'negative_news': {
                    '$sum': {'$cond': [{'$lt': ['$sentiment.overall', -0.2]}, 1, 0]}
                },
                'high_impact_news': {
                    '$sum': {'$cond': [{'$gt': ['$trading_relevance.score', 0.7]}, 1, 0]}
                }
            }}
        ]
        
        result = list(self.processed_news.aggregate(pipeline))
        
        if result:
            sentiment_data = result[0]
            sentiment_data['sentiment_label'] = (
                'Bullish' if sentiment_data['avg_sentiment'] > 0.1
                else 'Bearish' if sentiment_data['avg_sentiment'] < -0.1
                else 'Neutral'
            )
            return sentiment_data
        else:
            return {
                'avg_sentiment': 0,
                'total_news': 0,
                'positive_news': 0,
                'negative_news': 0,
                'high_impact_news': 0,
                'sentiment_label': 'No Data'
            }
            
    def get_symbol_news_summary(self, symbol, days=7):
        """Get news summary for a specific symbol"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        pipeline = [
            {
                '$match': {
                    'trading_relevance.affected_symbols': symbol,
                    'processed_date': {'$gte': cutoff}
                }
            },
            {
                '$group': {
                    '_id': '$classification.news_type',
                    'count': {'$sum': 1},
                    'avg_sentiment': {'$avg': '$sentiment.overall'},
                    'avg_relevance': {'$avg': '$trading_relevance.score'}
                }
            },
            {'$sort': {'count': -1}}
        ]
        
        news_by_type = list(self.processed_news.aggregate(pipeline))
        
        # Get recent high-impact news
        high_impact = self.get_latest_news(symbols=[symbol], hours=24*days)
        high_impact = [n for n in high_impact if n['relevance'] > 0.7][:5]
        
        return {
            'symbol': symbol,
            'period_days': days,
            'news_by_type': news_by_type,
            'high_impact_news': high_impact,
            'total_news': sum(item['count'] for item in news_by_type)
        }


    # Example usage and testing
if __name__ == "__main__":
    # Initialize the scraper
    scraper = ComprehensiveNewsScraper(
        mongodb_uri="mongodb://localhost:27017/",
        db_name="trading_system"
    )
    
    # Start collection in a separate thread
    collection_thread = threading.Thread(target=scraper.start_collection)
    collection_thread.daemon = True
    collection_thread.start()
    
    print("News collection started. Press Ctrl+C to stop.")
    print("\nMonitoring news from:")
    print("- Economic Times, Business Standard, Moneycontrol")
    print("- LiveMint, Financial Express, Hindu Business Line")
    print("- Reuters, Bloomberg (India-relevant)")
    print("- NSE/BSE announcements")
    print("- And more...\n")
    
    # Example: Monitor for 5 minutes and show statistics
    try:
        time.sleep(300)  # Run for 5 minutes
        
        # Get market sentiment
        sentiment = scraper.get_market_sentiment(hours=1)
        print(f"\nMarket Sentiment (Last Hour): {sentiment['sentiment_label']}")
        print(f"Average Sentiment: {sentiment['avg_sentiment']:.3f}")
        print(f"Positive News: {sentiment['positive_news']}, Negative: {sentiment['negative_news']}")
        
        # Get latest high-impact news
        latest_news = scraper.get_latest_news(hours=1)
        print(f"\nLatest High-Impact News:")
        for news in latest_news[:5]:
            print(f"- [{news['source']}] {news['title'][:80]}...")
            print(f"  Sentiment: {news['sentiment']:.2f}, Relevance: {news['relevance']:.2f}")
            print(f"  Affected: {', '.join(news['affected_symbols'])}\n")
            
    except KeyboardInterrupt:
        print("\nStopping news collection...")