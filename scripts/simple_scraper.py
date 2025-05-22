#!/usr/bin/env python3
"""
Simple News Scraper for Stocks - Using the working Zerodha Markets logic
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
import hashlib
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import requests
import pymongo
from urllib.parse import urljoin

class SimpleNewsScraper:
    """
    Simple news scraper for stocks from multiple sources
    """
    
    def __init__(self, db_name="automated_trading"):
        """
        Initialize the scraper
        """
        # MongoDB connection
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.portfolio_collection = self.db.portfolio
        self.news_collection = self.db.news
        
        # Request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Scraping delay
        self.delay = 3
        
        print("News scraper initialized")
    
    def get_stock_info(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Get stock information from portfolio collection
        """
        stock_info = self.portfolio_collection.find_one({
            "symbol": symbol.upper(),
            "exchange": exchange.upper()
        })
        
        if not stock_info:
            print(f"Stock {symbol}:{exchange} not found in portfolio collection")
            return None
        
        print(f"Found stock: {stock_info.get('company_name', symbol)}")
        
        # Check if news_urls exist, if not create them
        if 'news_urls' not in stock_info:
            print(f"Creating news URLs for {symbol}...")
            
            # Use the known working URL patterns
            if symbol == 'SBIN':
                stock_info['news_urls'] = {
                    'zerodha_market_url': "https://zerodha.com/markets/stocks/state-bank-of-india/",
                    'times_of_india_url': "https://timesofindia.indiatimes.com/topic/state-bank-of-india",
                    'business_standard_url': "https://www.business-standard.com/topic/sbi"
                }
            else:
                # For other symbols, create URLs based on company name
                company_name = stock_info.get('company_name', symbol)
                company_url_name = company_name.lower().replace(' ', '-').replace('&', 'and')
                symbol_lower = symbol.lower()
                
                stock_info['news_urls'] = {
                    'zerodha_market_url': f"https://zerodha.com/markets/stocks/{company_url_name}/",
                    'times_of_india_url': f"https://timesofindia.indiatimes.com/topic/{company_url_name}",
                    'business_standard_url': f"https://www.business-standard.com/topic/{symbol_lower}"
                }
            
            print(f"Created URLs: {stock_info['news_urls']}")
        
        return stock_info
    
    def retry_request(self, url, max_retries=3, delay=1):
        """
        Make HTTP request with retry logic - Using the logic from your news_aggregator
        """
        for attempt in range(max_retries):
            try:
                print(f"Making request to {url} (attempt {attempt+1}/{max_retries})")
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    print(f"Request to {url} successful")
                    return response
                else:
                    print(f"Request to {url} failed with status {response.status_code}")
                    
            except requests.RequestException as e:
                print(f"Request to {url} failed: {e}")
            
            # Wait before retrying
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
        
        print(f"All {max_retries} request attempts to {url} failed")
        return None
    
    def scrape_zerodha_markets(self, url: str, keywords: List[str], symbol: str) -> List[Dict[str, Any]]:
        """
        Scrape news from Zerodha Markets - Using the working logic from your news_aggregator
        """
        news_items = []
        
        # Special cases for known symbols - from your working code
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
            base_url = "https://zerodha.com/markets/stocks/"
            url_formats.append(f"{base_url}{special_cases[symbol]}/")
        
        # Add the original URL
        url_formats.append(url)
        
        # Try with symbol directly 
        url_name = symbol.lower().replace('&', 'and').replace(' ', '-').replace('.', '').replace(',', '')
        url_formats.append(f"https://zerodha.com/markets/stocks/{url_name}/")
        
        # Try with keywords
        if keywords:
            # Get company name from keywords (excluding the symbol itself)
            company_names = [kw for kw in keywords if kw.lower() != symbol.lower()]
            
            # Try with first keyword that might be a company name
            for kw in company_names:
                if len(kw) > 3 and kw.lower() not in ['ltd', 'limited', 'inc', 'corp']:
                    url_name = kw.lower().replace('&', 'and').replace(' ', '-').replace('.', '').replace(',', '')
                    url_formats.append(f"https://zerodha.com/markets/stocks/{url_name}/")
        
        # Remove duplicates
        url_formats = list(dict.fromkeys(url_formats))
        
        # Try each URL format until we get a response
        response = None
        successful_url = None
        
        for try_url in url_formats:
            print(f"Trying Zerodha URL: {try_url}")
            response = self.retry_request(try_url)
            
            if response and response.status_code == 200:
                # Check if it's not a 404 page
                if "404" not in response.text and "Page Not Found" not in response.text:
                    successful_url = try_url
                    print(f"✓ Working Zerodha URL found: {successful_url}")
                    break
                else:
                    print(f"URL gave 404: {try_url}")
            else:
                print(f"Failed to get valid response from {try_url}")
        
        if not response or not successful_url:
            print("All Zerodha Markets URLs failed")
            return news_items
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find news container - Using your working logic
        news_section = None
        news_section_selectors = ['#news', '.news-section', '.company-news', '#company-news', '#related-news', '#recent-news']
        
        for selector in news_section_selectors:
            section = soup.select_one(selector)
            if section:
                news_section = section
                print(f"Found news section with selector: {selector}")
                break
        
        # Set articles to an empty list initially
        articles = []
        
        if not news_section:
            print("No explicit news section found, looking for articles in the main page")
            
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
                    print(f"Found {len(articles)} articles using selector: {selector}")
                    break
            
            if not articles:
                # Last resort - look for any UL/OL with LI elements that contain A tags
                list_elements = soup.select('ul, ol')
                for list_elem in list_elements:
                    list_items = list_elem.select('li')
                    if list_items and all(item.find('a') for item in list_items[:3]):  # Check if first few items have links
                        articles = list_items
                        print(f"Found {len(articles)} articles in list element")
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
                    print(f"Found {len(articles)} articles in news section using selector: {selector}")
                    break
        
        if not articles:
            print("No news items found on Zerodha Markets page")
            return news_items
        
        print(f"Successfully found {len(articles)} articles on Zerodha Markets page")
        
        for article in articles:
            try:
                # Extract title and URL
                title = ""
                article_url = ""
                
                # First, try to get from a tags directly
                if article.name == 'a':
                    title = article.text.strip()
                    if 'href' in article.attrs:
                        article_url = article['href']
                else:
                    # Otherwise look for a tag within the element
                    link_elem = article.find('a')
                    if link_elem:
                        title = link_elem.text.strip()
                        if 'href' in link_elem.attrs:
                            article_url = link_elem['href']
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
                if article_url and not article_url.startswith('http'):
                    article_url = urljoin(successful_url, article_url)
                
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
                        print(f"Error parsing date '{date_text}': {e}")
                        published_date = datetime.now()
                else:
                    published_date = datetime.now()
                
                # Create news item
                news_item = {
                    'title': title,
                    'description': description,
                    'url': article_url or successful_url,
                    'published_date': published_date,
                    'source': 'zerodha_markets',
                    'categories': [],
                    'symbols': [symbol]
                }
                
                # Add to results
                news_items.append(news_item)
                print(f"✓ Added article: {title[:60]}...")
                
            except Exception as e:
                print(f"Error parsing Zerodha Markets article: {e}")
        
        print(f"Found {len(news_items)} items from Zerodha Markets")
        return news_items
    
    def scrape_times_of_india(self, url: str, keywords: List[str], symbol: str) -> List[Dict[str, Any]]:
        """
        Scrape news from Times of India
        """
        news_items = []
        
        response = self.retry_request(url)
        if not response:
            return news_items
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for article containers
        articles = soup.select('.uwU81')
        if not articles:
            articles = soup.select('.Z7cWa, .Kl7Zq, .article, .news-card')
        
        print(f"Found {len(articles)} articles from Times of India")
        
        for article in articles[:15]:  # Limit to 15 articles
            try:
                title_elem = article.select_one('h1, h2, h3, h4, .title, .headline, a')
                if not title_elem:
                    continue
                
                title = title_elem.text.strip()
                if not title or len(title) < 10:
                    continue
                
                # Check keyword relevance
                title_lower = title.lower()
                is_relevant = False
                
                for keyword in keywords:
                    if len(keyword) > 2 and keyword.lower() in title_lower:
                        is_relevant = True
                        print(f"✓ Matched keyword '{keyword}' in: {title[:50]}...")
                        break
                
                if not is_relevant:
                    # Check for financial terms + symbol
                    financial_terms = ['bank', 'banking', 'financial', 'stock', 'share']
                    for term in financial_terms:
                        if term in title_lower and (symbol.lower() in title_lower or 'sbi' in title_lower):
                            is_relevant = True
                            print(f"✓ Matched financial term '{term}' in: {title[:50]}...")
                            break
                
                if not is_relevant:
                    continue
                
                # Extract URL
                article_url = ""
                link_elem = title_elem if title_elem.name == 'a' else article.find('a')
                if link_elem and link_elem.get('href'):
                    article_url = link_elem['href']
                    if not article_url.startswith('http'):
                        article_url = urljoin('https://timesofindia.indiatimes.com', article_url)
                
                # Extract description
                desc_elem = article.select_one('p, .summary, .content, .synopsis')
                description = desc_elem.text.strip() if desc_elem else ""
                
                # Extract and parse date
                published_date = datetime.now()
                date_elem = article.select_one('.date, time, .timestamp, .meta')
                if date_elem:
                    date_text = date_elem.text.strip()
                    # Handle relative dates
                    if 'ago' in date_text.lower():
                        if 'hour' in date_text.lower():
                            hours = int(re.search(r'(\d+)', date_text).group(1)) if re.search(r'(\d+)', date_text) else 1
                            published_date = datetime.now() - timedelta(hours=hours)
                        elif 'day' in date_text.lower():
                            days = int(re.search(r'(\d+)', date_text).group(1)) if re.search(r'(\d+)', date_text) else 1
                            published_date = datetime.now() - timedelta(days=days)
                
                news_items.append({
                    'title': title,
                    'description': description,
                    'url': article_url,
                    'published_date': published_date,
                    'source': 'times_of_india',
                    'symbols': [symbol],
                    'categories': ['market_news']
                })
                
            except Exception as e:
                continue
        
        print(f"Found {len(news_items)} relevant items from Times of India")
        return news_items
    
    def scrape_business_standard(self, url: str, keywords: List[str], symbol: str) -> List[Dict[str, Any]]:
        """
        Scrape news from Business Standard - Try different approaches for 403
        """
        news_items = []
        
        # Try different URLs if we get 403
        urls_to_try = [
            url,
            f"https://www.business-standard.com/search?q={symbol}",
            f"https://www.business-standard.com/companies/{symbol.lower()}",
            f"https://www.business-standard.com/topic/{symbol.lower()}-news"
        ]
        
        response = None
        for try_url in urls_to_try:
            print(f"Trying Business Standard URL: {try_url}")
            
            # Try with different headers to avoid 403
            temp_headers = self.headers.copy()
            temp_headers.update({
                'Referer': 'https://www.google.com/',
                'Cache-Control': 'no-cache'
            })
            
            try:
                session = requests.Session()
                session.headers.update(temp_headers)
                response = session.get(try_url, timeout=15)
                
                if response.status_code == 200:
                    print(f"✓ Successfully accessed: {try_url}")
                    break
                else:
                    print(f"HTTP {response.status_code} for {try_url}")
            except Exception as e:
                print(f"Request failed for {try_url}: {e}")
            
            time.sleep(2)
        
        if not response:
            print("All Business Standard URLs failed")
            return news_items
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for headlines
        headlines = soup.select('h1, h2, h3')
        print(f"Found {len(headlines)} potential headlines")
        
        for headline in headlines:
            try:
                title = headline.text.strip()
                
                if len(title) < 20:
                    continue
                
                # Check relevance
                title_lower = title.lower()
                is_relevant = False
                
                for keyword in keywords:
                    if len(keyword) > 2 and keyword.lower() in title_lower:
                        is_relevant = True
                        print(f"✓ Found BS news: {title[:50]}...")
                        break
                
                if not is_relevant:
                    continue
                
                # Get description from nearby paragraph
                description = ""
                parent = headline.parent
                if parent:
                    desc_elem = parent.find_next('p')
                    if desc_elem:
                        description = desc_elem.text.strip()
                
                # Extract date
                published_date = datetime.now()
                if parent:
                    date_match = re.search(r'(\d{1,2}\s+(May|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})', parent.get_text())
                    if date_match:
                        try:
                            published_date = datetime.strptime(date_match.group(1), '%d %b %Y')
                        except:
                            pass
                
                # Get URL
                article_url = url
                link_elem = headline.find('a') if headline.name != 'a' else headline
                if not link_elem:
                    link_elem = headline.find_parent('a')
                
                if link_elem and link_elem.get('href'):
                    article_url = link_elem['href']
                    if not article_url.startswith('http'):
                        article_url = urljoin('https://www.business-standard.com', article_url)
                
                news_items.append({
                    'title': title,
                    'description': description,
                    'url': article_url,
                    'published_date': published_date,
                    'source': 'business_standard',
                    'symbols': [symbol],
                    'categories': ['market_news']
                })
                
            except Exception as e:
                continue
        
        print(f"Found {len(news_items)} items from Business Standard")
        return news_items
    
    def save_to_database(self, news_items: List[Dict[str, Any]], force_save: bool = False) -> int:
        """
        Save news items to database with option to force save
        """
        if not news_items:
            print("No news items to save")
            return 0
        
        saved_count = 0
        duplicate_count = 0
        
        for item in news_items:
            try:
                # More lenient duplicate checking if force_save is True
                if not force_save:
                    existing = self.news_collection.find_one({
                        "title": item['title'],
                        "source": item['source']
                    })
                else:
                    # Only check for exact duplicates
                    existing = self.news_collection.find_one({
                        "title": item['title'],
                        "source": item['source'],
                        "url": item['url']
                    })
                
                if existing and not force_save:
                    duplicate_count += 1
                    continue
                
                # Add scraped timestamp and unique ID
                item['scraped_at'] = datetime.now()
                item['news_id'] = hashlib.md5(f"{item['title']}{item['source']}{datetime.now().isoformat()}".encode()).hexdigest()
                
                # Ensure required fields
                if 'categories' not in item:
                    item['categories'] = []
                if 'symbols' not in item:
                    item['symbols'] = []
                
                # Insert to database
                result = self.news_collection.insert_one(item)
                if result.inserted_id:
                    saved_count += 1
                    print(f"✓ Saved: {item['title'][:60]}...")
                
            except Exception as e:
                print(f"Error saving news item: {e}")
        
        print(f"Saved {saved_count} new items, {duplicate_count} duplicates skipped")
        return saved_count
    
    def scrape_stock_news(self, symbol: str, exchange: str = "NSE", force_save: bool = False) -> int:
        """
        Scrape news for a specific stock
        """
        print(f"Starting news scraping for {symbol}:{exchange}")
        if force_save:
            print("🔄 Force save mode enabled - will save items even if similar ones exist")
        
        # Get stock information
        stock_info = self.get_stock_info(symbol, exchange)
        if not stock_info:
            return 0
        
        # Get keywords
        keywords = []
        if 'company_details' in stock_info and 'keywords' in stock_info['company_details']:
            keywords = stock_info['company_details']['keywords']
        elif 'keywords' in stock_info:
            keywords = stock_info['keywords']
        else:
            keywords = [symbol]
            if 'company_name' in stock_info:
                keywords.append(stock_info['company_name'])
        
        print(f"Using keywords: {keywords}")
        
        # Get news URLs
        news_urls = stock_info.get('news_urls', {})
        if not news_urls:
            print("No news URLs found")
            return 0
        
        all_news = []
        
        # Scrape from each source
        if 'zerodha_market_url' in news_urls:
            print("\n" + "="*50)
            print("SCRAPING ZERODHA MARKETS")
            print("="*50)
            zerodha_news = self.scrape_zerodha_markets(news_urls['zerodha_market_url'], keywords, symbol)
            all_news.extend(zerodha_news)
            time.sleep(self.delay)
        
        if 'times_of_india_url' in news_urls:
            print("\n" + "="*50)
            print("SCRAPING TIMES OF INDIA")
            print("="*50)
            toi_news = self.scrape_times_of_india(news_urls['times_of_india_url'], keywords, symbol)
            all_news.extend(toi_news)
            time.sleep(self.delay)
        
        if 'business_standard_url' in news_urls:
            print("\n" + "="*50)
            print("SCRAPING BUSINESS STANDARD")
            print("="*50)
            bs_news = self.scrape_business_standard(news_urls['business_standard_url'], keywords, symbol)
            all_news.extend(bs_news)
            time.sleep(self.delay)
        
        # Save to database
        print("\n" + "="*50)
        print("SAVING TO DATABASE")
        print("="*50)
        print(f"Total news items collected: {len(all_news)}")
        
        if all_news:
            for i, item in enumerate(all_news, 1):
                print(f"{i}. {item['title'][:60]}... ({item['source']})")
        
        saved_count = self.save_to_database(all_news, force_save)
        
        print(f"\n🎉 News scraping completed for {symbol}. Saved {saved_count} items.")
        return saved_count
    
    def close(self):
        """Close database connection"""
        self.client.close()
        print("Database connection closed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple Stock News Scraper')
    parser.add_argument('symbol', help='Stock symbol to scrape news for')
    parser.add_argument('--exchange', '-e', default='NSE', help='Exchange (default: NSE)')
    parser.add_argument('--db', '-d', default='automated_trading', help='Database name')
    parser.add_argument('--force', '-f', action='store_true', help='Force save items even if similar ones exist')
    
    args = parser.parse_args()
    
    scraper = None
    try:
        scraper = SimpleNewsScraper(args.db)
        total_saved = scraper.scrape_stock_news(args.symbol.upper(), args.exchange.upper(), args.force)
        print(f"\n✅ Scraping completed. Total items saved: {total_saved}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Scraping interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if scraper:
            scraper.close()


if __name__ == "__main__":
    main()