#!/usr/bin/env python3
"""
Simple News Scraper for Stocks - Fixed Version
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
        
        # Request headers - More realistic browser simulation
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
        
        # Check if news_urls exist
        if 'news_urls' not in stock_info:
            print(f"Warning: No news_urls found for {symbol}. Creating default URLs...")
            
            # Create default news URLs based on company name and symbol
            company_name = stock_info.get('company_name', symbol)
            
            # Format company name for URLs
            company_url_name = company_name.lower().replace(' ', '-').replace('&', 'and')
            symbol_lower = symbol.lower()
            
            # Use the correct URLs based on what actually works
            stock_info['news_urls'] = {
                'zerodha_market_url': f"https://zerodha.com/markets/stocks/{company_url_name}/",
                'times_of_india_url': f"https://timesofindia.indiatimes.com/topic/{company_url_name}",
                'business_standard_url': f"https://www.business-standard.com/topic/{symbol_lower}"
            }
            
            print(f"Created URLs: {stock_info['news_urls']}")
        
        return stock_info
    
    def make_request(self, url: str, max_retries: int = 3) -> requests.Response:
        """
        Make HTTP request with retry logic
        """
        session = requests.Session()
        session.headers.update(self.headers)
        
        for attempt in range(max_retries):
            try:
                print(f"Fetching: {url} (attempt {attempt + 1})")
                response = session.get(url, timeout=15)
                
                if response.status_code == 200:
                    return response
                else:
                    print(f"HTTP {response.status_code} for {url}")
                    
            except requests.RequestException as e:
                print(f"Request failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(self.delay)
        
        print(f"All attempts failed for {url}")
        return None
    
    def scrape_zerodha_markets(self, url: str, keywords: List[str], symbol: str) -> List[Dict[str, Any]]:
        """
        Scrape news from Zerodha Markets
        """
        news_items = []
        
        response = self.make_request(url)
        if not response:
            return news_items
        
        # Check if it's a 404 page
        if "404" in response.text or "Page Not Found" in response.text:
            print(f"Got 404 for {url}")
            return news_items
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract company description as news - this works as shown in your image
        company_desc_selectors = [
            'p',  # The paragraph with company description
            '.company-description', 
            '.about-company p',
            'div p'  # Generic paragraph in div
        ]
        
        for selector in company_desc_selectors:
            company_desc = soup.select_one(selector)
            if company_desc:
                description = company_desc.text.strip()
                if len(description) > 100 and any(keyword.lower() in description.lower() for keyword in keywords[:3]):  # Check first 3 keywords
                    news_items.append({
                        'title': f"Company Profile: {symbol}",
                        'description': description,
                        'url': url,
                        'published_date': datetime.now(),
                        'source': 'zerodha_markets',
                        'symbols': [symbol],
                        'categories': ['company_profile']
                    })
                    print(f"Extracted company description for {symbol}")
                    break
        
        # Extract financial metrics as news items
        financial_metrics = {}
        
        # Look for the financial metrics table as shown in your image
        metrics_map = {
            'PE': ['PE', 'P/E'],
            'Sector PE': ['Sector PE'],
            'P/B': ['P/B', 'PB'],
            'Sector P/B': ['Sector P/B'],
            'Div.Yield': ['Div.Yield', 'Dividend Yield'],
            'ROE': ['ROE'],
            'Gross NPA': ['Gross NPA'],
            'Net NPA': ['Net NPA']
        }
        
        # Try to find these metrics in the page
        for metric_name, selectors in metrics_map.items():
            for selector in selectors:
                # Look for text containing the metric
                elements = soup.find_all(text=re.compile(selector, re.IGNORECASE))
                if elements:
                    # Try to find the value near this text
                    for element in elements:
                        parent = element.parent
                        if parent:
                            # Look for numbers in the parent or siblings
                            siblings = parent.find_next_siblings()
                            for sibling in siblings[:2]:  # Check next 2 siblings
                                text = sibling.get_text().strip()
                                if re.match(r'[\d.,]+%?', text):
                                    financial_metrics[metric_name] = text
                                    break
                            if metric_name in financial_metrics:
                                break
                    if metric_name in financial_metrics:
                        break
        
        if financial_metrics:
            metrics_desc = '\n'.join([f"{k}: {v}" for k, v in financial_metrics.items()])
            news_items.append({
                'title': f"Financial Ratios for {symbol}",
                'description': f"Key financial metrics:\n{metrics_desc}",
                'url': url,
                'published_date': datetime.now(),
                'source': 'zerodha_markets',
                'symbols': [symbol],
                'categories': ['financial_metrics']
            })
            print(f"Extracted {len(financial_metrics)} financial metrics for {symbol}")
        
        # Extract current price information
        price_elements = soup.find_all(text=re.compile(r'₹[\d,]+'))
        if price_elements:
            current_price = price_elements[0].strip()
            
            # Look for change information
            change_elements = soup.find_all(text=re.compile(r'[+-]?\d+\.\d+%'))
            change_info = change_elements[0].strip() if change_elements else ""
            
            news_items.append({
                'title': f"Current Price Update: {symbol}",
                'description': f"Current Price: {current_price}. Change: {change_info}",
                'url': url,
                'published_date': datetime.now(),
                'source': 'zerodha_markets',
                'symbols': [symbol],
                'categories': ['price_update']
            })
            print(f"Extracted price information for {symbol}")
        
        print(f"Found {len(news_items)} items from Zerodha Markets")
        return news_items
    
    def scrape_times_of_india(self, url: str, keywords: List[str], symbol: str) -> List[Dict[str, Any]]:
        """
        Scrape news from Times of India
        """
        news_items = []
        
        response = self.make_request(url)
        if not response:
            return news_items
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for article containers
        article_selectors = [
            '.uwU81',  # This worked according to your log
            '.Z7cWa', 
            '.Kl7Zq',
            '.article', 
            '.news-card', 
            '.story-card'
        ]
        
        articles = []
        for selector in article_selectors:
            articles = soup.select(selector)
            if articles:
                print(f"Found {len(articles)} articles using selector: {selector}")
                break
        
        if not articles:
            print("No articles found with any selector")
            return news_items
        
        # Process articles with more lenient keyword matching
        for article in articles[:15]:  # Check more articles
            try:
                title_elem = article.select_one('h1, h2, h3, h4, .title, .headline, a')
                if not title_elem:
                    continue
                
                title = title_elem.text.strip()
                if not title or len(title) < 10:
                    continue
                
                # More flexible keyword matching
                title_lower = title.lower()
                is_relevant = False
                
                # Check if any keyword appears in title
                for keyword in keywords:
                    if len(keyword) > 2 and keyword.lower() in title_lower:
                        is_relevant = True
                        print(f"Matched keyword '{keyword}' in title: {title[:50]}...")
                        break
                
                # Also check for common financial terms
                financial_terms = ['bank', 'banking', 'financial', 'stock', 'share', 'market', 'trading']
                if not is_relevant:
                    for term in financial_terms:
                        if term in title_lower and any(k.lower() in title_lower for k in keywords[:2]):
                            is_relevant = True
                            print(f"Matched financial term '{term}' with keyword in title: {title[:50]}...")
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
                
                # Extract date
                date_elem = article.select_one('.date, time, .timestamp, .meta, .publish-time')
                published_date = datetime.now()
                
                if date_elem:
                    date_text = date_elem.text.strip()
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
                print(f"Error parsing TOI article: {e}")
                continue
        
        print(f"Found {len(news_items)} relevant items from Times of India")
        return news_items
    
    def scrape_business_standard(self, url: str, keywords: List[str], symbol: str) -> List[Dict[str, Any]]:
        """
        Scrape news from Business Standard - with 403 handling
        """
        news_items = []
        
        # Try different URL formats since we're getting 403
        urls_to_try = [
            url,
            f"https://www.business-standard.com/search?q={symbol}",
            f"https://www.business-standard.com/companies/{symbol.lower()}"
        ]
        
        response = None
        for try_url in urls_to_try:
            print(f"Trying Business Standard URL: {try_url}")
            response = self.make_request(try_url)
            if response:
                break
            time.sleep(2)
        
        if not response:
            print("All Business Standard URLs failed")
            return news_items
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for article containers
        article_selectors = [
            '.cardHolder', 
            '.listing-page',
            '.article', 
            '.news-card', 
            '.story-card',
            '.search-result'
        ]
        
        articles = []
        for selector in article_selectors:
            articles = soup.select(selector)
            if articles:
                print(f"Found {len(articles)} articles using selector: {selector}")
                break
        
        for article in articles[:10]:
            try:
                title_elem = article.select_one('h1, h2, h3, h4, .title, .headline, a')
                if not title_elem:
                    continue
                
                title = title_elem.text.strip()
                if not title or len(title) < 10:
                    continue
                
                # Check relevance to keywords
                is_relevant = any(keyword.lower() in title.lower() for keyword in keywords[:3])
                if not is_relevant:
                    continue
                
                # Extract URL
                article_url = ""
                link_elem = title_elem if title_elem.name == 'a' else article.find('a')
                if link_elem and link_elem.get('href'):
                    article_url = link_elem['href']
                    if not article_url.startswith('http'):
                        article_url = urljoin('https://www.business-standard.com', article_url)
                
                # Extract description
                desc_elem = article.select_one('p, .summary, .content, .synopsis')
                description = desc_elem.text.strip() if desc_elem else ""
                
                news_items.append({
                    'title': title,
                    'description': description,
                    'url': article_url,
                    'published_date': datetime.now(),
                    'source': 'business_standard',
                    'symbols': [symbol],
                    'categories': ['market_news']
                })
                
            except Exception as e:
                print(f"Error parsing BS article: {e}")
                continue
        
        print(f"Found {len(news_items)} items from Business Standard")
        return news_items
    
    def save_to_database(self, news_items: List[Dict[str, Any]]) -> int:
        """
        Save news items to database
        """
        if not news_items:
            print("No news items to save")
            return 0
        
        saved_count = 0
        duplicate_count = 0
        
        for item in news_items:
            try:
                # Check for duplicates
                existing = self.news_collection.find_one({
                    "title": item['title'],
                    "source": item['source']
                })
                
                if existing:
                    duplicate_count += 1
                    continue
                
                # Add scraped timestamp
                item['scraped_at'] = datetime.now()
                
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
    
    def scrape_stock_news(self, symbol: str, exchange: str = "NSE") -> int:
        """
        Scrape news for a specific stock
        """
        print(f"Starting news scraping for {symbol}:{exchange}")
        
        # Get stock information
        stock_info = self.get_stock_info(symbol, exchange)
        if not stock_info:
            return 0
        
        # Get keywords from stock info
        keywords = []
        
        # Try to get keywords from company_details
        if 'company_details' in stock_info and 'keywords' in stock_info['company_details']:
            keywords = stock_info['company_details']['keywords']
        elif 'keywords' in stock_info:
            keywords = stock_info['keywords']
        else:
            # Create basic keywords
            keywords = [symbol]
            if 'company_name' in stock_info:
                keywords.append(stock_info['company_name'])
        
        print(f"Using keywords: {keywords}")
        
        # Get news URLs
        news_urls = stock_info.get('news_urls', {})
        if not news_urls:
            print("No news URLs found in stock information")
            return 0
        
        all_news = []
        
        # Scrape from each source
        if 'zerodha_market_url' in news_urls:
            print("\n" + "="*50)
            print("SCRAPING ZERODHA MARKETS")
            print("="*50)
            zerodha_news = self.scrape_zerodha_markets(
                news_urls['zerodha_market_url'], 
                keywords, 
                symbol
            )
            all_news.extend(zerodha_news)
            time.sleep(self.delay)
        
        if 'times_of_india_url' in news_urls:
            print("\n" + "="*50)
            print("SCRAPING TIMES OF INDIA")
            print("="*50)
            toi_news = self.scrape_times_of_india(
                news_urls['times_of_india_url'], 
                keywords, 
                symbol
            )
            all_news.extend(toi_news)
            time.sleep(self.delay)
        
        if 'business_standard_url' in news_urls:
            print("\n" + "="*50)
            print("SCRAPING BUSINESS STANDARD")
            print("="*50)
            bs_news = self.scrape_business_standard(
                news_urls['business_standard_url'], 
                keywords, 
                symbol
            )
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
        
        saved_count = self.save_to_database(all_news)
        
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
    
    args = parser.parse_args()
    
    scraper = None
    try:
        scraper = SimpleNewsScraper(args.db)
        total_saved = scraper.scrape_stock_news(args.symbol.upper(), args.exchange.upper())
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