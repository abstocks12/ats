import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import schedule
import argparse
from flask import Flask, render_template, jsonify, request
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin
# ----------------- CONFIGURATION -----------------
class Config:
    # Scraper Settings
    BASE_URL = "https://pulse.zerodha.com/"
    SEARCH_URL = "https://pulse.zerodha.com/?q=banking"
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    
    # Economic Times settings
    ET_MARKETS_URL = "https://economictimes.indiatimes.com/markets"
    ET_BANKING_URL = "https://economictimes.indiatimes.com/industry/banking/finance"
    INCLUDE_ET_SOURCES = True  # Set to False to disable Economic Times scraping
    
    # Additional news sources
   

    BLOOMBERG_URL = "https://www.bloombergquint.com/markets"
    HINDU_URL = "https://www.thehindu.com/business/markets"
    LIVEMINT_URL = "https://www.livemint.com/companies/company-results"
    GROWW_IDFC_URL = "https://groww.in/stocks/idfc-bank-ltd/market-news"
    INCLUDE_ADDITIONAL_SOURCES = True  # Set to False to disable these sources
    # Keywords for filtering
    BANKING_KEYWORDS = [
        "banking", "bank", "RBI", "Reserve Bank", "IDFC", "HDFC", "ICICI", 
        "SBI", "Axis", "Kotak", "Yes Bank", "IndusInd", "Federal Bank",
        "PSU banks", "private banks", "NBFCs", "NPA", "credit growth",
        "deposit", "lending", "monetary policy", "interest rate", "repo rate",
        "central bank", "financial services", "loan", "credit"
    ]
    
    IDFC_KEYWORDS = [
        "IDFC", "IDFC First", "IDFC Bank", "IDFC Ltd", "IDFC Limited",
        "IDFC Capital", "IDFC Securities", "IDFC AMC", "IDFC Asset Management",
        "IDFC Foundation", "IDFC Institute", "IDFC Alternatives"
    ]
    
    # File paths
    DATA_DIR = "data"
    DEBUG_DIR = "debug"
    LOG_FILE = os.path.join(DATA_DIR, "scraper.log")
    JSON_FILE = os.path.join(DATA_DIR, "banking_news.json")
    CSV_FILE = os.path.join(DATA_DIR, "banking_news.csv")
    DB_FILE = os.path.join(DATA_DIR, "banking_news.db")
    
    # Email settings (for alerts)
    EMAIL_ENABLED = False  # Set to True to enable email alerts
    EMAIL_FROM = "your-email@gmail.com"
    EMAIL_PASSWORD = "your-app-password"  # Use app password for Gmail
    EMAIL_TO = "recipient@example.com"
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    
    # Scheduling
    SCHEDULE_ENABLED = True
    SCHEDULE_INTERVAL = 60  # minutes
    
    # Flask web interface
    WEB_INTERFACE_ENABLED = True
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 5000
    
    # Sentiment analysis
    POSITIVE_KEYWORDS = [
        'rise', 'rally', 'gain', 'profit', 'growth', 'improve', 'strong',
        'boost', 'positive', 'increase', 'bullish', 'surge', 'jump',
        'upgrade', 'opportunity', 'outperform'
    ]
    
    NEGATIVE_KEYWORDS = [
        'fall', 'drop', 'loss', 'concern', 'weak', 'cut', 'hit',
        'decline', 'down', 'negative', 'bearish', 'plunge', 'crash',
        'downgrade', 'risk', 'underperform', 'pressure'
    ]
    
    # Debug mode
    DEBUG_MODE = True

# Add this new class for Bloomberg Quint and The Hindu scraping
class AdditionalNewsScraper:
    def __init__(self, debug_mode=True):
        self.bloomberg_url = Config.BLOOMBERG_URL
        self.hindu_markets_url = Config.HINDU_URL
        self.livemint_url = Config.LIVEMINT_URL
        self.groww_idfc_url = Config.GROWW_IDFC_URL
        self.debug_mode = debug_mode
        self.headers = {
            "User-Agent": Config.USER_AGENT
        }
        self.banking_keywords = Config.BANKING_KEYWORDS
        self.idfc_keywords = Config.IDFC_KEYWORDS
    
    def fetch_with_selenium(self, url, source_name):
        """Use Selenium to fetch JavaScript-rendered content"""
        chrome_options = Options()
        if not self.debug_mode:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            logger.info(f"Opening {source_name} URL with Selenium: {url}")
            driver.get(url)
            
            # Wait for page to load
            try:
                WebDriverWait(driver, 15).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
                logger.info(f"{source_name} page loaded (readyState is complete)")
            except:
                logger.warning(f"Timeout waiting for {source_name} page readyState, continuing anyway")
            
            # Additional wait for any dynamic content
            time.sleep(5)
            
            # Take a screenshot for debugging
            if self.debug_mode:
                driver.save_screenshot(os.path.join(Config.DEBUG_DIR, f"{source_name.lower().replace(' ', '_')}_page_load.png"))
                logger.info(f"Saved {source_name} page screenshot")
            
            # Try to scroll to load more content
            try:
                driver.execute_script("window.scrollTo(0, 500);")
                time.sleep(1)
                driver.execute_script("window.scrollTo(0, 1000);")
                time.sleep(1)
                driver.execute_script("window.scrollTo(0, 1500);")
                time.sleep(1)
                logger.info(f"Scrolled through {source_name} page")
            except Exception as e:
                logger.error(f"Error scrolling {source_name} page: {e}")
            
            # Save page source for debugging
            if self.debug_mode:
                with open(os.path.join(Config.DEBUG_DIR, f"{source_name.lower().replace(' ', '_')}_content.html"), "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                logger.info(f"Saved {source_name} page source")
            
            # Parse different sources based on their structures
            if source_name == "Bloomberg Quint":
                return self.parse_bloomberg(driver.page_source, url)
            elif source_name == "The Hindu":
                return self.parse_hindu(driver.page_source, url)
            elif source_name == "Livemint":
                return self.parse_livemint(driver.page_source, url)
            elif source_name == "Groww":
                return self.parse_groww(driver.page_source, url)
            else:
                logger.warning(f"Unknown source name: {source_name}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching {source_name} page: {e}")
            return []
        finally:
            try:
                driver.quit()
            except:
                pass
    
    def parse_bloomberg(self, html_content, base_url):
        """Parse Bloomberg Quint page content"""
        if not html_content:
            logger.warning("No Bloomberg HTML content to parse")
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        news_items = []
        
        # Bloomberg Quint selectors (adjust as needed based on actual HTML structure)
        articles = soup.select('.article-card, .top-story-card, .story-card, .featured-card')
        
        # Headline news section
        headline_news = soup.select('.headline, .latest-news-item, .breaking-news-item')
        
        # Combine all potential news items
        all_articles = articles + headline_news
        
        logger.info(f"Found {len(all_articles)} potential news items on Bloomberg Quint page")
        
        for article in all_articles:
            try:
                # Extract title
                title_elem = article.select_one('h1, h2, h3, h4, .headline, .title, a[title]')
                if not title_elem:
                    continue
                
                title = title_elem.get_text().strip()
                if not title:
                    continue
                
                # Extract link
                link = ""
                if title_elem.name == 'a' and title_elem.has_attr('href'):
                    link = title_elem['href']
                else:
                    link_elem = article.select_one('a')
                    if link_elem and link_elem.has_attr('href'):
                        link = link_elem['href']
                
                # Make link absolute
                if link and not link.startswith('http'):
                    link = urljoin(base_url, link)
                
                # Extract description
                description = ""
                desc_elem = article.select_one('p, .summary, .excerpt, .description')
                if desc_elem:
                    description = desc_elem.get_text().strip()
                
                # Extract date/time
                timestamp = "Unknown"
                time_elem = article.select_one('time, .date, .published-date, .timestamp')
                if time_elem:
                    timestamp = time_elem.get_text().strip()
                
                # Add to news items
                news_items.append({
                    'title': title,
                    'description': description,
                    'source': 'Bloomberg Quint',
                    'timestamp': timestamp,
                    'link': link,
                    'scraped_at': datetime.now().isoformat()
                })
                
                logger.info(f"Extracted from Bloomberg Quint: {title}")
                
            except Exception as e:
                logger.error(f"Error parsing Bloomberg article: {e}")
        
        return news_items
    
    def parse_hindu(self, html_content, base_url):
        """Parse The Hindu Markets page content"""
        if not html_content:
            logger.warning("No The Hindu HTML content to parse")
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        news_items = []
        
        # The Hindu selectors (adjust as needed based on actual HTML structure)
        articles = soup.select('.story-card, .story-card-33, .story-card-25, .story-card-50, article')
        
        # Headline news section
        headline_news = soup.select('.lead-story, .other-story, .just-in li')
        
        # Combine all potential news items
        all_articles = articles + headline_news
        
        logger.info(f"Found {len(all_articles)} potential news items on The Hindu page")
        
        for article in all_articles:
            try:
                # Extract title
                title_elem = article.select_one('h1, h2, h3, h4, .title, a')
                if not title_elem:
                    continue
                
                title = title_elem.get_text().strip()
                if not title:
                    continue
                
                # Extract link
                link = ""
                if title_elem.name == 'a' and title_elem.has_attr('href'):
                    link = title_elem['href']
                else:
                    link_elem = article.select_one('a')
                    if link_elem and link_elem.has_attr('href'):
                        link = link_elem['href']
                
                # Make link absolute
                if link and not link.startswith('http'):
                    link = urljoin(base_url, link)
                
                # Extract description
                description = ""
                desc_elem = article.select_one('p, .summary, .intro, .standfirst')
                if desc_elem:
                    description = desc_elem.get_text().strip()
                
                # Extract date/time
                timestamp = "Unknown"
                time_elem = article.select_one('time, .date-display, .dateline, .timestamp')
                if time_elem:
                    timestamp = time_elem.get_text().strip()
                
                # Add to news items
                news_items.append({
                    'title': title,
                    'description': description,
                    'source': 'The Hindu Business',
                    'timestamp': timestamp,
                    'link': link,
                    'scraped_at': datetime.now().isoformat()
                })
                
                logger.info(f"Extracted from The Hindu: {title}")
                
            except Exception as e:
                logger.error(f"Error parsing The Hindu article: {e}")
        
        return news_items
    
    def parse_livemint(self, html_content, base_url):
        """Parse Livemint Company Results page content"""
        if not html_content:
            logger.warning("No Livemint HTML content to parse")
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        news_items = []
        
        # Livemint selectors (adjust based on actual HTML structure)
        articles = soup.select('.headline, .linkclick, .listtotal .headtext, .headtext')
        
        # Additional article containers
        more_articles = soup.select('.companyresults .linkclick, .companyresults li, .resultWrap')
        
        # Combine all potential news items
        all_articles = articles + more_articles
        
        logger.info(f"Found {len(all_articles)} potential news items on Livemint page")
        
        for article in all_articles:
            try:
                # Extract title
                title_elem = article
                if article.name != 'a' and article.select_one('a'):
                    title_elem = article.select_one('a')
                
                title = title_elem.get_text().strip()
                if not title:
                    continue
                
                # Extract link
                link = ""
                if title_elem.name == 'a' and title_elem.has_attr('href'):
                    link = title_elem['href']
                else:
                    link_elem = article.select_one('a')
                    if link_elem and link_elem.has_attr('href'):
                        link = link_elem['href']
                
                # Make link absolute
                if link and not link.startswith('http'):
                    link = urljoin(base_url, link)
                
                # Extract description
                description = ""
                desc_elem = article.select_one('p, .synopsis, .summary')
                if desc_elem:
                    description = desc_elem.get_text().strip()
                
                # Extract date/time
                timestamp = "Unknown"
                time_elem = article.select_one('time, .date, .posted-on, .publishtime')
                if time_elem:
                    timestamp = time_elem.get_text().strip()
                
                # Add to news items
                news_items.append({
                    'title': title,
                    'description': description,
                    'source': 'Livemint',
                    'timestamp': timestamp,
                    'link': link,
                    'scraped_at': datetime.now().isoformat()
                })
                
                logger.info(f"Extracted from Livemint: {title}")
                
            except Exception as e:
                logger.error(f"Error parsing Livemint article: {e}")
        
        return news_items
    
    def parse_groww(self, html_content, base_url):
        """Parse Groww IDFC News page content"""
        if not html_content:
            logger.warning("No Groww HTML content to parse")
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        news_items = []
        
        # Groww selectors (adjust based on actual HTML structure)
        # The Groww website likely uses dynamic React components, so we need to try various selectors
        articles = soup.select('.newsCard, .newsItem, .newsFeed, div[class*="news"], article')
        
        # If standard selectors fail, try to find div elements with news-like content
        if not articles:
            potential_news_divs = soup.select('div:has(h3), div:has(h2), div:has(a)')
            # Filter to those that might be news items
            articles = [div for div in potential_news_divs if div.get_text().strip() and len(div.get_text().strip()) > 20]
        
        logger.info(f"Found {len(articles)} potential news items on Groww page")
        
        for article in articles:
            try:
                # Extract title
                title_elem = article.select_one('h1, h2, h3, h4, a, .newsTitle, .title')
                if not title_elem:
                    continue
                
                title = title_elem.get_text().strip()
                if not title or len(title) < 10:  # Skip very short titles as they might not be news
                    continue
                
                # Extract link
                link = ""
                if title_elem.name == 'a' and title_elem.has_attr('href'):
                    link = title_elem['href']
                else:
                    link_elem = article.select_one('a')
                    if link_elem and link_elem.has_attr('href'):
                        link = link_elem['href']
                
                # Make link absolute
                if link and not link.startswith('http'):
                    link = urljoin(base_url, link)
                
                # Extract description
                description = ""
                desc_elem = article.select_one('p, .newsContent, .content, .description')
                if desc_elem:
                    description = desc_elem.get_text().strip()
                    
                # If no description found but there's other text, use that
                if not description:
                    all_text = article.get_text().strip()
                    title_text = title.strip()
                    if len(all_text) > len(title_text):
                        description = all_text.replace(title_text, '').strip()
                
                # Extract date/time
                timestamp = "Unknown"
                time_elem = article.select_one('time, .dateTime, .date, .timeStamp, .publishedAt')
                if time_elem:
                    timestamp = time_elem.get_text().strip()
                
                # Add to news items - for Groww, we know it's specifically about IDFC
                news_items.append({
                    'title': title,
                    'description': description,
                    'source': 'Groww IDFC News',
                    'timestamp': timestamp,
                    'link': link,
                    'scraped_at': datetime.now().isoformat(),
                    'categories': ['banking', 'idfc']  # Pre-categorize as IDFC news
                })
                
                logger.info(f"Extracted from Groww: {title}")
                
            except Exception as e:
                logger.error(f"Error parsing Groww article: {e}")
        
        return news_items
    
    def filter_banking_news(self, news_items):
        """Filter news items to include only banking/IDFC related news"""
        banking_news = []
        
        for item in news_items:
            # Skip filtering for Groww items as they're already pre-categorized for IDFC
            if item.get('source') == 'Groww IDFC News':
                banking_news.append(item)
                continue
                
            # Check both title and description
            text_to_check = (item['title'] + " " + item.get('description', '')).lower()
            
            # Check for banking keywords
            is_banking = False
            matched_banking_keyword = None
            
            for keyword in self.banking_keywords:
                if keyword.lower() in text_to_check:
                    is_banking = True
                    matched_banking_keyword = keyword
                    break
            
            # Check for IDFC specifically
            is_idfc = False
            matched_idfc_keyword = None
            
            for keyword in self.idfc_keywords:
                if keyword.lower() in text_to_check:
                    is_idfc = True
                    matched_idfc_keyword = keyword
                    break
            
            if is_banking or is_idfc:
                categories = []
                if is_banking:
                    categories.append('banking')
                    logger.info(f"{item['source']} Banking match: '{matched_banking_keyword}' in article: {item['title']}")
                if is_idfc:
                    categories.append('idfc')
                    logger.info(f"{item['source']} IDFC match: '{matched_idfc_keyword}' in article: {item['title']}")
                
                item['categories'] = categories
                banking_news.append(item)
        
        return banking_news
    
    def run(self):
        """Run the additional news sources scraper"""
        additional_news = []
        
        # Scrape Bloomberg Quint
        logger.info(f"Fetching news from Bloomberg Quint")
        bloomberg_news = self.fetch_with_selenium(self.bloomberg_url, "Bloomberg Quint")
        if bloomberg_news:
            logger.info(f"Found {len(bloomberg_news)} items from Bloomberg Quint")
            additional_news.extend(bloomberg_news)
        
        # Scrape The Hindu Markets
        logger.info(f"Fetching news from The Hindu Markets")
        hindu_news = self.fetch_with_selenium(self.hindu_markets_url, "The Hindu")
        if hindu_news:
            logger.info(f"Found {len(hindu_news)} items from The Hindu Markets")
            additional_news.extend(hindu_news)
        
        # Scrape Livemint Company Results
        logger.info(f"Fetching news from Livemint Company Results")
        livemint_news = self.fetch_with_selenium(self.livemint_url, "Livemint")
        if livemint_news:
            logger.info(f"Found {len(livemint_news)} items from Livemint")
            additional_news.extend(livemint_news)
        
        # Scrape Groww IDFC News
        logger.info(f"Fetching news from Groww IDFC News")
        groww_news = self.fetch_with_selenium(self.groww_idfc_url, "Groww")
        if groww_news:
            logger.info(f"Found {len(groww_news)} items from Groww IDFC News")
            additional_news.extend(groww_news)
        
        # Filter to only banking/IDFC news
        filtered_news = self.filter_banking_news(additional_news)
        logger.info(f"Filtered to {len(filtered_news)} banking/IDFC related news items from additional sources")
        
        return filtered_news
# Add this new class for Economic Times scraping
class EconomicTimesScraper:
    def __init__(self, debug_mode=True):
        self.markets_url = Config.ET_MARKETS_URL
        self.banking_url = Config.ET_BANKING_URL
        self.debug_mode = debug_mode
        self.headers = {
            "User-Agent": Config.USER_AGENT
        }
        self.banking_keywords = Config.BANKING_KEYWORDS
        self.idfc_keywords = Config.IDFC_KEYWORDS
    
    def fetch_with_selenium(self, url):
        """Use Selenium to fetch JavaScript-rendered content"""
        chrome_options = Options()
        if not self.debug_mode:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            logger.info(f"Opening ET URL with Selenium: {url}")
            driver.get(url)
            
            # Wait for page to load
            try:
                WebDriverWait(driver, 15).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
                logger.info("ET page loaded (readyState is complete)")
            except:
                logger.warning("Timeout waiting for ET page readyState, continuing anyway")
            
            # Additional wait for any dynamic content
            time.sleep(5)
            
            # Take a screenshot for debugging
            if self.debug_mode:
                driver.save_screenshot(os.path.join(Config.DEBUG_DIR, "et_page_load.png"))
                logger.info("Saved ET page screenshot")
            
            # Try to scroll to load more content
            try:
                driver.execute_script("window.scrollTo(0, 500);")
                time.sleep(1)
                driver.execute_script("window.scrollTo(0, 1000);")
                time.sleep(1)
                driver.execute_script("window.scrollTo(0, 1500);")
                time.sleep(1)
                logger.info("Scrolled through ET page")
            except Exception as e:
                logger.error(f"Error scrolling ET page: {e}")
            
            # Save page source for debugging
            if self.debug_mode:
                with open(os.path.join(Config.DEBUG_DIR, "et_content.html"), "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                logger.info("Saved ET page source")
            
            # Parse the page
            html_content = driver.page_source
            return self.parse_et_page(html_content, url)
            
        except Exception as e:
            logger.error(f"Error fetching ET page: {e}")
            return []
        finally:
            try:
                driver.quit()
            except:
                pass
    
    def parse_et_page(self, html_content, base_url):
        """Parse Economic Times page content"""
        if not html_content:
            logger.warning("No ET HTML content to parse")
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        news_items = []
        
        # Try different selectors for Economic Times
        # Main news articles
        articles = soup.select('.newslist li article, .eachStory, .story_list .story_wrap, .top-news article')
        
        # Headline news section
        headline_news = soup.select('.headline, .topStories li, .featured-article')
        
        # Combine all potential news items
        all_articles = articles + headline_news
        
        logger.info(f"Found {len(all_articles)} potential news items on ET page")
        
        for article in all_articles:
            try:
                # Extract title
                title_elem = article.select_one('h2, h3, h4, .title, a[title]')
                if not title_elem:
                    continue
                
                title = title_elem.get_text().strip()
                if not title:
                    continue
                
                # Extract link
                link = ""
                if title_elem.name == 'a' and title_elem.has_attr('href'):
                    link = title_elem['href']
                else:
                    link_elem = article.select_one('a')
                    if link_elem and link_elem.has_attr('href'):
                        link = link_elem['href']
                
                # Make link absolute
                if link and not link.startswith('http'):
                    link = urljoin(base_url, link)
                
                # Extract description
                description = ""
                desc_elem = article.select_one('p, .summary, .synopsis')
                if desc_elem:
                    description = desc_elem.get_text().strip()
                
                # Extract date/time
                timestamp = "Unknown"
                time_elem = article.select_one('time, .date, .time')
                if time_elem:
                    timestamp = time_elem.get_text().strip()
                
                # Add to news items
                news_items.append({
                    'title': title,
                    'description': description,
                    'source': 'Economic Times',
                    'timestamp': timestamp,
                    'link': link,
                    'scraped_at': datetime.now().isoformat()
                })
                
                logger.info(f"Extracted from ET: {title}")
                
            except Exception as e:
                logger.error(f"Error parsing ET article: {e}")
        
        return news_items
    
    def filter_banking_news(self, news_items):
        """Filter news items to include only banking/IDFC related news"""
        banking_news = []
        
        for item in news_items:
            # Check both title and description
            text_to_check = (item['title'] + " " + item.get('description', '')).lower()
            
            # Check for banking keywords
            is_banking = False
            matched_banking_keyword = None
            
            for keyword in self.banking_keywords:
                if keyword.lower() in text_to_check:
                    is_banking = True
                    matched_banking_keyword = keyword
                    break
            
            # Check for IDFC specifically
            is_idfc = False
            matched_idfc_keyword = None
            
            for keyword in self.idfc_keywords:
                if keyword.lower() in text_to_check:
                    is_idfc = True
                    matched_idfc_keyword = keyword
                    break
            
            if is_banking or is_idfc:
                categories = []
                if is_banking:
                    categories.append('banking')
                    logger.info(f"ET Banking match: '{matched_banking_keyword}' in article: {item['title']}")
                if is_idfc:
                    categories.append('idfc')
                    logger.info(f"ET IDFC match: '{matched_idfc_keyword}' in article: {item['title']}")
                
                item['categories'] = categories
                banking_news.append(item)
        
        return banking_news
    
    def run(self):
        """Run the Economic Times scraper"""
        et_news = []
        
        # Scrape the Markets page
        logger.info(f"Fetching news from {self.markets_url}")
        markets_news = self.fetch_with_selenium(self.markets_url)
        if markets_news:
            logger.info(f"Found {len(markets_news)} items from ET Markets URL")
            et_news.extend(markets_news)
        
        # Scrape the Banking & Finance page
        logger.info(f"Fetching news from {self.banking_url}")
        banking_news = self.fetch_with_selenium(self.banking_url)
        if banking_news:
            logger.info(f"Found {len(banking_news)} items from ET Banking URL")
            et_news.extend(banking_news)
        
        # Filter to only banking/IDFC news
        filtered_news = self.filter_banking_news(et_news)
        logger.info(f"Filtered to {len(filtered_news)} banking/IDFC related news items from ET")
        
        return filtered_news
# ----------------- UTILITY FUNCTIONS -----------------
def setup_directories():
    """Create necessary directories"""
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    if Config.DEBUG_MODE:
        os.makedirs(Config.DEBUG_DIR, exist_ok=True)


def setup_logging():
    """Set up basic logging"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("banking_news_scraper")


def setup_database():
    """Initialize the SQLite database"""
    conn = sqlite3.connect(Config.DB_FILE)
    c = conn.cursor()
    
    # Create news table
    c.execute('''
    CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT UNIQUE,
        description TEXT,
        source TEXT,
        timestamp TEXT,
        link TEXT,
        scraped_at TEXT,
        categories TEXT,
        sentiment TEXT
    )
    ''')
    
    # Create summary table
    c.execute('''
    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        total_items INTEGER,
        processing_time TEXT,
        bank_mentions TEXT,
        sentiment_counts TEXT
    )
    ''')
    
    conn.commit()
    conn.close()


def send_email_alert(news_items, summary=None):
    """Send email alert with new news items"""
    if not Config.EMAIL_ENABLED:
        logger.info("Email alerts are disabled")
        return
    
    try:
        msg = MIMEMultipart()
        msg['From'] = Config.EMAIL_FROM
        msg['To'] = Config.EMAIL_TO
        msg['Subject'] = f"Banking News Alert - {len(news_items)} New Items"
        
        # Create HTML content
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .news-item { margin-bottom: 20px; padding: 10px; border-left: 3px solid #007bff; background-color: #f8f9fa; }
                .title { font-weight: bold; color: #007bff; }
                .meta { color: #6c757d; font-size: 0.9em; }
                .description { margin-top: 5px; }
                .sentiment-positive { color: green; }
                .sentiment-negative { color: red; }
                .sentiment-neutral { color: gray; }
                .summary { margin-top: 30px; padding: 15px; background-color: #e9ecef; border-radius: 5px; }
                .chart { margin-top: 20px; }
            </style>
        </head>
        <body>
            <h2>Banking News Alert</h2>
            <p>Found {total_items} new banking/IDFC related news items.</p>
            
            <div class="news-items">
        """.format(total_items=len(news_items))
        
        # Add news items
        for item in news_items:
            sentiment = "neutral"
            title = item['title'].lower()
            if any(word in title for word in Config.POSITIVE_KEYWORDS):
                sentiment = "positive"
            elif any(word in title for word in Config.NEGATIVE_KEYWORDS):
                sentiment = "negative"
            
            html += f"""
            <div class="news-item">
                <div class="title">{item['title']}</div>
                <div class="meta">
                    Source: {item.get('source', 'Unknown')} |
                    Time: {item.get('timestamp', 'Unknown')} |
                    Sentiment: <span class="sentiment-{sentiment}">{sentiment}</span>
                </div>
                <div class="description">{item.get('description', '')}</div>
                {f'<div><a href="{item["link"]}">Read more</a></div>' if item.get('link') else ''}
            </div>
            """
        
        # Add summary if provided
        if summary:
            html += """
            <div class="summary">
                <h3>Summary</h3>
                <p><strong>Most mentioned banks:</strong></p>
                <ul>
            """
            
            for bank, count in summary.get('bank_mentions', {}).items():
                html += f"<li>{bank}: {count} mentions</li>"
            
            html += """
                </ul>
                <p><strong>Sentiment analysis:</strong></p>
                <ul>
            """
            
            for sentiment, count in summary.get('sentiment', {}).items():
                html += f"<li>{sentiment}: {count} items</li>"
            
            html += """
                </ul>
            </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        # Attach HTML content
        msg.attach(MIMEText(html, 'html'))
        
        # If we have generated charts, attach them
        chart_files = [
            os.path.join(Config.DATA_DIR, "bank_mentions.png"),
            os.path.join(Config.DATA_DIR, "sentiment_distribution.png")
        ]
        
        for chart_file in chart_files:
            if os.path.exists(chart_file):
                with open(chart_file, 'rb') as f:
                    img = MIMEImage(f.read())
                    img_name = os.path.basename(chart_file)
                    img.add_header('Content-ID', f'<{img_name}>')
                    img.add_header('Content-Disposition', 'inline', filename=img_name)
                    msg.attach(img)
        
        # Send the email
        server = smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT)
        server.starttls()
        server.login(Config.EMAIL_FROM, Config.EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        logger.info("Email alert sent successfully")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")


def analyze_sentiment(title, description=""):
    """Analyze sentiment of a news item"""
    text = (title + " " + description).lower()
    
    if any(word in text for word in Config.POSITIVE_KEYWORDS):
        return "positive"
    elif any(word in text for word in Config.NEGATIVE_KEYWORDS):
        return "negative"
    else:
        return "neutral"


def save_to_database(news_items):
    """Save news items to SQLite database"""
    conn = sqlite3.connect(Config.DB_FILE)
    c = conn.cursor()
    
    new_items_count = 0
    
    for item in news_items:
        # Check if this news item already exists
        c.execute("SELECT id FROM news WHERE title = ?", (item['title'],))
        if c.fetchone() is None:
            # Add sentiment analysis
            sentiment = analyze_sentiment(item['title'], item.get('description', ''))
            
            # Insert new item
            c.execute('''
            INSERT INTO news (title, description, source, timestamp, link, scraped_at, categories, sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item['title'],
                item.get('description', ''),
                item.get('source', 'Unknown'),
                item.get('timestamp', 'Unknown'),
                item.get('link', ''),
                item['scraped_at'],
                ', '.join(item.get('categories', [])),
                sentiment
            ))
            new_items_count += 1
    
    conn.commit()
    conn.close()
    
    return new_items_count


def load_from_database(limit=100, offset=0, category=None, sentiment=None):
    """Load news items from the database with filtering options"""
    conn = sqlite3.connect(Config.DB_FILE)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    c = conn.cursor()
    
    query = "SELECT * FROM news"
    params = []
    
    # Apply filters
    filters = []
    if category:
        filters.append("categories LIKE ?")
        params.append(f"%{category}%")
    
    if sentiment:
        filters.append("sentiment = ?")
        params.append(sentiment)
    
    if filters:
        query += " WHERE " + " AND ".join(filters)
    
    # Add ordering and limits
    query += " ORDER BY scraped_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    c.execute(query, params)
    items = [dict(row) for row in c.fetchall()]
    
    # Get total count for pagination
    count_query = "SELECT COUNT(*) FROM news"
    if filters:
        count_query += " WHERE " + " AND ".join(filters)
    
    c.execute(count_query, params[:-2] if params else [])
    total_count = c.fetchone()[0]
    
    conn.close()
    
    return items, total_count


def process_banking_news(news_items):
    """Process news items to extract insights"""
    # Add a processing timestamp
    processing_time = datetime.now().isoformat()
    
    # Categorize by bank names
    bank_mentions = Counter()
    banks = [
        'sbi', 'state bank', 'hdfc', 'icici', 'pnb', 'punjab national', 
        'axis', 'kotak', 'yes bank', 'indusind', 'federal', 'idfc', 
        'rbi', 'reserve bank', 'union bank', 'bank of baroda'
    ]
    
    for item in news_items:
        title = item['title'].lower()
        description = item.get('description', '').lower()
        text = title + " " + description
        
        for bank in banks:
            if bank in text:
                bank_mentions[bank] += 1
    
    # Group by source
    sources = Counter([item.get('source', 'Unknown') for item in news_items])
    sources_filtered = Counter({k: v for k, v in sources.items() if k != 'Unknown'})    
    # Check for sentiment
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for item in news_items:
        title = item['title'].lower()
        description = item.get('description', '').lower()
        
        if any(word in title for word in Config.POSITIVE_KEYWORDS):
            sentiment_counts['positive'] += 1
        elif any(word in title for word in Config.NEGATIVE_KEYWORDS):
            sentiment_counts['negative'] += 1
        else:
            sentiment_counts['neutral'] += 1
    
    # Create summary
    summary = {
        'total_items': len(news_items),
        'processing_time': processing_time,
        'bank_mentions': dict(bank_mentions.most_common()),
        'sources': dict(sources_filtered.most_common()),
        'sentiment': sentiment_counts
    }
    
    # Save summary to database
    conn = sqlite3.connect(Config.DB_FILE)
    c = conn.cursor()
    c.execute('''
    INSERT INTO summaries (date, total_items, processing_time, bank_mentions, sentiment_counts)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d"),
        summary['total_items'],
        summary['processing_time'],
        json.dumps(summary['bank_mentions']),
        json.dumps(summary['sentiment'])
    ))
    conn.commit()
    conn.close()
    
    # Generate visualizations
    generate_visualizations(summary)
    
    return summary


def generate_visualizations(summary):
    """Generate charts based on the summary data"""
    # Visualize bank mentions
    if summary['bank_mentions']:
        plt.figure(figsize=(12, 6))
        banks = list(summary['bank_mentions'].keys())
        mentions = list(summary['bank_mentions'].values())
        
        # Sort by number of mentions
        banks_sorted = [x for _, x in sorted(zip(mentions, banks), reverse=True)]
        mentions_sorted = sorted(mentions, reverse=True)
        
        plt.bar(banks_sorted, mentions_sorted, color='skyblue')
        plt.title('Bank Mentions in News')
        plt.xlabel('Bank')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.DATA_DIR, 'bank_mentions.png'))
        plt.close()
    
    # Visualize sentiment
    plt.figure(figsize=(8, 6))
    sentiments = list(summary['sentiment'].keys())
    counts = list(summary['sentiment'].values())
    
    colors = ['green', 'red', 'gray']
    explode = (0.1, 0.1, 0)  # explode the positive and negative slices
    
    plt.pie(counts, labels=sentiments, autopct='%1.1f%%', 
            startangle=90, colors=colors, explode=explode)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Sentiment Distribution in Banking News')
    plt.savefig(os.path.join(Config.DATA_DIR, 'sentiment_distribution.png'))
    plt.close()


# ----------------- SCRAPER CLASS -----------------
class ZerodhaPulseScraper:
    def __init__(self, debug_mode=True):
        self.base_url = Config.BASE_URL
        self.search_url = Config.SEARCH_URL
        self.debug_mode = debug_mode
        self.headers = {
            "User-Agent": Config.USER_AGENT
        }
        self.banking_keywords = Config.BANKING_KEYWORDS
        self.idfc_keywords = Config.IDFC_KEYWORDS
    
    def fetch_with_selenium(self, url):
        """Use Selenium to fetch JavaScript-rendered content"""
        chrome_options = Options()
        if not self.debug_mode:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            logger.info(f"Opening URL with Selenium: {url}")
            driver.get(url)
            
            # Wait for page to load
            try:
                WebDriverWait(driver, 15).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
                logger.info("Page loaded (readyState is complete)")
            except:
                logger.warning("Timeout waiting for readyState, continuing anyway")
            
            # Additional wait for any dynamic content
            time.sleep(5)
            
            # Take a screenshot for debugging
            if self.debug_mode:
                driver.save_screenshot(os.path.join(Config.DEBUG_DIR, "page_load.png"))
            
            # Try to scroll to load more content
            try:
                driver.execute_script("window.scrollTo(0, 500);")
                time.sleep(1)
                driver.execute_script("window.scrollTo(0, 1000);")
                time.sleep(1)
                driver.execute_script("window.scrollTo(0, 1500);")
                time.sleep(1)
                logger.info("Scrolled through page")
            except Exception as e:
                logger.error(f"Error scrolling: {e}")
            
            # Save page source for debugging
            if self.debug_mode:
                with open(os.path.join(Config.DEBUG_DIR, "selenium_content.html"), "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
            
            # Directly extract news items using Selenium
            news_items = []
            
            # Find all articles
            articles = driver.find_elements(By.TAG_NAME, "article")
            logger.info(f"Found {len(articles)} article elements")
            
            if len(articles) == 0:
                # Try alternative selectors if no articles found
                headlines = driver.find_elements(By.TAG_NAME, "h2")
                logger.info(f"Found {len(headlines)} h2 elements as fallback")
                
                for headline in headlines:
                    try:
                        title = headline.text.strip()
                        if not title:
                            continue
                        
                        # Try to find a description
                        description = ""
                        parent = headline.find_element(By.XPATH, "./..")
                        
                        try:
                            desc_elem = parent.find_element(By.TAG_NAME, "p")
                            description = desc_elem.text.strip()
                        except:
                            pass
                        
                        # Try to find source and time
                        source = "Unknown"
                        timestamp = "Unknown"
                        
                        try:
                            footer = parent.find_element(By.TAG_NAME, "footer")
                            spans = footer.find_elements(By.TAG_NAME, "span")
                            
                            if spans and len(spans) > 0:
                                source = spans[0].text.strip()
                            if spans and len(spans) > 1:
                                timestamp = spans[1].text.strip()
                        except:
                            pass
                        
                        # Try to find a link
                        link = ""
                        try:
                            # Check if headline is within an <a> tag or has a parent <a>
                            parent_elem = headline
                            while parent_elem:
                                if parent_elem.tag_name == "a":
                                    link = parent_elem.get_attribute("href")
                                    break
                                try:
                                    parent_elem = parent_elem.find_element(By.XPATH, "./..")
                                except:
                                    parent_elem = None
                        except:
                            pass
                        
                        news_items.append({
                            'title': title,
                            'description': description,
                            'source': source,
                            'timestamp': timestamp,
                            'link': link,
                            'scraped_at': datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.error(f"Error processing headline: {e}")
            else:
                # Process each article
                for article in articles:
                    try:
                        # Get title
                        try:
                            title_elem = article.find_element(By.TAG_NAME, "h2")
                            title = title_elem.text.strip()
                        except:
                            # Try h3 if h2 not found
                            try:
                                title_elem = article.find_element(By.TAG_NAME, "h3")
                                title = title_elem.text.strip()
                            except:
                                continue  # Skip if no title found
                        
                        # Try to get description
                        description = ""
                        try:
                            desc_elem = article.find_element(By.TAG_NAME, "p")
                            description = desc_elem.text.strip()
                        except:
                            pass
                        
                        # Try to get source and timestamp
                        source = "Unknown"
                        timestamp = "Unknown"
                        
                        try:
                            footer = article.find_element(By.TAG_NAME, "footer")
                            spans = footer.find_elements(By.TAG_NAME, "span")
                            
                            if spans and len(spans) > 0:
                                source = spans[0].text.strip()
                            if spans and len(spans) > 1:
                                timestamp = spans[1].text.strip()
                        except:
                            pass
                        
                        # Try to get link
                        link = ""
                        try:
                            # Check if article is within an <a> tag
                            parent = article.find_element(By.XPATH, "./..")
                            if parent.tag_name == "a":
                                link = parent.get_attribute("href")
                            else:
                                # Try to find a link within the article
                                link_elem = article.find_element(By.TAG_NAME, "a")
                                link = link_elem.get_attribute("href")
                        except:
                            pass
                        
                        news_items.append({
                            'title': title,
                            'description': description,
                            'source': source,
                            'timestamp': timestamp,
                            'link': link,
                            'scraped_at': datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.error(f"Error processing article: {e}")
            
            return news_items
        except Exception as e:
            logger.error(f"Selenium error: {e}")
            return []
        finally:
            try:
                driver.quit()
            except:
                pass
    
    def filter_banking_news(self, news_items):
        """Filter news items to include only banking/IDFC related news"""
        if not news_items:
            logger.warning("No news items to filter")
            return []
        
        banking_news = []
        
        for item in news_items:
            # Check both title and description
            text_to_check = (item['title'] + " " + item.get('description', '')).lower()
            
            # Check for banking keywords
            is_banking = False
            matched_banking_keyword = None
            
            for keyword in self.banking_keywords:
                if keyword.lower() in text_to_check:
                    is_banking = True
                    matched_banking_keyword = keyword
                    break
            
            # Check for IDFC specifically
            is_idfc = False
            matched_idfc_keyword = None
            
            for keyword in self.idfc_keywords:
                if keyword.lower() in text_to_check:
                    is_idfc = True
                    matched_idfc_keyword = keyword
                    break
            
            if is_banking or is_idfc:
                categories = []
                if is_banking:
                    categories.append('banking')
                    logger.info(f"Banking match: '{matched_banking_keyword}' in article: {item['title']}")
                if is_idfc:
                    categories.append('idfc')
                    logger.info(f"IDFC match: '{matched_idfc_keyword}' in article: {item['title']}")
                
                item['categories'] = categories
                banking_news.append(item)
        
        return banking_news
    
    def run(self):
        """Run the scraper to collect banking/IDFC news"""
        all_news = []
        
        # Try the base URL first
        logger.info(f"Fetching news from {self.base_url}")
        base_news = self.fetch_with_selenium(self.base_url)
        
        if base_news:
            logger.info(f"Found {len(base_news)} items from base URL")
            all_news.extend(base_news)
        else:
            logger.warning("No news found from base URL")
        
        # Try the search URL for banking-specific news
        logger.info(f"Fetching banking-specific news from {self.search_url}")
        search_news = self.fetch_with_selenium(self.search_url)
        
        if search_news:
            logger.info(f"Found {len(search_news)} items from search URL")
            all_news.extend(search_news)
        else:
            logger.warning("No news found from search URL")
        
        # Filter to only banking/IDFC news
        banking_news = self.filter_banking_news(all_news)
        logger.info(f"Filtered to {len(banking_news)} banking/IDFC related news items")
        
        # Save the news items to JSON file
        with open(Config.JSON_FILE, 'w') as f:
            json.dump(banking_news, f, indent=2)
        
        # Save to CSV
        if banking_news:
            df = pd.DataFrame(banking_news)
            df.to_csv(Config.CSV_FILE, index=False)
        
        return banking_news


# ----------------- FLASK WEB INTERFACE -----------------
app = Flask(__name__)

@app.route('/')
def home():
    page = request.args.get('page', 1, type=int)
    category = request.args.get('category', None)
    sentiment = request.args.get('sentiment', None)
    
    items_per_page = 10
    news_items, total_count = load_from_database(
        limit=items_per_page, 
        offset=(page-1)*items_per_page,
        category=category,
        sentiment=sentiment
    )
    
    # Get latest summary
    conn = sqlite3.connect(Config.DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM summaries ORDER BY id DESC LIMIT 1")
    summary_row = c.fetchone()
    
    summary = None
    if summary_row:
        summary = dict(summary_row)
        summary['bank_mentions'] = json.loads(summary['bank_mentions'])
        summary['sentiment_counts'] = json.loads(summary['sentiment_counts'])
    
    conn.close()
    
    # Calculate pagination info
    total_pages = (total_count + items_per_page - 1) // items_per_page
    
    return render_template(
        'index.html',
        news=news_items,
        summary=summary,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
        category=category,
        sentiment=sentiment
    )

@app.route('/api/news')
def api_news():
    """API endpoint to get news items in JSON format"""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    category = request.args.get('category', None)
    sentiment = request.args.get('sentiment', None)
    
    news_items, total_count = load_from_database(
        limit=limit,
        offset=(page-1)*limit,
        category=category,
        sentiment=sentiment
    )
    
    return jsonify({
        'page': page,
        'limit': limit,
        'total': total_count,
        'total_pages': (total_count + limit - 1) // limit,
        'news': news_items
    })

@app.route('/api/summary')
def api_summary():
    """API endpoint to get latest summary"""
    conn = sqlite3.connect(Config.DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM summaries ORDER BY id DESC LIMIT 1")
    summary_row = c.fetchone()
    
    if summary_row:
        summary = dict(summary_row)
        summary['bank_mentions'] = json.loads(summary['bank_mentions'])
        summary['sentiment_counts'] = json.loads(summary['sentiment_counts'])
        return jsonify(summary)
    else:
        return jsonify({'error': 'No summary found'})


@app.route('/manual-scrape', methods=['POST'])
def manual_scrape():
    """Endpoint to manually trigger the scraper"""
    try:
        scraper = ZerodhaPulseScraper(debug_mode=Config.DEBUG_MODE)
        news_items = scraper.run()
        
        new_count = save_to_database(news_items)
        
        if new_count > 0:
            # Process the news and generate summary
            all_news, _ = load_from_database(limit=1000)  # Get all news for summary
            summary = process_banking_news(all_news)
            
            # Send email alert if configured
            if Config.EMAIL_ENABLED:
                send_email_alert(news_items[:10], summary)  # Send only the 10 newest items
            
            return jsonify({
                'status': 'success',
                'message': f'Scraping completed. Found {len(news_items)} items, {new_count} new.',
                'new_items': new_count
            })
        else:
            return jsonify({
                'status': 'success',
                'message': f'Scraping completed. Found {len(news_items)} items, but no new ones.',
                'new_items': 0
            })
    except Exception as e:
        logger.error(f"Error in manual scrape: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error during scraping: {str(e)}'
        }), 500


# ----------------- SCHEDULED TASKS -----------------
def scheduled_scrape():
    """Function to be run on schedule"""
    logger.info("Running scheduled scrape")
    
    try:
        all_news_items = []
        
        # Run Zerodha Pulse scraper
        scraper = ZerodhaPulseScraper(debug_mode=Config.DEBUG_MODE)
        zerodha_news = scraper.run()
        logger.info(f"Scheduled Zerodha Pulse scrape: Found {len(zerodha_news)} news items")
        all_news_items.extend(zerodha_news)
        
        # Run Economic Times scraper if enabled
        if Config.INCLUDE_ET_SOURCES:
            et_scraper = EconomicTimesScraper(debug_mode=Config.DEBUG_MODE)
            et_news = et_scraper.run()
            logger.info(f"Scheduled Economic Times scrape: Found {len(et_news)} news items")
            all_news_items.extend(et_news)
        
        # Run Bloomberg Quint and The Hindu scraper if enabled
        if Config.INCLUDE_ADDITIONAL_SOURCES:
            additional_scraper = AdditionalNewsScraper(debug_mode=Config.DEBUG_MODE)
            additional_news = additional_scraper.run()
            logger.info(f"Scheduled additional sources scrape: Found {len(additional_news)} news items")
            all_news_items.extend(additional_news)
        
        # Save to database
        new_count = save_to_database(all_news_items)
        logger.info(f"Scheduled scrape: Saved {new_count} new items to database")
        
        if new_count > 0:
            # Process the news and generate summary
            all_news, _ = load_from_database(limit=1000)
            summary = process_banking_news(all_news)
            
            # Send email alert if configured
            if Config.EMAIL_ENABLED:
                send_email_alert(all_news_items[:10], summary)
    except Exception as e:
        logger.error(f"Error in scheduled scrape: {e}")

def setup_scheduler():
    """Set up the scheduler for periodic scraping"""
    if Config.SCHEDULE_ENABLED:
        logger.info(f"Setting up scheduler to run every {Config.SCHEDULE_INTERVAL} minutes")
        schedule.every(Config.SCHEDULE_INTERVAL).minutes.do(scheduled_scrape)
        
        # Run in a separate thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
    else:
        logger.info("Scheduler is disabled")


# ----------------- TEMPLATES -----------------
def create_templates():
    """Create Flask templates if they don't exist"""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create base template
    base_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Banking News Monitor{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 20px; padding-bottom: 60px; }
        .news-item { margin-bottom: 20px; padding: 15px; border-left: 5px solid #007bff; background-color: #f8f9fa; }
        .title { font-weight: bold; font-size: 1.2em; }
        .meta { color: #6c757d; font-size: 0.9em; margin: 5px 0; }
        .sentiment-positive { color: green; border-color: green !important; }
        .sentiment-negative { color: red; border-color: red !important; }
        .sentiment-neutral { color: gray; border-color: gray !important; }
        .summary-section { margin-top: 30px; padding: 20px; background-color: #e9ecef; border-radius: 5px; }
        .chart-container { margin-top: 20px; text-align: center; }
        .banner { background-color: #007bff; color: white; padding: 20px; margin-bottom: 30px; border-radius: 5px; }
        .navigation { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
"""
    
    # Create index template
    index_template = """{% extends "base.html" %}

{% block title %}Banking News Monitor{% endblock %}

{% block content %}
    <div class="banner">
        <h1>Banking News Monitor</h1>
        <p>Stay updated with the latest in banking and financial news</p>
    </div>
    
    <div class="row">
        <div class="col-md-8">
            <div class="navigation">
                <h3>Latest Banking News</h3>
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <div class="filters">
                        <div class="btn-group">
                            <a href="{{ url_for('home') }}" class="btn btn-outline-primary {% if not category %}active{% endif %}">All</a>
                            <a href="{{ url_for('home', category='banking') }}" class="btn btn-outline-primary {% if category == 'banking' %}active{% endif %}">Banking</a>
                            <a href="{{ url_for('home', category='idfc') }}" class="btn btn-outline-primary {% if category == 'idfc' %}active{% endif %}">IDFC</a>
                        </div>
                        <div class="btn-group ms-2">
                            <a href="{{ url_for('home', category=category) }}" class="btn btn-outline-secondary {% if not sentiment %}active{% endif %}">All</a>
                            <a href="{{ url_for('home', category=category, sentiment='positive') }}" class="btn btn-outline-success {% if sentiment == 'positive' %}active{% endif %}">Positive</a>
                            <a href="{{ url_for('home', category=category, sentiment='negative') }}" class="btn btn-outline-danger {% if sentiment == 'negative' %}active{% endif %}">Negative</a>
                            <a href="{{ url_for('home', category=category, sentiment='neutral') }}" class="btn btn-outline-secondary {% if sentiment == 'neutral' %}active{% endif %}">Neutral</a>
                        </div>
                    </div>
                    
                    <form action="{{ url_for('manual_scrape') }}" method="post" class="d-inline">
                        <button type="submit" class="btn btn-primary">Refresh News</button>
                    </form>
                </div>
            </div>
            
            {% if news %}
                {% for item in news %}
                    <div class="news-item{% if item.sentiment %} sentiment-{{ item.sentiment }}{% endif %}">
                        <div class="title">{{ item.title }}</div>
                        <div class="meta">
                            Source: {{ item.source }} | 
                            Time: {{ item.timestamp }} | 
                            Categories: {{ item.categories }}
                            {% if item.sentiment %}
                                | Sentiment: <span class="badge {% if item.sentiment == 'positive' %}bg-success{% elif item.sentiment == 'negative' %}bg-danger{% else %}bg-secondary{% endif %}">{{ item.sentiment }}</span>
                            {% endif %}
                        </div>
                        {% if item.description %}
                            <div class="description">{{ item.description }}</div>
                        {% endif %}
                        {% if item.link %}
                            <div class="mt-2">
                                <a href="{{ item.link }}" target="_blank" class="btn btn-sm btn-outline-primary">Read more</a>
                            </div>
                        {% endif %}
                    </div>
                {% endfor %}
                
                <!-- Pagination -->
                {% if total_pages > 1 %}
                    <nav aria-label="Page navigation">
                        <ul class="pagination">
                            {% if page > 1 %}
                                <li class="page-item">
                                    <a class="page-link" href="{{ url_for('home', page=page-1, category=category, sentiment=sentiment) }}" aria-label="Previous">
                                        <span aria-hidden="true">&laquo;</span>
                                    </a>
                                </li>
                            {% endif %}
                            
                            {% for p in range(max(1, page-2), min(total_pages+1, page+3)) %}
                                <li class="page-item {% if p == page %}active{% endif %}">
                                    <a class="page-link" href="{{ url_for('home', page=p, category=category, sentiment=sentiment) }}">{{ p }}</a>
                                </li>
                            {% endfor %}
                            
                            {% if page < total_pages %}
                                <li class="page-item">
                                    <a class="page-link" href="{{ url_for('home', page=page+1, category=category, sentiment=sentiment) }}" aria-label="Next">
                                        <span aria-hidden="true">&raquo;</span>
                                    </a>
                                </li>
                            {% endif %}
                        </ul>
                    </nav>
                {% endif %}
            {% else %}
                <div class="alert alert-info">No news items found. Try refreshing or adjusting filters.</div>
            {% endif %}
        </div>
        
        <div class="col-md-4">
            {% if summary %}
                <div class="summary-section">
                    <h3>Summary</h3>
                    <p><strong>Total news items:</strong> {{ summary.total_items }}</p>
                    
                    <h4 class="mt-4">Sentiment Analysis</h4>
                    <div class="chart-container">
                        <canvas id="sentimentChart"></canvas>
                    </div>
                    
                    {% if summary.bank_mentions %}
                        <h4 class="mt-4">Most Mentioned Banks</h4>
                        <div class="chart-container">
                            <canvas id="bankChart"></canvas>
                        </div>
                        
                        <ul class="list-group mt-3">
                            {% for bank, count in summary.bank_mentions.items() %}
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    {{ bank }}
                                    <span class="badge bg-primary rounded-pill">{{ count }}</span>
                                </li>
                            {% endfor %}
                        </ul>
                    {% endif %}
                    
                    <div class="mt-4">
                        <p><small class="text-muted">Last updated: {{ summary.processing_time }}</small></p>
                    </div>
                </div>
            {% endif %}
        </div>
    </div>
{% endblock %}

{% block scripts %}
    {% if summary %}
        <script>
            // Sentiment Chart
            const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
            const sentimentChart = new Chart(sentimentCtx, {
                type: 'pie',
                data: {
                    labels: [{% for sentiment, count in summary.sentiment_counts.items() %}'{{ sentiment|capitalize }}'{% if not loop.last %}, {% endif %}{% endfor %}],
                    datasets: [{
                        data: [{% for sentiment, count in summary.sentiment_counts.items() %}{{ count }}{% if not loop.last %}, {% endif %}{% endfor %}],
                        backgroundColor: ['#28a745', '#dc3545', '#6c757d'],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom',
                        },
                    }
                }
            });
            
            {% if summary.bank_mentions %}
                // Bank Mentions Chart
                const bankCtx = document.getElementById('bankChart').getContext('2d');
                const bankChart = new Chart(bankCtx, {
                    type: 'bar',
                    data: {
                        labels: [{% for bank, count in summary.bank_mentions.items() %}'{{ bank }}'{% if not loop.last %}, {% endif %}{% endfor %}],
                        datasets: [{
                            label: 'Mentions',
                            data: [{% for bank, count in summary.bank_mentions.items() %}{{ count }}{% if not loop.last %}, {% endif %}{% endfor %}],
                            backgroundColor: 'rgba(54, 162, 235, 0.6)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        scales: {
                            x: {
                                beginAtZero: true,
                                ticks: {
                                    precision: 0
                                }
                            }
                        }
                    }
                });
            {% endif %}
        </script>
    {% endif %}
{% endblock %}
"""
    
    # Write templates to files
    with open(os.path.join(templates_dir, "base.html"), "w") as f:
        f.write(base_template)
    
    with open(os.path.join(templates_dir, "index.html"), "w") as f:
        f.write(index_template)



# ----------------- MAIN -----------------
def main():
    """Main entry point"""
    global logger
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Banking News Scraper")
    parser.add_argument("--no-web", action="store_true", help="Disable web interface")
    parser.add_argument("--no-schedule", action="store_true", help="Disable scheduled scraping")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-et", action="store_true", help="Disable Economic Times scraping")
    parser.add_argument("--no-additional", action="store_true", help="Disable Bloomberg Quint and The Hindu scraping")
    args = parser.parse_args()
    
    # Override config settings if specified in command line
    if args.no_web:
        Config.WEB_INTERFACE_ENABLED = False
    if args.no_schedule:
        Config.SCHEDULE_ENABLED = False
    if args.debug:
        Config.DEBUG_MODE = True
    if args.no_et:
        Config.INCLUDE_ET_SOURCES = False
    if args.no_additional:
        Config.INCLUDE_ADDITIONAL_SOURCES = False
    
    # Setup directories
    setup_directories()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Banking News Scraper")
    
    # Setup database
    setup_database()
    
    # Create templates
    create_templates()
    
    # Collect all news items
    all_news_items = []
    
    # Run Zerodha Pulse scraper
    scraper = ZerodhaPulseScraper(debug_mode=Config.DEBUG_MODE)
    zerodha_news = scraper.run()
    logger.info(f"Zerodha Pulse scrape: Found {len(zerodha_news)} news items")
    all_news_items.extend(zerodha_news)
    
    # Run Economic Times scraper if enabled
    if Config.INCLUDE_ET_SOURCES:
        et_scraper = EconomicTimesScraper(debug_mode=Config.DEBUG_MODE)
        et_news = et_scraper.run()
        logger.info(f"Economic Times scrape: Found {len(et_news)} news items")
        all_news_items.extend(et_news)
    
    # Run Bloomberg Quint and The Hindu scraper if enabled
    if Config.INCLUDE_ADDITIONAL_SOURCES:
        additional_scraper = AdditionalNewsScraper(debug_mode=Config.DEBUG_MODE)
        additional_news = additional_scraper.run()
        logger.info(f"Additional sources scrape: Found {len(additional_news)} news items")
        all_news_items.extend(additional_news)
    
    # Save to database
    new_count = save_to_database(all_news_items)
    logger.info(f"Initial scrape: Saved {new_count} new items to database")
    
    # Process and generate summary
    if all_news_items:
        db_news, _ = load_from_database(limit=1000)
        summary = process_banking_news(db_news)
        
        # Send email alert if configured
        if Config.EMAIL_ENABLED and new_count > 0:
            send_email_alert(all_news_items[:10], summary)
    
    # Setup scheduler if enabled
    if Config.SCHEDULE_ENABLED:
        setup_scheduler()
    
    # Start web interface if enabled
    if Config.WEB_INTERFACE_ENABLED:
        logger.info(f"Starting web interface on {Config.WEB_HOST}:{Config.WEB_PORT}")
        app.run(host=Config.WEB_HOST, port=Config.WEB_PORT, debug=Config.DEBUG_MODE)
    else:
        # If web interface is disabled, keep the script running for scheduler
        if Config.SCHEDULE_ENABLED:
            logger.info("Web interface disabled. Running scheduler only.")
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Scraper stopped by user")
        else:
            logger.info("Both web interface and scheduler are disabled. Exiting.")
if __name__ == "__main__":
    main()