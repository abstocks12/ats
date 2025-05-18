"""
Financial Scraper Module for the Automated Trading System.
Collects financial data from various sources.
"""

import os
import sys
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings
from database.connection_manager import get_db
from utils.logging_utils import setup_logger, log_error, log_data_collection
from utils.helper_functions import retry_function, normalize_symbol, sanitize_filename

class FinancialScraper:
    """
    Scraper for financial data from various sources including Screener.in, 
    Zerodha Markets, and Tijori Finance.
    """
    
    def __init__(self, symbol, exchange, db=None, debug_mode=False):
        """
        Initialize financial scraper
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            db: Database connector (optional, will use global connection if not provided)
            debug_mode (bool): Enable debug mode for verbose output
        """
        self.symbol = symbol
        self.exchange = exchange
        self.db = db or get_db()
        self.debug_mode = debug_mode
        self.logger = setup_logger(__name__)
        
        # Normalize symbol
        self.normalized_symbol = normalize_symbol(symbol, exchange)
        
        # Set user agent for requests
        self.headers = {
            'User-Agent': settings.USER_AGENT
        }
        
        # Set up URLs for different sources
        self.urls = self._setup_urls()
        
        # Define scraping delay to avoid rate limiting
        self.scraping_delay = settings.SCRAPING_DELAY
        
        # Initialize results storage
        self.financial_data = {
            'symbol': symbol,
            'exchange': exchange,
            'timestamp': datetime.now(),
            'quarterly_results': [],
            'annual_results': [],
            'key_metrics': {},
            'financial_ratios': {},
            'company_info': {},
            'shareholding_pattern': {}
        }
    
    def _setup_urls(self) -> Dict[str, str]:
        """
        Set up URLs for different data sources
        
        Returns:
            dict: Dictionary of URLs
        """
        # Convert symbol to lowercase and hyphenate for URL
        # Handle special cases like "&" and spaces
        url_symbol = self.symbol.lower().replace('&', 'and').replace(' ', '-')
        
        urls = {
            # Screener.in URL
            'screener': settings.FINANCIAL_SOURCES.get('screener', {}).get(
                'url_template', 'https://www.screener.in/company/{}/').format(url_symbol)
        }
        
        # Zerodha Markets URL
        urls['zerodha_markets'] = f'https://zerodha.com/markets/stocks/{url_symbol}/'
        
        # Tijori Finance URL
        urls['tijori_finance'] = f'https://www.tijorifinance.com/company/{url_symbol}/'
        
        # Add any additional sources here
        
        return urls
    
    def run(self) -> Dict[str, Any]:
        """
        Run the financial scraper and collect data from all sources
        
        Returns:
            dict: Collected financial data
        """
        self.logger.info(f"Starting financial data collection for {self.symbol}:{self.exchange}")
        
        try:
            # Screener.in data
            self._scrape_screener()
            
            # Zerodha Markets data
            self._scrape_zerodha_markets()
            
            # Tijori Finance data
            self._scrape_tijori_finance()
            
            # Save data to database
            self._save_to_database()
            
            # Return collected data
            return self.financial_data
            
        except Exception as e:
            log_error(e, context={"action": "run_financial_scraper", "symbol": self.symbol})
            return None
    
    def _scrape_screener(self) -> None:
        """Scrape financial data from Screener.in"""
        try:
            self.logger.info(f"Scraping Screener.in data for {self.symbol}")
            
            # Get URL
            url = self.urls['screener']
            
            # Make request
            response = self._make_request(url)
            
            if not response or response.status_code != 200:
                self.logger.warning(f"Failed to fetch data from Screener.in for {self.symbol}")
                return
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract company information
            self._extract_company_info(soup)
            
            # Extract key metrics
            self._extract_key_metrics(soup)
            
            # Extract quarterly results
            self._extract_quarterly_results(soup)
            
            # Extract annual results
            self._extract_annual_results(soup)
            
            # Extract financial ratios
            self._extract_financial_ratios(soup)
            
            # Extract shareholding pattern
            self._extract_shareholding_pattern(soup)
            
            self.logger.info(f"Successfully scraped Screener.in data for {self.symbol}")
            
        except Exception as e:
            log_error(e, context={"action": "scrape_screener", "symbol": self.symbol})
    
    def _scrape_zerodha_markets(self) -> None:
        """Scrape financial data from Zerodha Markets"""
        try:
            self.logger.info(f"Scraping Zerodha Markets data for {self.symbol}")
            
            # Get URL
            url = self.urls['zerodha_markets']
            
            # Make request
            response = self._make_request(url)
            
            if not response or response.status_code != 200:
                self.logger.warning(f"Failed to fetch data from Zerodha Markets for {self.symbol}")
                return
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract company information
            self._extract_zerodha_company_info(soup)
            
            # Extract key metrics
            self._extract_zerodha_key_metrics(soup)
            
            # Extract financial data
            self._extract_zerodha_financial_data(soup)
            
            self.logger.info(f"Successfully scraped Zerodha Markets data for {self.symbol}")
            
        except Exception as e:
            log_error(e, context={"action": "scrape_zerodha_markets", "symbol": self.symbol})
    
    def _scrape_tijori_finance(self) -> None:
        """Scrape financial data from Tijori Finance"""
        try:
            self.logger.info(f"Scraping Tijori Finance data for {self.symbol}")
            
            # Get URL
            url = self.urls['tijori_finance']
            
            # Make request
            response = self._make_request(url)
            
            if not response or response.status_code != 200:
                self.logger.warning(f"Failed to fetch data from Tijori Finance for {self.symbol}")
                return
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract company information
            self._extract_tijori_company_info(soup)
            
            # Extract key metrics
            self._extract_tijori_key_metrics(soup)
            
            # Extract financial data
            self._extract_tijori_financial_data(soup)
            
            self.logger.info(f"Successfully scraped Tijori Finance data for {self.symbol}")
            
        except Exception as e:
            log_error(e, context={"action": "scrape_tijori_finance", "symbol": self.symbol})
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic
        
        Args:
            url (str): URL to request
            
        Returns:
            requests.Response: Response object or None on failure
        """
        try:
            def request_with_timeout():
                return requests.get(url, headers=self.headers, timeout=settings.SCRAPING_TIMEOUT)
            
            # Use retry function from helper_functions
            response = retry_function(
                request_with_timeout,
                max_retries=settings.SCRAPING_RETRY_COUNT,
                delay=1,
                backoff=2,
                exceptions=(requests.RequestException,)
            )
            
            # Add delay to avoid rate limiting
            time.sleep(self.scraping_delay)
            
            return response
            
        except Exception as e:
            log_error(e, context={"action": "make_request", "url": url})
            return None
    
    def _extract_company_info(self, soup: BeautifulSoup) -> None:
        """
        Extract company information from Screener.in
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Extract company name
            company_name_elem = soup.select_one('h1.margin-0')
            if company_name_elem:
                self.financial_data['company_info']['name'] = company_name_elem.text.strip()
            
            # Extract company description
            description_elem = soup.select_one('.company-description')
            if description_elem:
                self.financial_data['company_info']['description'] = description_elem.text.strip()
            
            # Extract sector
            sector_elem = soup.select_one('li.widget-link p.sub')
            if sector_elem:
                sector_text = sector_elem.text.strip()
                if ':' in sector_text:
                    self.financial_data['company_info']['sector'] = sector_text.split(':', 1)[1].strip()
            
            # Extract market cap
            market_cap_elem = soup.select_one('li.flex.flex-space-between:contains("MARKET CAP")')
            if market_cap_elem:
                market_cap_text = market_cap_elem.select_one('.number').text.strip()
                self.financial_data['company_info']['market_cap'] = self._parse_numeric_value(market_cap_text)
            
            # Extract current price
            price_elem = soup.select_one('div.primary-value')
            if price_elem:
                self.financial_data['company_info']['current_price'] = self._parse_numeric_value(price_elem.text.strip())
            
        except Exception as e:
            log_error(e, context={"action": "extract_company_info", "symbol": self.symbol})
    
    def _extract_key_metrics(self, soup: BeautifulSoup) -> None:
        """
        Extract key metrics from Screener.in
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Extract ratios section
            ratios_section = soup.select_one('section.company-ratios')
            
            if not ratios_section:
                return
            
            # Extract each ratio
            ratios = ratios_section.select('li')
            
            for ratio in ratios:
                try:
                    # Extract name and value
                    name_elem = ratio.select_one('.name')
                    value_elem = ratio.select_one('.value')
                    
                    if not name_elem or not value_elem:
                        continue
                    
                    name = name_elem.text.strip().lower().replace(' ', '_')
                    value = self._parse_numeric_value(value_elem.text.strip())
                    
                    # Add to key metrics
                    self.financial_data['key_metrics'][name] = value
                    
                except Exception as e:
                    if self.debug_mode:
                        self.logger.debug(f"Error parsing ratio: {e}")
                    continue
            
        except Exception as e:
            log_error(e, context={"action": "extract_key_metrics", "symbol": self.symbol})
    
    def _extract_quarterly_results(self, soup: BeautifulSoup) -> None:
        """
        Extract quarterly results from Screener.in
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Find quarterly results section
            quarterly_section = soup.find('section', id='quarters')
            
            if not quarterly_section:
                return
            
            # Extract table
            table = quarterly_section.select_one('table')
            
            if not table:
                return
            
            # Extract headers
            headers = [th.text.strip() for th in table.select('thead th')]
            
            # Handle empty headers
            if not headers:
                return
            
            # Extract rows
            rows = table.select('tbody tr')
            
            # Process quarters from columns
            quarters = headers[1:]  # First column is metric name
            
            # Initialize results for each quarter
            quarterly_results = []
            for quarter in quarters:
                quarterly_results.append({
                    'quarter': quarter
                })
            
            # Extract data for each row
            for row in rows:
                cells = row.select('td')
                if len(cells) <= 1:
                    continue
                
                metric_name = cells[0].text.strip().lower().replace(' ', '_')
                
                # Process values for each quarter
                for i, cell in enumerate(cells[1:]):
                    if i >= len(quarterly_results):
                        break
                    
                    value = self._parse_numeric_value(cell.text.strip())
                    quarterly_results[i][metric_name] = value
            
            # Add to financial data
            self.financial_data['quarterly_results'] = quarterly_results
            
        except Exception as e:
            log_error(e, context={"action": "extract_quarterly_results", "symbol": self.symbol})
    
    def _extract_annual_results(self, soup: BeautifulSoup) -> None:
        """
        Extract annual results from Screener.in
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Find annual results section
            annual_section = soup.find('section', id='profit-loss')
            
            if not annual_section:
                # Try alternative section IDs
                annual_section = soup.find('section', id='annual-report')
                
                if not annual_section:
                    return
            
            # Extract table
            table = annual_section.select_one('table')
            
            if not table:
                return
            
            # Extract headers
            headers = [th.text.strip() for th in table.select('thead th')]
            
            # Handle empty headers
            if not headers:
                return
            
            # Extract rows
            rows = table.select('tbody tr')
            
            # Process years from columns
            years = headers[1:]  # First column is metric name
            
            # Initialize results for each year
            annual_results = []
            for year in years:
                annual_results.append({
                    'year': year
                })
            
            # Extract data for each row
            for row in rows:
                cells = row.select('td')
                if len(cells) <= 1:
                    continue
                
                metric_name = cells[0].text.strip().lower().replace(' ', '_')
                
                # Process values for each year
                for i, cell in enumerate(cells[1:]):
                    if i >= len(annual_results):
                        break
                    
                    value = self._parse_numeric_value(cell.text.strip())
                    annual_results[i][metric_name] = value
            
            # Add to financial data
            self.financial_data['annual_results'] = annual_results
            
        except Exception as e:
            log_error(e, context={"action": "extract_annual_results", "symbol": self.symbol})
    
    def _extract_financial_ratios(self, soup: BeautifulSoup) -> None:
        """
        Extract financial ratios from Screener.in
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Find ratios sections
            ratio_sections = soup.select('section.company-ratios, section.flex.flex-wrap')
            
            for section in ratio_sections:
                # Extract each ratio
                ratios = section.select('li.flex.flex-space-between')
                
                for ratio in ratios:
                    try:
                        # Extract name and value
                        name_elem = ratio.select_one('.name, .flex-column p')
                        value_elem = ratio.select_one('.value, .number')
                        
                        if not name_elem or not value_elem:
                            continue
                        
                        name = name_elem.text.strip().lower().replace(' ', '_')
                        value = self._parse_numeric_value(value_elem.text.strip())
                        
                        # Add to financial ratios
                        self.financial_data['financial_ratios'][name] = value
                        
                    except Exception as e:
                        if self.debug_mode:
                            self.logger.debug(f"Error parsing ratio: {e}")
                        continue
            
        except Exception as e:
            log_error(e, context={"action": "extract_financial_ratios", "symbol": self.symbol})
    
    def _extract_shareholding_pattern(self, soup: BeautifulSoup) -> None:
        """
        Extract shareholding pattern from Screener.in
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Find shareholding section
            shareholding_section = soup.find('section', id='shareholding')
            
            if not shareholding_section:
                return
            
            # Extract tables
            tables = shareholding_section.select('table')
            
            if not tables:
                return
            
            for table in tables:
                # Extract headers
                headers = [th.text.strip() for th in table.select('thead th')]
                
                # Extract rows
                rows = table.select('tbody tr')
                
                for row in rows:
                    cells = row.select('td')
                    if len(cells) < 2:
                        continue
                    
                    category = cells[0].text.strip()
                    percentage = self._parse_numeric_value(cells[-1].text.strip())
                    
                    # Add to shareholding pattern
                    self.financial_data['shareholding_pattern'][category] = percentage
            
        except Exception as e:
            log_error(e, context={"action": "extract_shareholding_pattern", "symbol": self.symbol})
    
    def _extract_zerodha_company_info(self, soup: BeautifulSoup) -> None:
        """
        Extract company information from Zerodha Markets
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Extract company name
            company_name_elem = soup.select_one('h1.stock-name')
            if company_name_elem:
                self.financial_data['company_info']['name'] = company_name_elem.text.strip()
            
            # Extract company description
            description_elem = soup.select_one('.company-description')
            if description_elem:
                self.financial_data['company_info']['description'] = description_elem.text.strip()
            
            # Extract sector
            sector_elem = soup.select_one('.stock-details .details-label:contains("Sector") + .details-value')
            if sector_elem:
                self.financial_data['company_info']['sector'] = sector_elem.text.strip()
            
            # Extract industry
            industry_elem = soup.select_one('.stock-details .details-label:contains("Industry") + .details-value')
            if industry_elem:
                self.financial_data['company_info']['industry'] = industry_elem.text.strip()
            
            # Extract market cap
            market_cap_elem = soup.select_one('.stock-details .details-label:contains("Market cap") + .details-value')
            if market_cap_elem:
                self.financial_data['company_info']['market_cap'] = self._parse_numeric_value(market_cap_elem.text.strip())
            
            # Extract current price
            price_elem = soup.select_one('.stock-price')
            if price_elem:
                self.financial_data['company_info']['current_price'] = self._parse_numeric_value(price_elem.text.strip())
            
        except Exception as e:
            log_error(e, context={"action": "extract_zerodha_company_info", "symbol": self.symbol})
    
    def _extract_zerodha_key_metrics(self, soup: BeautifulSoup) -> None:
        """
        Extract key metrics from Zerodha Markets
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Find key metrics section
            metrics_section = soup.select_one('.key-metrics')
            
            if not metrics_section:
                return
            
            # Extract each metric
            metrics = metrics_section.select('.metric')
            
            for metric in metrics:
                try:
                    # Extract name and value
                    name_elem = metric.select_one('.metric-name')
                    value_elem = metric.select_one('.metric-value')
                    
                    if not name_elem or not value_elem:
                        continue
                    
                    name = name_elem.text.strip().lower().replace(' ', '_')
                    value = self._parse_numeric_value(value_elem.text.strip())
                    
                    # Add to key metrics
                    self.financial_data['key_metrics'][name] = value
                    
                except Exception as e:
                    if self.debug_mode:
                        self.logger.debug(f"Error parsing metric: {e}")
                    continue
                    
            # Extract additional metrics from fundamentals section
            fundamentals_section = soup.select_one('#fundamentals')
            
            if fundamentals_section:
                # Extract each fundamental metric
                metrics = fundamentals_section.select('.fundamental-item')
                
                for metric in metrics:
                    try:
                        # Extract name and value
                        name_elem = metric.select_one('.item-name')
                        value_elem = metric.select_one('.item-value')
                        
                        if not name_elem or not value_elem:
                            continue
                        
                        name = name_elem.text.strip().lower().replace(' ', '_')
                        value = self._parse_numeric_value(value_elem.text.strip())
                        
                        # Add to key metrics
                        self.financial_data['key_metrics'][name] = value
                        
                    except Exception as e:
                        if self.debug_mode:
                            self.logger.debug(f"Error parsing fundamental metric: {e}")
                        continue
            
        except Exception as e:
            log_error(e, context={"action": "extract_zerodha_key_metrics", "symbol": self.symbol})
    
    def _extract_zerodha_financial_data(self, soup: BeautifulSoup) -> None:
        """
        Extract financial data from Zerodha Markets
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Find financials section
            financials_section = soup.select_one('#financials')
            
            if not financials_section:
                return
            
            # Extract quarterly results
            quarterly_section = financials_section.select_one('.quarterly-results')
            
            if quarterly_section:
                tables = quarterly_section.select('table')
                
                for table in tables:
                    self._process_zerodha_table(table, 'quarterly_results')
            
            # Extract annual results
            annual_section = financials_section.select_one('.annual-results')
            
            if annual_section:
                tables = annual_section.select('table')
                
                for table in tables:
                    self._process_zerodha_table(table, 'annual_results')
            
        except Exception as e:
            log_error(e, context={"action": "extract_zerodha_financial_data", "symbol": self.symbol})
    
    def _process_zerodha_table(self, table: BeautifulSoup, result_type: str) -> None:
        """
        Process financial table from Zerodha Markets
        
        Args:
            table (BeautifulSoup): Table element
            result_type (str): Type of result ('quarterly_results' or 'annual_results')
        """
        try:
            # Extract headers
            headers = [th.text.strip() for th in table.select('thead th')]
            
            # Handle empty headers
            if not headers:
                return
            
            # Extract rows
            rows = table.select('tbody tr')
            
            # Get existing results
            results = self.financial_data[result_type]
            
            # Process periods from columns
            periods = headers[1:]  # First column is metric name
            
            # Check if periods already exist in results
            existing_periods = [r.get('quarter' if result_type == 'quarterly_results' else 'year') for r in results]
            
            # Initialize results for each period if needed
            for period in periods:
                if period in existing_periods:
                    continue
                
                period_key = 'quarter' if result_type == 'quarterly_results' else 'year'
                results.append({
                    period_key: period
                })
            
            # Extract data for each row
            for row in rows:
                cells = row.select('td')
                if len(cells) <= 1:
                    continue
                
                metric_name = cells[0].text.strip().lower().replace(' ', '_')
                
                # Process values for each period
                for i, cell in enumerate(cells[1:]):
                    if i >= len(periods):
                        break
                    
                    period = periods[i]
                    value = self._parse_numeric_value(cell.text.strip())
                    
                    # Find matching result
                    period_key = 'quarter' if result_type == 'quarterly_results' else 'year'
                    result_entry = next((r for r in results if r.get(period_key) == period), None)
                    
                    if result_entry:
                        result_entry[metric_name] = value
            
            # Update financial data
            self.financial_data[result_type] = results
            
        except Exception as e:
            log_error(e, context={"action": "process_zerodha_table", "symbol": self.symbol})
    
    def _extract_tijori_company_info(self, soup: BeautifulSoup) -> None:
        """
        Extract company information from Tijori Finance
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Extract company name
            company_name_elem = soup.select_one('h1.company-name')
            if company_name_elem:
                self.financial_data['company_info']['name'] = company_name_elem.text.strip()
            
            # Extract company description
            description_elem = soup.select_one('.company-description')
            if description_elem:
                self.financial_data['company_info']['description'] = description_elem.text.strip()
            
            # Extract sector
            sector_elem = soup.select_one('.company-details .label:contains("Sector") + .value')
            if sector_elem:
                self.financial_data['company_info']['sector'] = sector_elem.text.strip()
            
            # Extract industry
            industry_elem = soup.select_one('.company-details .label:contains("Industry") + .value')
            if industry_elem:
                self.financial_data['company_info']['industry'] = industry_elem.text.strip()
            
            # Extract market cap
            market_cap_elem = soup.select_one('.company-metrics .label:contains("Market Cap") + .value')
            if market_cap_elem:
                self.financial_data['company_info']['market_cap'] = self._parse_numeric_value(market_cap_elem.text.strip())
            
            # Extract current price
            price_elem = soup.select_one('.current-price')
            if price_elem:
                self.financial_data['company_info']['current_price'] = self._parse_numeric_value(price_elem.text.strip())
                
            # Extract more company info
            info_section = soup.select_one('.company-overview')
            if info_section:
                info_items = info_section.select('.info-item')
                
                for item in info_items:
                    label_elem = item.select_one('.label')
                    value_elem = item.select_one('.value')
                    
                    if not label_elem or not value_elem:
                        continue
                    
                    label = label_elem.text.strip().lower().replace(' ', '_')
                    value = value_elem.text.strip()
                    
                    # Try to parse as numeric if possible
                    try:
                        value = self._parse_numeric_value(value)
                    except:
                        pass
                    
                    self.financial_data['company_info'][label] = value
            
        except Exception as e:
            log_error(e, context={"action": "extract_tijori_company_info", "symbol": self.symbol})
    
    def _extract_zerodha_market_depth(self, soup: BeautifulSoup) -> None:
        """
        Extract market depth information from Zerodha Markets
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Find market depth section
            depth_section = soup.select_one('#market-depth')
            
            if not depth_section:
                return
            
            # Extract buy side
            buy_side = {}
            buy_table = depth_section.select_one('.buy-side')
            
            if buy_table:
                buy_rows = buy_table.select('tr')
                
                for i, row in enumerate(buy_rows):
                    cells = row.select('td')
                    if len(cells) >= 2:
                        price = self._parse_numeric_value(cells[0].text.strip())
                        quantity = self._parse_numeric_value(cells[1].text.strip())
                        buy_side[f'level_{i+1}'] = {'price': price, 'quantity': quantity}
            
            # Extract sell side
            sell_side = {}
            sell_table = depth_section.select_one('.sell-side')
            
            if sell_table:
                sell_rows = sell_table.select('tr')
                
                for i, row in enumerate(sell_rows):
                    cells = row.select('td')
                    if len(cells) >= 2:
                        price = self._parse_numeric_value(cells[0].text.strip())
                        quantity = self._parse_numeric_value(cells[1].text.strip())
                        sell_side[f'level_{i+1}'] = {'price': price, 'quantity': quantity}
            
            # Add to financial data
            self.financial_data['market_depth'] = {
                'buy_side': buy_side,
                'sell_side': sell_side,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error(e, context={"action": "extract_zerodha_market_depth", "symbol": self.symbol})
    
    
    # Extract methods for Tijori Finance data
    
    def _extract_tijori_key_metrics(self, soup: BeautifulSoup) -> None:
        """
        Extract key metrics from Tijori Finance
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Find key metrics section
            metrics_section = soup.select_one('.key-metrics')
            
            if not metrics_section:
                return
            
            # Extract each metric
            metrics = metrics_section.select('.metric-item')
            
            for metric in metrics:
                try:
                    # Extract name and value
                    name_elem = metric.select_one('.metric-name')
                    value_elem = metric.select_one('.metric-value')
                    
                    if not name_elem or not value_elem:
                        continue
                    
                    name = name_elem.text.strip().lower().replace(' ', '_')
                    value = self._parse_numeric_value(value_elem.text.strip())
                    
                    # Add to key metrics
                    self.financial_data['key_metrics'][name] = value
                    
                except Exception as e:
                    if self.debug_mode:
                        self.logger.debug(f"Error parsing metric: {e}")
                    continue
            
            # Extract valuation metrics
            valuation_section = soup.select_one('.valuation-metrics')
            
            if valuation_section:
                metrics = valuation_section.select('.metric-item')
                
                for metric in metrics:
                    try:
                        # Extract name and value
                        name_elem = metric.select_one('.metric-name')
                        value_elem = metric.select_one('.metric-value')
                        
                        if not name_elem or not value_elem:
                            continue
                        
                        name = name_elem.text.strip().lower().replace(' ', '_')
                        value = self._parse_numeric_value(value_elem.text.strip())
                        
                        # Add to key metrics
                        self.financial_data['key_metrics'][name] = value
                        
                    except Exception as e:
                        if self.debug_mode:
                            self.logger.debug(f"Error parsing valuation metric: {e}")
                        continue
            
        except Exception as e:
            log_error(e, context={"action": "extract_tijori_key_metrics", "symbol": self.symbol})
    
    def _extract_tijori_financial_data(self, soup: BeautifulSoup) -> None:
        """
        Extract financial data from Tijori Finance
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Find financials section
            financials_section = soup.select_one('#financials')
            
            if not financials_section:
                return
            
            # Extract quarterly results
            quarterly_section = financials_section.select_one('.quarterly-financials')
            
            if quarterly_section:
                tables = quarterly_section.select('table')
                
                for table in tables:
                    self._process_tijori_table(table, 'quarterly_results')
            
            # Extract annual results
            annual_section = financials_section.select_one('.annual-financials')
            
            if annual_section:
                tables = annual_section.select('table')
                
                for table in tables:
                    self._process_tijori_table(table, 'annual_results')
            
            # Extract cash flow statement
            cash_flow_section = financials_section.select_one('.cash-flow')
            
            if cash_flow_section:
                tables = cash_flow_section.select('table')
                
                for table in tables:
                    self._process_tijori_table(table, 'cash_flow_statement')
            
            # Extract balance sheet
            balance_sheet_section = financials_section.select_one('.balance-sheet')
            
            if balance_sheet_section:
                tables = balance_sheet_section.select('table')
                
                for table in tables:
                    self._process_tijori_table(table, 'balance_sheet')
            
        except Exception as e:
            log_error(e, context={"action": "extract_tijori_financial_data", "symbol": self.symbol})
    
    def _extract_tijori_peer_comparison(self, soup: BeautifulSoup) -> None:
        """
        Extract peer comparison data from Tijori Finance
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Find peer comparison section
            peer_section = soup.select_one('#peer-comparison')
            
            if not peer_section:
                return
            
            # Extract table
            table = peer_section.select_one('table')
            
            if not table:
                return
            
            # Extract headers
            headers = [th.text.strip() for th in table.select('thead th')]
            
            # Handle empty headers
            if len(headers) <= 1:
                return
            
            # Extract rows
            rows = table.select('tbody tr')
            
            # Initialize peer comparison data
            peer_comparison = []
            
            # Extract data for each row (company)
            for row in rows:
                cells = row.select('td')
                if len(cells) < len(headers):
                    continue
                
                company_data = {}
                
                # Extract company name
                company_data['company'] = cells[0].text.strip()
                
                # Extract metrics
                for i, header in enumerate(headers[1:], 1):
                    metric_name = header.lower().replace(' ', '_')
                    value = self._parse_numeric_value(cells[i].text.strip())
                    company_data[metric_name] = value
                
                # Add to peer comparison
                peer_comparison.append(company_data)
            
            # Add to financial data
            self.financial_data['peer_comparison'] = peer_comparison
            
        except Exception as e:
            log_error(e, context={"action": "extract_tijori_peer_comparison", "symbol": self.symbol})
    
    def _extract_tijori_sector_metrics(self, soup: BeautifulSoup) -> None:
        """
        Extract sector metrics from Tijori Finance
        
        Args:
            soup (BeautifulSoup): Parsed HTML
        """
        try:
            # Find sector metrics section
            sector_section = soup.select_one('#sector-metrics')
            
            if not sector_section:
                return
            
            # Extract sector metrics
            metrics = sector_section.select('.sector-metric')
            
            sector_metrics = {}
            
            for metric in metrics:
                try:
                    # Extract name and values
                    name_elem = metric.select_one('.metric-name')
                    company_value_elem = metric.select_one('.company-value')
                    sector_value_elem = metric.select_one('.sector-value')
                    
                    if not name_elem or not company_value_elem or not sector_value_elem:
                        continue
                    
                    name = name_elem.text.strip().lower().replace(' ', '_')
                    company_value = self._parse_numeric_value(company_value_elem.text.strip())
                    sector_value = self._parse_numeric_value(sector_value_elem.text.strip())
                    
                    # Add to sector metrics
                    sector_metrics[name] = {
                        'company': company_value,
                        'sector_average': sector_value
                    }
                    
                except Exception as e:
                    if self.debug_mode:
                        self.logger.debug(f"Error parsing sector metric: {e}")
                    continue
            
            # Add to financial data
            self.financial_data['sector_comparison'] = sector_metrics
            
        except Exception as e:
            log_error(e, context={"action": "extract_tijori_sector_metrics", "symbol": self.symbol})
    
    def _process_tijori_table(self, table: BeautifulSoup, result_type: str) -> None:
        """
        Process financial table from Tijori Finance
        
        Args:
            table (BeautifulSoup): Table element
            result_type (str): Type of results ('quarterly_results', 'annual_results', etc.)
        """
        try:
            # Extract headers
            headers = [th.text.strip() for th in table.select('thead th')]
            
            # Handle empty headers
            if not headers:
                return
            
            # Extract rows
            rows = table.select('tbody tr')
            
            # Process periods from columns
            periods = headers[1:]  # First column is metric name
            
            # Check if we need to initialize or update existing results
            results = self.financial_data.get(result_type, [])
            
            # Initialize results for each period if empty
            if not results:
                for period in periods:
                    results.append({
                        'period': period
                    })
            
            # Extract data for each row
            for row in rows:
                cells = row.select('td')
                if len(cells) <= 1:
                    continue
                
                metric_name = cells[0].text.strip().lower().replace(' ', '_')
                
                # Process values for each period
                for i, cell in enumerate(cells[1:]):
                    if i >= len(results):
                        break
                    
                    value = self._parse_numeric_value(cell.text.strip())
                    results[i][metric_name] = value
            
            # Update financial data
            self.financial_data[result_type] = results
            
        except Exception as e:
            log_error(e, context={"action": "process_tijori_table", "symbol": self.symbol})
    
    def _parse_numeric_value(self, value_str: str) -> Union[float, int, str]:
        """
        Parse numeric value from string
        
        Args:
            value_str (str): String value
            
        Returns:
            Union[float, int, str]: Parsed value
        """
        try:
            # Remove commas, percentages, and other non-numeric characters
            clean_value = re.sub(r'[₹,]', '', value_str)
            
            # Handle percentages
            if '%' in clean_value:
                return float(clean_value.replace('%', '')) / 100.0
            
            # Handle crores and lakhs (Indian number format)
            if 'cr' in clean_value.lower():
                return float(clean_value.lower().replace('cr', '')) * 10000000
            
            if 'lakh' in clean_value.lower() or 'lac' in clean_value.lower():
                return float(re.sub(r'lakh|lac', '', clean_value.lower())) * 100000
            
            # Try to convert to int or float
            try:
                value = int(clean_value)
            except ValueError:
                value = float(clean_value)
            
            return value
            
        except Exception:
            # Return original string if parsing fails
            return value_str
    
    def _save_to_database(self) -> None:
        """Save the collected financial data to database using the FinancialData model"""
        try:
            # Import the FinancialData model
            from financial_data import FinancialData
            
            # Convert the scraper data to FinancialData objects
            financial_data_objects = FinancialData.from_financial_scraper(
                symbol=self.symbol,
                exchange=self.exchange,
                scraper_data=self.financial_data
            )
            
            # Save to database
            cursor = self.db.cursor()
            
            for financial_data_obj in financial_data_objects:
                # Convert to dictionary
                data_dict = financial_data_obj.to_dict()
                
                # Convert datetime objects to strings
                report_date_str = data_dict['report_date'].strftime('%Y-%m-%d %H:%M:%S')
                scraped_at_str = data_dict['scraped_at'].strftime('%Y-%m-%d %H:%M:%S')
                created_at_str = data_dict['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                updated_at_str = data_dict['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Convert data to JSON
                data_json = json.dumps(data_dict['data'])
                
                # Check if record exists
                cursor.execute(
                    """SELECT id FROM financial_data 
                       WHERE symbol = ? AND exchange = ? AND report_type = ? AND period = ?""",
                    (self.symbol, self.exchange, data_dict['report_type'], data_dict['period'])
                )
                
                existing_id = cursor.fetchone()
                
                if existing_id:
                    # Update existing record
                    cursor.execute(
                        """UPDATE financial_data 
                           SET data = ?, report_date = ?, scraped_at = ?, updated_at = datetime('now')
                           WHERE id = ?""",
                        (data_json, report_date_str, scraped_at_str, existing_id[0])
                    )
                    self.logger.info(
                        f"Updated {data_dict['report_type']} financial data for {self.symbol}:{self.exchange} - {data_dict['period']}"
                    )
                else:
                    # Insert new record
                    cursor.execute(
                        """INSERT INTO financial_data 
                           (symbol, exchange, report_type, period, report_date, data, 
                            scraped_at, created_at, updated_at) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            self.symbol, self.exchange, data_dict['report_type'], data_dict['period'], 
                            report_date_str, data_json, scraped_at_str, created_at_str, updated_at_str
                        )
                    )
                    self.logger.info(
                        f"Inserted new {data_dict['report_type']} financial data for {self.symbol}:{self.exchange} - {data_dict['period']}"
                    )
            
            # Create a unique filename based on symbol and date for the full data
            filename = sanitize_filename(f"{self.symbol}_{self.exchange}_financial_data_{datetime.now().strftime('%Y%m%d')}.json")
            
            # Also save the complete raw data for archival purposes
            full_data_json = json.dumps(self.financial_data)
            cursor.execute(
                """INSERT INTO financial_data_raw 
                   (symbol, exchange, data, timestamp, filename) 
                   VALUES (?, ?, ?, datetime('now'), ?)""",
                (self.symbol, self.exchange, full_data_json, filename)
            )
            
            # Commit changes
            self.db.commit()
            
            # Log data collection
            log_data_collection(
                data_type="financial_data",
                symbol=self.symbol,
                exchange=self.exchange,
                source="enhanced_financial_scraper"
            )
            
        except Exception as e:
            log_error(e, context={"action": "save_to_database", "symbol": self.symbol})
            # Rollback in case of error
            self.db.rollback()

# Example usage
if __name__ == "__main__":
    # Example usage
    scraper = FinancialScraper(
        symbol="RELIANCE",
        exchange="NSE",
        debug_mode=True
    )
    
    financial_data = scraper.run()
    
    if financial_data:
        print(f"Successfully collected financial data for RELIANCE:NSE")
        
        # Print some key metrics
        print("\nKey Metrics:")
        for metric, value in financial_data['key_metrics'].items():
            print(f"{metric}: {value}")
        
        # Print latest quarterly results
        if financial_data['quarterly_results']:
            latest_quarter = financial_data['quarterly_results'][0]
            print(f"\nLatest Quarter ({latest_quarter.get('quarter', 'Unknown')}):")
            for metric, value in latest_quarter.items():
                if metric != 'quarter':
                    print(f"{metric}: {value}")