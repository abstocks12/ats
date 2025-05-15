"""
Financial data models for the Automated Trading System.
Defines the structure for financial data collections.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union

class FinancialData:
    """Financial data model for storing company financial information"""
    
    def __init__(self, symbol: str, exchange: str, report_type: str, period: str,
                 report_date: datetime, data: Dict[str, Any]):
        """
        Initialize financial data model
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            report_type (str): Report type (e.g. 'quarterly', 'annual')
            period (str): Period (e.g. 'Q1-2023', 'FY-2023')
            report_date (datetime): Date of the report
            data (dict): Financial data
        """
        self.symbol = symbol
        self.exchange = exchange
        self.report_type = report_type
        self.period = period
        self.report_date = report_date
        self.data = data
        self.scraped_at = datetime.now()
        self.created_at = datetime.now()
        self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert financial data to dictionary"""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "report_type": self.report_type,
            "period": self.period,
            "report_date": self.report_date,
            "data": self.data,
            "scraped_at": self.scraped_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FinancialData':
        """Create financial data from dictionary"""
        return cls(
            symbol=data["symbol"],
            exchange=data["exchange"],
            report_type=data["report_type"],
            period=data["period"],
            report_date=data["report_date"],
            data=data["data"]
        )
    
    @classmethod
    def from_financial_scraper(cls, symbol: str, exchange: str, scraper_data: Dict[str, Any]) -> List['FinancialData']:
        """Create financial data from financial scraper output"""
        financial_data_list = []
        
        # Process quarterly results
        quarterly_results = scraper_data.get('quarterly_results', [])
        for result in quarterly_results:
            if 'quarter' in result:
                period = result['quarter']
                
                # Extract date from period (e.g. "Mar 2023" -> 2023-03-31)
                report_date = None
                try:
                    if 'Mar' in period:
                        month = 3
                    elif 'Jun' in period:
                        month = 6
                    elif 'Sep' in period:
                        month = 9
                    elif 'Dec' in period:
                        month = 12
                    else:
                        month = 1
                    
                    year_str = ''.join(filter(str.isdigit, period))
                    year = int(year_str) if year_str else datetime.now().year
                    
                    # Last day of the month
                    if month in [3, 5, 8, 10]:
                        day = 31
                    elif month == 2:
                        # Check for leap year
                        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                            day = 29
                        else:
                            day = 28
                    else:
                        day = 30
                    
                    report_date = datetime(year, month, day)
                except (ValueError, TypeError):
                    report_date = datetime.now()
                
                financial_data = cls(
                    symbol=symbol,
                    exchange=exchange,
                    report_type='quarterly',
                    period=period,
                    report_date=report_date,
                    data=result
                )
                
                financial_data_list.append(financial_data)
        
        # Process annual results
        annual_results = scraper_data.get('annual_results', [])
        for result in annual_results:
            if 'year' in result or 'mar' in result:
                period = result.get('year', result.get('mar', 'FY'))
                
                # Extract date from period
                report_date = None
                try:
                    year_str = ''.join(filter(str.isdigit, period))
                    year = int(year_str) if year_str else datetime.now().year
                    
                    # Fiscal year end - March 31
                    report_date = datetime(year, 3, 31)
                except (ValueError, TypeError):
                    report_date = datetime.now()
                
                financial_data = cls(
                    symbol=symbol,
                    exchange=exchange,
                    report_type='annual',
                    period=period,
                    report_date=report_date,
                    data=result
                )
                
                financial_data_list.append(financial_data)
        
        # Process key metrics
        if 'key_metrics' in scraper_data:
            metrics = scraper_data['key_metrics']
            
            financial_data = cls(
                symbol=symbol,
                exchange=exchange,
                report_type='key_metrics',
                period='current',
                report_date=datetime.now(),
                data=metrics
            )
            
            financial_data_list.append(financial_data)
        
        # Process ratios
        if 'financial_ratios' in scraper_data:
            ratios = scraper_data['financial_ratios']
            
            financial_data = cls(
                symbol=symbol,
                exchange=exchange,
                report_type='ratios',
                period='current',
                report_date=datetime.now(),
                data=ratios
            )
            
            financial_data_list.append(financial_data)
        
        return financial_data_list
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.symbol}:{self.exchange} - {self.report_type} - {self.period}"