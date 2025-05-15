"""
Market Hours Module - Manages market open/close timings
"""

import pytz
from datetime import datetime, time, timedelta

class MarketHours:
    """
    Manages market trading hours, including regular market hours,
    pre-market and after-hours sessions, and market holidays.
    """
    
    def __init__(self):
        """Initialize with default settings for NSE/BSE"""
        # Default timezone for Indian markets (NSE/BSE)
        self.timezone = pytz.timezone('Asia/Kolkata')
        
        # Regular market hours (9:15 AM to 3:30 PM IST)
        self.regular_open = time(9, 15, 0)  # 9:15 AM IST
        self.regular_close = time(15, 30, 0)  # 3:30 PM IST
        
        # Weekend days (0 = Monday, 6 = Sunday)
        self.weekend_days = [5, 6]  # Saturday and Sunday
        
        # Initialize holidays list
        self._load_holidays()
    
    def _load_holidays(self):
        """Load market holidays for the current year"""
        # This would typically load from a database or config file
        # Hardcoding 2024-2025 Indian market holidays for now
        current_year = datetime.now().year
        
        # Format: [(month, day), (month, day), ...]
        holiday_dates = [
            # 2024 holidays - Sample data, replace with actual holidays
            (1, 26),  # Republic Day
            (3, 25),  # Holi
            (4, 9),   # Ram Navami
            (4, 19),  # Good Friday
            (5, 1),   # Maharashtra Day
            (8, 15),  # Independence Day
            (10, 2),  # Gandhi Jayanti
            (11, 1),  # Diwali Laxmi Pujan
            (12, 25), # Christmas
            
            # 2025 holidays - Sample data, replace with actual holidays
            (1, 26),  # Republic Day
            (4, 18),  # Good Friday
            (5, 1),   # Maharashtra Day
            (8, 15),  # Independence Day
            (10, 2),  # Gandhi Jayanti
            (12, 25), # Christmas
        ]
        
        self.holidays = []
        for month, day in holiday_dates:
            # Only include holidays for current and next year
            if month >= datetime.now().month or current_year < datetime.now().year:
                holiday_date = datetime(current_year, month, day).date()
                self.holidays.append(holiday_date)
            
            # Include next year's holidays if we're in the second half of the year
            if datetime.now().month > 6:
                next_year = current_year + 1
                holiday_date = datetime(next_year, month, day).date()
                self.holidays.append(holiday_date)
    
    def is_market_open(self, check_time=None):
        """
        Check if the market is currently open
        
        Args:
            check_time (datetime, optional): Time to check (default: current time)
            
        Returns:
            bool: True if market is open, False otherwise
        """
        # Use current time if not provided
        if check_time is None:
            check_time = datetime.now(self.timezone)
        elif check_time.tzinfo is None:
            # Localize naive datetime
            check_time = self.timezone.localize(check_time)
        
        # Get current date and time components
        current_date = check_time.date()
        current_time = check_time.time()
        
        # Check if it's a weekend
        if check_time.weekday() in self.weekend_days:
            return False
        
        # Check if it's a holiday
        if current_date in self.holidays:
            return False
        
        # Check regular market hours
        if current_time >= self.regular_open and current_time < self.regular_close:
            return True
            
        return False
    
    def get_next_market_open(self, from_time=None):
        """
        Get the next market open datetime
        
        Args:
            from_time (datetime, optional): Starting time (default: current time)
            
        Returns:
            datetime: Next market open datetime
        """
        # Use current time if not provided
        if from_time is None:
            from_time = datetime.now(self.timezone)
        elif from_time.tzinfo is None:
            # Localize naive datetime
            from_time = self.timezone.localize(from_time)
        
        # If market is currently open, return next day's open time
        if self.is_market_open(from_time):
            from_time = from_time.replace(
                hour=self.regular_close.hour,
                minute=self.regular_close.minute,
                second=0,
                microsecond=0
            )
        
        # Start with the same day if before market open, or next day if after close
        check_date = from_time.date()
        check_time = from_time.time()
        
        if check_time >= self.regular_close:
            check_date += timedelta(days=1)
        
        # Find the next valid market open date
        max_attempts = 30  # Avoid infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            # Check if this date is valid (not weekend or holiday)
            check_datetime = datetime.combine(check_date, self.regular_open)
            check_datetime = self.timezone.localize(check_datetime)
            
            if self.is_valid_trading_day(check_date):
                return check_datetime
            
            # Try the next day
            check_date += timedelta(days=1)
            attempts += 1
        
        # If we got here, something is wrong
        raise ValueError("Could not find next market open time within 30 days")
    
    def is_valid_trading_day(self, check_date):
        """
        Check if a given date is a valid trading day
        
        Args:
            check_date (date): Date to check
            
        Returns:
            bool: True if it's a valid trading day, False otherwise
        """
        # If datetime was passed, extract the date
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        
        # Check if it's a weekend
        if check_date.weekday() in self.weekend_days:
            return False
        
        # Check if it's a holiday
        if check_date in self.holidays:
            return False
            
        return True
    
    def get_market_close_time(self, for_date=None):
        """
        Get market close time for a given date
        
        Args:
            for_date (date, optional): Date to check (default: today)
            
        Returns:
            datetime: Market close datetime or None if not a trading day
        """
        # Use today if not provided
        if for_date is None:
            for_date = datetime.now(self.timezone).date()
        elif isinstance(for_date, datetime):
            for_date = for_date.date()
        
        # Check if it's a valid trading day
        if not self.is_valid_trading_day(for_date):
            return None
        
        # Create a datetime with the close time
        close_datetime = datetime.combine(for_date, self.regular_close)
        close_datetime = self.timezone.localize(close_datetime)
        
        return close_datetime
    
    def get_time_until_market_close(self, from_time=None):
        """
        Get time remaining until market close
        
        Args:
            from_time (datetime, optional): Starting time (default: current time)
            
        Returns:
            timedelta: Time until market close, or zero if market is closed
        """
        # Use current time if not provided
        if from_time is None:
            from_time = datetime.now(self.timezone)
        elif from_time.tzinfo is None:
            # Localize naive datetime
            from_time = self.timezone.localize(from_time)
            
        # If market is not open, return zero time
        if not self.is_market_open(from_time):
            return timedelta(0)
        
        # Get today's close time
        close_time = self.get_market_close_time(from_time.date())
        
        # Calculate time difference
        time_remaining = close_time - from_time
        
        return time_remaining
    
    def get_trading_days_between(self, start_date, end_date):
        """
        Get a list of trading days between two dates (inclusive)
        
        Args:
            start_date (date): Start date
            end_date (date): End date
            
        Returns:
            list: List of dates that are valid trading days
        """
        # Ensure we have date objects
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
        
        # Validate dates
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")
        
        # Find all trading days between start and end dates
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if self.is_valid_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    def is_pre_market(self, check_time=None):
        """
        Check if it's the pre-market session
        
        Args:
            check_time (datetime, optional): Time to check (default: current time)
            
        Returns:
            bool: True if it's pre-market, False otherwise
        """
        # Currently no pre-market in Indian markets, but included for future use
        return False
    
    def is_after_hours(self, check_time=None):
        """
        Check if it's the after-hours session
        
        Args:
            check_time (datetime, optional): Time to check (default: current time)
            
        Returns:
            bool: True if it's after-hours, False otherwise
        """
        # Currently no after-hours in Indian markets, but included for future use
        return False
    
    def set_exchange(self, exchange):
        """
        Set market hours configuration for a specific exchange
        
        Args:
            exchange (str): Exchange code (e.g., 'NSE', 'BSE', 'NYSE')
        """
        if exchange in ['NSE', 'BSE']:
            # Indian markets
            self.timezone = pytz.timezone('Asia/Kolkata')
            self.regular_open = time(9, 15, 0)
            self.regular_close = time(15, 30, 0)
            self.weekend_days = [5, 6]  # Saturday and Sunday
            self._load_holidays()  # Reload Indian holidays
        elif exchange == 'NYSE':
            # US markets
            self.timezone = pytz.timezone('America/New_York')
            self.regular_open = time(9, 30, 0)
            self.regular_close = time(16, 0, 0)
            self.weekend_days = [5, 6]  # Saturday and Sunday
            # Would need to load US holidays here
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")