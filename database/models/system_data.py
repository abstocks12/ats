"""
System data models for the Automated Trading System.
Defines the structure for system related collections like trades and performance.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union

class TradeData:
    """Trade data model for storing executed trades"""
    
    def __init__(self, symbol: str, exchange: str, instrument_type: str, trade_type: str,
                 entry_price: float, entry_time: datetime, quantity: int,
                 exit_price: Optional[float] = None, exit_time: Optional[datetime] = None,
                 profit_loss: Optional[float] = None, profit_loss_percent: Optional[float] = None,
                 strategy: Optional[str] = None, timeframe: Optional[str] = None,
                 entry_signals: Optional[List[Dict[str, Any]]] = None,
                 exit_signals: Optional[List[Dict[str, Any]]] = None,
                 initial_stop_loss: Optional[float] = None,
                 final_stop_loss: Optional[float] = None,
                 target_price: Optional[float] = None,
                 notes: Optional[str] = None):
        """
        Initialize trade data model
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            instrument_type (str): Instrument type ('equity', 'futures', 'options')
            trade_type (str): Trade type ('buy', 'sell')
            entry_price (float): Entry price
            entry_time (datetime): Entry time
            quantity (int): Quantity
            exit_price (float, optional): Exit price
            exit_time (datetime, optional): Exit time
            profit_loss (float, optional): Profit or loss amount
            profit_loss_percent (float, optional): Profit or loss percentage
            strategy (str, optional): Strategy used
            timeframe (str, optional): Trading timeframe
            entry_signals (list, optional): Entry signals
            exit_signals (list, optional): Exit signals
            initial_stop_loss (float, optional): Initial stop loss
            final_stop_loss (float, optional): Final stop loss
            target_price (float, optional): Target price
            notes (str, optional): Additional notes
        """
        self.symbol = symbol
        self.exchange = exchange
        self.instrument_type = instrument_type
        self.trade_type = trade_type
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.quantity = quantity
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.profit_loss = profit_loss
        self.profit_loss_percent = profit_loss_percent
        self.strategy = strategy
        self.timeframe = timeframe
        self.entry_signals = entry_signals or []
        self.exit_signals = exit_signals or []
        self.initial_stop_loss = initial_stop_loss
        self.final_stop_loss = final_stop_loss
        self.target_price = target_price
        self.notes = notes
        self.status = "closed" if exit_time else "open"
        self.created_at = datetime.now()
        self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade data to dictionary"""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "instrument_type": self.instrument_type,
            "trade_type": self.trade_type,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "quantity": self.quantity,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "profit_loss": self.profit_loss,
            "profit_loss_percent": self.profit_loss_percent,
            "strategy": self.strategy,
            "timeframe": self.timeframe,
            "entry_signals": self.entry_signals,
            "exit_signals": self.exit_signals,
            "initial_stop_loss": self.initial_stop_loss,
            "final_stop_loss": self.final_stop_loss,
            "target_price": self.target_price,
            "notes": self.notes,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeData':
        """Create trade data from dictionary"""
        return cls(
            symbol=data["symbol"],
            exchange=data["exchange"],
            instrument_type=data["instrument_type"],
            trade_type=data["trade_type"],
            entry_price=data["entry_price"],
            entry_time=data["entry_time"],
            quantity=data["quantity"],
            exit_price=data.get("exit_price"),
            exit_time=data.get("exit_time"),
            profit_loss=data.get("profit_loss"),
            profit_loss_percent=data.get("profit_loss_percent"),
            strategy=data.get("strategy"),
            timeframe=data.get("timeframe"),
            entry_signals=data.get("entry_signals", []),
            exit_signals=data.get("exit_signals", []),
            initial_stop_loss=data.get("initial_stop_loss"),
            final_stop_loss=data.get("final_stop_loss"),
            target_price=data.get("target_price"),
            notes=data.get("notes")
        )
    
    def close_trade(self, exit_price: float, exit_time: Optional[datetime] = None) -> None:
        """
        Close an open trade
        
        Args:
            exit_price (float): Exit price
            exit_time (datetime, optional): Exit time, defaults to now
        """
        if self.status == "closed":
            return
        
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        
        # Calculate profit/loss
        if self.trade_type == "buy":
            self.profit_loss = (self.exit_price - self.entry_price) * self.quantity
        else:  # sell
            self.profit_loss = (self.entry_price - self.exit_price) * self.quantity
        
        # Calculate profit/loss percentage
        self.profit_loss_percent = (self.profit_loss / (self.entry_price * self.quantity)) * 100
        
        self.status = "closed"
        self.updated_at = datetime.now()
    
    def update_stop_loss(self, new_stop_loss: float) -> None:
        """
        Update the stop loss
        
        Args:
            new_stop_loss (float): New stop loss price
        """
        if self.status == "closed":
            return
        
        self.final_stop_loss = new_stop_loss
        self.updated_at = datetime.now()
    
    def __str__(self) -> str:
        """String representation"""
        status = self.status.upper()
        trade_type = self.trade_type.upper()
        if self.status == "closed" and self.profit_loss is not None:
            profit_loss_str = f"P/L: {self.profit_loss:.2f} ({self.profit_loss_percent:.2f}%)"
            return f"{self.symbol}:{self.exchange} - {trade_type} - {status} - {profit_loss_str}"
        else:
            return f"{self.symbol}:{self.exchange} - {trade_type} - {status} - Entry: {self.entry_price}"


class PerformanceData:
    """Performance data model for storing system performance metrics"""
    
    def __init__(self, date: datetime, portfolio_value: float, cash_balance: float, 
                 daily_pnl: float, daily_pnl_percent: float, 
                 total_trades: Optional[int] = None, winning_trades: Optional[int] = None,
                 metrics: Optional[Dict[str, Any]] = None, 
                 positions: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize performance data model
        
        Args:
            date (datetime): Performance date
            portfolio_value (float): Total portfolio value
            cash_balance (float): Cash balance
            daily_pnl (float): Daily profit or loss
            daily_pnl_percent (float): Daily profit or loss percentage
            total_trades (int, optional): Total trades for the day
            winning_trades (int, optional): Winning trades for the day
            metrics (dict, optional): Additional performance metrics
            positions (list, optional): End-of-day positions
        """
        self.date = date
        self.portfolio_value = portfolio_value
        self.cash_balance = cash_balance
        self.daily_pnl = daily_pnl
        self.daily_pnl_percent = daily_pnl_percent
        self.total_trades = total_trades or 0
        self.winning_trades = winning_trades or 0
        self.metrics = metrics or {}
        self.positions = positions or []
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert performance data to dictionary"""
        return {
            "date": self.date,
            "portfolio_value": self.portfolio_value,
            "cash_balance": self.cash_balance,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_percent": self.daily_pnl_percent,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            "metrics": self.metrics,
            "positions": self.positions,
            "created_at": self.created_at
       }
   
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceData':
        """Create performance data from dictionary"""
        return cls(
            date=data["date"],
            portfolio_value=data["portfolio_value"],
            cash_balance=data["cash_balance"],
            daily_pnl=data["daily_pnl"],
            daily_pnl_percent=data["daily_pnl_percent"],
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            metrics=data.get("metrics", {}),
            positions=data.get("positions", [])
        )
    
    @classmethod
    def calculate_daily_performance(cls, date: datetime, trades: List[Dict[str, Any]], 
                                    previous_performance: Optional[Dict[str, Any]] = None,
                                    current_positions: Optional[List[Dict[str, Any]]] = None,
                                    starting_balance: Optional[float] = None) -> 'PerformanceData':
        """
        Calculate daily performance from trades
        
        Args:
            date (datetime): Performance date
            trades (list): List of trades for the day
            previous_performance (dict, optional): Previous day's performance
            current_positions (list, optional): Current open positions
            starting_balance (float, optional): Starting balance if first day
            
        Returns:
            PerformanceData: Performance data for the day
        """
        # Initialize values
        if previous_performance:
            prev_portfolio_value = previous_performance.get("portfolio_value", 0)
            prev_cash_balance = previous_performance.get("cash_balance", 0)
        else:
            prev_portfolio_value = starting_balance or 1000000  # Default 1M if not specified
            prev_cash_balance = prev_portfolio_value
        
        # Calculate daily PnL from trades
        daily_pnl = sum(trade.get("profit_loss", 0) for trade in trades if trade.get("exit_time") and trade.get("exit_time").date() == date.date())
        
        # Calculate current positions value
        positions_value = 0
        positions_list = []
        
        if current_positions:
            for position in current_positions:
                symbol = position.get("symbol", "")
                quantity = position.get("quantity", 0)
                current_price = position.get("current_price", 0)
                entry_price = position.get("entry_price", 0)
                
                position_value = quantity * current_price
                positions_value += position_value
                
                # Calculate position PnL
                if entry_price > 0:
                    position_pnl = (current_price - entry_price) * quantity
                    position_pnl_percent = (position_pnl / (entry_price * quantity)) * 100
                else:
                    position_pnl = 0
                    position_pnl_percent = 0
                
                positions_list.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "position_value": position_value,
                    "position_pnl": position_pnl,
                    "position_pnl_percent": position_pnl_percent
                })
        
        # Calculate current portfolio value
        portfolio_value = prev_cash_balance + daily_pnl + positions_value
        
        # Calculate daily PnL percentage
        daily_pnl_percent = (daily_pnl / prev_portfolio_value * 100) if prev_portfolio_value > 0 else 0
        
        # Calculate trade metrics
        total_trades = len([t for t in trades if t.get("exit_time") and t.get("exit_time").date() == date.date()])
        winning_trades = len([t for t in trades if t.get("exit_time") and t.get("exit_time").date() == date.date() and t.get("profit_loss", 0) > 0])
        
        # Calculate additional metrics
        metrics = {
            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            "average_win": sum(t.get("profit_loss", 0) for t in trades if t.get("exit_time") and t.get("exit_time").date() == date.date() and t.get("profit_loss", 0) > 0) / winning_trades if winning_trades > 0 else 0,
            "average_loss": sum(t.get("profit_loss", 0) for t in trades if t.get("exit_time") and t.get("exit_time").date() == date.date() and t.get("profit_loss", 0) < 0) / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0,
            "largest_win": max([t.get("profit_loss", 0) for t in trades if t.get("exit_time") and t.get("exit_time").date() == date.date() and t.get("profit_loss", 0) > 0], default=0),
            "largest_loss": min([t.get("profit_loss", 0) for t in trades if t.get("exit_time") and t.get("exit_time").date() == date.date() and t.get("profit_loss", 0) < 0], default=0)
        }
        
        return cls(
            date=date,
            portfolio_value=portfolio_value,
            cash_balance=prev_cash_balance + daily_pnl,
            daily_pnl=daily_pnl,
            daily_pnl_percent=daily_pnl_percent,
            total_trades=total_trades,
            winning_trades=winning_trades,
            metrics=metrics,
            positions=positions_list
        )
    
    def __str__(self) -> str:
       """String representation"""
       win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
       return f"Performance {self.date.strftime('%Y-%m-%d')} - PnL: {self.daily_pnl:.2f} ({self.daily_pnl_percent:.2f}%) - Win Rate: {win_rate:.1f}%"