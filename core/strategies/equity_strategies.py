"""
Equity Trading Strategies Module

This module implements equity-focused trading strategies including:
- Growth stock strategies
- Value stock strategies
- Quality factor strategies
- Momentum stock strategies
- Sector rotation strategies
- Blend strategies combining multiple factors
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import math

class EquityStrategies:
    """
    Implements equity-focused trading strategies.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the strategy with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Strategy parameters (configurable)
        self.params = {
            # General parameters
            "min_data_points": 30,  # Minimum data points needed
            
            # Growth strategy parameters
            "growth_eps_threshold": 15,  # Minimum EPS growth (%)
            "growth_revenue_threshold": 10,  # Minimum revenue growth (%)
            "growth_pe_max": 30,  # Maximum P/E ratio
            
            # Value strategy parameters
            "value_pe_max": 15,  # Maximum P/E ratio for value stocks
            "value_pb_max": 1.5,  # Maximum P/B ratio
            "value_dividend_min": 2,  # Minimum dividend yield (%)
            
            # Quality strategy parameters
            "quality_roe_min": 15,  # Minimum ROE (%)
            "quality_debt_equity_max": 1.0,  # Maximum debt to equity ratio
            "quality_margin_min": 10,  # Minimum profit margin (%)
            
            # Momentum strategy parameters
            "momentum_lookback": 60,  # Days for momentum calculation
            "momentum_threshold": 10,  # Minimum price momentum (%)
            
            # Blend strategy parameters
            "blend_weights": {  # Weights for blend strategy
                "growth": 0.25,
                "value": 0.25,
                "quality": 0.25,
                "momentum": 0.25
            },
            
            # Sector rotation parameters
            "sector_momentum_lookback": 90,  # Days for sector momentum
            "sector_count": 3,  # Number of top sectors to include
            
            # Risk management
            "position_size": 0.05,  # Default position size (% of portfolio)
            "stop_loss_percent": 0.07,  # Default stop loss (7%)
            "max_sector_allocation": 0.25,  # Maximum allocation to a single sector
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for this module."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def analyze_growth_stock(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze a stock for growth attributes.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with growth analysis
        """
        try:
            # Get financial data
            financial_data = self._get_financial_data(symbol, exchange)
            
            if not financial_data:
                return {"status": "no_financial_data"}
            
            # Get latest price data
            latest_price_data = self._get_latest_price(symbol, exchange)
            current_price = latest_price_data.get("price") if latest_price_data else None
            
            if not current_price:
                return {"status": "no_price_data"}
            
            # Extract growth metrics
            eps_growth = financial_data.get("eps_growth_3yr")
            revenue_growth = financial_data.get("revenue_growth_3yr")
            pe_ratio = financial_data.get("pe_ratio")
            projected_eps_growth = financial_data.get("projected_eps_growth")
            peg_ratio = financial_data.get("peg_ratio")
            
            # Calculate growth score
            growth_score = 0
            max_score = 0
            
            # EPS growth
            if eps_growth is not None:
                max_score += 1
                if eps_growth >= self.params["growth_eps_threshold"]:
                    growth_score += 1
            
            # Revenue growth
            if revenue_growth is not None:
                max_score += 1
                if revenue_growth >= self.params["growth_revenue_threshold"]:
                    growth_score += 1
            
            # P/E ratio (should be reasonable for growth)
            if pe_ratio is not None:
                max_score += 1
                if pe_ratio < self.params["growth_pe_max"]:
                    growth_score += 1
            
            # PEG ratio (lower is better)
            if peg_ratio is not None:
                max_score += 1
                if peg_ratio < 1.5:
                    growth_score += 1
            
            # Projected EPS growth
            if projected_eps_growth is not None:
                max_score += 1
                if projected_eps_growth >= self.params["growth_eps_threshold"]:
                    growth_score += 1
            
            # Calculate normalized score (0-100)
            normalized_score = (growth_score / max_score * 100) if max_score > 0 else 0
            
            # Determine if it qualifies as a growth stock
            is_growth_stock = normalized_score >= 70
            
            # Generate trading signal
            signal = "neutral"
            if is_growth_stock:
                # Check technical confirmation
                technicals = self._check_technical_confirmation(symbol, exchange)
                
                if technicals:
                    signal = technicals.get("signal", "neutral")
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "current_price": current_price,
                "eps_growth": eps_growth,
                "revenue_growth": revenue_growth,
                "pe_ratio": pe_ratio,
                "projected_eps_growth": projected_eps_growth,
                "peg_ratio": peg_ratio,
                "growth_score": growth_score,
                "max_score": max_score,
                "normalized_score": normalized_score,
                "is_growth_stock": is_growth_stock,
                "signal": signal,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing growth stock {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def analyze_value_stock(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze a stock for value attributes.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with value analysis
        """
        try:
            # Get financial data
            financial_data = self._get_financial_data(symbol, exchange)
            
            if not financial_data:
                return {"status": "no_financial_data"}
            
            # Get latest price data
            latest_price_data = self._get_latest_price(symbol, exchange)
            current_price = latest_price_data.get("price") if latest_price_data else None
            
            if not current_price:
                return {"status": "no_price_data"}
            
            # Extract value metrics
            pe_ratio = financial_data.get("pe_ratio")
            pb_ratio = financial_data.get("pb_ratio")
            dividend_yield = financial_data.get("dividend_yield")
            ev_ebitda = financial_data.get("ev_ebitda")
            fcf_yield = financial_data.get("fcf_yield")
            
            # Calculate value score
            value_score = 0
            max_score = 0
            
            # P/E ratio
            if pe_ratio is not None:
                max_score += 1
                if pe_ratio <= self.params["value_pe_max"]:
                    value_score += 1
            
            # P/B ratio
            if pb_ratio is not None:
                max_score += 1
                if pb_ratio <= self.params["value_pb_max"]:
                    value_score += 1
            
            # Dividend yield
            if dividend_yield is not None:
                max_score += 1
                if dividend_yield >= self.params["value_dividend_min"]:
                    value_score += 1
            
            # EV/EBITDA (lower is better)
            if ev_ebitda is not None:
                max_score += 1
                if ev_ebitda < 10:
                    value_score += 1
            
            # FCF yield (higher is better)
            if fcf_yield is not None:
                max_score += 1
                if fcf_yield > 5:
                    value_score += 1
            
            # Calculate normalized score (0-100)
            normalized_score = (value_score / max_score * 100) if max_score > 0 else 0
            
            # Determine if it qualifies as a value stock
            is_value_stock = normalized_score >= 70
            
            # Generate trading signal
            signal = "neutral"
            if is_value_stock:
                # Check technical confirmation
                technicals = self._check_technical_confirmation(symbol, exchange)
                
                if technicals:
                    signal = technicals.get("signal", "neutral")
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "current_price": current_price,
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
                "dividend_yield": dividend_yield,
                "ev_ebitda": ev_ebitda,
                "fcf_yield": fcf_yield,
                "value_score": value_score,
                "max_score": max_score,
                "normalized_score": normalized_score,
                "is_value_stock": is_value_stock,
                "signal": signal,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing value stock {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def analyze_quality_stock(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze a stock for quality attributes.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with quality analysis
        """
        try:
            # Get financial data
            financial_data = self._get_financial_data(symbol, exchange)
            
            if not financial_data:
                return {"status": "no_financial_data"}
            
            # Get latest price data
            latest_price_data = self._get_latest_price(symbol, exchange)
            current_price = latest_price_data.get("price") if latest_price_data else None
            
            if not current_price:
                return {"status": "no_price_data"}
            
            # Extract quality metrics
            roe = financial_data.get("roe")
            debt_equity = financial_data.get("debt_equity")
            profit_margin = financial_data.get("net_profit_margin")
            interest_coverage = financial_data.get("interest_coverage")
            current_ratio = financial_data.get("current_ratio")
            
            # Calculate quality score
            quality_score = 0
            max_score = 0
            
            # Return on Equity
            if roe is not None:
                max_score += 1
                if roe >= self.params["quality_roe_min"]:
                    quality_score += 1
            
            # Debt to Equity
            if debt_equity is not None:
                max_score += 1
                if debt_equity <= self.params["quality_debt_equity_max"]:
                    quality_score += 1
            
            # Profit Margin
            if profit_margin is not None:
                max_score += 1
                if profit_margin >= self.params["quality_margin_min"]:
                    quality_score += 1
            
            # Interest Coverage
            if interest_coverage is not None:
                max_score += 1
                if interest_coverage > 5:
                    quality_score += 1
            
            # Current Ratio
            if current_ratio is not None:
                max_score += 1
                if current_ratio > 1.5:
                    quality_score += 1
            
            # Calculate normalized score (0-100)
            normalized_score = (quality_score / max_score * 100) if max_score > 0 else 0
            
            # Determine if it qualifies as a quality stock
            is_quality_stock = normalized_score >= 70
            
            # Generate trading signal
            signal = "neutral"
            if is_quality_stock:
                # Check technical confirmation
                technicals = self._check_technical_confirmation(symbol, exchange)
                
                if technicals:
                    signal = technicals.get("signal", "neutral")
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "current_price": current_price,
                "roe": roe,
                "debt_equity": debt_equity,
                "profit_margin": profit_margin,
                "interest_coverage": interest_coverage,
                "current_ratio": current_ratio,
                "quality_score": quality_score,
                "max_score": max_score,
                "normalized_score": normalized_score,
                "is_quality_stock": is_quality_stock,
                "signal": signal,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing quality stock {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def analyze_momentum_stock(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze a stock for price momentum.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with momentum analysis
        """
        try:
            # Get market data
            market_data = self._get_market_data(
                symbol, exchange, days=self.params["momentum_lookback"]
            )
            
            if not market_data or len(market_data) < self.params["min_data_points"]:
                return {"status": "insufficient_data"}
            
            # Convert to DataFrame
            df = pd.DataFrame(market_data).sort_values("timestamp")
            
            # Get latest price
            current_price = df['close'].iloc[-1]
            
            # Calculate momentum metrics
            # Calculate momentum metrics
            price_1m_ago = df['close'].iloc[-21] if len(df) >= 21 else df['close'].iloc[0]
            price_3m_ago = df['close'].iloc[-63] if len(df) >= 63 else df['close'].iloc[0]
            price_6m_ago = df['close'].iloc[-126] if len(df) >= 126 else df['close'].iloc[0]
            
            # Calculate momentum percentages
            momentum_1m = ((current_price / price_1m_ago) - 1) * 100
            momentum_3m = ((current_price / price_3m_ago) - 1) * 100
            momentum_6m = ((current_price / price_6m_ago) - 1) * 100
            
            # Calculate relative strength
            # Compare to index performance
            index_momentum = self._get_index_momentum(exchange)
            
            rel_strength_1m = momentum_1m - index_momentum.get("momentum_1m", 0)
            rel_strength_3m = momentum_3m - index_momentum.get("momentum_3m", 0)
            rel_strength_6m = momentum_6m - index_momentum.get("momentum_6m", 0)
            
            # Calculate momentum score
            momentum_score = 0
            max_score = 0
            
            # 1-month momentum
            max_score += 1
            if momentum_1m >= self.params["momentum_threshold"]:
                momentum_score += 1
            
            # 3-month momentum
            max_score += 1
            if momentum_3m >= self.params["momentum_threshold"]:
                momentum_score += 1
            
            # 6-month momentum
            max_score += 1
            if momentum_6m >= self.params["momentum_threshold"]:
                momentum_score += 1
            
            # 1-month relative strength
            max_score += 1
            if rel_strength_1m >= 0:
                momentum_score += 1
            
            # 3-month relative strength
            max_score += 1
            if rel_strength_3m >= 0:
                momentum_score += 1
            
            # Calculate normalized score (0-100)
            normalized_score = (momentum_score / max_score * 100) if max_score > 0 else 0
            
            # Determine if it qualifies as a momentum stock
            is_momentum_stock = normalized_score >= 60  # Lower threshold for momentum
            
            # Generate trading signal
            signal = "neutral"
            
            if is_momentum_stock:
                # For momentum stocks, we don't need additional technical confirmation
                # The trend itself is the signal
                if momentum_1m > 0 and momentum_3m > 0:
                    signal = "bullish"
                elif momentum_1m < 0 and momentum_3m < 0:
                    signal = "bearish"
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "current_price": current_price,
                "momentum_1m": momentum_1m,
                "momentum_3m": momentum_3m,
                "momentum_6m": momentum_6m,
                "rel_strength_1m": rel_strength_1m,
                "rel_strength_3m": rel_strength_3m,
                "rel_strength_6m": rel_strength_6m,
                "momentum_score": momentum_score,
                "max_score": max_score,
                "normalized_score": normalized_score,
                "is_momentum_stock": is_momentum_stock,
                "signal": signal,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum stock {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def analyze_blend_strategy(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze a stock using a blend of different strategies.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with blend strategy analysis
        """
        try:
            # Get analyses from different strategies
            growth_analysis = self.analyze_growth_stock(symbol, exchange)
            value_analysis = self.analyze_value_stock(symbol, exchange)
            quality_analysis = self.analyze_quality_stock(symbol, exchange)
            momentum_analysis = self.analyze_momentum_stock(symbol, exchange)
            
            # Check if all analyses were successful
            all_successful = (
                "status" not in growth_analysis or growth_analysis["status"] != "error"
            ) and (
                "status" not in value_analysis or value_analysis["status"] != "error"
            ) and (
                "status" not in quality_analysis or quality_analysis["status"] != "error"
            ) and (
                "status" not in momentum_analysis or momentum_analysis["status"] != "error"
            )
            
            if not all_successful:
                return {"status": "error", "message": "One or more analyses failed"}
            
            # Get latest price data (use from growth analysis if available)
            current_price = growth_analysis.get("current_price")
            
            # Calculate weighted score
            weights = self.params["blend_weights"]
            
            weighted_score = 0
            total_weight = 0
            
            if "normalized_score" in growth_analysis:
                weighted_score += growth_analysis["normalized_score"] * weights["growth"]
                total_weight += weights["growth"]
            
            if "normalized_score" in value_analysis:
                weighted_score += value_analysis["normalized_score"] * weights["value"]
                total_weight += weights["value"]
            
            if "normalized_score" in quality_analysis:
                weighted_score += quality_analysis["normalized_score"] * weights["quality"]
                total_weight += weights["quality"]
            
            if "normalized_score" in momentum_analysis:
                weighted_score += momentum_analysis["normalized_score"] * weights["momentum"]
                total_weight += weights["momentum"]
            
            # Calculate overall score
            overall_score = weighted_score / total_weight if total_weight > 0 else 0
            
            # Determine if it qualifies as a blend strategy stock
            is_blend_stock = overall_score >= 70
            
            # Generate trading signal
            signal = "neutral"
            
            if is_blend_stock:
                # Check individual signals
                growth_signal = growth_analysis.get("signal")
                value_signal = value_analysis.get("signal")
                quality_signal = quality_analysis.get("signal")
                momentum_signal = momentum_analysis.get("signal")
                
                # Count bullish and bearish signals
                bullish_signals = sum(1 for s in [growth_signal, value_signal, quality_signal, momentum_signal] if s == "bullish")
                bearish_signals = sum(1 for s in [growth_signal, value_signal, quality_signal, momentum_signal] if s == "bearish")
                
                if bullish_signals > bearish_signals:
                    signal = "bullish"
                elif bearish_signals > bullish_signals:
                    signal = "bearish"
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "current_price": current_price,
                "growth_score": growth_analysis.get("normalized_score"),
                "value_score": value_analysis.get("normalized_score"),
                "quality_score": quality_analysis.get("normalized_score"),
                "momentum_score": momentum_analysis.get("normalized_score"),
                "overall_score": overall_score,
                "is_blend_stock": is_blend_stock,
                "growth_signal": growth_analysis.get("signal"),
                "value_signal": value_analysis.get("signal"),
                "quality_signal": quality_analysis.get("signal"),
                "momentum_signal": momentum_analysis.get("signal"),
                "signal": signal,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing blend strategy for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def analyze_sector_rotation(self, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze sectors for sector rotation strategy.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            Dictionary with sector rotation analysis
        """
        try:
            # Get sector data
            sectors = self._get_sectors(exchange)
            
            if not sectors:
                return {"status": "no_sector_data"}
            
            # Calculate momentum for each sector
            sector_momentum = []
            
            for sector in sectors:
                sector_name = sector.get("name")
                
                # Get sector price data
                sector_data = self._get_sector_price_data(sector_name, exchange, self.params["sector_momentum_lookback"])
                
                if not sector_data or len(sector_data) < 21:  # Need at least 1 month of data
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(sector_data).sort_values("timestamp")
                
                # Calculate momentum
                current_price = df['close'].iloc[-1]
                price_1m_ago = df['close'].iloc[-21] if len(df) >= 21 else df['close'].iloc[0]
                price_3m_ago = df['close'].iloc[-63] if len(df) >= 63 else df['close'].iloc[0]
                
                momentum_1m = ((current_price / price_1m_ago) - 1) * 100
                momentum_3m = ((current_price / price_3m_ago) - 1) * 100
                
                # Calculate relative strength against broader market
                index_momentum = self._get_index_momentum(exchange)
                
                rel_strength_1m = momentum_1m - index_momentum.get("momentum_1m", 0)
                rel_strength_3m = momentum_3m - index_momentum.get("momentum_3m", 0)
                
                # Calculate combined momentum score
                momentum_score = (rel_strength_1m + rel_strength_3m) / 2
                
                sector_momentum.append({
                    "sector": sector_name,
                    "momentum_1m": momentum_1m,
                    "momentum_3m": momentum_3m,
                    "rel_strength_1m": rel_strength_1m,
                    "rel_strength_3m": rel_strength_3m,
                    "momentum_score": momentum_score
                })
            
            # Sort sectors by momentum score
            sector_momentum.sort(key=lambda x: x["momentum_score"], reverse=True)
            
            # Select top sectors
            top_sectors = sector_momentum[:self.params["sector_count"]]
            
            # Get stocks in top sectors
            top_sector_stocks = {}
            best_stocks = []
            
            for sector_data in top_sectors:
                sector = sector_data["sector"]
                
                # Get stocks in this sector
                stocks = self._get_sector_stocks(sector, exchange)
                
                # Analyze each stock for momentum
                sector_stocks = []
                
                for stock in stocks:
                    symbol = stock.get("symbol")
                    
                    # Analyze with momentum strategy
                    analysis = self.analyze_momentum_stock(symbol, exchange)
                    
                    if "status" in analysis and analysis["status"] == "error":
                        continue
                    
                    if analysis.get("is_momentum_stock") and analysis.get("signal") == "bullish":
                        stock_data = {
                            "symbol": symbol,
                            "momentum_score": analysis.get("normalized_score", 0),
                            "current_price": analysis.get("current_price"),
                            "momentum_1m": analysis.get("momentum_1m"),
                            "momentum_3m": analysis.get("momentum_3m")
                        }
                        
                        sector_stocks.append(stock_data)
                        best_stocks.append(stock_data)
                
                # Sort stocks by momentum score
                sector_stocks.sort(key=lambda x: x["momentum_score"], reverse=True)
                
                # Take top 5 stocks from each sector
                top_sector_stocks[sector] = sector_stocks[:5]
            
            # Sort all stocks by momentum score
            best_stocks.sort(key=lambda x: x["momentum_score"], reverse=True)
            
            # Take top 10 overall
            best_stocks = best_stocks[:10]
            
            return {
                "exchange": exchange,
                "top_sectors": top_sectors,
                "top_sector_stocks": top_sector_stocks,
                "best_stocks": best_stocks,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sector rotation: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_equity_signals(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Generate comprehensive equity trading signals.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with trading signals
        """
        try:
            # Get latest price data
            latest_price_data = self._get_latest_price(symbol, exchange)
            current_price = latest_price_data.get("price") if latest_price_data else None
            
            if not current_price:
                return {"status": "error", "message": "Could not determine current price"}
            
            # Analyze stock with all strategies
            growth_analysis = self.analyze_growth_stock(symbol, exchange)
            value_analysis = self.analyze_value_stock(symbol, exchange)
            quality_analysis = self.analyze_quality_stock(symbol, exchange)
            momentum_analysis = self.analyze_momentum_stock(symbol, exchange)
            blend_analysis = self.analyze_blend_strategy(symbol, exchange)
            
            # Determine the best-fit strategy
            strategy_scores = {
                "growth": growth_analysis.get("normalized_score", 0) if "status" not in growth_analysis else 0,
                "value": value_analysis.get("normalized_score", 0) if "status" not in value_analysis else 0,
                "quality": quality_analysis.get("normalized_score", 0) if "status" not in quality_analysis else 0,
                "momentum": momentum_analysis.get("normalized_score", 0) if "status" not in momentum_analysis else 0,
                "blend": blend_analysis.get("overall_score", 0) if "status" not in blend_analysis else 0
            }
            
            # Find best strategy
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
            strategy_type = best_strategy[0]
            strategy_score = best_strategy[1]
            
            # Get signal from best strategy
            if strategy_type == "growth":
                signal = growth_analysis.get("signal")
                analysis = growth_analysis
            elif strategy_type == "value":
                signal = value_analysis.get("signal")
                analysis = value_analysis
            elif strategy_type == "quality":
                signal = quality_analysis.get("signal")
                analysis = quality_analysis
            elif strategy_type == "momentum":
                signal = momentum_analysis.get("signal")
                analysis = momentum_analysis
            else:  # blend
                signal = blend_analysis.get("signal")
                analysis = blend_analysis
            
            # Default to neutral if no signal
            signal = signal if signal else "neutral"
            
            # Calculate risk management parameters
            stop_loss = None
            target_price = None
            
            if signal == "bullish":
                # Set stop loss based on stock volatility
                volatility = self._calculate_stock_volatility(symbol, exchange)
                
                if volatility:
                    # Stop loss based on ATR
                    stop_loss = current_price * (1 - 2 * volatility / 100)
                else:
                    # Default stop loss
                    stop_loss = current_price * (1 - self.params["stop_loss_percent"])
                
                # Set target based on historical performance
                if strategy_type == "growth":
                    # Growth stocks typically have higher upside
                    target_price = current_price * 1.20  # 20% upside
                elif strategy_type == "value":
                    # Value stocks typically have moderate upside
                    target_price = current_price * 1.15  # 15% upside
                elif strategy_type == "momentum":
                    # Momentum stocks can have significant upside
                    target_price = current_price * 1.25  # 25% upside
                else:
                    # Default target
                    target_price = current_price * 1.15  # 15% upside
            
            elif signal == "bearish":
                # Short position stop loss
                volatility = self._calculate_stock_volatility(symbol, exchange)
                
                if volatility:
                    # Stop loss based on ATR
                    stop_loss = current_price * (1 + 2 * volatility / 100)
                else:
                    # Default stop loss
                    stop_loss = current_price * (1 + self.params["stop_loss_percent"])
                
                # Short position target
                if strategy_type == "momentum":
                    # Momentum stocks can have significant downside when trend reverses
                    target_price = current_price * 0.80  # 20% downside
                else:
                    # Default target
                    target_price = current_price * 0.85  # 15% downside
            
            # Calculate risk-reward ratio
            risk_reward_ratio = None
            if stop_loss and target_price:
                risk = abs(current_price - stop_loss)
                reward = abs(target_price - current_price)
                
                if risk > 0:
                    risk_reward_ratio = reward / risk
            
            # Get stock sector for portfolio allocation
            stock_info = self._get_stock_info(symbol, exchange)
            sector = stock_info.get("sector") if stock_info else None
            
            # Generate final signal
            trading_signal = {
                "strategy": f"equity_{strategy_type}",
                "symbol": symbol,
                "exchange": exchange,
                "sector": sector,
                "timestamp": datetime.now(),
                "current_price": current_price,
                "signal": signal,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "risk_reward_ratio": risk_reward_ratio,
                "strategy_score": strategy_score,
                "strategy_type": strategy_type,
                "analysis": analysis,
                "all_scores": strategy_scores
            }
            
            return trading_signal
            
        except Exception as e:
            self.logger.error(f"Error generating equity signals for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def scan_for_equity_opportunities(self, symbols: List[str], exchange: str = "NSE") -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan for equity trading opportunities across multiple symbols.
        
        Args:
            symbols: List of stock symbols to analyze
            exchange: Stock exchange
            
        Returns:
            Dictionary with opportunities for each strategy type
        """
        results = {
            "growth": [],
            "value": [],
            "quality": [],
            "momentum": [],
            "blend": []
        }
        
        for symbol in symbols:
            try:
                # Get equity signals
                signals = self.generate_equity_signals(symbol, exchange)
                
                if not signals or "status" in signals and signals["status"] == "error":
                    continue
                
                # Check if there's a valid trading signal
                if signals.get("signal") == "neutral" or not signals.get("risk_reward_ratio", 0) >= 1.5:
                    continue
                
                # Add to appropriate category
                strategy_type = signals.get("strategy_type")
                
                if strategy_type in results:
                    results[strategy_type].append({
                        "symbol": symbol,
                        "exchange": exchange,
                        "sector": signals.get("sector"),
                        "signal": signals.get("signal"),
                        "current_price": signals.get("current_price"),
                        "stop_loss": signals.get("stop_loss"),
                        "target_price": signals.get("target_price"),
                        "risk_reward_ratio": signals.get("risk_reward_ratio"),
                        "strategy_score": signals.get("strategy_score")
                    })
            
            except Exception as e:
                self.logger.error(f"Error scanning equity opportunities for {symbol}: {e}")
        
        # Sort each category by strategy score
        for strategy_type in results:
            results[strategy_type].sort(key=lambda x: x.get("strategy_score", 0), reverse=True)
        
        return results
    
    def _get_financial_data(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get financial data from database.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with financial data
        """
        try:
            # Query the financial data collection
            financial_data = self.db.financial_data_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            }, sort=[("timestamp", -1)])
            
            return financial_data or {}
            
        except Exception as e:
            self.logger.error(f"Error getting financial data for {symbol}: {e}")
            return {}
    
    def _get_market_data(self, symbol: str, exchange: str, days: int = 100) -> List[Dict[str, Any]]:
        """
        Get market data from database.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            days: Number of days to retrieve
            
        Returns:
            List of market data documents
        """
        try:
            # Calculate the start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Create query
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "day",
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            # Get data from database
            cursor = self.db.market_data_collection.find(query).sort("timestamp", 1)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return []
    
    def _get_latest_price(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get latest price data for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with latest price data
        """
        try:
            # Query the market data collection
            latest_data = self.db.market_data_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "day"
            }, sort=[("timestamp", -1)])
            
            if latest_data:
                return {
                    "price": latest_data.get("close"),
                    "timestamp": latest_data.get("timestamp"),
                    "open": latest_data.get("open"),
                    "high": latest_data.get("high"),
                    "low": latest_data.get("low"),
                    "volume": latest_data.get("volume")
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return {}
    
    def _check_technical_confirmation(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Check technical indicators for confirmation.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with technical confirmation
        """
        try:
            # Get market data
            market_data = self._get_market_data(symbol, exchange, days=100)
            
            if not market_data or len(market_data) < 30:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(market_data).sort_values("timestamp")
            
            # Calculate moving averages
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            df['ma_200'] = df['close'].rolling(window=200).mean() if len(df) >= 200 else None
            
            # Get current values
            current_price = df['close'].iloc[-1]
            ma_20 = df['ma_20'].iloc[-1]
            ma_50 = df['ma_50'].iloc[-1]
            ma_200 = df['ma_200'].iloc[-1] if len(df) >= 200 else None
            
            # Check price relative to moving averages
            above_ma20 = current_price > ma_20
            above_ma50 = current_price > ma_50
            above_ma200 = current_price > ma_200 if ma_200 is not None else None
            
            # Determine trend
            trend = "neutral"
            
            if above_ma20 and above_ma50 and (above_ma200 or above_ma200 is None):
                trend = "bullish"
            elif not above_ma20 and not above_ma50 and (not above_ma200 or above_ma200 is None):
                trend = "bearish"
            
            # Check for recent breakouts
            breakout = "none"
            
            if len(df) > 2:
                prev_ma20 = df['ma_20'].iloc[-2]
                prev_ma50 = df['ma_50'].iloc[-2]
                prev_price = df['close'].iloc[-2]
                
                if current_price > ma_20 and prev_price <= prev_ma20:
                    breakout = "bullish_ma20"
                elif current_price > ma_50 and prev_price <= prev_ma50:
                    breakout = "bullish_ma50"
                elif current_price < ma_20 and prev_price >= prev_ma20:
                    breakout = "bearish_ma20"
                elif current_price < ma_50 and prev_price >= prev_ma50:
                    breakout = "bearish_ma50"
            
            # Check volume confirmation
            volume_confirmation = False
            
            if 'volume' in df.columns and len(df) > 20:
                recent_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].iloc[-20:-1].mean()
                
                volume_confirmation = recent_volume > avg_volume * 1.5
            
            # Generate signal
            signal = "neutral"
            
            if trend == "bullish" or breakout.startswith("bullish"):
                signal = "bullish"
            elif trend == "bearish" or breakout.startswith("bearish"):
                signal = "bearish"
            
            # Increase confidence if volume confirms
            confidence = 0.6  # Base confidence
            if volume_confirmation:
                confidence = 0.8
            
            return {
                "trend": trend,
                "breakout": breakout,
                "above_ma20": above_ma20,
                "above_ma50": above_ma50,
                "above_ma200": above_ma200,
                "volume_confirmation": volume_confirmation,
                "signal": signal,
                "confidence": confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error checking technical confirmation for {symbol}: {e}")
            return {}
    
    def _get_index_momentum(self, exchange: str) -> Dict[str, float]:
        """
        Get market index momentum.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            Dictionary with index momentum
        """
        try:
            # Determine index symbol based on exchange
            index_symbol = "NIFTY" if exchange == "NSE" else "SENSEX"
            
            # Get index data
            index_data = self._get_market_data(index_symbol, exchange, days=180)
            
            if not index_data or len(index_data) < 63:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(index_data).sort_values("timestamp")
            
            # Calculate momentum
            current_price = df['close'].iloc[-1]
            price_1m_ago = df['close'].iloc[-21] if len(df) >= 21 else df['close'].iloc[0]
            price_3m_ago = df['close'].iloc[-63] if len(df) >= 63 else df['close'].iloc[0]
            price_6m_ago = df['close'].iloc[-126] if len(df) >= 126 else df['close'].iloc[0]
            
            # Calculate momentum percentages
            momentum_1m = ((current_price / price_1m_ago) - 1) * 100
            momentum_3m = ((current_price / price_3m_ago) - 1) * 100
            momentum_6m = ((current_price / price_6m_ago) - 1) * 100
            
            return {
                "momentum_1m": momentum_1m,
                "momentum_3m": momentum_3m,
                "momentum_6m": momentum_6m
            }
            
        except Exception as e:
            self.logger.error(f"Error getting index momentum for {exchange}: {e}")
            return {}
    
    def _calculate_stock_volatility(self, symbol: str, exchange: str) -> Optional[float]:
        """
        Calculate stock volatility.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Volatility percentage
        """
        try:
            # Get market data
            market_data = self._get_market_data(symbol, exchange, days=30)
            
            if not market_data or len(market_data) < 10:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(market_data).sort_values("timestamp")
            
            # Calculate daily returns
            df['returns'] = df['close'].pct_change()
            
            # Calculate volatility (annualized standard deviation)
            daily_volatility = df['returns'].std()
            annualized_volatility = daily_volatility * math.sqrt(252) * 100  # As percentage
            
            return annualized_volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating stock volatility for {symbol}: {e}")
            return None
    
    def _get_sectors(self, exchange: str) -> List[Dict[str, Any]]:
        """
        Get list of sectors.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            List of sectors
        """
        try:
            # Query the sectors collection
            cursor = self.db.sectors_collection.find({
                "exchange": exchange
            })
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting sectors for {exchange}: {e}")
            return []
    
    def _get_sector_price_data(self, sector: str, exchange: str, days: int = 90) -> List[Dict[str, Any]]:
        """
        Get sector price data.
        
        Args:
            sector: Sector name
            exchange: Stock exchange
            days: Number of days to retrieve
            
        Returns:
            List of sector price data
        """
        try:
            # Calculate the start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Create query
            query = {
                "sector": sector,
                "exchange": exchange,
                "timeframe": "day",
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            # Get data from database
            # Get data from database
            cursor = self.db.sector_price_collection.find(query).sort("timestamp", 1)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting sector price data for {sector}: {e}")
            return []
    
    def _get_sector_stocks(self, sector: str, exchange: str) -> List[Dict[str, Any]]:
        """
        Get stocks in a sector.
        
        Args:
            sector: Sector name
            exchange: Stock exchange
            
        Returns:
            List of stocks in the sector
        """
        try:
            # Query the portfolio collection for stocks in the sector
            cursor = self.db.portfolio_collection.find({
                "sector": sector,
                "exchange": exchange,
                "status": "active"
            })
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting stocks in sector {sector}: {e}")
            return []
    
    def _get_stock_info(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get basic stock information.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with stock information
        """
        try:
            # Query the portfolio collection for stock info
            stock_info = self.db.portfolio_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            })
            
            return stock_info or {}
            
        except Exception as e:
            self.logger.error(f"Error getting stock info for {symbol}: {e}")
            return {}